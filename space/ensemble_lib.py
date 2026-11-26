"""Pools for the ensemble ("prompt intersection") prototype: plain PyTorch, no Modal.

Used by modal/app.py on the GPU (Qwen3 14B) and by scripts/ensemble_pool.py on a CPU (stand-in models).
See docs/handoff-ensemble-demo.md (A3, A7) for what the page does with these numbers.

For prompts A and B (and an optional explicit-intersection prompt C), draw N completions i.i.d. from the
token-level product of A and B at temperature T':

    r(y_t | y_<t) ∝ (p_A(y_t | A, y_<t) · p_B(y_t | B, y_<t))^(1/T')

and record, for each completion y (the same token ids under every prompt):

    logpA, logpB, logpC   Σ_t log p_X(y_t | X, y_<t): string-level scores (the global ensemble's inputs)
    logr                  Σ_t log r(y_t | y_<t): the proposal, for importance weights
    loglocal              Σ_t log normalise(p_A · p_B)(y_t | ...): the local (token-level) product ensemble
                          (equal to logr at T' = 1)

A completion is one sentence or one line (see "where a completion ends" below: pySBD's English rules that need
only the next token), or stops at max_new_tokens (then "ended" is false). An ending contributes the probability of
its whole ending set of next tokens, so every score is the exact probability of the completion ending there. The prompts are
plain text continued by the model (no chat template), as in the paper's Fig. 1, unless chat=True.
"""


def prompt_ids(tok, prompt, chat=False, chat_kw=None):
    if chat:
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, **(chat_kw or {}))
    return tok(prompt, add_special_tokens=False).input_ids


def _eos_ids(tok, model):
    """The tokens that end a completion whatever the text: every special or added token of the tokenizer.
    For Qwen3: <|im_end|> (the normal end of a reply), <|endoftext|>, <|im_start|> (a new turn: the reply is
    over even if the model skipped <|im_end|>), <think>/</think>, tool-call, vision and fill-in-the-middle tokens.
    A completion is then ordinary text only, so the text shown is exactly the tokens scored, and its score
    includes the probability that some special token comes next. Nothing is blocked or thrown away, which would
    change the model's distribution."""
    ids = model.generation_config.eos_token_id
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    ids = [i for i in ids if i is not None] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    ids += list(tok.all_special_ids) + [int(i) for i in getattr(tok, "added_tokens_decoder", {})]
    V = model.get_output_embeddings().weight.shape[0]
    return sorted(set(int(i) for i in ids if 0 <= int(i) < V))


# ---------------------------------------------------------------- where a completion ends
# A completion is one sentence, or one line. At every step the possible next tokens split into those that END the
# completion (it ends there; nothing is added) and those that continue it. Which tokens end it depends on how the
# text so far ends (its "ending kind", below), following the English rules of pySBD (the Golden Rules sentence
# splitter, github.com/nipunsadvilkar/pySBD): every rule there that needs only the next word fits, because the
# sampler sees the next token. When an ending token is drawn, every score gets the total probability of that
# kind's ending set, so each score stays the exact probability of the completion ending there.
#
#   kind      the text ends with ...                                    the completion ends if the next token ...
#   "any"     an ordinary word + "." or "?"                              starts with whitespace, or ends the text
#   "upper"   "!" or "...", or an abbreviation that can end a sentence   starts with whitespace + a capital letter,
#             (Inc., etc., Jr., Jan., Calif., U.S., a.m., Ph.D., ...)    or a new line, or ends the text
#   "nodigit" No., Nos., p., pp., Art., Ext.                             starts with whitespace, not before a digit or "("
#   "line"    anything else: mid-sentence, a title (Dr.), an initial     starts a new line, or ends the text
#             (J.), e.g., i.e.
#
# Closing quotes and brackets after the mark are part of it ("Whiskers." ends). A token with a line break after
# other text (".\n") ends the completion at the line break, scored with that token's own probability.
PREPOSITIVE = {"adm", "attys", "brig", "capt", "cmdr", "col", "cpl", "det", "dr", "gen", "gov", "ing", "lt", "maj",
               "mr", "mrs", "ms", "mt", "messrs", "mssrs", "prof", "ph", "rep", "reps", "rev", "sen", "sens", "sgt",
               "st", "supt", "v", "vs", "fig", "fr"}          # pySBD's prepositive list (+ Fr.): always before a name
NUMBER_ABBR = {"art", "ext", "no", "nos", "p", "pp"}           # before a number
# pySBD's general list, without everyday words that end sentences far more often than they abbreviate
# (is, me, man, may, miss, id, ill, con, ok, ore, wash, mass, ken, la, pa, mo, md, ind, min, sec, ...).
ABBREVIATIONS = {"adj", "adv", "al", "ala", "alta", "apr", "ariz", "ark", "assn", "asst", "aug", "ave", "bart",
                 "bld", "bldg", "blvd", "bros", "btw", "calif", "cl", "co", "colo", "comdr", "conn", "corp", "cres",
                 "ct", "dak", "dec", "dept", "dist", "drs", "ens", "esp", "esq", "etc", "exp", "expy", "feb", "fla",
                 "ft", "fwy", "fy", "ga", "hon", "hosp", "hr", "hway", "hwy", "ia", "ida", "inc", "insp", "jan",
                 "jr", "jul", "jun", "kan", "kans", "ky", "ltd", "mar", "mex", "mfg", "mich", "minn", "mlle", "mme",
                 "mont", "msgr", "mtn", "neb", "nebr", "nev", "nov", "nr", "oct", "okla", "ont", "op", "ord", "penn",
                 "penna", "pfc", "pl", "plz", "pvt", "que", "rd", "ref", "res", "rt", "sask", "sep", "sept", "sfc",
                 "sr", "surg", "tce", "tenn", "tex", "univ", "usafa", "ut", "va", "ver", "viz", "vt", "wis", "wisc",
                 "wy", "wyo", "yuk"}
MULTI_ENDING = {"u.s", "u.k", "e.u", "u.s.a", "a.m", "p.m", "ph.d"}   # multi-period abbreviations that can end a sentence
CLOSERS = "\"'”’)]»"


def ending_kind(text):
    """How the text ends, which decides which next tokens end the completion (see the table above)."""
    import re
    t = text.rstrip(CLOSERS)
    if t.endswith("...") or t.endswith("…") or t.endswith("!"):
        return "upper"
    if t.endswith("?"):
        return "any"
    if not t.endswith("."):
        return "line"
    m = re.search(r"(\S+)\.$", t)
    w = (m.group(1) if m else "").lstrip("\"'(“‘[").lower()
    if re.fullmatch(r"[a-z](\.[a-z])+", w):                    # U.S., e.g., a.m., Ph.D (ph.d)
        return "upper" if w in MULTI_ENDING else "line"
    if (len(w) == 1 and w.isalpha() and m and m.group(1).lstrip("\"'(“‘[").isupper()):
        return "line"                                           # an initial: "J."
    if w in PREPOSITIVE:
        return "line"
    if w in NUMBER_ABBR:
        return "nodigit"
    if w in ABBREVIATIONS:
        return "upper"
    return "any"


KINDS = ("any", "upper", "nodigit", "line", "eos")   # "eos": chat replies end only at the end-of-turn token


def ending_masks(tok, model, eos, dev):
    """For each ending kind, the tokens that end the completion (sized to the model's output layer)."""
    import torch
    V = model.get_output_embeddings().weight.shape[0]
    ws, nl, up, dig = (torch.zeros(V, dtype=torch.bool) for _ in range(4))
    for i in range(len(tok)):
        t = tok.decode([i])
        if not t[:1].isspace():
            continue
        ws[i] = True
        lead = t[: len(t) - len(t.lstrip())]
        rest = t.lstrip()
        nl[i] = "\n" in lead
        up[i] = rest[:1].isalpha() and rest[:1].isupper()
        dig[i] = rest[:1].isdigit() or rest[:1] == "("
    e = torch.zeros(V, dtype=torch.bool)
    e[list(eos)] = True
    m = {"any": ws | e, "upper": up | nl | e, "nodigit": (ws & ~dig) | e, "line": nl | e, "eos": e}
    return {k: v.to(dev) for k, v in m.items()}


def _cut_line(text):
    """Text up to its first line break (a token like ".\\n" ends the completion there)."""
    return text.split("\n", 1)[0]


def ensemble_sample(tok, model, A, B, n, t_prime=1.0, max_new_tokens=32, seed=0, batch_size=100,
                    chat=False, chat_kw=None):
    """N completions i.i.d. from the token-level product of prompts A and B (see the module docstring)."""
    import inspect
    import torch
    dev = next(model.parameters()).device
    ia = torch.tensor([prompt_ids(tok, A, chat, chat_kw)], device=dev)
    ib = torch.tensor([prompt_ids(tok, B, chat, chat_kw)], device=dev)
    eos = _eos_ids(tok, model)
    eos_set = set(eos)
    masks = ending_masks(tok, model, eos, dev)
    has_nl = [("\n" in tok.decode([i])) for i in range(len(tok))]
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    cache_kw = "cache" if "cache" in inspect.signature(type(model).forward).parameters else "past_key_values"
    out = []
    with torch.no_grad():
        while len(out) < n:
            Bn = min(batch_size, n - len(out))
            acc = {k: torch.zeros(Bn, dtype=torch.float64) for k in ("A", "B", "r", "local")}
            # Text: the sentence rules above. Chat: a reply ends where the model ends its turn.
            done, kind, end = [False] * Bn, ["eos" if chat else "line"] * Bn, ["max"] * Bn
            end_tok = [None] * Bn                                        # the special token that ended it, if any
            seqs = [[] for _ in range(Bn)]
            pa = pb = None
            sa, sb = ia.repeat(Bn, 1), ib.repeat(Bn, 1)
            for _ in range(max_new_tokens + 1):
                ra = model(input_ids=sa, use_cache=True, **{cache_kw: pa})
                rb = model(input_ids=sb, use_cache=True, **{cache_kw: pb})
                pa = getattr(ra, "cache", None) or ra.past_key_values
                pb = getattr(rb, "cache", None) or rb.past_key_values
                la = torch.log_softmax(ra.logits[:, -1, :].float(), -1)
                lb = torch.log_softmax(rb.logits[:, -1, :].float(), -1)
                local = torch.log_softmax(la + lb, -1)                  # normalised token-level product
                lr = torch.log_softmax((la + lb) / t_prime, -1)         # the proposal at T'
                nxt = torch.multinomial(lr.exp(), 1, generator=gen).squeeze(1)
                L = {"A": la, "B": lb, "r": lr, "local": local}
                tokp = {k: l.gather(1, nxt[:, None]).squeeze(1).double().cpu() for k, l in L.items()}
                # log P(an ending token comes next), per ending kind and distribution
                endp = {kd: {k: torch.logsumexp(l.masked_fill(~masks[kd], float("-inf")), -1).double().cpu()
                             for k, l in L.items()} for kd in KINDS}
                isend = {kd: masks[kd][nxt].cpu().tolist() for kd in KINDS}
                nl = nxt.cpu().tolist()
                for b in range(Bn):
                    if done[b]:
                        continue
                    if isend[kind[b]][b]:
                        for k in acc:
                            acc[k][b] += endp[kind[b]][k][b]
                        done[b], end[b] = True, kind[b]
                        if nl[b] in eos_set:
                            end_tok[b] = tok.convert_ids_to_tokens(nl[b])
                        continue
                    if len(seqs[b]) >= max_new_tokens:
                        done[b] = True                                   # cut off ("max")
                        continue
                    for k in acc:
                        acc[k][b] += tokp[k][b]
                    seqs[b].append(nl[b])
                    if not chat and nl[b] < len(has_nl) and has_nl[nl[b]]:
                        done[b], end[b] = True, "token"                  # e.g. ".\n": ends at the line break
                        continue
                    if not chat:
                        kind[b] = ending_kind(tok.decode(seqs[b]))
                if all(done):
                    break
                sa = sb = nxt[:, None]
            acc = {k: v.tolist() for k, v in acc.items()}
            for b in range(Bn):
                text = tok.decode([t for t in seqs[b] if t not in eos_set], skip_special_tokens=True)
                out.append({
                    "ids": seqs[b], "endKind": end[b] if end[b] in KINDS else None,
                    "text": _cut_line(text) if end[b] == "token" else text,
                    "logpA": acc["A"][b], "logpB": acc["B"][b], "logr": acc["r"][b], "loglocal": acc["local"][b],
                    "nTokens": len(seqs[b]), "ended": end[b] != "max", "endToken": end_tok[b],
                    "end": {"any": "sentence", "upper": "sentence", "nodigit": "sentence", "line": "line",
                            "token": "line", "eos": "eos", "max": "max"}[end[b]],
                })
    return out


def score_ids(tok, model, prompt, seqs, end_kinds=None, batch_size=50, chat=False, chat_kw=None):
    """Σ_t log p(y_t | prompt, y_<t) for each token-id sequence (teacher forcing, right padding), plus, where
    end_kinds[b] is an ending kind, log P(an ending token of that kind comes next) at the end."""
    import torch
    dev = next(model.parameters()).device
    h = prompt_ids(tok, prompt, chat, chat_kw)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    masks = ending_masks(tok, model, _eos_ids(tok, model), dev) if end_kinds and any(end_kinds) else None
    out = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            chunk = seqs[s:s + batch_size]
            L = len(h) + max(len(y) for y in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long, device=dev)
            mask = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
            for b, y in enumerate(chunk):
                ids[b, : len(h) + len(y)] = torch.tensor(h + y, device=dev)
                mask[b, : len(h) + len(y)] = 1
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
            for b, y in enumerate(chunk):
                v = 0.0
                if y:
                    pos = torch.arange(len(h) - 1, len(h) - 1 + len(y), device=dev)
                    v = lp[b, pos, torch.tensor(y, device=dev)].double().sum().item()
                kd = end_kinds[s + b] if end_kinds else None
                if kd:
                    last = lp[b, len(h) + len(y) - 1]
                    v += torch.logsumexp(last.masked_fill(~masks[kd], float("-inf")), -1).item()
                out.append(v)
    return out


CHAT_KW = {"enable_thinking": False}   # Qwen3: answer directly, no reasoning block


def ensemble_pool(tok, model, preset, fmt="text", n=200, t_prime=1.0, max_new_tokens=32, seed=0,
                  chat_kw=CHAT_KW, model_name="", check=True):
    """One preset's pool in one prompt format ("text": continue the prompt as plain text, as in the paper's
    Fig. 1; "chat": the prompts are user messages in the model's chat template and the completions are its
    replies): samples from the A×B product, scored under A, B and (if given) C."""
    chat = fmt == "chat"
    pid, label, preset = preset["id"], preset.get("label", ""), preset[fmt]
    import time
    t0 = time.time()
    S = ensemble_sample(tok, model, preset["A"], preset["B"], n, t_prime, max_new_tokens, seed,
                        chat=chat, chat_kw=chat_kw)
    t1 = time.time()
    seqs, bnd = [s["ids"] for s in S], [s["endKind"] for s in S]
    if preset.get("C"):
        for s, v in zip(S, score_ids(tok, model, preset["C"], seqs, bnd, chat=chat, chat_kw=chat_kw)):
            s["logpC"] = v
    check_err = None
    if check:
        # The sums accumulated while sampling must match teacher-forced scores of the same tokens.
        ra = score_ids(tok, model, preset["A"], seqs, bnd, chat=chat, chat_kw=chat_kw)
        rb = score_ids(tok, model, preset["B"], seqs, bnd, chat=chat, chat_kw=chat_kw)
        check_err = max(max(abs(a - s["logpA"]), abs(b - s["logpB"])) for a, b, s in zip(ra, rb, S))
    for s in S:
        del s["ids"], s["endKind"]
    return {
        "schema": 2, "kind": "ensemble-pool", "id": pid, "label": label, "format": fmt,
        "prompts": {k: preset[k] for k in ("A", "B", "C") if preset.get(k)},
        "model": model_name, "chat": chat, "proposal": "local_product", "Tprime": t_prime,
        "maxNewTokens": max_new_tokens, "stop": "sentence-or-line-v2", "N": len(S), "seed": seed,
        "checkMaxAbsErr": check_err,
        "timing": {"sample_s": round(t1 - t0, 2), "score_s": round(time.time() - t1, 2)},
        "samples": S,
    }
