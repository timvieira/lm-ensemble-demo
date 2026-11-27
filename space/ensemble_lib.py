"""Pools for the ensemble ("prompt intersection") prototype: plain PyTorch, no Modal.

This copy (lm-ensemble-demo's space/ensemble_lib.py) has moved on from dpp-lm-demo's modal/ensemble_lib.py: its
proposal is a family of local f-ensembles (schema 3). dpp-lm-demo's copy, used by Modal and by the cached pools,
still draws from the token-level product with its ending step summed token by token (schema 2).

For prompts A and B (and an optional explicit-intersection prompt C), draw N completions i.i.d. from a proposal r,
a local f-ensemble of A and B, so the page can reweight them by importance sampling to any global generalised mean
φ_τ(y) = (½ p_A(y)^τ + ½ p_B(y)^τ)^(1/τ) (Chan et al. 2026, "Ensembling Language Models with Sequential Monte
Carlo", Def. 5.1: weights φ_τ(y)/r(y)). The proposal has its own power τ', weight w_A (w_B = 1 − w_A) and
temperature T'. After the text so far x, the next token y' is drawn with

    r(y' | x) ∝ ( ŵ_A p_A(y' | A, x)^τ' + ŵ_B p_B(y' | B, x)^τ' )^(1/(τ' T'))

    weights "prefix":  ŵ_k ∝ w_k p_k(x)^τ', p_k(x) the probability of the text so far after prompt k. This is the
                       paper's proposal, its shaping function ψ(x' | x) = f(p_A(x x'), p_B(x x')) / f(p_A(x), p_B(x))
                       (§5.2, Prop. 5.2; prefix probabilities), here raised to 1/T'.
    weights "fixed":   ŵ_k = w_k at every step.

τ' = 0 is the weighted geometric mean (both weightings agree there), τ' = ±∞ are max and min over the prompts with
nonzero weight; w_A = 1 or 0 draws from A or B alone. "The completion ends here" is one choice, as the paper's end
symbol: its score is the same mean of each prompt's total probability of ending here (see "where a completion
ends"), not a sum of per-token scores. T' is this demo's own addition: the paper samples at temperature 1.

Recorded for each completion y (the same token ids under every prompt):

    logpA, logpB, logpC   log p_X(y | X): string-level scores (the global ensemble's inputs), ending included
    logr                  log r(y): the proposal's probability of drawing y, for importance weights
    loglocal              Σ_t log normalise(p_A · p_B)(y_t | ...): the local (token-level) product ensemble of the
                          paper's Fig. 1, its ending summed over the ending tokens (unchanged from schema 2)

A completion is one sentence or one line (see "where a completion ends" below: pySBD's English rules that need
only the next token), or stops at max_new_tokens (then "ended" is false). An ending contributes the probability of
its whole ending set of next tokens, so every score is the exact probability of the completion ending there. The
prompts are plain text continued by the model (no chat template), as in the paper's Fig. 1, unless chat=True.
"""
import math

NAMED_TAU = {"min": -math.inf, "harmonic": -1.0, "product": 0.0, "geometric": 0.0, "sum": 1.0, "mixture": 1.0,
             "quadratic": 2.0, "max": math.inf}
MAX_FINITE_TAU = 1000.0
# The default here: the product at T' = 1, √(p_A p_B) token by token. (The cached pools' p_A · p_B token by token
# is the product at T' = ½, drawn by the schema-2 sampler, whose ending step differs; see the module docstring.)
PROPOSAL_DEFAULT = {"tau": 0.0, "wA": 0.5, "weights": "prefix", "Tprime": 1.0}


def parse_tau(v):
    """τ' from a number or a string: a number, "inf", "-inf" (also ∞, −∞), or a name (min, harmonic, product, sum,
    quadratic, max). Raises ValueError for anything else, and for finite values beyond ±MAX_FINITE_TAU."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        t = float(v)
    else:
        s = str(v).strip().lower().replace("−", "-").replace("∞", "inf").replace("infinity", "inf")
        if s in NAMED_TAU:
            t = NAMED_TAU[s]
        elif s in ("inf", "+inf", "-inf"):
            t = -math.inf if s.startswith("-") else math.inf
        else:
            try:
                t = float(s)
            except ValueError:
                raise ValueError(f"τ' must be a number, -inf, inf, or one of {', '.join(NAMED_TAU)}; got {v!r}")
    if math.isnan(t) or (math.isfinite(t) and abs(t) > MAX_FINITE_TAU):
        raise ValueError(f"τ' must be a number between −{MAX_FINITE_TAU:g} and {MAX_FINITE_TAU:g}, or ±∞")
    return t


def tau_json(t):
    """τ' for a JSON record: JSON has no infinity, so ±∞ are the strings "inf" and "-inf"."""
    return t if math.isfinite(t) else ("inf" if t > 0 else "-inf")


def check_proposal(p):
    """The proposal's parameters, checked: {"tau": float, "wA": float in [0, 1], "weights": "prefix"|"fixed",
    "Tprime": float > 0}. Raises ValueError."""
    q = {"tau": parse_tau(p.get("tau", 0.0)), "wA": float(p.get("wA", 0.5)),
         "weights": str(p.get("weights", "prefix")), "Tprime": float(p.get("Tprime", 1.0))}
    if not 0.0 <= q["wA"] <= 1.0:
        raise ValueError("w_A must be between 0 and 1")
    if q["weights"] not in ("prefix", "fixed"):
        raise ValueError('weights must be "prefix" or "fixed"')
    if not (q["Tprime"] > 0 and math.isfinite(q["Tprime"])):
        raise ValueError("T' must be a positive number")
    return q


def local_f(la, lb, tau, wA, LA=None, LB=None):
    """Log of the generalised mean (ŵ_A p_A^τ + ŵ_B p_B^τ)^(1/τ), elementwise, of two log-probability tensors
    la, lb of shape [rows, X], up to a constant per row. Fixed weights ŵ = (w_A, 1 − w_A) if LA is None; with
    LA, LB (the log-probabilities of each row's prefix under A and B, float64 [rows]), the prefix-following
    weights ŵ_k ∝ w_k p_k(prefix)^τ. τ = 0: the weighted geometric mean (both weightings agree). τ = ±∞: max and
    min over the prompts with nonzero weight. A prompt with weight 0 is left out."""
    import torch
    wB = 1.0 - wA
    if wB == 0.0:
        return la
    if wA == 0.0:
        return lb
    if tau == 0.0:
        return wA * la + wB * lb
    if math.isinf(tau):
        a = la if LA is None else la + (LA - LB).to(la.dtype)[:, None]   # p_A(prefix) p_A(x') vs p_B(prefix) p_B(x')
        return torch.maximum(a, lb) if tau > 0 else torch.minimum(a, lb)
    if LA is None:
        lwA, lwB = math.log(wA), math.log(wB)
    else:   # log ŵ in float64, normalised per row, then in the logits' dtype
        lwA, lwB = math.log(wA) + tau * LA, math.log(wB) + tau * LB
        z = torch.logaddexp(lwA, lwB)
        lwA, lwB = (lwA - z).to(la.dtype)[:, None], (lwB - z).to(la.dtype)[:, None]
    return torch.logaddexp(lwA + tau * la, lwB + tau * lb) / tau


def proposal_logprobs(la, lb, ends, prop, LA=None, LB=None):
    """The proposal's next-step log-probabilities, [rows, V + 1]: column v < V is continuing with token v
    (−inf where token v would end the completion), column V is ending here. la, lb: next-token log-probabilities
    under A and B, [rows, V]; ends: bool [rows, V], the tokens that end each row's completion here; LA, LB: the
    prefixes' log-probabilities (float64 [rows]), needed for weights "prefix". prop: check_proposal's dict.
    Ending is one choice, scored with the mean of each prompt's total probability of the ending tokens."""
    import torch
    ninf = float("-inf")
    pre = prop["weights"] == "prefix"
    if pre and (LA is None or LB is None):
        raise ValueError('weights "prefix" needs the prefixes\' log-probabilities LA and LB')
    LA, LB = (LA, LB) if pre else (None, None)
    EA = torch.logsumexp(la.masked_fill(~ends, ninf), -1, keepdim=True)
    EB = torch.logsumexp(lb.masked_fill(~ends, ninf), -1, keepdim=True)
    s_tok = local_f(la, lb, prop["tau"], prop["wA"], LA, LB).masked_fill(ends, ninf)
    s_end = local_f(EA, EB, prop["tau"], prop["wA"], LA, LB)
    lr = torch.log_softmax(torch.cat([s_tok, s_end], -1) / prop["Tprime"], -1)
    if torch.isnan(lr).any():
        raise FloatingPointError("the proposal's probabilities came out NaN")
    return lr


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


def ensemble_sample(tok, model, A, B, n, proposal=None, max_new_tokens=32, seed=0, batch_size=100,
                    chat=False, chat_kw=None):
    """N completions i.i.d. from the proposal, a local f-ensemble of prompts A and B (see the module docstring)."""
    import inspect
    import torch
    prop = check_proposal(proposal if proposal is not None else PROPOSAL_DEFAULT)
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
                V = la.shape[1]
                local = torch.log_softmax(la + lb, -1)                  # normalised token-level product (Fig. 1)
                ends = torch.stack([masks[k] for k in kind])            # [Bn, V]: the tokens that end each one here
                # The proposal over [continue with token v | end here]; the prefixes' log-probabilities so far.
                lr = proposal_logprobs(la, lb, ends, prop, acc["A"].to(dev), acc["B"].to(dev))
                pick = torch.multinomial(lr.exp(), 1, generator=gen).squeeze(1)
                isend = (pick == V).cpu().tolist()
                nxt = pick.clamp(max=V - 1)                              # a token to feed the finished ones
                tokp = {k: l.gather(1, nxt[:, None]).squeeze(1).double().cpu()
                        for k, l in {"A": la, "B": lb, "local": local}.items()}
                tokp["r"] = lr.gather(1, pick[:, None]).squeeze(1).double().cpu()   # continuing or ending
                # log P(an ending token comes next) under A, B and the local product, for each one's ending kind
                endp = {k: torch.logsumexp(l.masked_fill(~ends, float("-inf")), -1).double().cpu()
                        for k, l in {"A": la, "B": lb, "local": local}.items()}
                endp["r"] = tokp["r"]
                nl = nxt.cpu().tolist()
                for b in range(Bn):
                    if done[b]:
                        continue
                    if isend[b]:
                        for k in acc:
                            acc[k][b] += endp[k][b]
                        done[b], end[b] = True, kind[b]
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
                    "nTokens": len(seqs[b]), "ended": end[b] != "max",
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


def ensemble_pool(tok, model, preset, fmt="text", n=200, proposal=None, max_new_tokens=32, seed=0,
                  chat_kw=CHAT_KW, model_name="", check=True):
    """One preset's pool in one prompt format ("text": continue the prompt as plain text, as in the paper's
    Fig. 1; "chat": the prompts are user messages in the model's chat template and the completions are its
    replies): samples from the proposal (a local f-ensemble of A and B), scored under A, B and (if given) C."""
    chat = fmt == "chat"
    pid, label, preset = preset["id"], preset.get("label", ""), preset[fmt]
    prop = check_proposal(proposal if proposal is not None else PROPOSAL_DEFAULT)
    import time
    t0 = time.time()
    S = ensemble_sample(tok, model, preset["A"], preset["B"], n, prop, max_new_tokens, seed,
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
        "schema": 3, "kind": "ensemble-pool", "id": pid, "label": label, "format": fmt,
        "prompts": {k: preset[k] for k in ("A", "B", "C") if preset.get(k)},
        "model": model_name, "chat": chat,
        "proposal": {"kind": "local_f", "tau": tau_json(prop["tau"]), "wA": prop["wA"], "weights": prop["weights"],
                     "Tprime": prop["Tprime"], "end": "one choice"},
        "maxNewTokens": max_new_tokens, "stop": "sentence-or-line-v2", "N": len(S), "seed": seed,
        "checkMaxAbsErr": check_err,
        "timing": {"sample_s": round(t1 - t0, 2), "score_s": round(time.time() - t1, 2)},
        "samples": S,
    }
