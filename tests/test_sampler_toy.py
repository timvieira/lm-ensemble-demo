"""The Space's sampling loop (space/ensemble_lib.py: ensemble_sample, ensemble_pool) on a toy tokenizer and model
small enough to list every complete outcome. Run: python3 tests/test_sampler_toy.py (needs torch and numpy).

tests/test_proposal.py checks one step of the proposal; this checks the loop around it: how the steps compose into
the probability of a whole completion under the three LM → LM transformations (the prompt; the ending rules, with
any special token ending; Max tokens, after which the end has probability 1). The reference below enumerates the
transformed LMs directly, in float64, from the same toy logits. The ending rules themselves (ending_kind,
ending_masks) are used as given: they are the rules' definition, not what is tested here.

Checked, for several proposals:
  1. the reference's outcomes are a distribution under A, B and the proposal (each sums to 1);
  2. every drawn completion is one of the reference's outcomes, and its recorded logpA, logpB, logr and loglocal
     equal the reference's (so, e.g., a completion that reaches Max tokens ends there, with nothing drawn);
  3. the draws' frequencies agree with the proposal's probabilities;
  4. ensemble_pool's self-check (sampling sums against teacher-forced scores) is near 0.
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "space"))
from ensemble_lib import (check_proposal, ending_kind, ending_masks, ensemble_pool,  # noqa: E402
                          ensemble_sample)

# token 0 is the special end token; "\n" always ends; ".\n" ends at its line break; " etc" + "." makes kind "upper"
VOCAB = ["<e>", " the", " cat", ".", " Dog", "\n", ".\n", " sat", " etc"]
V, L = len(VOCAB), 3
PROMPTS = {"pa": [1, 2], "pb": [4, 7], "pc": [2]}


class Tok:
    eos_token_id = pad_token_id = 0
    all_special_ids = [0]
    added_tokens_decoder = {}

    def __len__(self):
        return V

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids))

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(PROMPTS[text]))


class Model(torch.nn.Module):
    """Next-token logits: a fixed random vector for each full token sequence (prompt + completion so far)."""

    def __init__(self, seed=0):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.rng, self.table = np.random.default_rng(seed), {}

    def get_output_embeddings(self):
        return SimpleNamespace(weight=torch.zeros(V, 1))

    def logits_for(self, seq):
        k = tuple(seq)
        if k not in self.table:
            self.table[k] = self.rng.normal(size=V) * 1.5
        return self.table[k]

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        prev = past_key_values if past_key_values is not None else input_ids[:, :0]
        full = torch.cat([prev, input_ids], 1)
        off = prev.shape[1]
        lg = [[self.logits_for(full[b, : off + t + 1].tolist()) for t in range(input_ids.shape[1])]
              for b in range(input_ids.shape[0])]
        return SimpleNamespace(logits=torch.tensor(np.array(lg), dtype=torch.float32), past_key_values=full)


TOK, MODEL = Tok(), Model()
MASKS = {k: v.numpy() for k, v in ending_masks(TOK, MODEL, [0], "cpu").items()}
HAS_NL = ["\n" in t for t in VOCAB]


def softmax(z):
    z = np.exp(z - z.max())
    return z / z.sum()


def gm(a, b, tau, wA):
    """Weighted generalised mean of nonnegative arrays (as in tests/test_proposal.py)."""
    wB = 1 - wA
    if wB == 0:
        return a
    if wA == 0:
        return b
    if tau == 0:
        return a**wA * b**wB
    if math.isinf(tau):
        return np.maximum(a, b) if tau > 0 else np.minimum(a, b)
    with np.errstate(divide="ignore"):
        return (wA * a**tau + wB * b**tau) ** (1 / tau)


def reference(prop):
    """Every complete outcome of the transformed LMs: ids -> (log p_A, log p_B, log r, log local), float64."""
    out = {}

    def go(x, pre):    # pre: probabilities so far under A, B, the proposal, the local product
        if len(x) == L:                                     # Max tokens: the end, with probability 1
            out[tuple(x)] = pre
            return
        pa = softmax(MODEL.logits_for(PROMPTS["pa"] + x))
        pb = softmax(MODEL.logits_for(PROMPTS["pb"] + x))
        loc = softmax(np.log(pa) + np.log(pb))
        e = MASKS["line" if not x else ending_kind(TOK.decode(x))]
        if prop["weights"] == "prefix":
            a, b = pre[0] * np.r_[pa, pa[e].sum()], pre[1] * np.r_[pb, pb[e].sum()]
        else:
            a, b = np.r_[pa, pa[e].sum()], np.r_[pb, pb[e].sum()]
        s = gm(a, b, prop["tau"], prop["wA"])
        s[:V][e] = 0.0
        q = s ** (1 / prop["Tprime"])
        q = q / q.sum()
        out[tuple(x)] = (pre[0] * pa[e].sum(), pre[1] * pb[e].sum(), pre[2] * q[V], pre[3] * loc[e].sum())
        for v in np.flatnonzero(~e):
            nxt = (pre[0] * pa[v], pre[1] * pb[v], pre[2] * q[v], pre[3] * loc[v])
            if HAS_NL[v]:                                   # ".\n": the completion ends at the line break
                out[tuple(x + [int(v)])] = nxt
            else:
                go(x + [int(v)], nxt)

    go([], (1.0, 1.0, 1.0, 1.0))
    return {k: tuple(np.log(v)) for k, v in out.items()}


PROPOSALS = [{"tau": 0.0, "wA": 0.5, "weights": "prefix", "Tprime": 1.0},
             {"tau": -math.inf, "wA": 0.5, "weights": "prefix", "Tprime": 0.7},
             {"tau": 1.0, "wA": 0.3, "weights": "fixed", "Tprime": 1.3},
             {"tau": -1.0, "wA": 0.5, "weights": "prefix", "Tprime": 1.0},
             {"tau": 2.0, "wA": 1.0, "weights": "prefix", "Tprime": 1.0}]


def test_reference_is_a_distribution():
    for p in PROPOSALS:
        ref = reference(check_proposal(p))
        for i in range(3):
            tot = sum(math.exp(v[i]) for v in ref.values())
            assert abs(tot - 1) < 1e-12, (p, i, tot)
        assert any(len(k) == L for k in ref) and any(len(k) < L for k in ref)


def test_draws_match_the_reference():
    N = 4000
    for p in PROPOSALS:
        prop = check_proposal(p)
        ref = reference(prop)
        S = ensemble_sample(TOK, MODEL, "pa", "pb", N, prop, max_new_tokens=L, seed=1, batch_size=500)
        worst, counts = 0.0, {}
        for s in S:
            k = tuple(s["ids"])
            assert k in ref, (p, k, s["end"])
            rA, rB, rr, rl = ref[k]
            worst = max(worst, abs(s["logpA"] - rA), abs(s["logpB"] - rB), abs(s["logr"] - rr),
                        abs(s["loglocal"] - rl))
            assert (s["end"] == "max") == (len(k) == L and not HAS_NL[k[-1]]), (k, s["end"])
            counts[k] = counts.get(k, 0) + 1
        assert worst < 1e-5, (p, worst)                     # float32 logits in the library
        for k, v in ref.items():                            # frequencies: within 4.5 standard errors
            r = math.exp(v[2])
            assert abs(counts.get(k, 0) / N - r) < 4.5 * math.sqrt(r * (1 - r) / N) + 1e-3, (p, k, counts.get(k, 0), r)
        print(f"  {p}: {len(ref)} outcomes, {len(counts)} drawn, worst log error {worst:.1e}, "
              f"{sum(len(k) == L for k in counts)} distinct ending at Max tokens")


def test_pool_self_check():
    preset = {"id": "toy", "text": {"A": "pa", "B": "pb", "C": "pc"}}
    rec = ensemble_pool(TOK, MODEL, preset, "text", n=300, proposal=PROPOSALS[0], max_new_tokens=L, seed=2)
    assert rec["schema"] == 3 and rec["N"] == 300 and rec["checkMaxAbsErr"] < 1e-5, rec["checkMaxAbsErr"]
    assert all(s["end"] in ("sentence", "line", "eos", "max") and "ended" not in s for s in rec["samples"])
    assert all(math.isfinite(s["logpC"]) for s in rec["samples"])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"ok  {name}")
