"""The Space's proposal (space/ensemble_lib.py: local_f, proposal_logprobs) on a toy model small enough to enumerate
every string. Run: python3 tests/test_proposal.py (needs torch and numpy; no model download).

The toy: two "prompts" are two random next-token tables over V tokens, keyed by the prefix. At each prefix some
tokens end the completion (an ending set that depends on the prefix, like the page's sentence rules), and at
length L every token ends it. A complete string's probability under prompt k is the product of its tokens'
probabilities and its ending set's total probability, as in the sampler's records.

Checked, for many proposals (τ', w_A, weights, T'):
  1. the proposal's probabilities of all complete strings add up to 1;
  2. each step's probabilities equal an independent NumPy implementation of the definition (module docstring of
     ensemble_lib): the paper's shaping function ψ for weights "prefix", fixed weights otherwise, ending as one
     choice scored with each prompt's total ending probability;
  3. τ' = 0: "prefix" and "fixed" are the same proposal;
  4. τ' = 1, w_A = ½, "prefix", T' = 1 is exactly the global mixture ½ p_A + ½ p_B (all weights equal);
  5. w_A = 1 is prompt A alone (at T' = 1, exactly p_A).
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "space"))
from ensemble_lib import check_proposal, local_f, parse_tau, proposal_logprobs  # noqa: E402

V, L = 4, 3
rng = np.random.default_rng(1)
TABLE = {}


def cond(k, x):
    """Next-token probabilities after prefix x under prompt k (fixed random tables)."""
    if (k, x) not in TABLE:
        TABLE[(k, x)] = rng.dirichlet([0.6] * V)
    return TABLE[(k, x)]


def ends(x):
    """The tokens that end the completion after prefix x: token 0 always, token 1 after an odd-length prefix,
    every token at length L."""
    e = np.zeros(V, bool)
    e[0] = True
    e[1] = len(x) % 2 == 1
    if len(x) == L:
        e[:] = True
    return e


def complete_strings():
    """Every complete string: (continuing tokens, the prefixes' ending sets as decided at each step)."""
    out = []

    def go(x):
        out.append(x)                                   # ending right after x
        if len(x) < L:
            for v in np.flatnonzero(~ends(x)):
                go(x + (int(v),))
    go(())
    return out


def string_prob(k, y):
    p = 1.0
    for t in range(len(y)):
        p *= cond(k, y[:t])[y[t]]
    return p * cond(k, y)[ends(y)].sum()


def gm(a, b, tau, wA):
    """Reference generalised mean of nonnegative numbers, weights (wA, 1 − wA), float64."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wB = 1 - wA
    if wB == 0:
        return a
    if wA == 0:
        return b
    if tau == 0:
        return a**wA * b**wB
    if tau == math.inf:
        return np.maximum(a, b)
    if tau == -math.inf:
        return np.minimum(a, b)
    with np.errstate(divide="ignore"):
        return (wA * a**tau + wB * b**tau) ** (1 / tau)


def reference_step(x, prop, preA, preB):
    """The definition, directly: unnormalised scores of [continue with v | end], then ^(1/T'), normalised."""
    pa, pb, e = cond("A", x), cond("B", x), ends(x)
    if prop["weights"] == "prefix":    # ψ(x'|x) = f(p_A(x x'), p_B(x x')) / f(p_A(x), p_B(x)): f of prefix probabilities
        a, b = preA * np.r_[pa, pa[e].sum()], preB * np.r_[pb, pb[e].sum()]
    else:
        a, b = np.r_[pa, pa[e].sum()], np.r_[pb, pb[e].sum()]
    s = gm(a, b, prop["tau"], prop["wA"])
    s[:V][e] = 0.0
    s = s ** (1 / prop["Tprime"])
    return s / s.sum()


def lib_step(x, prop, preA, preB):
    la = torch.tensor(np.log(cond("A", x)), dtype=torch.float32)[None]
    lb = torch.tensor(np.log(cond("B", x)), dtype=torch.float32)[None]
    e = torch.tensor(ends(x))[None]
    LA = torch.tensor([math.log(preA)], dtype=torch.float64)
    LB = torch.tensor([math.log(preB)], dtype=torch.float64)
    return proposal_logprobs(la, lb, e, prop, LA, LB)[0].double().exp().numpy()


def proposal_prob(y, prop, step):
    r, preA, preB = 1.0, 1.0, 1.0
    for t in range(len(y) + 1):
        q = step(y[:t], prop, preA, preB)
        if t == len(y):
            return r * q[V]
        r *= q[y[t]]
        preA *= cond("A", y[:t])[y[t]]
        preB *= cond("B", y[:t])[y[t]]


STRINGS = complete_strings()
PA = np.array([string_prob("A", y) for y in STRINGS])
PB = np.array([string_prob("B", y) for y in STRINGS])


def test_toy_model_is_a_distribution():
    assert len(STRINGS) > 20
    assert abs(PA.sum() - 1) < 1e-12 and abs(PB.sum() - 1) < 1e-12


def proposals():
    for tau in (-math.inf, -3.0, -1.0, -0.4, 0.0, 0.5, 1.0, 2.0, 7.0, math.inf):
        for wA in (0.0, 0.3, 0.5, 1.0):
            for weights in ("prefix", "fixed"):
                for Tp in (0.5, 1.0, 1.7):
                    yield check_proposal({"tau": tau, "wA": wA, "weights": weights, "Tprime": Tp})


def test_steps_match_the_definition_and_strings_sum_to_one():
    worst = 0.0
    for prop in proposals():
        r = np.array([proposal_prob(y, prop, lib_step) for y in STRINGS])
        ref = np.array([proposal_prob(y, prop, reference_step) for y in STRINGS])
        assert abs(r.sum() - 1) < 1e-5, (prop, r.sum())
        worst = max(worst, float(np.max(np.abs(np.log(r) - np.log(ref)))))
    assert worst < 1e-4, worst     # float32 logits in the library, float64 in the reference
    print(f"  worst |log r − log r_ref| over all proposals and strings: {worst:.1e}")


def test_prefix_equals_fixed_at_the_product():
    for wA in (0.2, 0.5, 0.9):
        for Tp in (0.5, 1.0):
            p = {"tau": 0.0, "wA": wA, "Tprime": Tp}
            r1 = [proposal_prob(y, check_proposal({**p, "weights": "prefix"}), lib_step) for y in STRINGS]
            r2 = [proposal_prob(y, check_proposal({**p, "weights": "fixed"}), lib_step) for y in STRINGS]
            assert np.allclose(r1, r2, rtol=1e-6, atol=0)


def test_the_paper_proposal_is_exact_for_the_mixture():
    prop = check_proposal({"tau": 1.0, "wA": 0.5, "weights": "prefix", "Tprime": 1.0})
    r = np.array([proposal_prob(y, prop, lib_step) for y in STRINGS])
    w = (0.5 * PA + 0.5 * PB) / r                      # importance weights φ_1 / r
    assert np.max(np.abs(np.log(w))) < 1e-5, np.max(np.abs(np.log(w)))
    fixed = check_proposal({**prop, "weights": "fixed"})
    rf = np.array([proposal_prob(y, fixed, lib_step) for y in STRINGS])
    wf = (0.5 * PA + 0.5 * PB) / rf
    ess = lambda v: (r * v).sum() ** 2 / (r * v**2).sum()   # large-n ESS/n under the proposal: 1 / (1 + χ²)
    print(f"  mixture target: spread of log w, prefix {np.ptp(np.log(w)):.1e}, fixed {np.ptp(np.log(wf)):.2f};"
          f" large-n ESS/n, prefix {ess(w):.3f}, fixed {(rf * wf).sum() ** 2 / (rf * wf**2).sum():.3f}")
    assert np.ptp(np.log(wf)) > 0.1                    # fixed weights are not exact here


def test_one_prompt_alone():
    for weights in ("prefix", "fixed"):
        for tau in (-1.0, 0.0, 2.0, math.inf):
            prop = check_proposal({"tau": tau, "wA": 1.0, "weights": weights, "Tprime": 1.0})
            r = np.array([proposal_prob(y, prop, lib_step) for y in STRINGS])
            assert np.allclose(r, PA, rtol=1e-5, atol=0)


def test_parse_tau():
    assert parse_tau("min") == -math.inf and parse_tau("−∞") == -math.inf and parse_tau("inf") == math.inf
    assert parse_tau("product") == 0 and parse_tau(" -1 ") == -1 and parse_tau(2) == 2.0
    for bad in ("", "nan", "abc", "1e9"):
        try:
            parse_tau(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_tau accepted {bad!r}")


def test_local_f_limits_and_zero_weights():
    la = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
    lb = torch.log(torch.tensor([[0.6, 0.1, 0.3]]))
    for tau in (40.0, -40.0):
        lim = local_f(la, lb, math.copysign(math.inf, tau), 0.5)
        assert torch.allclose(local_f(la, lb, tau, 0.5), lim, atol=0.02)
    assert torch.equal(local_f(la, lb, -2.0, 0.0), lb) and torch.equal(local_f(la, lb, 3.0, 1.0), la)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print(f"ok  {name}")
