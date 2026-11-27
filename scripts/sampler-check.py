"""End-to-end check of the Space's sampler (space/ensemble_lib.py) with GPT-2 on a CPU.

For several proposals, draw a few completions with ensemble_sample (batched, with the KV cache), then recompute
each completion's logr, logpA and logpB independently: one full forward pass per step (no cache, one completion at
a time), the ending kind from the decoded text, and the proposal from proposal_logprobs (its maths is checked
against the definition in tests/test_proposal.py). Fails if any recorded value is off by more than 1e-3 nats.

    python3 scripts/sampler-check.py [model]     (default openai-community/gpt2; downloads it if needed)
"""
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "space"))
from ensemble_lib import (_eos_ids, check_proposal, ending_kind, ending_masks, ensemble_sample,  # noqa: E402
                          prompt_ids, proposal_logprobs)

repo = sys.argv[1] if len(sys.argv) > 1 else "openai-community/gpt2"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
A, B = "My favorite physicist is", "My favorite author is"
masks = ending_masks(tok, model, _eos_ids(tok, model), "cpu")


def recompute(ids, end, prop):
    """logr, logpA, logpB of one completion, step by step from scratch."""
    ha, hb = prompt_ids(tok, A), prompt_ids(tok, B)
    LA = LB = R = 0.0
    with torch.no_grad():
        for t in range(len(ids) + 1):
            if t == len(ids) and end in ("max", "token"):   # cut off, or ended by a token with a line break
                break
            la = torch.log_softmax(model(input_ids=torch.tensor([ha + ids[:t]])).logits[0, -1].float(), -1)[None]
            lb = torch.log_softmax(model(input_ids=torch.tensor([hb + ids[:t]])).logits[0, -1].float(), -1)[None]
            kind = ending_kind(tok.decode(ids[:t])) if t else "line"
            e = masks[kind][None]
            lr = proposal_logprobs(la, lb, e, prop, torch.tensor([LA], dtype=torch.float64),
                                   torch.tensor([LB], dtype=torch.float64))[0]
            if t == len(ids):                                # ended here: the "end" choice
                ninf = float("-inf")
                return (R + lr[-1].item(), LA + torch.logsumexp(la[0].masked_fill(~e[0], ninf), -1).item(),
                        LB + torch.logsumexp(lb[0].masked_fill(~e[0], ninf), -1).item())
            assert not e[0, ids[t]], "a continuing token is in the ending set"
            R += lr[ids[t]].item()
            LA += la[0, ids[t]].item()
            LB += lb[0, ids[t]].item()
    return R, LA, LB


worst = 0.0
for p in ({"tau": 0.0, "wA": 0.5, "weights": "prefix", "Tprime": 1.0},
          {"tau": 0.0, "wA": 0.5, "weights": "fixed", "Tprime": 0.5},
          {"tau": -math.inf, "wA": 0.5, "weights": "prefix", "Tprime": 1.0},
          {"tau": -1.0, "wA": 0.3, "weights": "prefix", "Tprime": 1.3},
          {"tau": 1.0, "wA": 0.5, "weights": "prefix", "Tprime": 1.0},
          {"tau": 1.0, "wA": 0.5, "weights": "fixed", "Tprime": 1.0},
          {"tau": math.inf, "wA": 0.5, "weights": "prefix", "Tprime": 1.0},
          {"tau": 2.0, "wA": 1.0, "weights": "fixed", "Tprime": 1.0}):
    prop = check_proposal(p)
    S = ensemble_sample(tok, model, A, B, 6, prop, max_new_tokens=12, seed=3, batch_size=3)
    err = 0.0
    for s in S:
        how = "max" if not s["ended"] else ("token" if s["endKind"] is None else "choice")   # endKind None: ".\n"
        R, LA, LB = recompute(s["ids"], how, prop)
        err = max(err, abs(R - s["logr"]), abs(LA - s["logpA"]), abs(LB - s["logpB"]))
    ends = sorted({s["end"] for s in S})
    print(f"{str(p):75s} max err {err:.1e}  ends {ends}  e.g. {S[0]['text']!r}")
    worst = max(worst, err)
assert worst < 1e-3, worst
print(f"OK: worst {worst:.1e} nats")
