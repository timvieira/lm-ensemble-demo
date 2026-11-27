"""GPU backend for the ensemble demo ("prompt intersection"), a Hugging Face ZeroGPU Space.

N completions i.i.d. from a proposal, a local f-ensemble of prompts A and B with its own τ', w_A, weights and T'
(ensemble_lib.py's module docstring), each scored under A, B and (if given) C: the pool record of ensemble_lib.py
(schema 3). The page (https://timvieira.github.io/lm-ensemble-demo/) calls the API "ensemble_pool" with the
visitor's Hugging Face sign-in, so ZeroGPU charges the visitor's own daily GPU quota.

ensemble_lib.py started as a copy of dpp-lm-demo's modal/ensemble_lib.py, the Modal backend's, and has moved on:
Modal still draws from the token-level product only (schema 2). The older API "ensemble_sample" (the Modal
backend's arguments) stays for pages loaded before the change (26 Nov 2026): it draws from the product at T' = t_prime / 2,
which is Modal's (p_A p_B)^(1/t_prime) token by token, but with the ending step as one choice (schema 3).

On ZeroGPU it serves Qwen3 14B (the AWQ checkpoint unpacked to fp16, exactly its 4-bit weights, as on Modal and in
the cached pools) and GPT-2. Anywhere else (a CPU, for testing) it serves GPT-2 and Qwen3 0.6B under its own name:
nothing stands in for Qwen3 14B, so asking for it there is an error.
"""
# On ZeroGPU, `spaces` must be imported before torch initializes CUDA.
try:
    import spaces
except Exception:  # running locally
    spaces = None

import os
import random

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ensemble_lib import check_proposal, ensemble_pool, tau_json

try:  # the spaces package knows whether this Space runs on ZeroGPU
    from spaces.config import Config as _SpacesConfig
    ON_ZEROGPU = bool(_SpacesConfig.zero_gpu)
except Exception:
    ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
# On ZeroGPU the GPU only exists inside @spaces.GPU calls; don't probe CUDA at import time there.
DEVICE = "cuda" if ON_ZEROGPU or torch.cuda.is_available() else "cpu"

# key -> (repo, prompt formats). The keys are the page's and the Modal backend's.
GPT2 = {"gpt2": ("openai-community/gpt2", ("text",))}          # the paper's Fig. 1 model; no chat template
LARGE = {"qwen3-14b": ("Qwen/Qwen3-14B-AWQ", ("text", "chat"))}  # 4-bit: the weights of Modal and the cached pools
SMALL = {"qwen3-0.6b": ("Qwen/Qwen3-0.6B", ("text", "chat"))}   # for testing on a CPU only
MODELS = {**(LARGE if ON_ZEROGPU else SMALL), **GPT2}
MAX_N, MAX_TOKENS, MAX_PROMPT = 200, 256, 2000   # the Modal backend's limits


def unpack_awq(qweight, qzeros, scales, group_size, bits=4):
    """AWQ "gemm" packing (AutoAWQ): 8 4-bit values per int32, in the order 0 2 4 6 1 3 5 7.
    Returns the linear layer's weight [out, in] = ((q - zero) * scale)^T, in the scales' dtype.
    (Copied from dpp-lm-demo's modal/app.py and space/app.py.)"""
    shifts = torch.arange(0, 32, bits, device=qweight.device)

    def unpack(x):
        v = torch.bitwise_right_shift(x[:, :, None], shifts[None, None, :]).view(x.shape[0], -1)
        order = torch.arange(v.shape[1], device=x.device).view(-1, 32 // bits)[:, [0, 4, 1, 5, 2, 6, 3, 7]].reshape(-1)
        return torch.bitwise_and(v[:, order], 2**bits - 1)
    q, z = unpack(qweight), unpack(qzeros)
    w = (q - z.repeat_interleave(group_size, 0)).to(scales.dtype) * scales.repeat_interleave(group_size, 0)
    return w.t().contiguous()


def load_awq_unpacked(repo, device, path=None):
    """The AWQ checkpoint as a plain fp16 model: every 4-bit weight unpacked to (q - zero) * scale.
    These are exactly the 4-bit model's weights; only the matrix kernels differ from an int4 engine.
    The model is built on `device` without random initialisation, then filled tensor by tensor.
    (Copied from dpp-lm-demo's modal/app.py and space/app.py.)"""
    import glob
    import json
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.initialization import no_init_weights
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo)
    qc = json.load(open(os.path.join(path, "config.json")))["quantization_config"]
    assert qc["quant_method"] == "awq" and qc["version"] == "gemm" and qc["bits"] == 4 and qc["zero_point"]
    cfg = AutoConfig.from_pretrained(path)
    cfg.__dict__.pop("quantization_config", None)
    with no_init_weights(), torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    params = dict(model.named_parameters())
    done, packed = set(), {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device=str(device)) as st:
            for k in st.keys():
                if k.endswith((".qweight", ".qzeros", ".scales")):
                    base, part = k.rsplit(".", 1)
                    packed.setdefault(base, {})[part] = st.get_tensor(k)
                    t = packed[base]
                    if len(t) == 3:
                        w = unpack_awq(t["qweight"], t["qzeros"], t["scales"].to(torch.float16), qc["group_size"])
                        params[base + ".weight"].data.copy_(w)
                        done.add(base + ".weight")
                        del packed[base], w
                else:
                    params[k].data.copy_(st.get_tensor(k))
                    done.add(k)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
        done |= {k for k in params if k.startswith("lm_head")}
    missing = set(params) - done
    assert not missing and not packed, (sorted(missing)[:5], list(packed)[:5])
    return model.eval()


def load(key):
    """Tokenizer and model. Qwen3 14B in fp16 (its 4-bit weights unpacked), everything else in float32: the
    dtypes of the Modal backend."""
    repo = MODELS[key][0]
    tok = AutoTokenizer.from_pretrained(repo)
    if repo.endswith("-AWQ"):   # unpacked on the CPU; moved to the GPU below
        model = load_awq_unpacked(repo, "cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    if DEVICE == "cuda":
        model.to("cuda")   # ZeroGPU wants models placed on cuda at module level
    return tok, model


LOADED = {k: load(k) for k in MODELS}


def _duration(model, fmt, prompts, n, prop, max_new_tokens, seed):
    """GPU seconds to reserve for one pool: start-up, then sampling (two prompts per step) and scoring (A and B
    again, and C). Not measured on this Space yet; Modal's L40S takes about 15 s for n = 200, 32 tokens."""
    return int(min(240, 30 + 0.4 * int(max_new_tokens) * (1 + int(n) // 100)))


if spaces is not None:
    gpu = spaces.GPU(duration=_duration)  # a no-op outside ZeroGPU
else:
    def gpu(f):
        return f


@gpu
def run(model, fmt, prompts, n, prop, max_new_tokens, seed):
    tok, m = LOADED[model]
    preset = {"id": "custom", "label": "", fmt: prompts}
    return ensemble_pool(tok, m, preset, fmt, n=n, proposal=prop, max_new_tokens=max_new_tokens, seed=seed,
                         model_name=MODELS[model][0])


def _as_gr_error(e):
    # ZeroGPU quota errors keep their own text, with no code; the page recognises them by "quota".
    if "quota" in type(e).__name__.lower() or "quota" in str(e).lower():
        return gr.Error(str(e))
    return gr.Error(f"server: {type(e).__name__}: {e}")


def sample_pool(model: str, fmt: str, A: str, B: str, C: str = "", n: float = 200, tau: str = "0",
                wA: float = 0.5, weights: str = "prefix", t_prime: float = 1.0, max_new_tokens: float = 32,
                seed: float = -1):
    """The API "ensemble_pool": N completions from the proposal (τ' = tau, as a number or "inf", "-inf" or a name;
    w_A = wA; weights "prefix" or "fixed"; T' = t_prime), scored under A, B and (if given) C. seed < 0 draws one.
    Errors start with a code ("bad_request: …", "server: …") so the page can explain them."""
    model, fmt = (model or "").strip(), (fmt or "text").strip()
    if model not in MODELS:
        raise gr.Error(f"bad_request: Unknown model {model!r}. This Space offers: {', '.join(MODELS)}.")
    if fmt not in MODELS[model][1]:
        raise gr.Error(f"bad_request: {model} supports the formats: {', '.join(MODELS[model][1])}.")
    prompts = {k: (v or "").strip() for k, v in (("A", A), ("B", B), ("C", C)) if (v or "").strip()}
    if "A" not in prompts or "B" not in prompts or any(len(v) > MAX_PROMPT for v in prompts.values()):
        raise gr.Error(f"bad_request: Prompts A and B are needed, each at most {MAX_PROMPT} characters (C is optional).")
    if None in (n, wA, t_prime, max_new_tokens) or n != int(n) or max_new_tokens != int(max_new_tokens):
        raise gr.Error("bad_request: n and max_new_tokens must be whole numbers; wA and t_prime must be numbers.")
    n, max_new_tokens = int(n), int(max_new_tokens)
    try:
        prop = check_proposal({"tau": tau if tau not in (None, "") else "0", "wA": wA, "weights": (weights or "").strip(),
                               "Tprime": t_prime})
    except ValueError as e:
        raise gr.Error(f"bad_request: {e}.")
    if not (2 <= n <= MAX_N and 8 <= max_new_tokens <= MAX_TOKENS and 0.1 <= prop["Tprime"] <= 2.0):
        raise gr.Error(f"bad_request: Allowed: 2 ≤ n ≤ {MAX_N}, 8 ≤ max_new_tokens ≤ {MAX_TOKENS}, 0.1 ≤ T′ ≤ 2.")
    seed = random.randrange(2**31) if seed is None or seed < 0 else int(seed)
    print(f"ensemble run: {model} {fmt} n={n} tau'={tau_json(prop['tau'])} wA={prop['wA']} {prop['weights']} "
          f"T'={prop['Tprime']} max={max_new_tokens}", flush=True)
    try:
        return run(model, fmt, prompts, n, prop, max_new_tokens, seed)
    except gr.Error:
        raise
    except Exception as e:  # pass the reason on to API callers instead of a bare "error"
        print(f"ensemble run failed: {type(e).__name__}: {e}", flush=True)
        raise _as_gr_error(e)


def ensemble_sample(model: str, fmt: str, A: str, B: str, C: str = "", n: float = 200, t_prime: float = 1.0,
                    max_new_tokens: float = 32, seed: float = -1):
    """The older API "ensemble_sample" (the Modal backend's arguments): the product proposal with equal weights at
    T' = t_prime / 2, i.e. (p_A p_B)^(1/t_prime) token by token, the ending step as one choice. For pages opened
    before the change (26 Nov 2026)."""
    if t_prime is None or not (0.2 <= float(t_prime) <= 2.0):
        raise gr.Error("bad_request: Allowed: 0.2 ≤ t_prime ≤ 2.")
    return sample_pool(model, fmt, A, B, C, n, "0", 0.5, "fixed", float(t_prime) / 2, max_new_tokens, seed)


def models():
    """The models this Space offers, with their prompt formats."""
    return {k: {"repo": v[0], "formats": list(v[1])} for k, v in MODELS.items()}


with gr.Blocks(title="LM ensemble demo backend") as demo:
    gr.Markdown("Backend for the [prompt-intersection demo](https://timvieira.github.io/lm-ensemble-demo/): "
                "completions drawn from a local ensemble of two prompts, with exact log-probabilities, as JSON.")
    with gr.Row():
        model = gr.Dropdown(choices=list(MODELS), value=next(iter(MODELS)), label="Model")
        fmt = gr.Dropdown(choices=["text", "chat"], value="text", label="Prompt format")
    with gr.Row():
        A = gr.Textbox(label="Prompt A", value="My favorite physicist is")
        B = gr.Textbox(label="Prompt B", value="My favorite author is")
        C = gr.Textbox(label="Prompt C (optional, for checking)", value="My favorite physicist and author is")
    with gr.Row():
        tau = gr.Textbox(value="product", label="Proposal τ′ (a number, -inf, inf, or min/harmonic/product/sum/max)")
        wA = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Weight of A, w_A")
        weights = gr.Dropdown(choices=["prefix", "fixed"], value="prefix", label="Weights (prefix: the paper's)")
        t_prime = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature T′")
    with gr.Row():
        n = gr.Slider(2, MAX_N, value=20, step=1, label="Samples")
        max_new = gr.Slider(8, MAX_TOKENS, value=32, step=8, label="Max tokens")
        seed = gr.Number(value=0, precision=0, label="Seed (< 0: random)")
    out = gr.JSON(label="Pool")
    # The UI button has a name of its own: the page looks the API up by name and must not find this one.
    gr.Button("Sample").click(sample_pool, [model, fmt, A, B, C, n, tau, wA, weights, t_prime, max_new, seed], out,
                              api_name="ensemble_pool_ui", api_visibility="private")
    # The API takes free text and unbounded numbers so that sample_pool, not the dropdowns and sliders, refuses
    # what is outside the limits, with a "bad_request: …" message (their own validation errors have no code).
    model_api, fmt_api, tau_api, weights_api = (gr.Textbox(visible=False) for _ in range(4))
    n_api, wA_api, t_api, max_api, seed_api = (gr.Number(visible=False) for _ in range(5))
    gr.Button("Pool (API)", visible=False).click(
        sample_pool, [model_api, fmt_api, A, B, C, n_api, tau_api, wA_api, weights_api, t_api, max_api, seed_api],
        out, api_name="ensemble_pool")
    gr.Button("Sample (older API)", visible=False).click(
        ensemble_sample, [model_api, fmt_api, A, B, C, n_api, t_api, max_api, seed_api], out, api_name="ensemble_sample")
    gr.Button("List models", visible=False).click(models, None, out, api_name="models")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(show_error=True)
