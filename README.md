# Ensembling language models: demo (prompt intersection)

An interactive companion to
[*Ensembling Language Models with Sequential Monte Carlo*](https://arxiv.org/abs/2603.05432)
(Chan, Liu, Kiegeland, Pasti, Hoover Vigly, O'Donnell, Cotterell, Vieira), built on its Figure 1.
A static page on GitHub Pages: https://timvieira.github.io/lm-ensemble-demo/

It started as a lab page of [dpp-lm-demo](https://github.com/timvieira/dpp-lm-demo) (`lab/ensemble.html`,
now a redirect here) and moved on 2026-11-26 from dpp-lm-demo@5d42446, with the history of its files.
The brief, and how the sibling demo has been steered: `docs/handoff-ensemble-demo.md`. Open items: `TODO.md`.

## Where things are

| here | was in dpp-lm-demo |
|---|---|
| `index.html` (the page) | `lab/ensemble.html` |
| `js/ensemble.js` | `lab/ensemble.js` |
| `js/ensemble-math.js` (the maths; tested in `tests/ensemble.test.mjs`) | `lab/ensemble-math.js` |
| `data/` (`presets.json` and the cached pools) | `data/ensemble/` |
| `js/explain.js` ("?" explainers), `js/hfauth.js` (Hugging Face sign-in) | `js/`, copied at 5d42446; `explain.js` without the DPP demo's explainers |

## Backends: Modal (in dpp-lm-demo) and a Hugging Face Space (here)

The settings' "Runs on" picks where a live run happens. Both need the visitor's Hugging Face sign-in, and both run
the same weights with the same pool code.

- **Modal**, dpp-lm-demo's app (`dpp-lm-sampler`, `modal/app.py` and `modal/ensemble_lib.py` there): the page calls
  its `/ensemble_sample` route (`MODAL_URL` in `js/ensemble.js`). Paid from the demo's Modal budget; 20 runs per
  visitor a day, shared with the diversity demo. Its CORS list allows `https://timvieira.github.io` and
  `http://localhost:8765`.
- **The Hugging Face Space** [xtimv/LM-ensemble-demo](https://huggingface.co/spaces/xtimv/LM-ensemble-demo), from
  `space/` here: Gradio on ZeroGPU, called through Gradio's queue (`js/space.js`) with the visitor's token, so a run
  uses the visitor's own daily GPU quota. `space/ensemble_lib.py` is a copy of dpp-lm-demo's `modal/ensemble_lib.py`;
  keep the two the same. Deploy by uploading the folder (with a Hugging Face token that can write to the Space):
  `hf upload xtimv/LM-ensemble-demo space . --repo-type space`. Check a running Space with
  `node scripts/space-check.mjs https://xtimv-lm-ensemble-demo.hf.space [model] [format] [n]`.

The cached pools are drawn by dpp-lm-demo's "Ensemble prototype pools" workflow on Modal, which reads *that*
repository's `data/ensemble/presets.json` and commits the pools there; a change to the presets or a new set of pools
has to be copied between the two repositories by hand (`data/ensemble/` there is `data/` here).

## Develop

```sh
npm test                          # the ensemble maths
python3 -m http.server 8765       # then open http://localhost:8765 (the port the Modal backend accepts)
```
