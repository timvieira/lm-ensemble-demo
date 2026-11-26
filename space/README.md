---
title: LM Ensemble Demo
emoji: 🏆
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.32.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: https://timvieira.github.io/lm-ensemble-demo/
---

Backend for the interactive demo of *Ensembling Language Models with Sequential Monte Carlo*
([arXiv:2603.05432](https://arxiv.org/abs/2603.05432)), Fig. 1 ("prompt intersection"):
https://timvieira.github.io/lm-ensemble-demo/

`/ensemble_sample` draws N completions from the token-level product of prompts A and B and returns each one's
exact log p under A, B and (if given) C, its log-probability under the proposal, and its local (token-level)
ensemble score. The page does the rest.

On ZeroGPU it serves Qwen3 14B (the AWQ checkpoint's 4-bit weights, unpacked to fp16) and GPT-2; the page calls it
with the visitor's Hugging Face sign-in, so runs use the visitor's own daily GPU quota. Elsewhere it serves GPT-2
and Qwen3 0.6B (for testing). Source: https://github.com/timvieira/lm-ensemble-demo (`space/`).
