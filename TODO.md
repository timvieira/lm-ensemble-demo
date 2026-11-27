# To do

Started as dpp-lm-demo's TODO.md, moved here on 2026-11-26 (dpp-lm-demo@5d42446); cleaned up on 27 Nov (resolved items
moved to Done). T′ is given throughout in the Space's units: the proposal is ∝ f(…)^{1/T′}, so at τ′ = 0, T′ = ½ is
p_A·p_B token by token and T′ = 1 is √(p_A p_B); Modal's t_prime is 2T′. Paths in the last section are dpp-lm-demo's:
lab/ensemble.* is now index.html and js/, data/ensemble/ is now data/; modal/ and scripts/ are still in dpp-lm-demo.

## Needs Tim

- **The paper's conference version.** Everything this TODO and the page say about the paper was checked against
  the arXiv v1 only. The NeurIPS 2026 version (openreview.net/forum?id=zWigIoVsHo) is not read: OpenReview refuses
  scripts and is blocked in the Chrome extension. Needs Tim to save its PDF or allow openreview.net. To recheck
  against it: ρ = 0.958 and 0.090 (§2), quoted by the explicit-prompt explainer; α = 1 − τ (Thm. 4.1, stated for
  α ∉ {0, 1}; Table 1 gives KL(Φ‖p_k) at τ = 0 and KL(p_k‖Φ) at τ = 1); Fig. 1's product being p_A·p_B; §6
  evaluating min, product, sum and max side by side; the proposal ψ (§5.2, Prop. 5.2); temperature 1.0 (App. E).
- **What "product" means at τ = 0.** The page's product is √(p_A p_B), normalised (the τ explainer says so). The
  paper (arXiv v1) uses both. Fig. 1 and §2 use the unweighted product, locally ("multiplies the token-level
  probabilities ... then normalizes", the sampler's `local`) and globally ("multiplies these string-level
  probabilities"): p_A·p_B. Table 1 and Eq. (6), with w ∈ Δ, give Π p_k^{w_k}: √(p_A p_B) at w = ½. Rankings and ρ
  are the same either way; "ensemble %", log Ẑ and "Uses n of N" are not. Open: which one the page follows.

## Open

- **The cached pools are schema 2** (the product at T′ = ½, p_A·p_B token by token, the earlier sampler), so the
  page's defaults stay at that proposal (DEFAULTS in js/ensemble.js) and the status line says "(the earlier
  sampler)". Tim (27 Nov): no need to redraw them yet. Their effective sample size is 1.6 to 7.1 of 200 on the text
  pools (see "Importance weights"). When they are redrawn: dpp-lm-demo's pools workflow (its defaults are the
  page's proposal), then copy them here by hand.
- **Importance weights.**
  - The cached pools (the page's default; proposal p_A·p_B token by token, T′ = ½) put the weight on a few draws, so
    log Ẑ and every card's "ensemble %" rest on them. Effective sample size of 200, computed from data/ on 26 Nov:

    | pool | √(p_A p_B) (the page) | p_A·p_B |
    |---|---|---|
    | Qwen3 14B, text (5 pairs) | 1.6–7.1 | 2.5–24.2 |
    | GPT-2, text (5 pairs) | 1.7–5.8 | 2.2–4.5 |
    | Qwen3 14B, chat (5 pairs) | 1.0–200 | 1.0–194 |

    Physicist × author (Qwen3, text), the page's default: ESS 1.6, and one draw ("Toni Morrison.") carries 79% of
    the weight. Dessert × vegetable (chat): ESS 1.0 ("Cucumber" 99%). The page shows "Uses n of N", but nothing says
    how uncertain the percentages are (open).
  - Other proposals, Qwen3 14B on ZeroGPU (26 Nov; physicist × author, text, N = 200, seed 0, w_A = ½; one pool
    each, so the numbers are noisy; scripts/space-check.mjs). ESS at each target, bold where τ′ = τ:

    | proposal | min | harmonic | product | sum | max | distinct |
    |---|---|---|---|---|---|---|
    | cached pool: earlier sampler, product, T′ = ½ | 4.7 | 4.9 | 1.6 | 1.0 | 1.0 | 84 |
    | earlier sampler, product, T′ = 1 | 24.9 | 28.7 | 52.1 | 7.4 | 7.2 | 184 |
    | product, T′ = 1 | 25.8 | 30.2 | **61.3** | 24.1 | 23.6 | 177 |
    | min, ψ weights, T′ = 1 | **33.2** | 33.4 | 16.8 | 4.2 | 4.2 | 181 |
    | harmonic, ψ weights, T′ = 1 | 33.5 | **37.0** | 20.1 | 5.8 | 5.7 | 182 |
    | sum, ψ weights, T′ = 1 | 3.7 | 4.7 | 20.0 | **200.0** | 199.6 | 156 |
    | max, ψ weights, T′ = 1 | 2.2 | 3.0 | 11.1 | 192.5 | **194.5** | 148 |
    | sum, fixed weights, T′ = 1 | 8.9 | 11.9 | 32.1 | 53.3 | 52.6 | 169 |
    | product, T′ = 1, on Modal (27 Nov) | 23.4 | 27.0 | **56.9** | 31.8 | 30.9 | 171 |

    The second row was drawn on the Space before the new sampler; its largest share at product was 4% (79% in the
    cached pool). It and the third row differ only in the ending step (summed token by token, or one choice);
    their sum and max (7 vs 24) differ for a reason not looked into (that, or noise). Max with fixed weights was not
    run: Tim's ZeroGPU quota ran out. The last row is the Modal check after the port (scripts/modal-check.mjs, seed 0,
    with the Max-tokens fix; another GPU, so other draws): the same proposal as the third row, so the two give a
    first idea of the spread between pools (min 23 vs 26, product 57 vs 61, sum 32 vs 24).
  - Where the gain comes from (27 Nov). For min, harmonic and product, most of it is T′ = 1 instead of ½ (rows 2
    and 3); matching τ′ adds about a third (33.2, 37.0 against 25.8, 30.2), which may be within one pool's noise
    (unmeasured). For sum and max, matching τ′ with the ψ weights is decisive (24 → 200 and 194): at τ′ = τ = 1 the
    proposal is the target. Why min and harmonic stay near 17%: with ψ weights and T′ = 1, a draw's importance weight
    is the product of its per-step normalisers Σ_{x′} ψ(x′ | x), which are 1 at every step only at τ = 1 (otherwise
    only where the two prompts agree on the next token); no local proposal removes that spread (the paper's SMC
    resamples on exactly these).
  - Open: repeat with several seeds before reading anything into 26 vs 33. Modal runs every proposal now: about 17 s
    of L40S per pool (plus about 3 minutes to start a cold GPU), from the demo's budget and Tim's 20 runs a day.
- **The chat pools are nearly deterministic: 1 to 6 distinct completions in 200** (the cached pools, T′ = ½).
  Painter × inventor is "Leonardo da Vinci" 200 times (hence ESS 200: equal weights, one string); physicist × author
  is 3 strings (Albert Einstein ×157, Stephen Hawking ×41, J.K. Rowling ×2); dessert × vegetable is "Carrot cake" in
  three spellings plus one more. The token-level p_A·p_B of two chat replies is very peaked, so these pools show
  little of either prompt's spread. Open: what the chat format should show; T′ = 1 and the new proposals are not
  tried on chat yet.
- **"____." ranks 3rd for physicist × author** (Qwen3, text): a fill-in-the-blank completion. Relates to the preset
  wording item below.
- **Preset wording.** "A famous scientist is" continues as "…coming to our school next week."; better phrasings
  (e.g. "My favorite scientist is") or a different pair list.
- **Plot labels crowd, and on a phone drift far from their dots.** "Richard Feynman." and "Stephen Hawking."
  (physicist × author) sit one under the other, and Hawking's label is 16 px below its dot: the labels are only pushed
  apart vertically (rechecked after the full-page layout: unchanged). On a narrow frame most labels sit left of their
  dots and are pushed down one by one (Tim's screenshot, 26 Nov: "Albert Einstein." about 50 px from its point).
- **The self-check differs by up to 0.13 nats on ZeroGPU** (sampling sums against teacher-forced scores, both in
  fp16: 0.049 text, 0.106 chat, 0.073 text at n = 200 with the earlier sampler; 0.07 to 0.13 in the proposal runs of
  26 Nov, Qwen3 14B text; the code it checks did not change). The record reports it as checkMaxAbsErr; nothing here
  filters on it. dpp-lm-demo's pools workflow fails a pool above 0.5 nats and reports 0.09 seen on Modal. The page
  uses the sampling sums (logpA, logpB).
- **The GPU time reserved per run is about 4 times what a run takes** (space/app.py, _duration: 68 s for n = 200 at
  32 tokens; 17 to 21 s measured from the client). ZeroGPU checks before the run: on 26 Nov it refused a run because
  it "requested" 102 s with 91 s of Tim's quota left (102 = 1.5 × 68; where the 1.5 comes from is not known). So a
  visitor needs about 100 s of quota left for a 20 s run. Open: whether the quota is then charged by the time
  reserved or used, and a tighter estimate (a run that outlasts its reservation is cut off).
- **Whether the GPU code moves here** (see Notes: where it lives now).

## Untested

- A run of the new proposals from the page itself, signed in. The Space's API was checked with the page's own client
  (js/space.js) and the page's argument order by scripts/space-check.mjs, not by pressing ↑ while signed in.
- A sign-in and a run on Modal from this page. The route was checked after the port with the page's request body
  (scripts/modal-check.mjs), not by pressing ↑ while signed in.
- A page opened before 26 Nov and still loaded in a browser: its script sends the older request and now gets a
  schema-3 record, which it was not written for (it reads `ended`, absent now, so its cards would all say "cut off").
  Not tried; such a page goes away on reload.
- The page's message when the visitor's ZeroGPU quota is used up: js/space.js classifies ZeroGPU's error as "quota"
  (seen in scripts/space-check.mjs on 26 Nov), but the page's message for it has not been seen.
- The page on a real phone; pinch and two-finger moves on a real touch screen (not simulated: d3.zoom's own touch
  handling).

## Notes

- Where the GPU code lives. Decided for now (Tim, 26 Nov): Modal's stays in dpp-lm-demo (`dpp-lm-sampler`:
  modal/app.py, modal/ensemble_lib.py, scripts/ensemble_pool.py) with the pools workflow, which reads dpp-lm-demo's
  data/ensemble/presets.json and commits pools there, so a preset change or new pools must be copied by hand between
  dpp-lm-demo's data/ensemble/ and data/ here. The second backend, the ZeroGPU Space xtimv/lm-ensemble-demo, is
  space/ here. Since 27 Nov space/ensemble_lib.py and modal/ensemble_lib.py are the same file (dpp-lm-demo 3023c02);
  keep them identical, changing it here first (tests/ run here). Nothing checks that they agree.
- The unused copies in dpp-lm-demo stay (Tim, 26 Nov): lab/ensemble*.js, data/ensemble/, tests/ensemble.test.mjs and
  docs/handoff-ensemble-demo.md. They will drift from the copies here; lab/ensemble.js there still has the explainers'
  backslash bug (fixed here, see Done).
- js/explain.js and js/hfauth.js are copies of dpp-lm-demo's (explain.js without the DPP explainers). They differ both
  ways now: fixes there do not come here, and explain.js here scrolls a long explainer (26 Nov), dpp-lm-demo's does not.
- The repository was created as timvieira/prompt-intersection and renamed lm-ensemble-demo the same day (Tim);
  https://timvieira.github.io/prompt-intersection/ no longer exists, and nothing should link to it.
- On Tim's laptop the `gh` token has no `workflow` scope, so pushes that touch .github/workflows go over SSH
  (or after `gh auth refresh -s workflow`).

## Decided: the proposal (26 Nov)

- Tim: importance sampling, not SMC. The proposal is a local f-ensemble with its own τ′, weights and T′, reweighted
  to the global target τ, like the DPP demo's T′ and T.
- Against the paper (arXiv v1, §5): Def. 5.1 is this sampler, any proposal r with weights φ/r. The paper's own
  proposal (§5.2, Prop. 5.2, §6) is r ∝ ψ(x′ | x) = f(p⃗_A(x x′), p⃗_B(x x′)) / f(p⃗_A(x), p⃗_B(x)): a local
  f-ensemble whose weights follow the prefix, ŵ_k(x) ∝ w_k p⃗_k(x)^τ. It equals the fixed-weight one only at τ = 0.
  At τ = 1 it is exact: the weights are the posterior over the two prompts, and every importance weight is equal.
- Toy check (scripts/proposal-check.py; two random LMs, 3 symbols, length ≤ 5, one seed). Large-n ESS/n, fixed
  weights vs ψ:

  | τ | fixed | ψ |
  |---|---|---|
  | ≈ min (−8) | 0.59 | 0.61 |
  | harmonic (−1) | 0.62 | 0.63 |
  | product (0) | 0.73 | 0.73 |
  | sum (1) | 0.63 | 1.00 |
  | ≈ max (8) | 0.58 | 0.95 |

- Tim ("Go"): both weightings, with the paper's ψ weights as the default.
- The paper has no proposal temperature (App. E: temperature 1.0), so T′ is the demo's own addition.
- Tim (27 Nov): prompting is an LM → LM transformation, and so is redefining the end (the sentence and line rules,
  any special token). The ensemble combines the two transformed LMs, each with a single end symbol, so which base
  token ended a completion is not part of its outcome: the Space's sampler draws the end as one choice and records
  no token. Modal's endToken went with the port.
- Tim (27 Nov): Max tokens is also an LM → LM transformation, "it should be implemented that way". After
  max_new_tokens tokens the end has probability 1 under A, B, the local product and the proposal; nothing is drawn
  there, and the record's end is "max" (the field ended is gone).

## Done (26–27 Nov)

- **Max tokens is an ending rule** (27 Nov, Tim's view above): space/ensemble_lib.py ends a completion that reaches
  max_new_tokens tokens with probability 1, with no draw and no forward pass there. Before, the sampler still drew
  at that step: a completion could end there by the sentence rules (scored with each prompt's ending probability)
  or be cut off (scored with its prefix probability, and its proposal probability left out the chance of not
  ending), so one text was two outcomes and the scores were not those of any one LM. Found when implementing it; my
  earlier statement (27 Nov) that the weights were already exact for the transformed LM was wrong. Measured on the
  cached pools: 4 of their 3000 draws ended by the sentence rules at exactly 32 tokens (55 were cut off, scored as
  the transformed LM would), and at min, harmonic, product, sum and max neither group carries more than 0.005% of
  any pool's weight; the proposal explainer says so. tests/test_sampler_toy.py runs the real loop on a toy
  tokenizer and model and checks every draw's logpA, logpB, logr and loglocal against an enumeration of the
  transformed LMs (to 1.6e-6) and the draws' frequencies; on the earlier loop it fails (306 of 4000 draws). It runs
  in the Space deploy workflow. scripts/sampler-check.py (GPT-2, 8 proposals) still agrees to 7.9e-5 nats.
- **The sampler on Modal** (Tim: "port to modal", 27 Nov; dpp-lm-demo 3023c02): modal/ensemble_lib.py is the Space's
  file; /ensemble_sample takes proposal {tau, wA, weights, Tprime}, and the older t_prime gets the product at
  T′ = t_prime / 2; ensemble_cache and scripts/ensemble_pool.py take the proposal, defaulting to the page's. The
  page sends any proposal to Modal (the gate is gone). Checked on the deployed app with scripts/modal-check.mjs:
  GPT-2 (8 draws, sum proposal: ESS 8.0 of 8 at the sum target), the older request, an invalid proposal (400, before
  any GPU work), and Qwen3 14B text, 200 draws (the row above; self-check 0.072 nats). The DPP demo's code did not
  change; its routes share the redeployed app.
- **The proposal family, on the Space and the page:** `space/ensemble_lib.py` (local_f, proposal_logprobs; schema 3),
  the Space API `ensemble_pool` (the older `ensemble_sample` maps to the product at T′ = t/2), and the settings ("The
  proposal: a local ensemble": its weights, τ′, w_A, T′). The ending step is one choice, scored with the mean of each
  prompt's total ending probability, as for the paper's end symbol; the schema-2 sampler summed it token by token.
  Checked: tests/test_proposal.py (every step against the definition on an enumerable toy model, to 2.6e-6 in log r;
  the mixture proposal exact; runs in the Space deploy workflow) and scripts/sampler-check.py (GPT-2 on a CPU: 8
  proposals, recorded logr, logpA, logpB equal an uncached step-by-step recomputation to 7e-5 nats). Local Space (CPU,
  GPT-2, 6 draws, sum proposal): ESS 6.0 of 6 at the sum target. Measured on Qwen3 14B: see "Importance weights".
- **The page's claims about the paper, against the arXiv v1** (the conference version: see Needs Tim): ρ = 0.958 and
  0.090 (§2) match; α = 1 − τ matches Thm. 4.1; "the paper's main ensemble" overstated it (Fig. 1 uses the product,
  but §6 evaluates min, product, sum and max side by side, and the consensus-seeking two, min and product, do best).
  The τ explainer now says the paper's Fig. 1 uses the product, there p_A·p_B without the square root (the same
  ranking).
- **Hugging Face sign-in accepts this address** (26 Nov, 14:32). The OAuth app "Diverse sampling demo" (client
  43d09d55-…; huggingface.co/settings/connected-applications, Developer Applications, Settings) lists
  https://timvieira.github.io/dpp-lm-demo/, https://timvieira.github.io/lm-ensemble-demo/ and https://timvieira.github.io/,
  each in its own box (the + button; several in one box are read as one address). Hugging Face matches the address
  exactly: a parent such as the homepage does not cover the pages under it (checked). Each new demo page that signs in
  needs its own line.
- **Sign-in and a run on the Space, from the page, end to end** (Tim's screenshot, 26 Nov 14:38): favorite actor ×
  favorite author, GPT-2, 200 completions in 9 s, "drawn on the Hugging Face Space".
- **A second backend, a Hugging Face ZeroGPU Space** (Tim): space/ (app.py, ensemble_lib.py, README.md as the Space
  card), deployed by .github/workflows/space.yml; js/space.js calls it through Gradio's queue with the visitor's token;
  "Runs on" in the settings picks Modal or the Space; the Space's errors (quota, not answering, busy) have their own
  messages, with "Run on Modal". The status line says where a pool was drawn.
  - The Space is xtimv/lm-ensemble-demo (Tim renamed it; it was created as xtimv/LM-ensemble-demo, which now
    redirects). Its address, xtimv-lm-ensemble-demo.hf.space (the page's SPACE_URL), did not change.
  - The deploy token: the repository secret HF_TOKEN is set (Tim); the first deploy uploaded space/ to the Space (its
    commit b7bb356).
  - Tested on ZeroGPU through the page's own client (scripts/space-check.mjs, on Tim's quota): the Space built and ran
    on zero-a10g in about 3 minutes; it offers Qwen3 14B (the unpacked AWQ weights) and GPT-2; Gradio 6.32's queue
    protocol works with js/space.js. Qwen3 14B text n = 200 at 32 tokens (the page's default) took 17.5 s (9 s
    sampling, 8 s rescoring). With seed 0 the first Qwen3 completion starts as in the cached Modal pool ("Kip Thorne,
    and I'm trying to understand...") and then diverges: the same weights on another GPU round differently.
- **Explainers fit the window:** a long one (the proposal explainer is 830 px) scrolls inside its box, and is placed
  wherever it fits; before, the wide layout (which cannot scroll) cut off its bottom.
- **The status line comes back** when the settings return to those of the plot shown; it used to keep saying "The
  plot is from earlier prompts or settings".
- **MathJax for every symbol** (Tim, 26 Nov: "use mathjax"; his global rule). MathJax 3.2.2, SVG output
  (tex-svg-full, fontCache local) replaces KaTeX. The page's own text is typeset at start-up, explainers when they open
  (js/explain.js), and the text the page builds (cards, readouts, the τ readout, tooltip, ρ line) gets cached
  tex2svg markup; the axis labels inside both plots' SVG are tex2svg SVGs sized from their ex units and set on the
  baseline. The explainers' Unicode τ, α, T′, Ẑ, N are TeX now, and y_{<t} is escaped (it was swallowed as an HTML tag).
  Checked in headless Chrome at 1600 × 1000 and 390 × 844: no \( \[ $ left in the rendered text, no MathJax errors,
  all 16 explainers clean. If MathJax does not load, the page says so. Not maths: native hover titles and the
  backends' error messages (e.g. "Allowed: 2 ≤ n ≤ 200"), shown as sent.
- **The explainers' maths was broken in 9 of 16** (since they were written in dpp-lm-demo ad96dde): the TeX sat in
  JS template literals with single backslashes, so \varphi, \tau, \rho, \frac became control characters and \sum,
  \mathrm, \hat lost their backslash. Doubled (54 backslashes); rendered cleanly by KaTeX then, and by MathJax now.
- **Stale text fixed:** the banner now says "Work in progress. This demo is still being built." (it said only the
  example pairs had completions); the sign-in message names the Space when "Runs on" is the Space (it always said
  Modal); code comments name both backends.
- **The title is "Ensembling Language Models"** (Tim; was "Prompt intersection", which now opens the subtitle). The
  controls start 10 px under the title block, measured by its bottom edge as on the diversity demo (they overlapped it
  by 2 px).
- **Card meta lines** break between fields, never inside one ("log p_A −1.1" stays together).
- **The diversity demo's full-page layout** (Tim): on wide screens the plot is the page. The title and the controls
  float top left, the ranked rail runs down the right, the readouts (Uses n of N, log Ẑ) and View sit in a bar along
  the bottom. The frame is the open area between them; the grid, the level lines and the diagonal continue under the
  panels and fade out; two axes with arrowheads, no box. Narrower screens: one column (composer, plot with the bar
  under it, rail), τ docked at the bottom of the screen. The page shows after its first render (html.booting).
  - Differs from the diversity demo: the breakpoint is 1100 px, not 901 px (a 2-D scatter with axes needs the room
    between the controls and the rail; at 1100 px the frame is about 320 px wide).
  - Checked at 1920 × 1080, 1600 × 1000, 1280 × 800, 1100 × 760, and as a touch device at 1024 × 768 and 390 × 844:
    nothing wider than the window, no console errors; τ at min, Scored local and an explainer checked at 1600 × 1000.
    Checked with a DevTools-protocol script outside the repo: headless Chrome's --window-size lays the page out
    87 px shorter than its screenshot and no narrower than ~500 px. This repo has no overflow test like
    dpp-lm-demo's tests/overflow.cjs.
- **Level lines across the whole window** (Tim): no longer faded with the grid, and continued past the completions'
  score range at the same spacing (a ninth of that range); at most about 120. Cost: zooming out to 0.33× draws about
  48 lines and a wheel step takes 50 to 65 ms (was about 20).
- **Pan and zoom, as on the diversity demo** (its Plot2D; missed in the first port of the layout, Tim asked): the wheel
  or a pinch zooms (0.2× to 50×); Shift-drag, right-drag or two fingers move; double-click returns to the first view;
  a plain drag does nothing. Axes, grid, level lines and the diagonal follow the view; points and labels are clipped
  to the frame, and only points inside it are labelled. A resize keeps the view; another pool starts from the first
  view. A mouse reads a point by hovering, a finger by tapping (28 px radius). On narrow screens one finger scrolls
  the page over the plot (touch-action: pan-y). Checked headless at 1600 × 1000 with synthetic events (wheel ×5 =
  2×, Shift-drag, plain drag, double-click; about 20 ms from a wheel step to the next drawn frame) and a tap at
  390 × 844.
- **Menus on phones** (Tim's screenshot: View opened off the left edge): View sits at the right end of its line, so its
  menu opens leftward on screen; the gesture hint gets a line of its own instead of being cut; the readouts wrap; below
  480 px the composer's menus open from its left edge. Checked at dpp-lm-demo's overflow-test widths (320, 360, 390,
  430, 600, 700, 768, 900, 1024, 1280, 1440) and 1100: every menu on screen, nothing cut, no sideways scroll. This repo
  still has no overflow test of its own.
- **The redirect** from dpp-lm-demo's lab/ensemble.html keeps the query string and fragment (checked live:
  `?x=1#y` lands on https://timvieira.github.io/lm-ensemble-demo/?x=1#y).
- **Robin Chan has the link** (Tim messaged him himself).
- **A local clone** at ~/projects/lm-ensemble-demo on Tim's laptop (Tim).

## The prototype in dpp-lm-demo (25–26 Nov): decisions and history

- Brief: docs/handoff-ensemble-demo.md (in dpp-lm-demo). Built as an unlinked lab page of dpp-lm-demo
  (lab/ensemble.html), the ensemble demo of prompt intersection (arXiv:2603.05432 Fig. 1); moved here on 26 Nov.
- Decided (Tim): Qwen3 14B (the GPU server's model); cached pools for the presets during development
  (data/ensemble/presets.json).
- No longer waiting on Manuel (was: `/score` and a token-level product sampler on his GPU server, which had neither on
  26 Nov): the pools and live runs come from Modal (the token-level product) and, since 26 Nov, the Hugging Face Space
  (the proposal family); both score under A, B and C.
- Done (26 Nov): Qwen3 14B pools for the five presets, drawn on Modal (modal/ensemble_lib.py through the manual
  "Ensemble prototype pools" workflow; about 15 s of GPU per preset). Physicist × author reproduces the paper's
  direction: against the explicit prompt, ρ = 0.94 global (product) vs 0.77 local.
- Done: the composer (prompts A and B, examples, settings, ↑), the τ/K panel with ticks at the named means, a View
  menu (Scored global/local) and "?" explainers. (↑ then said live runs weren't connected yet; they came the same day.)
- Decided (Tim): completions are one sentence, with exceptions for non-sentence punctuation. A period ends a
  sentence when the next token starts with whitespace or ends the text, and it is not an abbreviation (Dr., Mr.,
  e.g., U.S., …) or an initial ("J."). The ending event's probability (all such next tokens) is part of every
  score. Known trade-off: "Vitamin C." reads as an initial.
- Decided (Tim): no K and no Sample in the ensemble prototype. The rail ranks every distinct completion by its
  ensemble score (exact), with its probability under the ensemble (φ/Ẑ, Ẑ estimated from the pool); ten shown
  until "Show all"; the five best are red and labelled in the plot.
- Decided (Tim): pySBD's English rules that need only the next word, adopted as next-token ending sets
  (ensemble_lib.py, ending_kind and ending_masks): "." and "?" end before whitespace; "!", "..." and
  sentence-ending abbreviations (Inc., Jr., U.S., Ph.D., …) only before a capital or a new line; No./pp. not
  before a digit; titles, initials, e.g., i.e. never; a new line always ends a completion; closing quotes count.
  Exact except tokens holding text plus a line break (".\n"), scored by their own probability. Not adopted:
  pySBD's list items, file extensions, coordinates and its everyday-word abbreviations (is, me, man, may, …).
- Decided (Tim): no in-browser models (GPT-2 in Transformers.js was considered). Design for a phone: one
  WebAssembly thread on GitHub Pages, roughly 3–5 min per pool at N = 200, and exact weights too big to download.
  Cached pools for the examples, and live runs on a server.
- Decided (Tim): two prompt formats with the same Qwen3 14B weights, chosen in the settings: "Continue text"
  (the default; plain-text continuation as in the paper's Fig. 1; one sentence or one line) and "Chat" (the
  prompts are user messages in the chat template, reasoning off; a reply ends at the end-of-turn token). Each
  example pair has both phrasings (data/ensemble/presets.json); pools are data/ensemble/<id>.<text|chat>.json.
  A chat model is still a language model: raw continuation is valid, just a different distribution.
- Decided (Tim): any special or added token ends a completion, in both formats (Qwen3: <|im_end|>,
  <|endoftext|>, <|im_start|>, <think>, tool-call, vision, FIM tokens: 26 in all). The ending set's probability is
  in every score; nothing is blocked or dropped. Each completion records which special token ended it
  (endToken), to spot e.g. replies that run into <|im_start|>; superseded on 27 Nov (see "Decided: the proposal"),
  and only Modal's pools still do. Before,
  only <|im_end|>/<|endoftext|> ended one and other special tokens were kept but hidden by decoding.
- Done: live runs on Modal. /ensemble_sample on the Modal web app (Hugging Face sign-in, the shared daily run limit
  and monthly GPU budget), two models: Qwen3 14B (the existing L40S container; text and chat formats) and GPT-2
  (124M, its own T4 container, charged to the meter at T4 prices; text only). The page offers both in the settings
  and runs anything not cached; pools are data/ensemble/<pair>.<model>.<format>.json, drawn for both models by the
  "Ensemble prototype pools" workflow.
- Done differently: the lab page moved here, and sign-in works from https://timvieira.github.io/lm-ensemble-demo/.
  The sign-in is shared with the diversity demo (same site).
