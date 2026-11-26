# Handoff: a "prompt intersection" demo for *Ensembling Language Models with SMC*

A brief for whoever builds a companion demo for *Ensembling Language Models with Sequential Monte Carlo*
(Chan, Liu, Kiegeland, Pasti, Vigly, O'Donnell, Cotterell, Vieira; NeurIPS 2026;
[arXiv:2603.05432](https://arxiv.org/abs/2603.05432); code: github.com/genlm/ensemble-neurips-2026).

It has two halves:

- **Part A** proposes the demo, built on the paper's Figure 1.
- **Part B** writes down how Tim has steered the existing demo for *A Model of Diverse Sampling from
  Language Models* (this repository, https://timvieira.github.io/dpp-lm-demo/): its look and feel, and
  the decisions behind it. The new demo should feel like a sibling of that one. When in doubt, do what
  this repository does and read its code; most of what Part B describes is already built and can be
  copied.

Nothing in Part A is decided. It is a proposal for Tim to react to. Items marked **Open** need his call.

---

# Part A. The proposed demo

## A1. What the paper's Figure 1 shows

**The setup (paper, §2).** GPT-2 is prompted with two prefixes whose plausible completions overlap:
"My favorite physicist is" (prompt A) and "My favorite author is" (prompt B). The goal is strings that
are probable under *both*: physicists who are also authors. 200 completions ending in a period are
generated with best-first beam search (beam width 20) from the *locally normalised token-level
product* of the two prompts.

Each completion is scored two ways:

- **Local ensemble** p_ens,local: at every step, multiply the two prompts' next-token probabilities and
  renormalise. A string's probability is the product of those locally normalised factors. This is what
  most prior ensembling does.
- **Global ensemble** p_ens,global: each prompt's probability of the *whole* string, multiplied, then
  normalised across the 200 completions. This is the paper's target, the product at the string level.

**Left panel.** A scatter of log p(· | physicist prompt) (x) against log p(· | author prompt) (y), one
dot per completion. The ensemble's completions are coloured by p_ens,global / p_ens,local, the global
probability relative to the local one, on a 0–1 colour bar. Completions drawn from each prompt alone
that the ensemble did not produce are shown in two other marks. Labelled points include "Carl Sagan.",
"Bill Nye.", "Einstein.", "Richard Feynman.", "David Hilbert.", "Robert Jordan.", "James Baldwin.",
"a great guy." and "a very good one.".

**Right panel.** Each completion's log p under an explicit intersection prompt, "My favorite physicist
and author is", plotted against the global ensemble (ρ = 0.958) and against the local one (ρ = 0.090,
p = 0.24).

**The point.**
1. The ensemble favours strings probable under both prompts. "James Baldwin." is likely for authors,
   unlikely for physicists, so it is unlikely under the ensemble.
2. Local, token-level ensembling is *not* the string-level ensemble. It over-rewards generic openings
   ("a", "my") that both prompts like token by token, so "a great guy." scores well locally but badly
   globally, and "Carl Sagan." the other way round.
3. The global ensemble is the one that matches the explicit "physicist and author" prompt.

**The general framework (paper, §4).** An f-ensemble of K experts is Φ(x) ∝ f(p_1(x), …, p_K(x)). The
paper focuses on the generalised mean with power τ:

  Φ_τ(x) ∝ ( Σ_k w_k p_k(x)^τ )^{1/τ}

- τ → −∞: the minimum. τ = −1: harmonic. τ → 0: the product (product of experts). τ = 1: the sum
  (mixture). τ = 2: quadratic. τ → +∞: the maximum.
- Theorem 4.1: Φ_τ is the unique minimiser of Σ_k w_k D_α(Φ ‖ p_k) with α = 1 − τ.
- τ ≤ 0 is *consensus-seeking*: mass on the intersection of the experts' supports. τ > 0 is
  *coverage-seeking*: mass on the union. |τ| sets how strongly.

**Inference (paper, §5).** Sampling from Φ is intractable. Importance sampling, sequential importance
sampling and sequential Monte Carlo approximate it. With M particles from a proposal r, the weights are
w = φ(x)/r(x). The estimate Ẑ = mean(w) is unbiased, and Φ̂ is consistent. SMC resamples when the
effective sample size (Σw)²/Σw² falls below η·M.

**The headline findings.**
- Ensembles beat their best single model.
- Consensus-seeking f (the product) consistently beats the field's default, token-level probability
  averaging.
- Better posterior approximation (more particles) gives better accuracy.

## A2. The demo in one paragraph

Type two prompts, or pick a preset pair like physicist and author. The page draws a pool of completions
and shows each completion as a point at (log p under prompt A, log p under prompt B), exactly Figure 1's
left panel. A single slider, **τ**, runs from *agreement* (min, product) to *coverage* (sum, max). As it
moves, the ensemble's level curves sweep across the plane: diagonal lines for the product, L-shaped
corners for the minimum, rounded curves in between. The points re-weight, and the **K most likely**
completions, or K **samples**, light up in red with their labels and cards. A **Scored: global / local**
switch shows Figure 1's other lesson: under "local", "a great guy." climbs and "Carl Sagan." falls. A
readout compares both against the explicit "A and B" prompt (Figure 1, right).

It is the DPP demo's structure with a different scoring rule:

| DPP demo | Ensemble demo |
|---|---|
| Prompt | Two prompts, A and B (optionally a third, the explicit "A and B" prompt, for validation) |
| Draw N completions from p°_{T′} | Draw N completions from a proposal r: the local product (as in the paper), A alone, B alone, or a mixture |
| Embedding φ, similarity matrix G | Two scores per completion: log p_A(y), log p_B(y), plus the proposal's log r(y) and the local ensemble's log p_local(y) |
| T (quality ↔ diversity) | τ (agreement ↔ coverage) |
| Target quality: Global / Global and local | Scored: Global (string level, the paper) / Local (token level, prior work) |
| K, Most likely / Sample | K, Most likely / Sample (the same controls) |
| Volume box (shown only when exact) | The ensemble's level curves in the (log p_A, log p_B) plane (exact everywhere: it is a 2-D plot of a function of two numbers) |
| Pool fit θ̂ | Effective sample size of the importance weights ("the ensemble effectively uses 7 of these 200 completions") |
| Inclusion probability on each card | The completion's share of the estimated ensemble, Φ̂(y) |
| log P(S), quality, diversity | log Ẑ, ESS, ρ with the explicit "A and B" prompt (global and local) |

## A3. The maths the page computes (in the browser, from the pool)

For each completion y_i (i = 1…N) the backend returns ℓ_A(y_i) = log p_A(y_i), ℓ_B(y_i) = log p_B(y_i),
log r(y_i) under the proposal it was drawn from, and log p_local(y_i). Then, in log space throughout:

- **Generalised mean.** log φ_τ(y) = (1/τ) · logsumexp_k(τ·ℓ_k + log w_k). At τ = 0 this is Σ_k w_k ℓ_k,
  the weighted product, taken as the limit. At ±∞ it is max_k ℓ_k or min_k ℓ_k. The expert weights w
  default to ½, ½ (an advanced knob, see A5).
- **Global score.** log φ_τ, as above. **Local score.** log p_local, which is defined for the product
  only. **Open:** what "local" means for τ ≠ 0. The natural generalisation applies the token-level
  generalised mean at each step and renormalises. That needs the full next-token distributions, so the
  backend would compute it for a fixed set of τ ticks, or the switch is offered only at τ = 0.
- **Most likely.** The K distinct strings with the largest φ (ties by ℓ_A + ℓ_B).
- **Sample.** Importance resampling from the pool: weights v_i = φ_τ(y_i)/r(y_i), K draws without
  replacement in proportion to v. This is the same estimator as the paper's importance sampling, on a
  fixed pool. It is not the paper's SMC, which runs *during* generation and needs the backend (see A7).
- **Estimates.** Ẑ = mean(v) (unbiased for Z, given i.i.d. draws), Φ̂(y_i) = v_i / Σv,
  ESS = (Σv)²/Σv².
- **Level curves.** Contours of log φ_τ(ℓ_A, ℓ_B) on the plot's own grid, computed analytically: no
  sampling, no projection.
- **Validation (Figure 1, right).** If a third prompt C ("My favorite physicist and author is") is
  given, the backend also returns ℓ_C(y_i). The readout gives Spearman ρ between log Φ̂ and ℓ_C, for
  global and for local scoring. A small inset or a Details panel plots it.

Copies of the same string: as in the DPP demo, group identical strings for display, and say "×3" on
the card. For Sample, decide whether identical strings may both be drawn (**Open**; the DPP demo sums
inclusion over copies).

Tests: brute-force references in `tests/`, run before each deploy, as the DPP demo does. Check the
generalised mean at its named limits, log-space stability for very negative ℓ (−200 and below), ESS
bounds (1 ≤ ESS ≤ N), and resampling frequencies against v.

## A4. Layout (copy the DPP demo's)

- **The plot fills the page.** It's a 2-D scatter on the soft floor grid, the same as the DPP demo's
  2-D view. x = log p(· | A), y = log p(· | B), with the prompts written along the axes in small sans.
  - Points are grey (ink). The selected K are red (the one accent), with first-five-word labels, as now.
  - Point size or opacity follows Φ̂. The ensemble's level curves are thin grey lines, labelled sparsely.
  - No arrows from the origin: there is no origin to speak of here.
  - A diagonal guide where ℓ_A = ℓ_B is worth considering (it is where the product and the min agree).
- **Upper panel (left), the composer.** Everything here needs new completions.
  - Prompts A and B. Two short fields stacked in the composer, or one field per line, with a "×"
    between them, read as "A and B".
  - The send ↑ button, presets under +, and the settings chip holding: model, proposal r (local
    product / A / B / mixture), N, T′, max tokens, stop at the first period (on by default, as in the
    paper), and the optional explicit prompt C.
- **Lower panel, two rows, compact on phones.** Everything here re-weights the same pool and never
  draws.
  - Row 1: τ, a slider with ticks at the named means (min · harmonic · product · sum · max), "agreement"
    and "coverage" under its ends, the value shown as a number or a name.
  - Row 2: K, and Most likely / Sample.
- **Advanced, in the View menu (as Target quality is now):** Scored: global / local; the expert
  weights w_A : w_B; colour by global ÷ local (Figure 1's colour, on the grey ramp).
- **Right rail:** "Selected" cards (serif text; meta line: log p_A · log p_B · share of ensemble), then
  "All completions", collapsed. On phones a tap expands a card in place.
- **Details drawer:** readouts (log Ẑ, ESS as "uses n of N", ρ with C for global and for local), the
  validation inset, and the View menu.

The demo's one "aha" should take no reading: drag τ from product to min and watch the diagonal level
lines fold into corners, and the red points move from the "both moderately likely" diagonal to the
"neither terrible" corner. Then flip Scored to local and watch "a great guy." rise.

## A5. Knobs, and what each one needs

| Knob | Panel | New draws? | Notes |
|---|---|---|---|
| Prompt A, prompt B | upper | yes | The pair is the "question". |
| Explicit prompt C | upper, settings | scoring only (no new samples) | Optional; presets supply it. |
| Model | upper, settings | yes | The paper used GPT-2 for Fig. 1; a base model continues prompts best. **Open:** Qwen3 14B (what the GPU server has) or a base model. |
| Proposal r | upper, settings | yes | Default: the local product, as in the paper. A, B and the 50/50 mixture show how the proposal limits what the pool can represent (ESS). |
| N, T′, max tokens, stop at "." | upper, settings | yes | Draws are i.i.d. and kept in order, so changing N reuses a prefix, as in the DPP demo. |
| τ | lower | no | The headline knob. |
| K, Most likely / Sample | lower | no | As now. |
| Scored: global / local | View menu | no (if local is precomputed) | See A3 for what "local" means at τ ≠ 0. |
| Expert weights w | View menu | no | Default ½ : ½. |

Note: the paper's Figure 1 uses beam search. The demo must draw i.i.d. samples, because importance
weights need a known proposal probability. Say so in the "?" for the proposal.

## A6. Presets

Pairs with a clear intersection, precomputed like `data/cache/` so they show instantly:

- "My favorite physicist is" × "My favorite author is", with C = "My favorite physicist and author is"
  (the paper's).
- "A famous painter:" × "A famous inventor:" (Leonardo).
- "A fruit that is" × "A colour that is" (orange).
- "A city in Europe:" × "A city on a river:".
- "A word that rhymes with cat:" × "An animal:" (rat, bat).
- A disagreement pair where the product and the sum differ sharply, e.g. "My favorite dessert is" ×
  "My favorite vegetable is".

**Open:** the final list. Check each one: the intersection must be non-empty and visible in the plot
at N = 200.

## A7. The backend

Reuse the DPP demo's contract style (docs/BACKEND.md: `/status`, versioned, exact numbers, stable
error codes) and its three backends (GPU server, Modal, the HF Space). New routes:

- `POST /ensemble_sample` `{prompts: [A, B], proposal: "local_product"|"A"|"B"|"mixture", n, t_prime,
  max_new_tokens, stop, seed, model}` returns `{samples: [{text, logp: [ℓ_A, ℓ_B], logr, loglocal,
  nTokens, ended}], timing}`. All log-probabilities are exact sums over the generated tokens, with EOS
  or the stop handled as in docs/BACKEND.md.
- `POST /score` `{prompts: [...], texts: [...]}` returns `{logp: [[...]]}`, for prompt C and any
  re-scoring.
- **Later:** `POST /ensemble_smc` `{prompts, tau, weights, particles M, ess_threshold η, …}` to run
  the paper's SMC itself and return the particles with weights. This is the honest way to show "better
  posterior approximation → better results" (the paper's RQ3): a particles slider in the upper panel
  (it needs new draws), comparing ESS and ρ with C as M grows. It is also where τ must be fixed before
  drawing, so a run is per τ. **Open:** whether this belongs in v1.

Byte-level ensembling of models with different tokenisers (paper §5, §E) is out of scope for v1. A
cross-model pair is a natural v2 ("Llama × Qwen"), and would need the byte-level machinery on the
server.

## A8. Explainers to write (the "?" buttons, plain words, maths in KaTeX)

τ; the generalised mean family (table of names, consensus vs coverage); global vs local (the "a great
guy." story); the proposal and importance weights; ESS ("uses n of N"); log Ẑ; the validation prompt
and ρ; Most likely vs Sample; why the plot is in log space; share of ensemble.

---

# Part B. How the DPP demo has been steered

These are the decisions Tim has made while building https://timvieira.github.io/dpp-lm-demo/, and the
taste behind them. Treat them as the brief for the new demo unless Tim says otherwise.

## B1. Faithful first

- **Compute what the paper defines, exactly, and say what is approximate.** Log-space routines are
  checked against brute-force enumeration in `tests/`, and the tests run before every deploy. Exact
  numbers come from the backends: exact log p and log q, and exact G rather than vectors projected on
  the client.
- **Draw a picture only when it is the real thing.** The volume box is drawn only when the plot's axes
  contain the selected vectors (K ≤ 2 in 2-D, K ≤ 3 in 3-D), so it *is* the scored volume and not a
  shadow of it. Projections are labelled as projections.
- **No arbitrary thresholds or cutoffs.** Tim asked repeatedly "why have any cutoff?". Arrows, labels and
  readouts used to vanish past K = 3 or K = 6, and the pool-fit warning had a 1-nat threshold; all of
  these were removed. Keep a limit only when it is real, and then say what the real limit is ("volume
  drawn for K ≤ 2" in 2-D).
- **Plain numbers over verdicts.** Show the pool fit as a number with a "?", not a red/green judgement.
  The preferred form is an effective count ("uses 4 of 200 completions").
- **Theory before features.** Ideas are written up first (docs/matrix-views.md) with **Decided** and
  **Open** marks, and prototyped in a separate, unlinked lab page (`lab/`) before anything reaches the
  main page.

## B2. A curated experience

- **The page is curated by Tim.** Collaborators' features (Manuel's fork) are their own exploration and
  come in only when Tim picks them. Many were explicitly cut: an expert mode, preset T ticks, colour
  switches, extra models.
- **Advanced options are tucked away.** Target quality was moved out of the main panel into the View
  menu because "we don't need an average user to play with it". The main panel shows only the knobs
  that carry the idea.
- **Semantic split of the controls.** The upper panel holds what needs new completions (prompt, model,
  N, T′, max tokens, embedding). The lower panel holds what re-weights the same completions (T, K, Most
  likely / Sample). Changing N reuses the same i.i.d. draws (completions 1–N; "Draw new ones" takes the
  next N).
- **One idea per view.** The DPP demo's hero is the volume. In the new demo it should be the ensemble's
  level curves.

## B3. Respect the visitor

- **Nothing moves unless the visitor asks.** Axes, zoom and framing are set once, when a pool is first
  drawn. Mode, K and T never turn or re-zoom the view. Points sit at √κ in both modes, so switching mode
  does not move them. When the axes *could* be refitted, there is a "Fit axes to selection" button rather
  than an automatic turn: Tim rejected the automatic version because points moved when K changed. A
  "follow the action" camera is a possible later option, never a default.
- **Explicit (docs/run-flow.md).** Nothing changes the visitor's settings, switches backends or opens a
  sign-in page unless they clicked something that says it will. The plot always matches the settings it
  claims, or says "The plot is from earlier settings. Press ↑."
- **Every failure says what happened in plain words and offers the next step as a button:** Try again,
  Use the GPU server, Sign in, Dismiss. Long waits show elapsed time and "taking longer than usual", and
  Stop always works. A result that arrives after the settings changed is kept and offered ("Show it").
- **No surprise costs or sign-ins.** The GPU server needs no sign-in. Modal and the Space need a free
  Hugging Face account, asked for only on ↑.

## B4. Look and feel (the house style of timvieira.github.io)

From `css/style.css`:

- **Colour and type.**
  - Warm paper background `#f6f1ea`, surface `#fbfaf8`, text `#333`, secondary text `#555`, muted
    `#999`.
  - EB Garamond for reading text: titles, completions, explainers. A small system sans for controls, and
    a mono face for numbers.
- **Near-monochrome.** Ink-grey controls and points; one red accent `#c8473c` for the selection, its
  vectors and its volume. Light theme only, like the blog. Any added colour (the lab's theme bars) is
  muted, validated and kept clear of the accent.
- **Layout.**
  - The plot fills the window with the panels floating over it.
  - The composer is top-left: a prompt textarea, + for presets, a settings chip and a round red ↑.
  - The T/K panel sits under it, and docks at the bottom on phones.
  - The rail on the right holds the cards. The Details drawer holds the readouts, Spectrum and View.
  - A soft floor grid that fades out, axes with small arrowheads, 3-D that turns about the origin with
    one gentle opening turn and no auto-rotation.
- **Controls.**
  - Small pill switches (Most likely / Sample), a K stepper in the same style, and "?" buttons that open
    explainers.
  - The explainers are short, in plain language, with the maths typeset in KaTeX, and every readout has
    one.
- **Cards.** The completion in serif, clipped to four lines. A meta line in small sans (log p ·
  inclusion), and "cut off at 128 tokens" in italics when the model did not finish. Copies are shown as
  "×n". Selected cards are sorted by inclusion probability.
- **Labels and hover.** Selected points are labelled with their first five words. Hover shows a tooltip;
  a click opens the full text (wide screens) or expands the card in place (phones).
- **Phones count.**
  - The docked bottom panel must be compact: two rows. Tim asked for the third (Target) row to go.
  - A tap expands a card in place; the inspector is far up the page on a phone.
  - Two fingers pan, and nothing spills out of its box at any width (`tests/overflow.cjs` checks 11
    widths).
- **Header.**
  - The title in Garamond, then "Paper ↗" and "Feedback ↗". Feedback opens a GitHub issue pre-filled
    with the current settings, which the visitor can edit; issues are public.
  - A small "Work in progress" note.
- **Loading.** The page appears in one step after its first complete render. There's no layout shift
  when the scroll bar appears.

## B5. Engineering conventions

- **Static page plus backends.** The page is a static site on GitHub Pages, deployed by a workflow on
  every push to `main`. Scripts sit in a per-commit folder, so a reload never mixes versions.
  - It is plain ES modules with no build step. d3 and three.js come from a CDN with pinned versions,
    plus KaTeX.
- **Precomputed pools** for every preset (`data/cache/`, with an index), so the first view is instant
  and needs no GPU.
- **Backends behind one contract** (docs/BACKEND.md), with `scripts/conformance.mjs` to check a backend
  against it. There are three:
  - **The GPU server (Manuel's):** no sign-in. It logs runs, and the page says "Runs here are logged
    for research." while a run is in progress. Tim chose to say no more than that.
  - **Modal:** Hugging Face sign-in. There's a $30 credit, so avoid warm-ups on every visit.
  - **The Hugging Face Space:** Hugging Face sign-in.
- **Limits:** N ≤ 200 and max tokens ≤ 256 on every backend. Typing more keeps the number and explains
  the cap.
- **Usage reporting** (from Manuel's PR, merged): only to the GPU server, only when its `/status` says
  `"usage": true`, and fire-and-forget. It sends a random visitor id (`localStorage`), a per-load
  session id, a load event, a few UI events (at most one per name per 10 s), and the ids on `/sample`
  and `/embed`. Reuse `js/usage.js` and `js/usage-lib.js`.
- **Tests:** `npm test` (maths against brute force), `tests/smoke.cjs` (headless Playwright),
  `tests/overflow.cjs`.
- **Repository rules.**
  - `TODO.md` is the running record: Done, Next, Open, Later, and **Decided (Tim)** lines. Write
    decisions down when they are made.
  - Design notes go in `docs/`. Commit messages say what changed for the visitor.
- **How Tim works with Claude.**
  - Claude recommends one option, Tim decides; then do exactly that, without extras he did not ask for.
  - Don't claim something was tested in a browser if it wasn't; say what was and wasn't checked.
  - Push to `main` when asked; it deploys.

## B6. Things learned the hard way

- **The pool limits what a reweighting can represent.** A target far from the proposal leans on a few
  completions; that is what pool fit / ESS measures. In the new demo the proposal choice (local product
  vs A vs B) makes this visible, so show ESS prominently.
- **Make the default view interesting and honest at once.** The DPP demo's default T is still Open:
  T = 20 gives diverse picks but poor fit for Sample. Choose the new demo's default τ (the product, as
  in the paper) and proposal so that the first screen shows the paper's point.
- **Expect repeat runs to hit the GPU server's rate limit.** Six 200-completion runs in about eight
  minutes did. The page must explain it.
- **Phones cache aggressively.** After a deploy, ask testers to reload.

---

# Open questions for Tim, in one list

1. The model for presets and live runs: a base model (closer to the paper's GPT-2) or Qwen3 14B (what
   the GPU server serves today)?
2. What "local" means for τ ≠ 0, or offer the global/local switch only at the product?
3. Is SMC itself (a particles knob, paper RQ3) in v1, or is v1 importance resampling from a fixed pool?
4. The preset pairs, and whether each comes with an explicit "A and B" prompt C.
5. Expert weights w: hidden at ½ : ½, or in the View menu?
6. Separate repository and site (suggested), or a second page in this one?
