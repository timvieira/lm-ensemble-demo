# To do

Moved from dpp-lm-demo's TODO.md on 2026-11-26 (dpp-lm-demo@5d42446), as it was. Paths in it are dpp-lm-demo's:
lab/ensemble.* is now index.html and js/, data/ensemble/ is now data/; modal/ and scripts/ are still in dpp-lm-demo.

## Open after the move (26 Nov): nothing here is resolved yet

### Needs Tim
- **Done: Hugging Face sign-in accepts this address** (26 Nov, 14:32). The OAuth app "Diverse sampling demo" (client
  43d09d55-…; huggingface.co/settings/connected-applications, Developer Applications, Settings) lists
  https://timvieira.github.io/dpp-lm-demo/, https://timvieira.github.io/lm-ensemble-demo/ and https://timvieira.github.io/,
  each in its own box (the + button; several in one box are read as one address). Hugging Face matches the address
  exactly: a parent such as the homepage does not cover the pages under it (checked). Each new demo page that signs in
  needs its own line. Still untested: a sign-in on the page, end to end, and a live run after it (Tim).
- **The cloud session that built the prototype still points at dpp-lm-demo.** Point it at this repository. Its
  commits to dpp-lm-demo's lab/ensemble* no longer reach any page (that address redirects here). One commit landed
  there after the move (3af271a, empty completions labelled) and was ported here (72d3fb3). Before closing this item,
  check dpp-lm-demo for later ensemble commits: `git log 3af271a.. -- lab/ensemble.html lab/ensemble.js lab/ensemble-math.js data/ensemble`.
  26 Nov: it is the cloud session "Missing arrows" (session_015dhVokNfYW6EJNMhx5kX4W; its 30 commits in dpp-lm-demo run from
  25 Nov 21:20 to 26 Nov 16:49 UTC). A message telling it about the move (paths, what stays in dpp-lm-demo, this
  TODO) was sent from Tim's laptop session; not confirmed read, and it asks the session to confirm with Tim.
  Still needed: Tim tells it in that session. It may also need the Claude GitHub app to have access to this new
  repository (not checked: the laptop's gh token cannot list app installations). No ensemble commits landed in
  dpp-lm-demo after the move (checked 26 Nov).
- **Where the GPU code lives.** Decided for now (Tim, 26 Nov): it stays in dpp-lm-demo's Modal app (`dpp-lm-sampler`:
  modal/app.py, modal/ensemble_lib.py, scripts/ensemble_pool.py) with the pools workflow, which reads dpp-lm-demo's
  data/ensemble/presets.json and commits pools there. So a preset change or new pools must be copied by hand between
  dpp-lm-demo's data/ensemble/ and data/ here. Open: whether it moves here later.
  26 Nov (Tim): a second backend lives here, a Hugging Face ZeroGPU Space (xtimv/lm-ensemble-demo, from space/), next
  to Modal; the visitor picks it under "Runs on". space/ensemble_lib.py is a copy of dpp-lm-demo's
  modal/ensemble_lib.py (at 099e77e): two copies to keep the same.
- **The unused copies in dpp-lm-demo.** Decided (Tim, 26 Nov): lab/ensemble*.js, data/ensemble/,
  tests/ensemble.test.mjs and docs/handoff-ensemble-demo.md stay there. They will drift from the copies here.
- **Maths typesetting.** The page typesets its explainers with KaTeX and writes symbols in labels, axes and cards as
  Unicode (τ, Ẑ, log p(· | …)). Tim's global rule asks for MathJax (SVG) for every symbol in every page; the
  handoff (B4, A8) says KaTeX. Does the rule apply to the demos?
- **What "product" means at τ = 0.** The page's product is the weighted geometric mean √(p_A p_B) (weights ½; the
  τ explainer says so), normalised. Whether the paper's Fig. 1 uses that or p_A·p_B has not been checked against
  the paper. Rankings and ρ are the same either way; "ensemble %", log Ẑ and "Uses n of N" are not.

### Found in the review of the page (26 Nov)
- **The importance weights are concentrated on most continue-text pools**, so log Ẑ and every card's "ensemble %"
  rest on a few draws. Effective sample size of 200 (proposal: the token-level product), computed from data/ on
  26 Nov (dpp-lm-demo@5d42446 pools):

  | pool | √(p_A p_B) (the page) | p_A·p_B |
  |---|---|---|
  | Qwen3 14B, text (5 pairs) | 1.6–7.1 | 2.5–24.2 |
  | GPT-2, text (5 pairs) | 1.7–5.8 | 2.2–4.5 |
  | Qwen3 14B, chat (5 pairs) | 1.0–200 | 1.0–194 |

  Physicist × author (Qwen3, text), the page's default: ESS 1.6, and one draw ("Toni Morrison.") carries 79% of the
  weight. Dessert × vegetable (chat): ESS 1.0 ("Cucumber" 99%). The page shows "Uses n of N", so this is disclosed,
  but nothing says how uncertain the percentages are. Open: what to do (more draws, a better proposal, SMC as in
  the paper, or showing the uncertainty).
- **The chat pools are nearly deterministic: 1 to 6 distinct completions in 200.** Painter × inventor is
  "Leonardo da Vinci" 200 times (hence ESS 200: equal weights, one string); physicist × author is 3 strings
  (Albert Einstein ×157, Stephen Hawking ×41, J.K. Rowling ×2); dessert × vegetable is "Carrot cake" in three
  spellings plus one more. The token-level product of two chat replies is very peaked at T′ = 1, so these pools
  show little of either prompt's spread. Open: what the chat format should show (a higher T′ for the proposal?).
- **"____." ranks 3rd for physicist × author** (Qwen3, text): a fill-in-the-blank completion. Relates to the open
  preset-wording item below.
- **Plot labels still crowd:** "Richard Feynman." and "Stephen Hawking." (physicist × author) sit one under the other,
  and Hawking's label is 16 px below its dot (the labels are only pushed apart vertically). Rechecked after the
  full-page layout: unchanged.
- **The page makes claims about the paper that have not been checked against it** (they came from the handoff):
  the τ explainer says the product is "the paper's main ensemble" and that each τ is the distribution closest to
  both prompts under an α-divergence with α = 1 − τ; the explicit-prompt explainer quotes the paper's ρ = 0.96
  (global) and ρ = 0.09 (local). Check each against the paper's PDF (the conference version).
- **Zoomed out, the plot shows impossible values.** The zoom goes down to 0.2× as on the diversity demo, but here that
  shows log p up to +40 (Tim's phone, 26 Nov), where no completion can be. Open: stop zooming out at the first view (1×)?
- **On a phone the labels drift far from their dots** (Tim's screenshot, 26 Nov: "Albert Einstein." about 50 px from
  its point): on a narrow frame most labels sit left of their dots and are pushed down one by one. Same cause as the
  crowding item above.
- **The banner is stale:** "only the example pairs have completions so far", but live runs exist since
  dpp-lm-demo e54c873 (they need Hugging Face sign-in; see the first item).

### Open: the Hugging Face Space backend (26 Nov)
- **Done: the Space is xtimv/lm-ensemble-demo** (Tim renamed it, 26 Nov; it was created as xtimv/LM-ensemble-demo,
  which now redirects). Its address, xtimv-lm-ensemble-demo.hf.space (the page's SPACE_URL), did not change.
- **Done: the deploy token** (26 Nov). The repository secret HF_TOKEN is set (Tim); the first deploy uploaded space/ to
  the Space (its commit b7bb356). (Before the rename, the upload followed Hugging Face's redirect from the lowercase name.)
- **Tested on ZeroGPU** (26 Nov, through the page's own client, scripts/space-check.mjs, on Tim's quota): the Space built
  and ran on zero-a10g in about 3 minutes; it offers Qwen3 14B (the unpacked AWQ weights) and GPT-2; Gradio 6.32's
  queue protocol works with js/space.js. Pools: GPT-2 text n = 4; Qwen3 14B text n = 8, chat n = 4, and text n = 200 at
  32 tokens (the page's default) in 17.5 s (9 s sampling, 8 s rescoring). With seed 0 the first Qwen3 completion
  starts as in the cached Modal pool ("Kip Thorne, and I'm trying to understand...") and then diverges: the same
  weights on another GPU round differently.
- **The self-check differs by up to 0.11 nats on ZeroGPU** (sampling sums against teacher-forced scores, both in fp16:
  0.049 text, 0.106 chat, 0.073 text at n = 200). The record reports it as checkMaxAbsErr; nothing here filters on it.
  dpp-lm-demo's pools workflow fails a pool above 0.5 nats and reports 0.09 seen on Modal. The page uses the sampling
  sums (logpA, logpB).
- **The GPU time reserved per run is about 4 times what a run takes** (space/app.py, _duration: 68 s reserved for n =
  200 at 32 tokens; 17 s measured). Open: whether ZeroGPU charges the visitor's quota by the time reserved or used; if
  reserved, a tighter estimate saves visitors' quota (but a run that outlasts its reservation is cut off).
- **Untested on ZeroGPU:** the quota message (ZeroGPU's own error text, matched by "quota"), and a run from the page
  itself (it needs sign-in, which needs the redirect URI above).

### Untested after the move
- Hugging Face sign-in end to end, and a live run (/ensemble_sample) from this address.
- The page on a real phone; pinch and two-finger moves on a real touch screen (not simulated: d3.zoom's own touch handling).

### Notes
- The repository was created as timvieira/prompt-intersection and renamed lm-ensemble-demo the same day (Tim);
  https://timvieira.github.io/prompt-intersection/ no longer exists, and nothing should link to it.
- js/explain.js and js/hfauth.js are copies of dpp-lm-demo's (explain.js without the DPP explainers); fixes there do
  not come here.
- On Tim's laptop the `gh` token has no `workflow` scope, so pushes that touch .github/workflows go over SSH
  (or after `gh auth refresh -s workflow`).

### In progress (26 Nov)
- When it has settled: send Robin Chan the link on Zulip (Tim). Not sent yet: Tim chose to fix the Hugging Face
  sign-in first (the redirect URI, above). The text is approved (26 Nov); it goes as a direct message to Robin
  (Zulip user 253) through the Zulip API with ~/zuliprc on Tim's laptop.

### Done (26 Nov)
- **A local clone** at ~/projects/lm-ensemble-demo on Tim's laptop (Tim, 26 Nov).
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
- **The redirect** from dpp-lm-demo's lab/ensemble.html keeps the query string and fragment (checked live:
  `?x=1#y` lands on https://timvieira.github.io/lm-ensemble-demo/?x=1#y).
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
- **A second backend, a Hugging Face ZeroGPU Space** (Tim): space/ (app.py, a copy of dpp-lm-demo's ensemble_lib.py,
  README.md as the Space card), deployed by .github/workflows/space.yml; js/space.js calls it through Gradio's queue with
  the visitor's token; "Runs on" in the settings picks Modal or the Space; the Space's errors (quota, not answering,
  busy) have their own messages, with "Run on Modal". The status line says where a pool was drawn.
- **Card meta lines** break between fields, never inside one ("log p_A −1.1" stays together).
- **The explainers' maths was broken in 9 of 16** (since they were written in dpp-lm-demo ad96dde): the TeX sat in
  JS template literals with single backslashes, so \varphi, \tau, \rho, \frac became control characters and
  \sum, \mathrm, \hat lost their backslash. Doubled (54 backslashes); KaTeX now renders them with no errors.
  dpp-lm-demo's copy (lab/ensemble.js) still has the bug.

## Prototype: the ensemble demo (prompt intersection, arXiv:2603.05432 Fig. 1)
- Brief: docs/handoff-ensemble-demo.md. Built here as an unlinked lab page (lab/ensemble.html), to move
  to its own repository once it works.
- Decided (Tim): Qwen3 14B (the GPU server's model); cached pools for the presets during development
  (data/ensemble/presets.json).
- Waiting on Manuel: `/score` (texts scored under given prompts) first; the token-level product sampler
  with local scores second. Today's server has neither (checked 26 Nov). With `/score` alone, a 50/50
  mixture of A and B draws gives the global ensemble exactly; local scores wait for the sampler.
- Done (26 Nov): Qwen3 14B pools for the five presets, drawn on Modal (modal/ensemble_lib.py through the manual
  "Ensemble prototype pools" workflow; about 15 s of GPU per preset). The token-level product sampler and local
  scores run there, so the GPU server's missing routes no longer block the prototype. Physicist × author
  reproduces the paper's direction: against the explicit prompt, ρ = 0.94 global (product) vs 0.77 local.
- Done: lab/ensemble.html has the composer (prompts A and B, examples, settings, ↑), the τ/K panel with ticks at
  the named means, a View menu (Scored global/local) and "?" explainers; ↑ on anything but an example pair at
  its settings says live runs aren't connected yet and offers an example.
- Open: preset wording. "A famous scientist is" continues as "…coming to our school next week."; better
  phrasings (e.g. "My favorite scientist is") or a different pair list.
- Decided (Tim): completions are one sentence, with exceptions for non-sentence punctuation. A period ends a
  sentence when the next token starts with whitespace or ends the text, and it is not an abbreviation (Dr., Mr.,
  e.g., U.S., …) or an initial ("J."). The ending event's probability (all such next tokens) is part of every
  score (modal/ensemble_lib.py, sentence_period). Known trade-off: "Vitamin C." reads as an initial.
- Decided (Tim): no K and no Sample in the ensemble prototype. The rail ranks every distinct completion by its
  ensemble score (exact), with its probability under the ensemble (φ/Ẑ, Ẑ estimated from the pool); ten shown
  until "Show all"; the five best are red and labelled in the plot.
- Decided (Tim): pySBD's English rules that need only the next word, adopted as next-token ending sets
  (modal/ensemble_lib.py, ending_kind and ending_masks): "." and "?" end before whitespace; "!", "..." and
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
  (endToken), to spot e.g. replies that run into <|im_start|>. Before, only <|im_end|>/<|endoftext|> ended one
  and other special tokens were kept but hidden by decoding.
- Done: live runs on Modal for the ensemble prototype. /ensemble_sample on the Modal web app (Hugging Face
  sign-in, the shared daily run limit and monthly GPU budget), two models: Qwen3 14B (the existing L40S
  container; text and chat formats) and GPT-2 (124M, its own T4 container, charged to the meter at T4 prices;
  text only). The page offers both in the settings and runs anything not cached; pools are
  data/ensemble/<pair>.<model>.<format>.json, drawn for both models by the "Ensemble prototype pools" workflow.
- Check: Hugging Face sign-in from the lab page redirects back to lab/ensemble.html; if the OAuth app does not
  list that address as a redirect URL, add it (huggingface.co/settings/connected-applications, Developer Applications, "Diverse sampling demo", Settings). The token is shared with
  the main page (same origin), so signing in there works too.

