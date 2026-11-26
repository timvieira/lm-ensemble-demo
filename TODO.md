# To do

Moved from dpp-lm-demo's TODO.md on 2026-11-26 (dpp-lm-demo@5d42446), as it was. Paths in it are dpp-lm-demo's:
lab/ensemble.* is now index.html and js/, data/ensemble/ is now data/; modal/ and scripts/ are still in dpp-lm-demo.

## Left over from the move (26 Nov)
- Hugging Face sign-in: the page asks Hugging Face to send the visitor back to its own address, so
  https://timvieira.github.io/prompt-intersection/ must be a redirect URI of the OAuth app
  (huggingface.co/settings/applications). Until then, signing in here fails.
- Pools and presets are drawn in dpp-lm-demo (README, "The backend is still dpp-lm-demo's"). Decide whether the
  GPU code moves here too.

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
  list that address as a redirect URL, add it (huggingface.co/settings/applications). The token is shared with
  the main page (same origin), so signing in there works too.

