// Small "?" buttons that open a plain-language explainer with the math typeset.
// Any element with class "xbtn" and data-x="<key>" opens EXPLAIN[key].

// Filled by the page (js/ensemble.js adds its own explainers). The DPP demo's entries are not copied.
export const EXPLAIN = {};

let openBtn = null;

function typeset(el) {
  window.renderMathInElement?.(el, {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
    throwOnError: false,
  });
}

export function closeExplainer() {
  const pop = document.getElementById('xpop');
  pop.hidden = true;
  openBtn?.setAttribute('aria-expanded', 'false');
  openBtn = null;
}

function openExplainer(btn) {
  const item = EXPLAIN[btn.dataset.x];
  if (!item) return;
  const pop = document.getElementById('xpop');
  pop.innerHTML = `<div class="xp-title">${item.title}</div>${item.body}<button type="button" class="xp-close" aria-label="Close">×</button>`;
  typeset(pop);
  pop.querySelector('.xp-close').onclick = closeExplainer;
  pop.hidden = false;
  openBtn?.setAttribute('aria-expanded', 'false');
  openBtn = btn;
  btn.setAttribute('aria-expanded', 'true');
  // Place below the button (above if there is no room), clamped to the viewport.
  const r = btn.getBoundingClientRect();
  const w = Math.min(360, window.innerWidth - 24);
  pop.style.width = w + 'px';
  const left = Math.max(12, Math.min(window.innerWidth - w - 12, r.left + r.width / 2 - w / 2));
  const h = pop.offsetHeight;
  let top = r.bottom + 8;
  if (top + h > window.innerHeight - 8 && r.top - h - 8 > 8) top = r.top - h - 8;
  pop.style.left = left + window.scrollX + 'px';
  pop.style.top = Math.max(8, top) + window.scrollY + 'px';
}

export function initExplainers() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.xbtn');
    if (btn) {
      e.preventDefault(); // don't toggle the checkbox or focus the input of a surrounding <label>
      e.stopPropagation();
      if (openBtn === btn) closeExplainer(); else openExplainer(btn);
      return;
    }
    if (!e.target.closest('#xpop')) closeExplainer();
  }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeExplainer(); });
  window.addEventListener('resize', closeExplainer);
}
