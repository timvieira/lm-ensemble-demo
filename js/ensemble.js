// Prompt intersection: a prototype for "Ensembling Language Models with SMC" (arXiv:2603.05432, Fig. 1).
// Brief: docs/handoff-ensemble-demo.md. Pools: data/ensemble/<preset>.json (modal/ensemble_lib.py).
import * as d3 from 'd3';
import { logGenMean, tauOf, uOf, tauName, NAMED, summarize, spearman } from './ensemble-math.js';
import { EXPLAIN, initExplainers } from '../js/explain.js';

const DATA = '../data/ensemble/';
const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
const xb = (k) => `<button type="button" class="xbtn" data-x="${k}" aria-label="Explain">?</button>`;
const words = (t, n = 5) => { const w = t.trim().split(/\s+/); return w.length > n ? w.slice(0, n).join(' ') + '…' : w.join(' '); };

const S = { presets: [], pool: null, groups: [], u: 0.5, K: 5, mode: 'map', score: 'global', draw: 1, sel: [], open: new Set(), hover: -1 };

// ---------------------------------------------------------------- data
async function init() {
  Object.assign(EXPLAIN, ENSEMBLE_EXPLAIN);
  initExplainers();
  S.presets = await (await fetch(DATA + 'presets.json')).json();
  const ex = $('#examples');
  ex.innerHTML = '<div class="menu-head">Example pairs</div>' + S.presets.map((p, i) =>
    `<button type="button" role="menuitem" data-i="${i}"><b>${esc(p.label)}</b><span>${esc(p.A)} · ${esc(p.B)}</span></button>`).join('');
  ex.querySelectorAll('button').forEach((b) => (b.onclick = () => { $('#exmenu').open = false; fill(S.presets[+b.dataset.i]); settingsChanged(); }));
  wire();
  fill(S.presets[0]);
  await settingsChanged();
}

// ---------------------------------------------------------------- the composer
// Cached pools exist only for the example pairs at their settings; anything else needs a live run, which this
// prototype does not have yet (docs/handoff-ensemble-demo.md, A7). The page says so instead of hiding the controls.
const DEFAULTS = { N: 200, Tprime: 1.0, maxNewTokens: 32 };
function fill(p) { $('#pa').value = p.A; $('#pb').value = p.B; $('#pc').value = p.C ?? ''; syncGo(); }
function readConfig() {
  return { A: $('#pa').value.trim(), B: $('#pb').value.trim(), C: $('#pc').value.trim(),
    N: Math.round(+$('#n').value), Tprime: Math.round(10 * +$('#tp').value) / 10, maxNewTokens: Math.round(+$('#maxtok').value) };
}
function cachedFor(c) {
  const p = S.presets.find((x) => x.A === c.A && x.B === c.B);
  const same = c.N === DEFAULTS.N && c.Tprime === DEFAULTS.Tprime && c.maxNewTokens === DEFAULTS.maxNewTokens;
  return p && same && (!c.C || c.C === p.C) ? p : null;
}
function syncGo() { $('#go').disabled = !($('#pa').value.trim() && $('#pb').value.trim()); $('#adv-sum').textContent = `Qwen3 14B · ${$('#n').value} samples`; }
async function settingsChanged() {
  syncGo();
  const c = readConfig(), p = cachedFor(c);
  if (p) { if (S.pool?.id !== p.id) await load(p.id); stageMsg(null); return; }
  if (S.pool) setStatus('The plot is from earlier prompts or settings. Press ↑.');
}
function run() {
  const c = readConfig();
  if (!c.A || !c.B) return;
  const p = cachedFor(c);
  if (p) { load(p.id); stageMsg(null); return; }
  const near = S.presets.find((x) => x.A === c.A && x.B === c.B);
  stageMsg(near
    ? { title: 'No completions for these settings yet.',
        detail: `This prototype has completions for the example pairs at N ${DEFAULTS.N}, T′ ${DEFAULTS.Tprime.toFixed(1)} and ${DEFAULTS.maxNewTokens} tokens only.`,
        actions: '<button type="button" class="btn" data-act="reset">Use those settings</button>' }
    : { title: 'Live runs aren’t connected yet.',
        detail: 'This prototype only has completions for the example pairs. Running your own prompts needs the ensemble route on Modal, which is not live yet.',
        actions: '<button type="button" class="btn" data-act="example">Use an example</button>' });
}
function stageMsg(m) {
  const el = $('#stage-msg');
  if (!m) { el.hidden = true; el.innerHTML = ''; return; }
  el.className = 'stage-msg error';
  el.innerHTML = `<p class="t">${m.title}</p><p class="d">${m.detail}</p><div class="a">${m.actions}${S.pool ? '<button type="button" class="btn" data-act="dismiss">Dismiss</button>' : ''}</div>`;
  el.hidden = false;
  el.querySelectorAll('[data-act]').forEach((b) => (b.onclick = () => {
    const a = b.dataset.act;
    if (a === 'dismiss') stageMsg(null);
    if (a === 'example') { stageMsg(null); setTimeout(() => { $('#exmenu').open = true; }); }   // after this click's close-menus pass
    if (a === 'reset') { $('#n').value = DEFAULTS.N; $('#tp').value = DEFAULTS.Tprime.toFixed(1); $('#maxtok').value = DEFAULTS.maxNewTokens; settingsChanged(); }
  }));
}
function setStatus(t) { $('#status').textContent = t; }

async function load(id) {
  const r = await fetch(DATA + id + '.json').catch(() => null);
  if (!r || !r.ok) { stageMsg({ title: 'These completions could not be loaded.', detail: 'Reload the page to try again.', actions: '' }); return; }
  const pool = await r.json();
  S.pool = pool;
  setStatus(`${pool.standIn ? 'Stand-in data' : 'Cached'}: ${pool.N} completions from ${pool.model.split('/').pop()}, drawn on ${pool.standIn ? 'a CPU' : 'Modal'}.`);
  // One group per distinct string; copies share every score.
  const byText = new Map();
  pool.samples.forEach((s, i) => { const g = byText.get(s.text); if (g) g.idx.push(i); else byText.set(s.text, { i, idx: [i] }); });
  S.groups = [...byText.values()];
  S.open = new Set(); S.showAll = false;
  plot.reset();
  update();
}

// ---------------------------------------------------------------- the ensemble
function tau() { return tauOf(S.u); }
function scoreOf(s) { return S.score === 'local' ? s.loglocal : logGenMean([s.logpA, s.logpB], tau()); }

function update() {
  if (!S.pool) return;
  const P = S.pool, T = tau(), name = tauName(T);
  document.querySelectorAll('#ticks span').forEach((t) => t.classList.toggle('on', S.score === 'global' && Math.abs(+t.dataset.u - S.u) < 1e-9));
  $('#tau-out').textContent = S.score === 'local' ? 'local product' : `${name ?? 'τ = ' + fmt(T, 2)}${name && Number.isFinite(T) ? ` (τ = ${T})` : ''}`;
  const score = P.samples.map(scoreOf);
  const logv = P.samples.map((s, i) => score[i] - s.logr);    // importance weights: target / proposal
  const sum = summarize(logv);
  const gScore = S.groups.map((g) => score[g.i]);
  // Every distinct completion, ranked by its score under the ensemble. Its probability under the ensemble is
  // φ(y)/Z, with Z estimated from the pool by importance weighting (Ẑ = mean of φ/r); the rank is exact.
  const rank = d3.range(S.groups.length).sort((a, b) => gScore[b] - gScore[a]);
  const gProb = gScore.map((v) => Math.exp(v - sum.logZ));
  S.sel = rank.slice(0, LABELLED).map((g) => S.groups[g].i);   // the best five: red, and labelled in the plot
  Object.assign(S, { scoreArr: score, sum, rank, gProb });
  readouts(score);
  cards(rank, gProb);
  plot.draw();
  valid.draw(score);
}

function readouts(score) {
  const P = S.pool, N = P.N, distinct = S.groups.length;
  $('#sel-sub').textContent = `${N} completions, ${distinct} distinct · ${S.score === 'local' ? 'local scores' : 'scored at ' + (tauName(tau()) ?? 'τ = ' + fmt(tau(), 2))}`;
  const rows = [
    [`Uses${xb('ess')}`, `${fmt(S.sum.ess, 1)} of ${N} completions`],
    [`log Ẑ${xb('logZ')}`, fmt(S.sum.logZ, 2)],
  ];
  $('#readout').innerHTML = rows.map(([k, v]) => `<span>${k}</span><span class="v">${v}</span>`).join('');
}

const SHOWN = 10, LABELLED = 5;   // display only: the list shows ten until "Show all"; the plot labels five
const pctOf = (p) => (p >= 0.001 ? `${fmt(100 * p, p >= 0.1 ? 0 : 1)}%` : '< 0.1%');
function cards(rank, gProb) {
  const P = S.pool, top = new Set(S.sel), list = S.showAll ? rank : rank.slice(0, SHOWN);
  $('#cards').innerHTML = list.map((g, r) => {
    const G = S.groups[g], s = P.samples[G.i];
    return `<li class="card${S.open.has(G.i) ? ' open' : ''}" data-i="${G.i}">
      <div class="txt">${esc(s.text.trim())}${s.ended ? '' : ' …'}</div>
      <div class="meta"><span class="rank">${r + 1}.</span>${top.has(G.i) ? '<span class="dot"></span>' : ''}${G.idx.length > 1 ? `×${G.idx.length} · ` : ''}log p<sub>A</sub> ${fmt(s.logpA)} · log p<sub>B</sub> ${fmt(s.logpB)} · ensemble ${pctOf(gProb[g])}${xb('prob')}${s.ended ? '' : `<span class="cutoff">cut off at ${s.nTokens} tokens</span>`}</div></li>`;
  }).join('');
  const more = $('#more');
  more.hidden = rank.length <= SHOWN;
  more.textContent = S.showAll ? `Show the top ${SHOWN}` : `Show all ${rank.length}`;
  more.onclick = () => { S.showAll = !S.showAll; cards(S.rank, S.gProb); };
  $('#cards').querySelectorAll('.card').forEach((el) => {
    const i = +el.dataset.i;
    el.onmouseenter = () => plot.highlight(i);
    el.onmouseleave = () => plot.highlight(-1);
    el.onclick = (e) => { if (e.target.closest('.xbtn')) return; el.classList.toggle('open'); if (el.classList.contains('open')) S.open.add(i); else S.open.delete(i); };
  });
}

// ---------------------------------------------------------------- the plot: (log p_A, log p_B)
const plot = {
  svg: null,
  reset() { this.x = null; },
  build() {
    const svg = d3.select('#plot');
    svg.selectAll('*').remove();
    this.gGrid = svg.append('g');
    this.gContour = svg.append('g');
    this.gDiag = svg.append('g');
    this.gDot = svg.append('g');
    this.gSel = svg.append('g');
    this.gLbl = svg.append('g');
    this.gAxis = svg.append('g');
    this.svg = svg;
    svg.on('pointermove', (ev) => { const i = this.nearest(ev); this.highlight(i); tip(i, ev); })
      .on('pointerleave', () => { this.highlight(-1); tip(-1); });
  },
  frame() {
    // Framed once per pool (the view never moves with τ, K or the mode), over every completion.
    const el = $('#plot'), W = el.clientWidth, H = el.clientHeight, P = S.pool.samples;
    const m = { l: 56, r: 16, t: 14, b: 46 };
    const ext = (k) => { const v = P.map((s) => s[k]); const lo = d3.min(v), hi = d3.max(v), p = 0.04 * (hi - lo || 1); return [lo - p, hi + p]; };
    this.W = W; this.H = H; this.m = m;
    this.x = d3.scaleLinear().domain(ext('logpA')).range([m.l, W - m.r]);
    this.y = d3.scaleLinear().domain(ext('logpB')).range([H - m.b, m.t]);
    const ax = this.gAxis;
    ax.selectAll('*').remove();
    ax.append('g').attr('class', 'tick-x').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(this.x).ticks(6).tickSize(4));
    ax.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(this.y).ticks(6).tickSize(4));
    ax.append('text').attr('class', 'axis-lab').attr('x', (m.l + W - m.r) / 2).attr('y', H - 8).attr('text-anchor', 'middle')
      .text(`log p( · | “${S.pool.prompts.A}”)`);
    ax.append('text').attr('class', 'axis-lab').attr('transform', `translate(14,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle')
      .text(`log p( · | “${S.pool.prompts.B}”)`);
    // Soft grid, as on the main page's floor.
    this.gGrid.selectAll('*').remove();
    const g = this.gGrid, grid = css('--grid');
    this.x.ticks(12).forEach((v) => g.append('line').attr('x1', this.x(v)).attr('x2', this.x(v)).attr('y1', m.t).attr('y2', H - m.b).attr('stroke', grid));
    this.y.ticks(12).forEach((v) => g.append('line').attr('y1', this.y(v)).attr('y2', this.y(v)).attr('x1', m.l).attr('x2', W - m.r).attr('stroke', grid));
    // The diagonal log p_A = log p_B, where every member of the family agrees.
    const [a0, a1] = this.x.domain(), [b0, b1] = this.y.domain(), lo = Math.max(a0, b0), hi = Math.min(a1, b1);
    this.gDiag.selectAll('*').remove();
    if (hi > lo) this.gDiag.append('line').attr('x1', this.x(lo)).attr('y1', this.y(lo)).attr('x2', this.x(hi)).attr('y2', this.y(hi))
      .attr('stroke', css('--muted')).attr('stroke-dasharray', '3 4').attr('stroke-width', 1);
    this.pos = P.map((s) => [this.x(s.logpA), this.y(s.logpB)]);
  },
  draw() {
    if (!this.svg) this.build();
    if (!this.x) this.frame();
    const P = S.pool.samples, sel = new Set(S.sel);
    this.contours();
    this.gDot.selectAll('circle').data(d3.range(P.length)).join('circle')
      .attr('cx', (i) => this.pos[i][0]).attr('cy', (i) => this.pos[i][1]).attr('r', 2.3)
      .attr('fill', css('--ink')).attr('fill-opacity', 0.55);
    this.gSel.selectAll('circle').data(S.sel, (i) => i).join('circle')
      .attr('cx', (i) => this.pos[i][0]).attr('cy', (i) => this.pos[i][1]).attr('r', 4.2)
      .attr('fill', css('--accent')).attr('stroke', css('--surface')).attr('stroke-width', 1);
    this.labels(sel);
  },
  contours() {
    this.gContour.selectAll('*').remove();
    if (S.score === 'local') return;   // the local score is not a function of (log p_A, log p_B)
    const T = tau(), n = 120, [a0, a1] = this.x.domain(), [b0, b1] = this.y.domain();
    const vals = new Float64Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const a = a0 + ((i + 0.5) / n) * (a1 - a0), b = b1 - ((j + 0.5) / n) * (b1 - b0);
      vals[j * n + i] = logGenMean([a, b], T);
    }
    const sc = S.scoreArr, lo = d3.min(sc), hi = d3.max(sc);
    const levels = d3.range(1, 9).map((k) => lo + (k / 9) * (hi - lo));
    const proj = d3.geoTransform({ point(px, py) {
      this.stream.point(plot.x.range()[0] + (px / n) * (plot.x.range()[1] - plot.x.range()[0]), plot.y.range()[1] + (py / n) * (plot.y.range()[0] - plot.y.range()[1]));
    } });
    const path = d3.geoPath(proj), cont = d3.contours().size([n, n]);
    const draw = (lv, strong) => this.gContour.append('path').attr('d', path(cont.contour(vals, lv)))
      .attr('fill', 'none').attr('stroke', strong ? css('--accent') : css('--text-2'))
      .attr('stroke-opacity', strong ? 0.6 : 0.22).attr('stroke-width', strong ? 1.2 : 1)
      .attr('stroke-dasharray', strong ? '5 3' : null);
    levels.forEach((lv) => draw(lv, false));
  },
  labels(sel) {
    const items = S.sel.map((i) => ({ i, x: this.pos[i][0], y: this.pos[i][1] })).sort((a, b) => a.y - b.y);
    const placed = [];
    for (const it of items) {
      let y = it.y + 4;
      for (const q of placed) if (Math.abs(q.x - it.x) < 160 && Math.abs(q.y - y) < 16) y = q.y + 16;
      placed.push({ x: it.x, y });
      it.ly = y;
      it.anchor = it.x > this.W - 200 ? 'end' : 'start';
    }
    this.gLbl.selectAll('text').data(items, (d) => d.i).join('text').attr('class', 'lbl')
      .attr('x', (d) => d.x + (d.anchor === 'end' ? -8 : 8)).attr('y', (d) => d.ly).attr('text-anchor', (d) => d.anchor)
      .text((d) => words(S.pool.samples[d.i].text));
  },
  nearest(ev) {
    if (!this.pos) return -1;
    const [px, py] = d3.pointer(ev, this.svg.node());
    let best = -1, bd = 14 * 14;
    this.pos.forEach(([a, b], i) => { const d = (a - px) ** 2 + (b - py) ** 2; if (d < bd) { bd = d; best = i; } });
    return best;
  },
  highlight(i) {
    S.hover = i;
    if (!this.svg) return;
    this.gDot.selectAll('circle').attr('stroke', (j) => (j === i ? css('--text') : null)).attr('stroke-width', (j) => (j === i ? 1.5 : null));
    document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i));
    valid.highlight(i);
  },
};

function tip(i, ev) {
  const el = $('#tip');
  if (i < 0 || !S.pool) { el.hidden = true; return; }
  const s = S.pool.samples[i];
  el.innerHTML = `${esc(s.text.trim())}${s.ended ? '' : ' …'}<div class="m">log p_A ${fmt(s.logpA, 2)} · log p_B ${fmt(s.logpB, 2)}${s.logpC != null ? ` · log p_C ${fmt(s.logpC, 2)}` : ''} · local ${fmt(s.loglocal, 2)}</div>`;
  el.hidden = false;
  const r = el.getBoundingClientRect();
  el.style.left = Math.min(ev.clientX + 14, innerWidth - r.width - 8) + 'px';
  el.style.top = Math.min(ev.clientY + 14, innerHeight - r.height - 8) + 'px';
}

// ---------------------------------------------------------------- Fig. 1, right: against the explicit prompt
const valid = {
  draw(score) {
    const P = S.pool, svg = d3.select('#valid');
    svg.selectAll('*').remove();
    if (!P.prompts.C || P.samples[0].logpC == null) { $('#valid-sub').textContent = 'No explicit prompt for this pair.'; return; }
    const glob = P.samples.map((s) => logGenMean([s.logpA, s.logpB], tau())), loc = P.samples.map((s) => s.loglocal), c = P.samples.map((s) => s.logpC);
    const rg = spearman(glob, c), rl = spearman(loc, c);
    $('#valid-sub').innerHTML = `“${esc(P.prompts.C)}”. Rank correlation ρ = <b>${fmt(rg, 2)}</b> global${S.score === 'global' && tauName(tau()) ? ` (${tauName(tau())})` : ''}, <b>${fmt(rl, 2)}</b> local.`;
    const el = $('#valid'), W = el.clientWidth, H = el.clientHeight, m = { l: 40, r: 8, t: 6, b: 30 };
    const xs = S.score === 'local' ? loc : glob;
    const x = d3.scaleLinear().domain(d3.extent(xs)).nice().range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain(d3.extent(c)).nice().range([H - m.b, m.t]);
    svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(4).tickSize(3));
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(3));
    svg.append('text').attr('class', 'axis-lab').attr('x', (m.l + W - m.r) / 2).attr('y', H - 2).attr('text-anchor', 'middle')
      .text(S.score === 'local' ? 'local score' : 'global score (τ as set)');
    svg.append('text').attr('class', 'axis-lab').attr('transform', `translate(10,${(m.t + H - m.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('log p_C');
    const sel = new Set(S.sel);
    this.dots = svg.append('g').selectAll('circle').data(d3.range(P.N)).join('circle')
      .attr('cx', (i) => x(xs[i])).attr('cy', (i) => y(c[i])).attr('r', (i) => (sel.has(i) ? 3.6 : 2))
      .attr('fill', (i) => (sel.has(i) ? css('--accent') : css('--ink'))).attr('fill-opacity', (i) => (sel.has(i) ? 1 : 0.5));
  },
  highlight(i) { this.dots?.attr('stroke', (j) => (j === i ? css('--text') : null)); },
};

// ---------------------------------------------------------------- controls
function wire() {
  const ticks = $('#ticks');
  ticks.innerHTML = NAMED.filter((t) => t.name !== 'quadratic').map((t) => `<span style="--u:${uOf(t.tau)}" data-u="${uOf(t.tau)}">${t.name}</span>`).join('');
  ticks.querySelectorAll('span').forEach((s) => (s.onclick = () => { S.u = +s.dataset.u; $('#tau').value = Math.round(1000 * S.u); update(); }));
  let raf = 0;
  $('#tau').oninput = (e) => {
    let u = +e.target.value / 1000;
    for (const t of NAMED) { const v = uOf(t.tau); if (Math.abs(u - v) < 0.012) u = v; }   // snap to the named means
    S.u = u;
    cancelAnimationFrame(raf); raf = requestAnimationFrame(update);
  };
  ['#pa', '#pb', '#pc', '#n', '#tp', '#maxtok'].forEach((id) => $(id).addEventListener('input', () => { clearTimeout(S.ct); S.ct = setTimeout(settingsChanged, 250); }));
  ['#pa', '#pb'].forEach((id) => $(id).addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); } }));
  $('#go').onclick = run;
  document.addEventListener('click', (e) => { for (const id of ['#exmenu', '#adv', '#viewmenu']) { const m = $(id); if (m.open && !e.composedPath().includes(m) && !e.target.closest('.xpop')) m.open = false; } });
  document.querySelectorAll('[data-score]').forEach((b) => (b.onclick = () => {
    S.score = b.dataset.score;
    document.querySelectorAll('[data-score]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    $('#tau').disabled = S.score === 'local';
    update();
  }));
  let rt = 0;
  new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { if (S.pool) { plot.reset(); plot.draw(); valid.draw(S.scoreArr); } }, 150); }).observe($('#plot'));
}

// ---------------------------------------------------------------- the "?" explainers (js/explain.js renders them)
const ENSEMBLE_EXPLAIN = {
  prompts: {
    title: 'Two prompts',
    body: `<p>The same language model, prompted two ways. Each completion $y$ gets a probability from each: $p_A(y)$ after prompt A and $p_B(y)$ after prompt B. The page looks for completions that are likely under <i>both</i>: for “My favorite physicist is” and “My favorite author is”, physicists who also write.</p>
      <p>Combining the two is <b>ensembling</b> the model with itself.</p>`,
  },
  tau: {
    title: 'τ: how the two prompts are combined',
    body: `<p>Each completion's score combines its two probabilities with a generalised mean:
      $$\varphi_\tau(y) = \Big(\tfrac12\, p_A(y)^\tau + \tfrac12\, p_B(y)^\tau\Big)^{1/\tau}.$$</p>
      <ul>
        <li><b>min</b> ($\tau = -\infty$): only as good as the prompt that likes it less. Strict agreement.</li>
        <li><b>harmonic</b> ($\tau = -1$) and <b>product</b> ($\tau \to 0$, $\sqrt{p_A p_B}$): agreement, less strictly. The product is the paper's main ensemble.</li>
        <li><b>sum</b> ($\tau = 1$): a mixture; either prompt liking it helps.</li>
        <li><b>max</b> ($\tau = +\infty$): good if either prompt likes it. Coverage.</li>
      </ul>
      <p>Each value of $\tau$ is the distribution closest to both prompts under one α-divergence ($\alpha = 1 - \tau$). The grey lines in the plot are where $\varphi_\tau$ is constant.</p>`,
  },
  scored: {
    title: 'Global or local scores',
    body: `<p><b>Global</b> scores each whole string: $\varphi_\tau(y)$ from the two prompts' probabilities of the complete completion. This is the ensemble the paper targets.</p>
      <p><b>Local</b> uses the token-level product: at every step, multiply the two prompts' next-token probabilities and renormalise. That is how ensembling is usually done, and it is not the same thing. It rewards tokens both prompts like one at a time, so generic openings (“a”, “the”) gain and specific answers can lose (the paper's Fig. 1: “a great guy.” rises, “Carl Sagan.” falls). The local score exists only for the product, so τ is off while it is on.</p>`,
  },
  proposal: {
    title: 'Where the completions come from',
    body: `<p>The completions are drawn from the token-level product of A and B: each next token is picked with probability proportional to $p_A \cdot p_B$ (at temperature $T'$). The probability of drawing a completion, $r(y)$, is recorded, so the ensemble's probabilities can be estimated from the pool (see <b>ensemble %</b> on the cards).</p>
      <p>The paper's Fig. 1 used beam search; drawing at random is needed here because the corrections need $r(y)$. Other choices (A alone, B alone, a mixture) are planned.</p>`,
  },
  model: {
    title: 'The model',
    body: `<p>Qwen3 14B with 4-bit (AWQ) weights, the same weights as the diversity demo, run on Modal's GPU. The prompts are continued as plain text, without a chat template, like the paper's GPT-2 example.</p>`,
  },
  N: { title: 'Samples N', body: `<p>How many completions are drawn. The ranking and every readout are computed from these N; more completions represent the ensemble better.</p>` },
  Tprime: { title: 'Sampling temperature T′', body: `<p>The temperature at which each token is drawn: $r(y_t \mid y_{<t}) \propto (p_A \cdot p_B)^{1/T'}$. At $T' = 1$ the draws follow the token-level product exactly.</p>` },
  maxtok: { title: 'Max tokens', body: `<p>A completion stops at the end of its first sentence, or after this many tokens. One that hits the limit is marked “cut off”.</p>` },
  stop: {
    title: 'One sentence',
    body: `<p>As in the paper, every completion is a short answer ending in a period. A period ends it when both hold:</p>
      <ul>
        <li>the next token starts with a space or a new line, or ends the text, so “0.5” is not cut at “0.”;</li>
        <li>the period does not belong to an abbreviation or an initial: Dr., Mr., Mrs., Ms., Prof., St., Jr., Sr., Mt., vs., Gen., Sen., Rev., Fr., Lt., Col., Capt., Sgt., e.g., i.e., U.S., U.K., a.k.a., or a single letter (“J. Robert Oppenheimer”).</li>
      </ul>
      <p>So “Dr. Richard Feynman.” is one completion. Otherwise it stops after the number of tokens in <b>Max tokens</b>, marked “cut off”.</p>
      <p>Whether a period ends the sentence depends on what comes next, so that choice has a probability too: the total probability of all next tokens that start with a space or a new line, or end the text. It is part of every score, so each score is still the exact probability of the completion ending there.</p>
      <p><b>Known gaps</b> (a heuristic, not a parser): a completion does not end at “?” or “!”, at a period inside quotes (“Whiskers.”), or at a line break without a period, so those run on; “Vitamin C.” is read as an initial; “Jr.” and “Sr.” never end one.</p>` },
  promptC: {
    title: 'The explicit prompt',
    body: `<p>A third prompt that asks for both at once, e.g. “My favorite physicist and author is”. It is not used to choose anything; it checks which ensemble behaves like asking directly. In the paper the global ensemble agrees with it closely ($\rho = 0.96$) and the local one does not ($\rho = 0.09$).</p>`,
  },
  valid: {
    title: 'Checking against the explicit prompt',
    body: `<p>Each dot is a completion: across, its ensemble score (global at the current $\tau$, or local); up, its log-probability after the explicit “A and B” prompt. $\rho$ is Spearman's rank correlation: 1 means both put the completions in the same order.</p>`,
  },
  ess: {
    title: 'Uses n of N completions',
    body: `<p>The effective sample size of the importance weights,
      $$\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2},$$
      between 1 and N. It says how many completions the estimate of $Z$ (and so every “ensemble %”) really rests on: when a few weights dominate, it is only as good as those few. The ranking itself does not depend on it.</p>`,
  },
  logZ: {
    title: 'log Ẑ',
    body: `<p>The estimated normaliser of the ensemble, $\hat Z = \frac1N\sum_i w_i$: an unbiased estimate of $Z = \sum_y \varphi_\tau(y)$ over all strings. Small means the two prompts share little probability.</p>`,
  },
  prob: {
    title: 'Probability under the ensemble',
    body: `<p>The completion's probability under the ensemble, $\Phi(y) = \varphi_\tau(y)/Z$. The score $\varphi_\tau(y)$ is exact; the normaliser $Z$, a sum over all strings, is estimated from the N completions ($\hat Z$, see log Ẑ). The list is ranked by $\varphi_\tau$, so the ranking is exact even when $\hat Z$ is rough.</p>`,
  },
};

init();
