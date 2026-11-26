// Prompt intersection: a prototype for "Ensembling Language Models with SMC" (arXiv:2603.05432, Fig. 1).
// Brief: docs/handoff-ensemble-demo.md. Pools: data/ensemble/<preset>.json (modal/ensemble_lib.py).
import * as d3 from 'd3';
import { logGenMean, tauOf, uOf, tauName, NAMED, summarize, sampleWOR, spearman, mulberry32, logsumexp } from './ensemble-math.js';

const DATA = '../data/ensemble/';
const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
const words = (t, n = 5) => { const w = t.trim().split(/\s+/); return w.length > n ? w.slice(0, n).join(' ') + '…' : w.join(' '); };

const S = { presets: [], pool: null, groups: [], u: 0.5, K: 5, mode: 'map', score: 'global', draw: 1, sel: [], open: new Set(), hover: -1 };

// ---------------------------------------------------------------- data
async function init() {
  S.presets = await (await fetch(DATA + 'presets.json')).json();
  $('#preset').innerHTML = S.presets.map((p) => `<option value="${p.id}">${esc(p.label)}</option>`).join('');
  $('#preset').onchange = () => load($('#preset').value);
  wire();
  await load(S.presets[0].id);
}

async function load(id) {
  const r = await fetch(DATA + id + '.json').catch(() => null);
  if (!r || !r.ok) {
    S.pool = null;
    $('#plot').style.display = 'none';
    $('#empty').hidden = false;
    $('#empty').textContent = 'No completions for this pair yet.';
    $('#cards').innerHTML = ''; $('#readout').innerHTML = ''; $('#sel-sub').textContent = '';
    return;
  }
  const pool = await r.json();
  S.pool = pool;
  $('#plot').style.display = ''; $('#empty').hidden = true;
  $('#data-note').textContent = pool.standIn
    ? `stand-in data: ${pool.N} completions from ${pool.model.split('/').pop()} on a CPU; the demo will use Qwen3 14B`
    : `${pool.N} cached completions from ${pool.model.split('/').pop()}`;
  // One group per distinct string; copies share every score.
  const byText = new Map();
  pool.samples.forEach((s, i) => { const g = byText.get(s.text); if (g) g.idx.push(i); else byText.set(s.text, { i, idx: [i] }); });
  S.groups = [...byText.values()];
  S.open = new Set(); S.draw = 1;
  plot.reset();
  update();
}

// ---------------------------------------------------------------- the ensemble
function tau() { return tauOf(S.u); }
function scoreOf(s) { return S.score === 'local' ? s.loglocal : logGenMean([s.logpA, s.logpB], tau()); }

function update() {
  if (!S.pool) return;
  const P = S.pool, T = tau(), name = tauName(T);
  $('#tau-out').textContent = S.score === 'local' ? 'local product' : `${name ?? 'τ = ' + fmt(T, 2)}${name && Number.isFinite(T) ? ` (τ = ${T})` : ''}`;
  const score = P.samples.map(scoreOf);
  const logv = P.samples.map((s, i) => score[i] - s.logr);    // importance weights: target / proposal
  const sum = summarize(logv);
  const gScore = S.groups.map((g) => score[g.i]);
  const gLogv = S.groups.map((g) => logsumexp(g.idx.map((i) => logv[i])));
  const gShare = S.groups.map((g) => g.idx.reduce((a, i) => a + sum.share[i], 0));
  const K = Math.min(S.K, S.groups.length);
  let sel;
  if (S.mode === 'map') sel = d3.range(S.groups.length).sort((a, b) => gScore[b] - gScore[a]).slice(0, K);
  else sel = sampleWOR(gLogv, K, mulberry32(1000 * S.draw + 7));
  S.sel = sel.map((g) => S.groups[g].i);
  Object.assign(S, { scoreArr: score, gShare, sum, selGroups: sel });
  readouts(score);
  cards(sel, gShare);
  plot.draw();
  valid.draw(score);
}

function readouts(score) {
  const P = S.pool, N = P.N, distinct = S.groups.length;
  $('#sel-sub').textContent = `K = ${S.sel.length} · ${S.mode === 'map' ? 'most likely' : 'sampled'} · ${N} completions, ${distinct} distinct`;
  const rows = [
    ['Uses', `${fmt(S.sum.ess, 1)} of ${N} completions`],
    ['log Ẑ', fmt(S.sum.logZ, 2)],
  ];
  $('#readout').innerHTML = rows.map(([k, v]) => `<span>${k}</span><span class="v">${v}</span>`).join('');
}

function cards(sel, gShare) {
  const P = S.pool;
  $('#cards').innerHTML = sel.map((g) => {
    const G = S.groups[g], s = P.samples[G.i];
    return `<li class="card${S.open.has(G.i) ? ' open' : ''}" data-i="${G.i}">
      <div class="txt">${esc(s.text.trim())}${s.ended ? '' : ' …'}</div>
      <div class="meta"><span class="dot"></span>${G.idx.length > 1 ? `×${G.idx.length} · ` : ''}log p<sub>A</sub> ${fmt(s.logpA)} · log p<sub>B</sub> ${fmt(s.logpB)} · share ${fmt(100 * gShare[g], 1)}%${s.ended ? '' : `<span class="cutoff">cut off at ${s.nTokens} tokens</span>`}</div></li>`;
  }).join('');
  $('#cards').querySelectorAll('.card').forEach((el) => {
    const i = +el.dataset.i;
    el.onmouseenter = () => plot.highlight(i);
    el.onmouseleave = () => plot.highlight(-1);
    el.onclick = () => { el.classList.toggle('open'); if (el.classList.contains('open')) S.open.add(i); else S.open.delete(i); };
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
    const kth = S.mode === 'map' && S.sel.length ? Math.min(...S.sel.map((i) => sc[i])) : null;
    const proj = d3.geoTransform({ point(px, py) {
      this.stream.point(plot.x.range()[0] + (px / n) * (plot.x.range()[1] - plot.x.range()[0]), plot.y.range()[1] + (py / n) * (plot.y.range()[0] - plot.y.range()[1]));
    } });
    const path = d3.geoPath(proj), cont = d3.contours().size([n, n]);
    const draw = (lv, strong) => this.gContour.append('path').attr('d', path(cont.contour(vals, lv)))
      .attr('fill', 'none').attr('stroke', strong ? css('--accent') : css('--text-2'))
      .attr('stroke-opacity', strong ? 0.6 : 0.22).attr('stroke-width', strong ? 1.2 : 1)
      .attr('stroke-dasharray', strong ? '5 3' : null);
    levels.forEach((lv) => draw(lv, false));
    if (kth != null) draw(kth - 1e-9, true);   // Most likely: everything past this line is selected
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
  ticks.innerHTML = NAMED.filter((t) => t.name !== 'quadratic').map((t) => `<span style="left:${100 * uOf(t.tau)}%" data-u="${uOf(t.tau)}">${t.name}</span>`).join('');
  ticks.querySelectorAll('span').forEach((s) => (s.onclick = () => { S.u = +s.dataset.u; $('#tau').value = Math.round(1000 * S.u); update(); }));
  let raf = 0;
  $('#tau').oninput = (e) => {
    let u = +e.target.value / 1000;
    for (const t of NAMED) { const v = uOf(t.tau); if (Math.abs(u - v) < 0.012) u = v; }   // snap to the named means
    S.u = u;
    cancelAnimationFrame(raf); raf = requestAnimationFrame(update);
  };
  const setK = (k) => { S.K = Math.max(1, Math.min(20, k)); $('#k').textContent = S.K; update(); };
  $('#k-minus').onclick = () => setK(S.K - 1);
  $('#k-plus').onclick = () => setK(S.K + 1);
  document.querySelectorAll('[data-mode]').forEach((b) => (b.onclick = () => {
    if (b.dataset.mode === 'sample' && S.mode === 'sample') S.draw++;   // Sample again: a new draw
    S.mode = b.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    update();
  }));
  document.querySelectorAll('[data-score]').forEach((b) => (b.onclick = () => {
    S.score = b.dataset.score;
    document.querySelectorAll('[data-score]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    $('#tau').disabled = S.score === 'local';
    update();
  }));
  let rt = 0;
  new ResizeObserver(() => { clearTimeout(rt); rt = setTimeout(() => { if (S.pool) { plot.reset(); plot.draw(); valid.draw(S.scoreArr); } }, 150); }).observe($('#plot'));
}

init();
