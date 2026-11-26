// Prompt intersection: a prototype for "Ensembling Language Models with SMC" (arXiv:2603.05432, Fig. 1).
// Brief: docs/handoff-ensemble-demo.md. Cached pools: data/<pair>.<model>.<format>.json; live runs: the
// dpp-lm-demo Modal app's /ensemble_sample (modal/app.py there), both made by modal/ensemble_lib.py there.
import * as d3 from 'd3';
import { logGenMean, tauOf, uOf, tauName, NAMED, summarize, spearman } from './ensemble-math.js';
import { EXPLAIN, initExplainers } from './explain.js';
import { getAuth, signIn, signOut, finishSignIn } from './hfauth.js';

const DATA = 'data/';
const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (x, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : String(x));
const xb = (k) => `<button type="button" class="xbtn" data-x="${k}" aria-label="Explain">?</button>`;
// A completion can be empty: the text ends right after the prompt (a new line or a special token comes first).
const EMPTY = '(nothing: it ends right away)';
const shown = (s) => (s.text.trim() ? esc(s.text.trim()) : `<i class="empty-txt">${EMPTY}</i>`);
const words = (t, n = 5) => { const w = t.trim().split(/\s+/); return w.length > n ? w.slice(0, n).join(' ') + '…' : w.join(' '); };

const S = { presets: [], pool: null, groups: [], u: 0.5, K: 5, mode: 'map', score: 'global', draw: 1, sel: [], open: new Set(), hover: -1 };

// ---------------------------------------------------------------- data
async function init() {
  Object.assign(EXPLAIN, ENSEMBLE_EXPLAIN);
  initExplainers();
  S.presets = await (await fetch(DATA + 'presets.json')).json();
  wire();
  syncModel();
  fill(S.presets[0]);
  // Back from signing in with Hugging Face: restore the settings and run, as on the diversity demo.
  const back = await finishSignIn().catch(() => null);
  if (back?.stash?.cfg) restore(back.stash.cfg);
  await settingsChanged();
  if (back?.auth && back.stash?.cfg) run();
  else if (back?.error) stageMsg({ title: 'Signing in was cancelled.', detail: esc(back.error), actions: signInBtn() });
}

// The model and the prompt format. "text" continues the prompts as plain text (the paper's Fig. 1); "chat" puts
// them in the model's chat template as user messages, and the completions are its replies. GPT-2 has no chat
// template, so it offers "text" only.
const MODELS = {
  'qwen3-14b': { label: 'Qwen3 14B', formats: ['text', 'chat'] },
  gpt2: { label: 'GPT-2', formats: ['text'] },
};
const MODAL_URL = 'https://timvieira--dpp-lm-sampler-web.modal.run';
const pmodel = () => $('#model').value;
const pfmt = () => $('#fmt').value;   // the prompt format
const FORMAT_UI = {
  text: { a: 'Prompt A, e.g. My favorite physicist is', b: 'Prompt B, e.g. My favorite author is', c: 'e.g. My favorite physicist and author is',
    note: 'Each completion is one sentence.<button type="button" class="xbtn" data-x="stop" aria-label="When does a completion end?">?</button>' },
  chat: { a: 'Message A, e.g. Name a famous physicist.', b: 'Message B, e.g. Name a famous author.', c: 'e.g. Name a famous physicist who is also an author.',
    note: 'Each reply ends where the model ends its turn.<button type="button" class="xbtn" data-x="specials" aria-label="When does a reply end?">?</button>' },
};
function fillExamples() {
  const ex = $('#examples'), f = pfmt();
  ex.innerHTML = '<div class="menu-head">Example pairs</div>' + S.presets.map((p, i) =>
    `<button type="button" role="menuitem" data-i="${i}"><b>${esc(p.label)}</b><span>${esc(p[f].A)} · ${esc(p[f].B)}</span></button>`).join('');
  ex.querySelectorAll('button').forEach((b) => (b.onclick = () => { $('#exmenu').open = false; fill(S.presets[+b.dataset.i]); settingsChanged(); }));
  const u = FORMAT_UI[f];
  $('#pa').placeholder = u.a; $('#pb').placeholder = u.b; $('#pc').placeholder = u.c;
  $('#fmt-note').innerHTML = u.note;
}
// The formats the chosen model offers; a format it lacks is shown but disabled, and the choice moves to one it has.
function syncModel() {
  const m = MODELS[pmodel()];
  [...$('#fmt').options].forEach((o) => { o.disabled = !m.formats.includes(o.value); });
  if (!m.formats.includes(pfmt())) { const c = readConfig(); $('#fmt').value = m.formats[0]; keepExample(c, 'chat'); }
  fillExamples();
}
// An example pair stays that example in another format; typed prompts stay as typed.
function keepExample(c, was) {
  const p = S.presets.find((x) => x[was].A === c.A && x[was].B === c.B);
  if (p) fill(p);
}
function setFormat() {
  const was = pfmt() === 'chat' ? 'text' : 'chat', c = readConfig();
  fillExamples();
  keepExample(c, was);
  settingsChanged();
}
const DEFAULTS = { N: 200, Tprime: 1.0, maxNewTokens: 32 };
function fill(p) { const q = p[pfmt()]; $('#pa').value = q.A; $('#pb').value = q.B; $('#pc').value = q.C ?? ''; syncGo(); }
function readConfig() {
  return { model: pmodel(), fmt: pfmt(), A: $('#pa').value.trim(), B: $('#pb').value.trim(), C: $('#pc').value.trim(),
    N: Math.round(+$('#n').value), Tprime: Math.round(10 * +$('#tp').value) / 10, maxNewTokens: Math.round(+$('#maxtok').value) };
}
function restore(c) {
  $('#model').value = c.model; syncModel(); $('#fmt').value = c.fmt; fillExamples();
  $('#pa').value = c.A; $('#pb').value = c.B; $('#pc').value = c.C;
  $('#n').value = c.N; $('#tp').value = c.Tprime.toFixed(1); $('#maxtok').value = c.maxNewTokens;
  syncGo();
}
function cachedFor(c) {
  const p = S.presets.find((x) => x[c.fmt].A === c.A && x[c.fmt].B === c.B);
  const same = c.N === DEFAULTS.N && c.Tprime === DEFAULTS.Tprime && c.maxNewTokens === DEFAULTS.maxNewTokens;
  return p && same && (!c.C || c.C === p[c.fmt].C) ? p : null;
}
const poolKey = (p, c) => `${p.id}.${c.model}.${c.fmt}`;
const runKey = (c) => JSON.stringify([c.model, c.fmt, c.A, c.B, c.C, c.N, c.Tprime, c.maxNewTokens]);
const RUNS = new Map();   // this visit's live runs, by their settings
function syncGo() { $('#go').disabled = !($('#pa').value.trim() && $('#pb').value.trim()); $('#adv-sum').textContent = `${MODELS[pmodel()].label} · ${$('#n').value} samples`; }
async function settingsChanged() {
  syncGo();
  const c = readConfig(), p = cachedFor(c);
  if (p) {
    const k = poolKey(p, c);
    if (S.poolKey === k || await load(k)) { stageMsg(null); return; }
  }
  if (RUNS.has(runKey(c))) { show(RUNS.get(runKey(c)), runKey(c), 'Your run'); stageMsg(null); return; }
  if (S.pool) setStatus('The plot is from earlier prompts or settings. Press ↑.');
}

// ---------------------------------------------------------------- running on Modal
// As on the diversity demo (docs/run-flow.md): nothing runs, signs in or switches without a click that says so;
// every failure says what happened and offers the next step.
const signInBtn = () => '<button type="button" class="btn" data-act="signin">Sign in with Hugging Face</button>';
let abort = null;
async function run() {
  const c = readConfig();
  if (!c.A || !c.B || abort) return;
  const p = cachedFor(c);
  if (p && (S.poolKey === poolKey(p, c) || await load(poolKey(p, c)))) { stageMsg(null); return; }
  if (RUNS.has(runKey(c))) { show(RUNS.get(runKey(c)), runKey(c), 'Your run'); stageMsg(null); return; }
  const auth = getAuth();
  if (!auth) {
    stageMsg({ title: 'Running new prompts needs a free Hugging Face account.',
      detail: 'Runs use the demo’s GPU on Modal. You will be sent to Hugging Face to sign in, then back here, where the run starts.',
      actions: signInBtn() });
    return;
  }
  abort = new AbortController();
  const t0 = performance.now(), secs = () => Math.round((performance.now() - t0) / 1000);
  const running = () => stageMsg({ running: true, title: `Sampling ${c.N} completions… ${secs()} s`,
    detail: `${MODELS[c.model].label} on Modal.${secs() >= 60 ? ' The first run starts a GPU, which can take a minute or two.' : ''}`,
    actions: '<button type="button" class="btn" data-act="stop">Stop</button>' });
  running();
  const tick = setInterval(running, 1000);
  $('#go').disabled = true;
  try {
    const r = await fetch(`${MODAL_URL}/ensemble_sample`, {
      method: 'POST', signal: abort.signal,
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${auth.token}` },
      body: JSON.stringify({ model: c.model, format: c.fmt, prompts: { A: c.A, B: c.B, ...(c.C ? { C: c.C } : {}) },
        n: c.N, t_prime: c.Tprime, max_new_tokens: c.maxNewTokens }),
    });
    const out = await r.json().catch(() => null);
    if (!r.ok || !out?.samples) throw Object.assign(new Error(out?.detail || ''), { code: out?.code || (r.ok ? 'bad_answer' : 'server') });
    RUNS.set(runKey(c), out);
    if (runKey(readConfig()) === runKey(c)) { stageMsg(null); show(out, runKey(c), `Sampled in ${secs()} s`); }
    else { stageMsg(null); setStatus('Done, but for earlier settings. Set them back to see it.'); }
  } catch (e) {
    if (e.name === 'AbortError') { stageMsg(null); setStatus('Stopped.'); }
    else failed(e);
  } finally {
    clearInterval(tick); abort = null; syncGo();
  }
}
function failed(e) {
  const code = e.code || 'unreachable';
  const why = {
    auth: ['Your Hugging Face sign-in has expired.', 'Sign in again to run.', signInBtn()],
    rate_limit: ['You have used today’s runs.', esc(e.message) || 'Please come back tomorrow.', ''],
    budget: ['This month’s GPU budget for the demo is used up.', 'The example pairs still work.', '<button type="button" class="btn" data-act="example">Use an example</button>'],
    bad_request: ['The server could not run these settings.', esc(e.message), ''],
    unreachable: ['The GPU on Modal could not be reached.', 'Check your connection and try again.', '<button type="button" class="btn" data-act="retry">Try again</button>'],
  }[code] ?? ['Something went wrong on the server.', esc(e.message), '<button type="button" class="btn" data-act="retry">Try again</button>'];
  if (code === 'auth') signOut();
  stageMsg({ title: why[0], detail: why[1], actions: why[2] });
}
function stageMsg(m) {
  const el = $('#stage-msg');
  if (!m) { el.hidden = true; el.innerHTML = ''; return; }
  el.className = 'stage-msg' + (m.running ? '' : ' error');
  el.innerHTML = `${m.running ? '<div class="spinner"></div>' : ''}<p class="t">${m.title}</p><p class="d">${m.detail}</p><div class="a">${m.actions}${S.pool && !m.running ? '<button type="button" class="btn" data-act="dismiss">Dismiss</button>' : ''}</div>`;
  el.hidden = false;
  el.querySelectorAll('[data-act]').forEach((b) => (b.onclick = () => {
    const a = b.dataset.act;
    if (a === 'dismiss') stageMsg(null);
    if (a === 'stop') abort?.abort();
    if (a === 'retry') { stageMsg(null); run(); }
    if (a === 'signin') signIn({ cfg: readConfig() }).catch(() => stageMsg({ title: 'The Hugging Face sign-in could not be loaded.', detail: 'Check your connection and try again.', actions: signInBtn() }));
    if (a === 'example') { stageMsg(null); setTimeout(() => { $('#exmenu').open = true; }); }   // after this click's close-menus pass
  }));
}
function setStatus(t) { $('#status').textContent = t; }

// A cached pool, or false when there is none (not drawn yet for this model and format: ↑ runs it).
async function load(key) {
  const r = await fetch(DATA + key + '.json').catch(() => null);
  if (r && r.status === 404) return false;
  if (!r || !r.ok) { stageMsg({ title: 'These completions could not be loaded.', detail: 'Check your connection and reload the page.', actions: '' }); return true; }
  show(await r.json(), key, 'Cached');
  return true;
}
function show(pool, key, how) {
  S.pool = pool; S.poolKey = key;
  setStatus(`${how}: ${pool.N} completions from ${pool.model.split('/').pop()}${pool.standIn ? ' (stand-in, on a CPU)' : ', drawn on Modal'}.`);
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
      <div class="txt">${shown(s)}${s.ended ? '' : ' …'}</div>
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
      .text((d) => words(S.pool.samples[d.i].text) || EMPTY);
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
  el.innerHTML = `${shown(s)}${s.ended ? '' : ' …'}<div class="m">log p_A ${fmt(s.logpA, 2)} · log p_B ${fmt(s.logpB, 2)}${s.logpC != null ? ` · log p_C ${fmt(s.logpC, 2)}` : ''} · local ${fmt(s.loglocal, 2)}</div>`;
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
  $('#fmt').onchange = setFormat;
  $('#model').onchange = () => { syncModel(); settingsChanged(); };
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
    body: `<p><b>Qwen3 14B</b> with 4-bit (AWQ) weights, the same weights as the diversity demo. It is a chat model, but like any language model it gives a probability to every continuation of any text, so it can be used either way: see <b>Prompt format</b>.</p>
      <p><b>GPT-2</b> (124M parameters), the model of the paper's Fig. 1. It only continues text: it has no chat template.</p>
      <p>Both run on the demo's GPUs on Modal. Running new prompts needs a free Hugging Face account; the example pairs are cached and need nothing.</p>`,
  },
  format: {
    title: 'Prompt format',
    body: `<p><b>Continue text</b>: the prompts are the beginning of a text (“My favorite physicist is”), and each completion continues it, as with GPT-2 in the paper's Fig. 1. Each completion is one sentence.</p>
      <p><b>Chat</b>: the prompts are messages to the model (“Name a famous physicist.”) in its chat template, with reasoning switched off, and each completion is its reply, ending where the model ends its turn.</p>
      <p>Same weights, two different distributions: a chat model continuing bare text often writes as if in a worksheet or a forum; asked in its template, it answers. In both, the scores are exact probabilities of the same reply under each prompt.</p>`,
  },
  N: { title: 'Samples N', body: `<p>How many completions are drawn. The ranking and every readout are computed from these N; more completions represent the ensemble better.</p>` },
  Tprime: { title: 'Sampling temperature T′', body: `<p>The temperature at which each token is drawn: $r(y_t \mid y_{<t}) \propto (p_A \cdot p_B)^{1/T'}$. At $T' = 1$ the draws follow the token-level product exactly.</p>` },
  maxtok: { title: 'Max tokens', body: `<p>A completion stops at the end of its first sentence, or after this many tokens. One that hits the limit is marked “cut off”.</p>` },
  stop: {
    title: 'One sentence',
    body: `<p>As in the paper, every completion is a short answer: one sentence, or one line. At every step, some next tokens would end it and the rest continue it; which ones end it depends on how the text so far ends. The rules follow the English rules of the pySBD sentence splitter that need only the next word:</p>
      <ul>
        <li><b>A word and “.” or “?”</b> (“Richard Feynman.”): ends if the next token starts with a space or a new line, or ends the text. So “0.5” is not cut at “0.”.</li>
        <li><b>“!”, “...”, or an abbreviation that can end a sentence</b> (Inc., etc., Jr., Jan., Calif., U.S., a.m., Ph.D., …): ends only before a capital letter, a new line or the end. “Apple Inc. and Google” goes on; “… Inc. The company …” ends.</li>
        <li><b>No., pp., p., Art.</b>: not before a number (“No. 5”).</li>
        <li><b>Titles and initials</b> (Dr., Mr., Mrs., Prof., Gov., Sen., St., …; “J.”; e.g.; i.e.): never end a sentence, so “Dr. Richard Feynman.” is one completion.</li>
        <li><b>A new line</b> always ends a completion, with or without punctuation.</li>
        <li><b>Any special token</b> of the model (end of text, a new chat turn, …) ends it too; see the chat format's note.</li>
      </ul>
      <p>Closing quotes and brackets count with the mark: “Whiskers.” ends. Otherwise a completion stops after <b>Max tokens</b>, marked “cut off”.</p>
      <p>Each ending is itself a choice with a probability: the total probability of all next tokens that would end the completion there. It is part of every score, so each score is still the exact probability of the completion ending there. One exception: a token that holds both text and a line break (such as “.” followed by a new line) ends the completion with that token's own probability, which leaves out other ways of writing the same text.</p>
      <p><b>Known gaps</b> (a heuristic, not a parser): “Vitamin C.” reads as an initial; abbreviations missing from the lists end like ordinary words; a space on its own as a token counts as “starts with a space” whatever follows.</p>` },
  specials: {
    title: 'Where a reply ends',
    body: `<p>The model marks the end of its reply with a special token, <code>&lt;|im_end|&gt;</code>. A reply ends there, and also at any other special token: the end of a document, the start of a new turn (the model skipped the end mark), a reasoning tag or a tool call. None of these is text, so the reply shown is exactly what was scored.</p>
      <p>Ending is part of the reply's probability: its score includes the probability that some special token comes next. Nothing is blocked or thrown away, which would change the model's distribution. Each reply records which special token ended it.</p>`,
  },
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
