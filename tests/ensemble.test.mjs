import { test } from 'node:test';
import assert from 'node:assert/strict';
import { logsumexp, logGenMean, tauOf, uOf, summarize, sampleWOR, spearman, mulberry32 } from '../lab/ensemble-math.js';

const close = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} vs ${b}`);

test('generalised mean: named members against direct formulas', () => {
  const pa = 0.02, pb = 0.3, la = Math.log(pa), lb = Math.log(pb);
  close(logGenMean([la, lb], 1), Math.log((pa + pb) / 2));                 // sum (mixture)
  close(logGenMean([la, lb], 0), Math.log(Math.sqrt(pa * pb)));             // product (geometric)
  close(logGenMean([la, lb], -1), Math.log(1 / ((1 / pa + 1 / pb) / 2)));   // harmonic
  close(logGenMean([la, lb], 2), Math.log(Math.sqrt((pa * pa + pb * pb) / 2)));
  close(logGenMean([la, lb], Infinity), lb);
  close(logGenMean([la, lb], -Infinity), la);
  // Continuity at 0 and towards ±∞, and monotone in τ (power-mean inequality).
  close(logGenMean([la, lb], 1e-7), logGenMean([la, lb], 0), 1e-6);
  close(logGenMean([la, lb], 400), lb, 1e-2);
  close(logGenMean([la, lb], -400), la, 1e-2);
  let prev = -Infinity;
  for (const t of [-Infinity, -5, -1, -0.1, 0, 0.1, 1, 2, 5, Infinity]) { const v = logGenMean([la, lb], t); assert.ok(v >= prev - 1e-12); prev = v; }
});

test('generalised mean stays finite for very unlikely strings', () => {
  for (const t of [-3, -1, 0, 1, 3]) {
    const v = logGenMean([-900, -1200], t);
    assert.ok(Number.isFinite(v) && v <= -900 + 1e-9 && v >= -1200 - 1e-9, `${t}: ${v}`);
  }
  close(logsumexp([-1000, -1000]), -1000 + Math.log(2));
});

test('slider mapping: named points and inverse', () => {
  assert.equal(tauOf(0), -Infinity); assert.equal(tauOf(1), Infinity); assert.equal(tauOf(0.5), 0);
  close(tauOf(0.75), 1); close(tauOf(0.25), -1);
  for (const t of [-7, -1, -0.3, 0, 0.5, 1, 2, 9]) close(tauOf(uOf(t)), t, 1e-9);
});

test('importance summaries: Z, ESS bounds, shares', () => {
  const lv = [0, 0, 0, 0];
  const s = summarize(lv);
  close(s.ess, 4); close(s.logZ, 0); s.share.forEach((x) => close(x, 0.25));
  const one = summarize([0, -800, -800]);
  close(one.ess, 1, 1e-6);
  const v = [1, 2, 3, 4], t = summarize(v.map(Math.log));
  close(t.ess, 100 / 30); close(Math.exp(t.logZ), 2.5);
});

test('sampling without replacement: distinct, and first-draw frequencies ∝ weight', () => {
  const w = [1, 2, 3, 4], lw = w.map(Math.log), rand = mulberry32(7), first = [0, 0, 0, 0], T = 40000;
  for (let t = 0; t < T; t++) {
    const s = sampleWOR(lw, 3, rand);
    assert.equal(new Set(s).size, 3);
    first[s[0]]++;
  }
  first.forEach((c, i) => close(c / T, w[i] / 10, 0.05));
  assert.deepEqual(sampleWOR([0, -Infinity, 0], 3, mulberry32(1)).sort(), [0, 2]);
});

test('spearman: perfect, reversed, ties', () => {
  close(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1);
  close(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1);
  close(spearman([1, 1, 2, 3], [5, 5, 6, 7]), 1);
});
