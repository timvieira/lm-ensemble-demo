// The ensemble prototype's maths (docs/handoff-ensemble-demo.md, A3), in log space throughout.
// Pure functions: tested in tests/ensemble.test.mjs.

export function logsumexp(xs) {
  let m = -Infinity;
  for (const x of xs) if (x > m) m = x;
  if (m === -Infinity) return -Infinity;
  if (m === Infinity) return Infinity;
  let s = 0;
  for (const x of xs) s += Math.exp(x - m);
  return m + Math.log(s);
}

// log of the weighted generalised mean (Σ_k w_k p_k^τ)^(1/τ), from log p_k and weights w (summing to 1).
// τ = 0 is the weighted geometric mean (the product of experts, normalised later); ±∞ are max and min.
export function logGenMean(logps, tau, w = null) {
  const K = logps.length, W = w ?? logps.map(() => 1 / K);
  if (tau === Infinity) return Math.max(...logps);
  if (tau === -Infinity) return Math.min(...logps);
  if (tau === 0) return logps.reduce((s, l, k) => s + W[k] * l, 0);
  return logsumexp(logps.map((l, k) => tau * l + Math.log(W[k]))) / tau;
}

// The slider's position u ∈ [0, 1] ↔ τ: τ = 1.5x/(1 − x²) with x = 2u − 1, so the ends are ∓∞, the middle
// is 0 (the product), u = 0.25 is τ = −1 (harmonic) and u = 0.75 is τ = 1 (the sum, a mixture).
export function tauOf(u) {
  const x = 2 * u - 1;
  if (x <= -1) return -Infinity;
  if (x >= 1) return Infinity;
  return (1.5 * x) / (1 - x * x);
}
export function uOf(tau) {
  if (tau === -Infinity) return 0;
  if (tau === Infinity) return 1;
  if (tau === 0) return 0.5;
  // Solve τx² + 1.5x − τ = 0 for x in (−1, 1).
  const x = (-1.5 + Math.sqrt(2.25 + 4 * tau * tau)) / (2 * tau);
  return (x + 1) / 2;
}
export const NAMED = [
  { tau: -Infinity, name: 'min' }, { tau: -1, name: 'harmonic' }, { tau: 0, name: 'product' },
  { tau: 1, name: 'sum' }, { tau: 2, name: 'quadratic' }, { tau: Infinity, name: 'max' },
];
export function tauName(tau) {
  const hit = NAMED.find((x) => x.tau === tau || (Number.isFinite(tau) && Number.isFinite(x.tau) && Math.abs(tau - x.tau) < 0.02));
  return hit ? hit.name : null;
}

// Importance-sampling summaries of log weights: log Ẑ = log mean(v), the effective sample size
// (Σv)²/Σv² (between 1 and N), and the normalised shares v/Σv.
export function summarize(logv) {
  const N = logv.length, ls = logsumexp(logv);
  const ess = Math.exp(2 * ls - logsumexp(logv.map((x) => 2 * x)));
  return { logZ: ls - Math.log(N), ess, share: logv.map((x) => Math.exp(x - ls)) };
}

// K distinct items drawn without replacement with probability ∝ weight at each draw (successive
// sampling), from log weights; Efraimidis–Spirakis keys: the K smallest log(−log u) − log w.
export function sampleWOR(logw, K, rand) {
  const keys = logw.map((lw, i) => ({ i, k: lw === -Infinity ? Infinity : Math.log(-Math.log(1 - rand())) - lw }));
  keys.sort((a, b) => a.k - b.k);
  return keys.slice(0, K).filter((x) => x.k < Infinity).map((x) => x.i);
}

// Spearman's rank correlation (average ranks for ties).
export function spearman(a, b) {
  const rank = (v) => {
    const idx = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]), r = new Array(v.length);
    for (let s = 0; s < idx.length;) {
      let e = s;
      while (e + 1 < idx.length && idx[e + 1][0] === idx[s][0]) e++;
      for (let j = s; j <= e; j++) r[idx[j][1]] = (s + e) / 2;
      s = e + 1;
    }
    return r;
  };
  const ra = rank(a), rb = rank(b), n = a.length;
  const ma = ra.reduce((s, x) => s + x, 0) / n, mb = rb.reduce((s, x) => s + x, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) ** 2; sbb += (rb[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
