// Check a running Space backend (space/app.py) through the page's own client (js/space.js):
//
//   node scripts/space-check.mjs <space url> [model] [format] [n]
//   e.g. node scripts/space-check.mjs https://xtimv-lm-ensemble-demo.hf.space gpt2 text 8
//
// Lists its models, draws one small pool and checks the record the page reads (docs: space/app.py,
// space/ensemble_lib.py). With HF_TOKEN set, the run uses that account's ZeroGPU quota, as the page does with the
// visitor's sign-in. Prints what it found; exits 1 on the first problem.
import { callSpace } from '../js/space.js';

const [url, model = 'gpt2', fmt = 'text', n = '8'] = process.argv.slice(2);
if (!url) { console.error('usage: node scripts/space-check.mjs <space url> [model] [format] [n]'); process.exit(2); }
const token = process.env.HF_TOKEN || undefined;
const die = (msg) => { console.error('FAIL', msg); process.exit(1); };
const prompts = { A: 'My favorite physicist is', B: 'My favorite author is', C: 'My favorite physicist and author is' };

const models = await callSpace(url, 'models', [], { token }).catch((e) => die(`models: ${e.code} ${e.message}`));
console.log('models:', JSON.stringify(models));
if (!models[model]) die(`the Space does not offer ${model}`);

const t0 = Date.now();
const pool = await callSpace(url, 'ensemble_sample', [model, fmt, prompts.A, prompts.B, prompts.C, +n, 1.0, 32, 0], { token })
  .catch((e) => die(`ensemble_sample: ${e.code} ${e.message}`));
const secs = ((Date.now() - t0) / 1000).toFixed(1);

const S = pool.samples;
if (pool.kind !== 'ensemble-pool' || pool.schema !== 2) die(`not a pool record: kind ${pool.kind}, schema ${pool.schema}`);
if (!Array.isArray(S) || S.length !== +n || pool.N !== +n) die(`expected ${n} samples, got ${S?.length} (N = ${pool.N})`);
for (const [i, s] of S.entries()) {
  for (const k of ['logpA', 'logpB', 'logr', 'loglocal', 'logpC']) {
    if (!Number.isFinite(s[k]) || s[k] > 1e-9) die(`sample ${i}: ${k} = ${s[k]} is not a finite log-probability`);
  }
  if (typeof s.text !== 'string' || typeof s.ended !== 'boolean') die(`sample ${i}: text or ended missing`);
}
// At T' = 1 the proposal is the local product itself, so logr and loglocal are the same sums.
const gap = Math.max(...S.map((s) => Math.abs(s.logr - s.loglocal)));
console.log(`pool: ${pool.N} samples of ${pool.model} (${pool.format}) in ${secs} s (Space timing ${JSON.stringify(pool.timing)})`);
console.log(`checkMaxAbsErr (sampling sums vs teacher-forced scores, nats): ${pool.checkMaxAbsErr}`);
console.log(`max |logr - loglocal| at T' = 1: ${gap.toExponential(2)}`);
console.log('first completions:', S.slice(0, 4).map((s) => JSON.stringify(s.text)).join(' | '));
console.log('OK');
