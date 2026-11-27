// Check a running Space backend (space/app.py) through the page's own client (js/space.js):
//
//   node scripts/space-check.mjs <space url> [model] [format] [n] [tau'] [w_A] [weights] [T'] [--old]
//   e.g. node scripts/space-check.mjs https://xtimv-lm-ensemble-demo.hf.space gpt2 text 8 sum 0.5 prefix 1
//
// Lists its models, draws one pool with the API "ensemble_pool" (the proposal: τ' as a number, inf, -inf or a
// name; w_A; weights prefix or fixed; T'; defaults product 0.5 prefix 1) and checks the record the page reads
// (docs: space/app.py, space/ensemble_lib.py). Prints the importance weights' effective sample size at each named
// target τ. With --old, also checks the older API "ensemble_sample" with a small pool. With HF_TOKEN set, the runs
// use that account's ZeroGPU quota, as the page does with the visitor's sign-in. Exits 1 on the first problem.
import { callSpace } from '../js/space.js';
import { logGenMean, summarize, NAMED } from '../js/ensemble-math.js';

const args = process.argv.slice(2).filter((a) => a !== '--old'), old = process.argv.includes('--old');
const [url, model = 'gpt2', fmt = 'text', n = '8', tau = 'product', wA = '0.5', weights = 'prefix', Tp = '1'] = args;
if (!url) { console.error('usage: node scripts/space-check.mjs <space url> [model] [format] [n] [tau\'] [w_A] [weights] [T\'] [--old]'); process.exit(2); }
const token = process.env.HF_TOKEN || undefined;
const die = (msg) => { console.error('FAIL', msg); process.exit(1); };
const prompts = { A: 'My favorite physicist is', B: 'My favorite author is', C: 'My favorite physicist and author is' };

function checkPool(pool, N, what, withC = true) {
  const S = pool.samples;
  if (pool.kind !== 'ensemble-pool' || pool.schema !== 3) die(`${what}: not a schema-3 pool record: kind ${pool.kind}, schema ${pool.schema}`);
  if (pool.proposal?.kind !== 'local_f') die(`${what}: proposal ${JSON.stringify(pool.proposal)}`);
  if (!Array.isArray(S) || S.length !== N || pool.N !== N) die(`${what}: expected ${N} samples, got ${S?.length} (N = ${pool.N})`);
  for (const [i, s] of S.entries()) {
    for (const k of ['logpA', 'logpB', 'logr', 'loglocal', ...(withC ? ['logpC'] : [])]) {
      if (!Number.isFinite(s[k]) || s[k] > 1e-9) die(`${what}, sample ${i}: ${k} = ${s[k]} is not a finite log-probability`);
    }
    if (typeof s.text !== 'string' || !['sentence', 'line', 'eos', 'max'].includes(s.end)) die(`${what}, sample ${i}: text or end missing`);
    if ('ended' in s) die(`${what}, sample ${i}: has the schema-2 field ended`);
    if (s.end === 'max' && s.nTokens !== pool.maxNewTokens) die(`${what}, sample ${i}: ended by Max tokens at ${s.nTokens} tokens`);
  }
}

const models = await callSpace(url, 'models', [], { token }).catch((e) => die(`models: ${e.code} ${e.message}`));
console.log('models:', JSON.stringify(models));
if (!models[model]) die(`the Space does not offer ${model}`);

const t0 = Date.now();
const pool = await callSpace(url, 'ensemble_pool', [model, fmt, prompts.A, prompts.B, prompts.C, +n, tau, +wA, weights, +Tp, 32, 0], { token })
  .catch((e) => die(`ensemble_pool: ${e.code} ${e.message}`));
const secs = ((Date.now() - t0) / 1000).toFixed(1);
checkPool(pool, +n, 'ensemble_pool');
const S = pool.samples;
console.log(`pool: ${pool.N} samples of ${pool.model} (${pool.format}) in ${secs} s (Space timing ${JSON.stringify(pool.timing)})`);
console.log(`proposal: ${JSON.stringify(pool.proposal)}`);
console.log(`checkMaxAbsErr (sampling sums vs teacher-forced scores, nats): ${pool.checkMaxAbsErr}`);
console.log(`distinct completions: ${new Set(S.map((s) => s.text)).size} of ${S.length}`);
console.log(`ESS at each target τ (of ${S.length}): ` + NAMED.map(({ tau: t, name }) =>
  `${name} ${summarize(S.map((s) => logGenMean([s.logpA, s.logpB], t) - s.logr)).ess.toFixed(1)}`).join(' · '));
console.log('first completions:', S.slice(0, 4).map((s) => JSON.stringify(s.text)).join(' | '));

if (old) {
  const p2 = await callSpace(url, 'ensemble_sample', [model, fmt, prompts.A, prompts.B, '', 4, 1.0, 16, 0], { token })
    .catch((e) => die(`ensemble_sample: ${e.code} ${e.message}`));
  checkPool(p2, 4, 'ensemble_sample', false);
  const q = p2.proposal;
  if (!(q.tau === 0 && q.wA === 0.5 && q.Tprime === 0.5)) die(`ensemble_sample: proposal ${JSON.stringify(q)}, expected the product at T' = 0.5`);
  console.log(`older API ensemble_sample: OK, proposal ${JSON.stringify(q)}`);
}
console.log('OK');
