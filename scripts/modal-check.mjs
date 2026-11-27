// Check the deployed Modal backend (dpp-lm-demo's modal/app.py, route /ensemble_sample) with the page's request:
//
//   HF_TOKEN=... node scripts/modal-check.mjs [model] [format] [n] [tau'] [w_A] [weights] [T'] [--old] [--bad]
//   e.g. node scripts/modal-check.mjs gpt2 text 8 sum 0.5 prefix 1
//
// Draws one pool with proposal {tau, wA, weights, Tprime} (defaults product 0.5 prefix 1) and checks the record
// (schema 3, as space/ensemble_lib.py writes it). Prints the importance weights' effective sample size at each named
// target τ. --old also sends the older request (t_prime = 1, no proposal), which should give the product at T' = ½
// with fixed weights. --bad also sends an invalid proposal, which should be refused (400) before any GPU work.
// Each accepted request is one of the token's account's Modal runs for the day and costs the demo's GPU budget.
import { logGenMean, summarize, NAMED } from '../js/ensemble-math.js';

const MODAL_URL = 'https://timvieira--dpp-lm-sampler-web.modal.run';
const flags = new Set(process.argv.slice(2).filter((a) => a.startsWith('--')));
const [model = 'gpt2', fmt = 'text', n = '8', tau = 'product', wA = '0.5', weights = 'prefix', Tp = '1'] =
  process.argv.slice(2).filter((a) => !a.startsWith('--'));
const token = process.env.HF_TOKEN;
if (!token) { console.error('HF_TOKEN is not set'); process.exit(2); }
const die = (msg) => { console.error('FAIL', msg); process.exit(1); };
const prompts = { A: 'My favorite physicist is', B: 'My favorite author is', C: 'My favorite physicist and author is' };

async function post(body) {
  const r = await fetch(`${MODAL_URL}/ensemble_sample`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
  return { status: r.status, out: await r.json().catch(() => null) };
}

function checkPool(pool, N, what) {
  const S = pool?.samples;
  if (pool?.kind !== 'ensemble-pool' || pool.schema !== 3) die(`${what}: not a schema-3 pool: ${JSON.stringify(pool)?.slice(0, 300)}`);
  if (pool.proposal?.kind !== 'local_f') die(`${what}: proposal ${JSON.stringify(pool.proposal)}`);
  if (!Array.isArray(S) || S.length !== N || pool.N !== N) die(`${what}: expected ${N} samples, got ${S?.length}`);
  for (const [i, s] of S.entries()) {
    for (const k of ['logpA', 'logpB', 'logpC', 'logr', 'loglocal']) {
      if (!Number.isFinite(s[k]) || s[k] > 1e-9) die(`${what}, sample ${i}: ${k} = ${s[k]} is not a finite log-probability`);
    }
    if (typeof s.text !== 'string' || !['sentence', 'line', 'eos', 'max'].includes(s.end) || 'ended' in s) die(`${what}, sample ${i}: text or end`);
    if (s.end === 'max' && s.nTokens !== pool.maxNewTokens) die(`${what}, sample ${i}: ended by Max tokens at ${s.nTokens} tokens`);
  }
}

const status = await (await fetch(`${MODAL_URL}/status`)).json();
console.log('status:', JSON.stringify({ ensemble: status.ensemble, gpu_minutes_left_this_month: status.gpu_minutes_left_this_month }));

if (flags.has('--bad')) {
  const { status: st, out } = await post({ model, format: fmt, prompts, n: 4, proposal: { tau: 'abc' }, max_new_tokens: 16 });
  if (st !== 400 || out?.code !== 'bad_request') die(`invalid proposal: status ${st}, ${JSON.stringify(out)}`);
  console.log(`invalid proposal refused: ${out.detail}`);
}

const t0 = Date.now();
const { status: st, out: pool } = await post({ model, format: fmt, prompts, n: +n, max_new_tokens: 32, seed: 0,
  proposal: { tau, wA: +wA, weights, Tprime: +Tp } });
if (st !== 200) die(`ensemble_sample: status ${st}, ${JSON.stringify(pool)}`);
checkPool(pool, +n, 'ensemble_sample');
const S = pool.samples;
console.log(`pool: ${pool.N} samples of ${pool.model} (${pool.format}) in ${((Date.now() - t0) / 1000).toFixed(1)} s (timing ${JSON.stringify(pool.timing)})`);
console.log(`proposal: ${JSON.stringify(pool.proposal)}`);
console.log(`checkMaxAbsErr (sampling sums vs teacher-forced scores, nats): ${pool.checkMaxAbsErr}`);
console.log(`ends: ${JSON.stringify(S.reduce((m, s) => ({ ...m, [s.end]: (m[s.end] || 0) + 1 }), {}))}; distinct completions: ${new Set(S.map((s) => s.text)).size} of ${S.length}`);
console.log(`ESS at each target τ (of ${S.length}): ` + NAMED.map(({ tau: t, name }) =>
  `${name} ${summarize(S.map((s) => logGenMean([s.logpA, s.logpB], t) - s.logr)).ess.toFixed(1)}`).join(' · '));

if (flags.has('--old')) {
  const { status: s2, out: p2 } = await post({ model, format: fmt, prompts, n: 4, t_prime: 1.0, max_new_tokens: 16, seed: 0 });
  if (s2 !== 200) die(`older request: status ${s2}, ${JSON.stringify(p2)}`);
  checkPool(p2, 4, 'older request');
  const q = p2.proposal;
  if (!(q.tau === 0 && q.wA === 0.5 && q.weights === 'fixed' && q.Tprime === 0.5)) die(`older request: proposal ${JSON.stringify(q)}`);
  console.log(`older request (t_prime = 1): OK, proposal ${JSON.stringify(q)}`);
}
console.log('OK');
