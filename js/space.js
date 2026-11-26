// Calling the Hugging Face Space (space/app.py) from the page, as the diversity demo does (dpp-lm-demo js/main.js,
// callSpace): Gradio's queue protocol. Join the queue, then read the event stream. (Gradio's simpler /call endpoint
// drops error messages, and its JS client sends credentials, which the Space's CORS answer does not allow for other
// sites.) A failure is an Error with a `code` the page explains: auth, starting, busy, quota, bad_request, server,
// unreachable, bad_answer.
const fail = (code, message = '') => Object.assign(new Error(message), { code });

export async function callSpace(url, api, data, { signal, token } = {}) {
  const hdr = token ? { Authorization: `Bearer ${token}` } : {};
  const net = async (u, init) => {
    try { return await fetch(u, { ...init, signal }); }
    catch (e) { if (e.name === 'AbortError') throw e; throw fail('unreachable'); }
  };
  let r = await net(`${url}/config`, { headers: hdr });
  if (!r.ok) throw fail(r.status === 401 || r.status === 403 ? 'auth' : 'starting');
  const cfg = await r.json().catch(() => null);
  const fn = cfg?.dependencies?.findIndex((d) => d.api_name === api) ?? -1;
  if (fn < 0) throw fail('bad_answer', `The Space has no "${api}" function.`);
  const session = Math.random().toString(36).slice(2);
  r = await net(`${url}/gradio_api/queue/join`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...hdr },
    body: JSON.stringify({ data, fn_index: fn, trigger_id: cfg.dependencies[fn].id ?? null, session_hash: session, event_data: null }),
  });
  if (!r.ok) throw fail(r.status === 401 || r.status === 403 ? 'auth' : r.status === 429 ? 'busy' : 'server');
  r = await net(`${url}/gradio_api/queue/data?session_hash=${session}`, { headers: hdr });
  if (!r.ok || !r.body) throw fail('server');
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let k;
    while ((k = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, k).split('\n').find((l) => l.startsWith('data:'));
      buf = buf.slice(k + 2);
      if (!line) continue;
      const m = JSON.parse(line.slice(5));
      if (m.msg !== 'process_completed') continue;
      if (m.success && m.output?.data) return m.output.data[0];
      const msg = String(m.output?.error ?? m.output?.title ?? '');
      // The Space prefixes its own errors with a code; ZeroGPU's quota errors have none.
      const c = /^(bad_request|server|busy): (.*)$/s.exec(msg);
      if (c) throw fail(c[1], c[2]);
      throw fail(/quota|runs limit/i.test(msg) ? 'quota' : 'server', msg);
    }
  }
  throw fail('bad_answer', 'The Space ended without an answer.');
}
