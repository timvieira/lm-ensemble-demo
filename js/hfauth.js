// "Sign in with Hugging Face" for the cloud models (OAuth with PKCE; no server, no secret).
// The access token stays in this browser and is only sent to the Space, so ZeroGPU charges
// the visitor's own daily GPU quota instead of the tiny anonymous one.

// Client ID of the OAuth app (public by design): "Diverse sampling demo", shared with dpp-lm-demo; its redirect URLs
// are under huggingface.co/settings/connected-applications, Developer Applications, Settings.
export const HF_CLIENT_ID = '43d09d55-ee33-4a6f-8447-4731a6da25b9';
const SCOPES = 'openid profile';
const KEY = 'dpp-lm-hf-auth';
const HUB = 'https://cdn.jsdelivr.net/npm/@huggingface/hub@2.17.10/+esm';

export const authEnabled = () => !!HF_CLIENT_ID;

export function getAuth() {
  try {
    const a = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (a && new Date(a.expiresAt) > new Date(Date.now() + 60e3)) return a;
  } catch {}
  return null;
}

export function signOut() {
  try { localStorage.removeItem(KEY); } catch {}
}

// Send the visitor to Hugging Face; `stash` is kept across the redirect (e.g. the prompt).
export async function signIn(stash = {}) {
  const { oauthLoginUrl } = await import(HUB);
  try { sessionStorage.setItem(KEY + '-stash', JSON.stringify(stash)); } catch {}
  location.href = await oauthLoginUrl({ clientId: HF_CLIENT_ID, scopes: SCOPES, redirectUrl: location.origin + location.pathname });
}

// On page load: finish a sign-in if we are coming back from Hugging Face.
// Returns { auth, stash } after a redirect, null otherwise.
// A refused or failed sign-in comes back with ?error=…: returns { error, stash } so the page can restore
// the visitor's settings and say what happened.
const takeStash = () => {
  let stash = {};
  try { stash = JSON.parse(sessionStorage.getItem(KEY + '-stash') || '{}'); sessionStorage.removeItem(KEY + '-stash'); } catch {}
  return stash;
};
export async function finishSignIn() {
  if (!authEnabled()) return null;
  const q = new URLSearchParams(location.search);
  if (q.has('error')) {
    history.replaceState(null, '', location.pathname);
    return { error: q.get('error_description') || q.get('error'), stash: takeStash() };
  }
  if (!q.has('code')) return null;
  const { oauthHandleRedirectIfPresent } = await import(HUB);
  const r = await oauthHandleRedirectIfPresent();
  history.replaceState(null, '', location.pathname); // drop ?code=…&state=…
  if (!r) return null;
  const auth = {
    token: r.accessToken, expiresAt: r.accessTokenExpiresAt,
    user: r.userInfo?.preferred_username, isPro: !!r.userInfo?.isPro,
  };
  try { localStorage.setItem(KEY, JSON.stringify(auth)); } catch {}
  return { auth, stash: takeStash() };
}
