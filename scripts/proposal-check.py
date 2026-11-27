# Toy check: two "prompts" = two random LMs over V symbols + EOS, strings of length <= L (EOS forced at L).
# Compare two proposals for the global generalized-mean target phi_tau = (1/2 pA^tau + 1/2 pB^tau)^(1/tau):
#   fixed : local f-ensemble with fixed weights 1/2 (the design proposed for the demo)
#   psi   : the paper's shaping proposal r ∝ psi(x'|x) = f(prefA(xx'), prefB(xx')) / f(prefA(x), prefB(x))  (Sec. 5.2)
# Report KL(Phi || r) and the exact ESS fraction 1 / (1 + chi2(Phi||r)) by enumeration.
import itertools, numpy as np
rng = np.random.default_rng(0)
V, L = 3, 5
def rand_lm(conc):
    tab = {}
    for n in range(L):
        for x in itertools.product(range(V), repeat=n):
            tab[x] = rng.dirichlet([conc]*(V+1))   # last entry = EOS
    return tab
A, B = rand_lm(0.5), rand_lm(0.5)
def cond(lm, x):
    if len(x) == L: return np.r_[np.zeros(V), 1.0]
    return lm[x]
strings = [x for n in range(L+1) for x in itertools.product(range(V), repeat=n)]
def strprob(lm, x):
    p = 1.0
    for t in range(len(x)): p *= cond(lm, x[:t])[x[t]]
    return p * cond(lm, x)[V]
def gm(a, b, tau, wa=.5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if tau == 0: return a**wa * b**(1-wa)
    with np.errstate(divide='ignore'):
        return (wa*a**tau + (1-wa)*b**tau)**(1/tau)
def proposal_prob(x, tau, kind):
    r, pa, pb = 1.0, 1.0, 1.0      # pa, pb: prefix probabilities of x[:t]
    for t in range(len(x)+1):
        ca, cb = cond(A, x[:t]), cond(B, x[:t])
        if kind == 'fixed': s = gm(ca, cb, tau)
        else:               s = gm(pa*ca, pb*cb, tau)   # f of the extended prefixes; normalising removes f(pa, pb)
        s = s / s.sum()
        y = x[t] if t < len(x) else V
        r *= s[y]
        if t < len(x): pa *= ca[y]; pb *= cb[y]
    return r
pA = np.array([strprob(A, x) for x in strings]); pB = np.array([strprob(B, x) for x in strings])
print(f"total mass A={pA.sum():.6f} B={pB.sum():.6f}  (#strings={len(strings)})")
for tau, name in [(-8,'≈min'), (-1,'harmonic'), (0,'product'), (1,'sum'), (8,'≈max')]:
    phi = gm(pA, pB, tau); Phi = phi / phi.sum()
    out = []
    for kind in ['fixed', 'psi']:
        r = np.array([proposal_prob(x, tau, kind) for x in strings])
        m = Phi > 0
        kl = np.sum(Phi[m] * np.log(Phi[m] / r[m]))
        chi2 = np.sum(Phi[m]**2 / r[m]) - 1
        out.append(f"{kind}: KL={kl:.4f} ESS/n={1/(1+chi2):.3f}")
    print(f"tau={tau:>3} {name:9s} " + "   ".join(out))
