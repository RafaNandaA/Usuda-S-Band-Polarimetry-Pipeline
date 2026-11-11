"""Energy distribution with the goodness-of-fit test (Clauset, Shalizi & Newman 2009 semi-parametric bootstrap;
p <= 0.1 -> the power law is rejected and no index is reported).
Differential index α of dN/dE ∝ E^-α by MLE above E_min (α = 1 + n/Σ ln(E/E_min)); E_min = minimum KS distance over a candidate
grid [50 quantiles of E between the 0th and 90th percentile, refined over every data point between the neighbours of the best one; tail >= 50 pulses], and >= the selection coverage threshold at the 90th
percentile of W. Interval: pulse bootstrap with E_min re-selected in every draw. Comparison: Enoto et al. 2021 MP α = 3.02 ± 0.19."""
import numpy as np
from .stats import ci

ENOTO = (3.02, 0.19)


def _fit_fixed(Es, k):
    t = Es[k:]
    n = t.size
    a = 1 + n / np.sum(np.log(t / t[0]))
    F = 1 - (t / t[0]) ** (1 - a)
    i = np.arange(1, n + 1) / n
    D = max(np.max(np.abs(i - F)), np.max(np.abs(i - 1 / n - F)))
    return a, D, n


def fit(E, emin_floor=0.0, ncand=50, mintail=50):
    Es = np.sort(E[E > 0])
    qs = np.unique(np.searchsorted(Es, np.percentile(Es, np.linspace(0, 90, ncand))))
    best = None
    bk = None
    for j, k in enumerate(qs):
        if Es[k] < emin_floor or Es.size - k < mintail:
            continue
        a, D, n = _fit_fixed(Es, k)
        if best is None or D < best[1]:
            best, bk = (a, D, n, Es[k]), j
    if best is None:
        return None
    # refine over every data point between the neighbouring grid candidates (the coarse grid alone biases the GoF, see tests)
    lo = qs[max(bk - 1, 0)]
    hi = qs[min(bk + 1, qs.size - 1)]
    for k in range(lo, hi + 1):
        if Es[k] < emin_floor or Es.size - k < mintail:
            continue
        a, D, n = _fit_fixed(Es, k)
        if D < best[1]:
            best = (a, D, n, Es[k])
    return best


def fit_at(E, emin):
    Es = np.sort(E[E >= emin])
    a, D, n = _fit_fixed(Es, 0)
    return a, D, n


def run(E, W, rng, emin_floor=0.0, B=20000, ngof=1000):
    a0, D0, n0, em0 = fit(E, emin_floor)
    n = E.size
    ab = np.empty(B)
    for b in range(B):
        r = fit(E[rng.integers(0, n, n)], emin_floor)
        ab[b] = r[0] if r else np.nan
    # goodness of fit (Clauset et al. 2009 §4.1)
    body = E[E < em0]
    ptail = n0 / n
    Ds = np.empty(ngof)
    for k in range(ngof):
        nt = rng.binomial(n, ptail)
        tail = em0 * (1 - rng.random(nt)) ** (-1 / (a0 - 1))
        bod = body[rng.integers(0, body.size, n - nt)] if body.size else np.empty(0)
        r = fit(np.concatenate([tail, bod]), emin_floor)
        Ds[k] = r[1] if r else np.nan
    pgof = float(np.nanmean(Ds >= D0))
    diff = ab - (ENOTO[0] + ENOTO[1] * rng.normal(size=B))
    lo, hi = ci(diff)
    out = {'alpha': float(a0), 'alpha_CI': ci(ab), 'E_min': float(em0), 'n_above': int(n0), 'KS_D': float(D0),
           'gof_p': pgof, 'gof_n': ngof, 'gof_pass': pgof > 0.1, 'diff_vs_Enoto': float(a0 - ENOTO[0]), 'diff_CI': [lo, hi]}
    out['comparison'] = ('consistent with Enoto+21 (difference within [%.2f, %.2f])' % (lo, hi) if lo <= 0 <= hi
                         else ('steeper than Enoto+21' if lo > 0 else 'flatter than Enoto+21'))
    out['decision'] = ('index printed with its definition; ' + out['comparison']) if out['gof_pass'] else \
        'power law rejected by the goodness-of-fit test (p <= 0.1): no index reported'
    out['check_Emin_x1.5'] = fit_at(E, 1.5 * em0)[0]
    out['check_Emin_x2'] = fit_at(E, 2 * em0)[0]
    e = np.percentile(W, [100 / 3, 200 / 3])
    t = np.digitize(W, e)
    out['alpha_by_W_tertile'] = [(lambda r: float(r[0]) if r else None)(fit(E[t == k], emin_floor)) for k in range(3)]
    return out
