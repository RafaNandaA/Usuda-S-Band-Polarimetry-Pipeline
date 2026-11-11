"""Duration at fixed brightness: E[y|x] = exp(β0 + β_W ln W + β_B ln B + β_IP IP), y ∈ {l², v²} debiased with
σ_full + σ_self, Gaussian quasi-score with log link (unweighted IRLS). Models A (B = ρ_pk) and B (B = I_int); samples S0 and
H (bright half), each ± Q-guard. Three mirrored branches + 'not decided', each with a fixed wording."""
import numpy as np
from .stats import ci

WORDING_FALLS = ("<l²> of the integrated pulse falls with duration at fixed brightness after removal of the Gaussian "
                 "self-noise; the remaining decline is the expected behaviour of a pulse made of more independent elements (or "
                 "more statistical degrees of freedom), and non-Gaussian (amplitude-modulated) self-noise, which is not removed, "
                 "is degenerate with that (vS09 §4.2, §4.5; van Straten & Tiburzi 2017 §4.2); it does not show that the emitted "
                 "polarization state changes with duration")


def glm_log(X, y, it=100):
    b = np.zeros(X.shape[1])
    b[0] = np.log(max(np.mean(y), 1e-12))
    for _ in range(it):
        mu = np.exp(X @ b)
        g = X.T @ (y - mu)
        J = (X * mu[:, None]).T @ X
        step = np.linalg.solve(J, g)
        t = 1.0
        n0 = np.linalg.norm(g)
        while t > 1e-4:
            bn = b + t * step
            mun = np.exp(np.clip(X @ bn, -50, 50))
            if np.linalg.norm(X.T @ (y - mun)) <= n0 or t < 2e-4:
                break
            t /= 2
        b = bn
        if np.max(np.abs(t * step)) < 1e-10:
            break
    return b


def y_moments(D, noise_scale=1.0):
    I = D['I']
    sQ = D['s2Q'] + D['s2self']['Q']; sU = D['s2U'] + D['s2self']['U']; sV = D['s2V'] + D['s2self']['V']
    l2 = (D['Q'] ** 2 + D['U'] ** 2 - noise_scale * (sQ + sU)) / I ** 2
    v2 = (D['V'] ** 2 - noise_scale * sV) / I ** 2
    return l2, v2


def design(D, idx, model, Wkey='W', decile_fe=False):
    W = D[Wkey][idx]
    Bc = D['rho_pk'][idx] if model == 'A' else D['I'][idx]
    ip = D['LFIP'][idx].astype(float)
    cols = [np.ones(idx.size), np.log(W)]
    if decile_fe:
        e = np.percentile(Bc, np.arange(10, 100, 10))
        dc = np.digitize(Bc, e)
        cols += [(dc == k).astype(float) for k in range(1, 10)]
    else:
        cols.append(np.log(Bc))
    if ip.any() and not ip.all():
        cols.append(ip)
    return np.c_[tuple(cols)]


def fit_boot(D, idx, model, y, rng, B=2000, **kw):
    X = design(D, idx, model, **kw)
    yy = y[idx]
    b0 = glm_log(X, yy)
    n = idx.size
    bs = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n, n)
        bs[b] = glm_log(X[s], yy[s])[1]
    return float(b0[1]), ci(bs), bs


def decide(cis):
    if all(c[1] < 0 for c in cis):
        return 'falls'
    if all(c[0] > 0 for c in cis):
        return 'rises'
    if all(c[0] <= 0 <= c[1] and (c[1] - c[0]) / 2 <= 0.5 for c in cis):
        return 'no duration dependence, |beta_W| <= %.2f' % max(max(abs(c[0]), abs(c[1])) for c in cis)
    return 'not decided'


def run(D, samples, rng, B=2000, qguard=None, noise_scale=1.0, Wkey='W', decile_fe=False, moments=('l2', 'v2')):
    """samples: dict name -> index array. qguard: boolean mask of windows to exclude for the Q-guard fits (None = n/c)."""
    l2, v2 = y_moments(D, noise_scale)
    Y = {'l2': l2, 'v2': v2}
    out = {}
    for sname, idx in samples.items():
        res = {}
        cis = []
        for model in ('A', 'B'):
            sets = [('all', idx)]
            if qguard is not None:
                sets.append(('Q-guard', idx[~qguard[idx]]))
            for gname, ii in sets:
                for mom in moments:
                    b0, c, _ = fit_boot(D, ii, model, Y[mom], rng, B, Wkey=Wkey, decile_fe=decile_fe)
                    res['%s_%s_%s' % (mom, model, gname)] = {'beta_W': b0, 'CI': c, 'n': int(ii.size),
                                                              'p_W': -b0 if mom == 'l2' else None}
                    if mom == 'l2':
                        cis.append(c)
        res['n_fits_l2'] = len(cis)
        res['branch'] = decide(cis)
        res['four_fits_available'] = len(cis) == 4
        out[sname] = res
    return out


def statement(out, self_ok, qguard_ok):
    """S0 first, then H. A branch that holds only with σ_full does not count (self-noise must be removed);
    the four-fit rule needs the Q-guard fits."""
    notes = []
    if not self_ok:
        notes.append('Gaussian self-noise NOT removed (self-noise sums not supplied): the decision rule cannot be met')
    if not qguard_ok:
        notes.append('Q-guard fits not available (window monitors not supplied): two of the four fits missing')
    for s in ('S0', 'H'):
        if s in out and out[s]['branch'] != 'not decided':
            br = out[s]['branch']
            if notes:
                return 'not decided (rule preconditions unmet)', s, br, notes
            txt = WORDING_FALLS if br == 'falls' else br
            return txt, s, br, notes
    return 'not decided', None, None, notes
