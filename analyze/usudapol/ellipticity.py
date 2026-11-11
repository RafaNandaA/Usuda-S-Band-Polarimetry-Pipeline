"""Receptor ellipticity measured on the pulses: errors-in-variables corrected regression v = a0 + m_Q q + m_U u + s_Q q' + s_U u'
with feed-fixed (q, u) and sky-fixed (q', u') terms. Only a feed-fixed coupling can be instrumental, and the pulses rotate in the
feed frame, so the two are separable. Checks E1-E4, region R (union of 99 % ellipses, chi2 = 9.21), fallback F-E."""
import numpy as np
from scipy import stats as ss
from .stats import ci

CHI2_99_2DOF = 9.21


def design(I, Q, U, V, chi, psi0, extra=None):
    q, u, v = Q / I, U / I, V / I
    z = (q + 1j * u) * np.exp(-2j * chi) * np.exp(-2j * psi0)
    X = np.c_[np.ones_like(q), q, u, z.real, z.imag]
    if extra is not None:
        X = np.c_[X, extra]
    return X, v


def err_cov(I, Q, U, V, s2I, s2Q, s2U, s2V, chi, psi0, nextra=0):
    """Mean per-pulse measurement-error covariance S_uu of x = (1, q, u, q', u'[, extra]) and S_uv (x-error with v-error),
    first order in the common I noise."""
    q, u, v = Q / I, U / I, V / I
    rI = s2I / I ** 2
    Sqq = s2Q / I ** 2 + q * q * rI
    Suu = s2U / I ** 2 + u * u * rI
    Squ = q * u * rI
    th = -2 * (chi + psi0)
    c, s = np.cos(th), np.sin(th)
    n = I.size
    k = 5 + nextra
    S = np.zeros((n, k, k))
    Cqu = np.stack([np.stack([Sqq, Squ], -1), np.stack([Squ, Suu], -1)], -2)       # (n,2,2)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)                # rotation (q,u)->(q',u')
    S[:, 1:3, 1:3] = Cqu
    S[:, 3:5, 3:5] = R @ Cqu @ np.transpose(R, (0, 2, 1))
    S[:, 1:3, 3:5] = Cqu @ np.transpose(R, (0, 2, 1))
    S[:, 3:5, 1:3] = R @ Cqu
    Suv = np.zeros((n, k))
    cvq, cvu = v * q * rI, v * u * rI
    Suv[:, 1], Suv[:, 2] = cvq, cvu
    Suv[:, 3] = c * cvq - s * cvu
    Suv[:, 4] = s * cvq + c * cvu
    return S, Suv


def eiv_fit(X, v, S, Suv, idx=None):
    if idx is not None:
        X, v, S, Suv = X[idx], v[idx], S[idx], Suv[idx]
    n = X.shape[0]
    Sxx = X.T @ X / n - S.mean(0)
    Sxy = X.T @ v / n - Suv.mean(0)
    return np.linalg.solve(Sxx, Sxy)


def ols_fit(X, v, idx=None):
    if idx is not None:
        X, v = X[idx], v[idx]
    return np.linalg.lstsq(X, v, rcond=None)[0]


def fit_with_boot(X, v, S, Suv, rng, B=2000):
    b = eiv_fit(X, v, S, Suv)
    n = X.shape[0]
    bs = np.array([eiv_fit(X, v, S, Suv, rng.integers(0, n, n)) for _ in range(B)])
    return b, bs


def run(I, Q, U, V, s2I, s2Q, s2U, s2V, chi, psi0, el, sod, rng, B=2000, m_app=None, extra_E4=None):
    """Returns the full result: m̂ (EIV) and OLS, Σ_p, reliability λ, checks E1, E3, E4, region ellipse list."""
    X, v = design(I, Q, U, V, chi, psi0)
    S, Suv = err_cov(I, Q, U, V, s2I, s2Q, s2U, s2V, chi, psi0)
    b, bs = fit_with_boot(X, v, S, Suv, rng, B)
    bo = ols_fit(X, v)
    Sig = np.cov(bs[:, 1:3].T)
    lam = 1 - np.mean(S[:, 1, 1]) / np.var(X[:, 1])
    corr = np.corrcoef(X[:, 1:5].T)
    e1 = float(np.max(np.abs(corr[0:2, 2:4])))
    out = {'m_hat': b[1:3].tolist(), 's_hat': b[3:5].tolist(), 'a0': float(b[0]), 'm_ols': bo[1:3].tolist(),
           's_ols': bo[3:5].tolist(), 'Sigma_p': Sig.tolist(), 'sd_m': np.sqrt(np.diag(Sig)).tolist(),
           's_sd': bs[:, 3:5].std(0).tolist(), 's_CI': [ci(bs[:, 3]), ci(bs[:, 4])], 'reliability_q': float(lam),
           'E1_max_abs_corr_feed_sky': e1, 'E1_pass': e1 < 0.9, 'boot_s': bs[:, 3:5]}
    # E3: El tertiles
    edges = np.percentile(el, [100 / 3, 200 / 3])
    terc = np.digitize(el, edges)
    ell = [(b[1:3], Sig, 'all')]
    tpairs = []
    for t in range(3):
        m = terc == t
        bt, bst = fit_with_boot(X[m], v[m], S[m], Suv[m], rng, B)
        St = np.cov(bst[:, 1:3].T)
        tpairs.append((bt[1:3], St))
        ell.append((bt[1:3], St, 'El tertile %d' % (t + 1)))
        out['tertile_%d' % (t + 1)] = {'m_hat': bt[1:3].tolist(), 'sd': np.sqrt(np.diag(St)).tolist(),
                                       's_hat': bt[3:5].tolist(), 's_CI': [ci(bst[:, 3]), ci(bst[:, 4])]}
    W = sum(np.linalg.inv(St) for _, St in tpairs)
    mbar = np.linalg.solve(W, sum(np.linalg.inv(St) @ mt for mt, St in tpairs))
    chi2 = float(sum((mt - mbar) @ np.linalg.inv(St) @ (mt - mbar) for mt, St in tpairs))
    p3 = float(ss.chi2.sf(chi2, 4))
    out['E3_chi2'] = chi2
    out['E3_p'] = p3
    out['E3_pass'] = p3 > 0.01
    # E4 specification variant
    if extra_E4 is not None:
        X4, _ = design(I, Q, U, V, chi, psi0, extra=extra_E4)
        S4, Suv4 = err_cov(I, Q, U, V, s2I, s2Q, s2U, s2V, chi, psi0, nextra=extra_E4.shape[1])
        b4, bs4 = fit_with_boot(X4, v, S4, Suv4, rng, B)
        S4c = np.cov(bs4[:, 1:3].T)
        moved = bool(np.any(np.abs(b4[1:3] - b[1:3]) > np.sqrt(np.diag(Sig))))
        out['E4'] = {'m_hat': b4[1:3].tolist(), 'sd': np.sqrt(np.diag(S4c)).tolist(), 'moved_more_than_sigma': moved}
        if moved:
            ell.append((b4[1:3], S4c, 'E4 variant'))
    out['fallback_trigger_sd>0.05'] = bool(np.any(np.sqrt(np.diag(Sig)) > 0.05))
    out['ellipses'] = ell
    if m_app is not None:
        out['delta_m'] = (b[1:3] - np.asarray(m_app)).tolist()
    return out


def region(ell_result, m_app, neb=None, fallback=False, npts=360):
    """Adopted range R as boundary points: union of the pulse ellipses (and the nebula ellipse if converged), or the F-E
    rectangle (offsets -0.175/+0.175 in m_QV, -0.137/+0.213 in m_UV about m_app). Returns (list of (center, Sigma9, label)
    ellipses or rectangle, boundary point array (k,2), description)."""
    th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    circ = np.c_[np.cos(th), np.sin(th)]
    pts = []
    parts = []
    if fallback:
        mq, mu = m_app
        lo = np.array([mq - 0.175, mu - 0.137])
        hi = np.array([mq + 0.175, mu + 0.213])
        k = npts // 4
        s = np.linspace(0, 1, k, endpoint=False)
        edges = [np.c_[lo[0] + s * (hi[0] - lo[0]), np.full(k, lo[1])], np.c_[np.full(k, hi[0]), lo[1] + s * (hi[1] - lo[1])],
                 np.c_[hi[0] - s * (hi[0] - lo[0]), np.full(k, hi[1])], np.c_[np.full(k, lo[0]), hi[1] - s * (hi[1] - lo[1])]]
        return [('rectangle', lo.tolist(), hi.tolist())], np.vstack(edges), 'F-E stated range (fallback)'
    for c, Sg, lab in ell_result:
        Lc = np.linalg.cholesky(np.asarray(Sg) * CHI2_99_2DOF)
        pts.append(np.asarray(c)[None, :] + circ @ Lc.T)
        parts.append((np.asarray(c).tolist(), np.asarray(Sg).tolist(), lab))
    desc = 'pulse-measured region (99 % ellipses: ' + ', '.join(p[2] for p in parts) + ')'
    if neb is not None:
        c, Sg = neb
        Lc = np.linalg.cholesky(np.asarray(Sg) * CHI2_99_2DOF)
        pts.append(np.asarray(c)[None, :] + circ @ Lc.T)
        parts.append((list(c), np.asarray(Sg).tolist(), 'nebula (converged)'))
        desc += ' U nebula region'
    return parts, np.vstack(pts), desc


def inside_any(m, parts):
    for p in parts:
        if p[0] == 'rectangle':
            if np.all(m >= np.asarray(p[1])) and np.all(m <= np.asarray(p[2])):
                return True
        else:
            c, Sg = np.asarray(p[0]), np.asarray(p[1])
            d = m - c
            if d @ np.linalg.solve(Sg, d) <= CHI2_99_2DOF:
                return True
    return False
