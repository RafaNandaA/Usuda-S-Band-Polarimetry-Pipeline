"""Band-common fast term of the differential gain gamma(t).

After the settling curve, the gain step and the ellipticity leak are removed, the nebular band V/I still varies on 20-60 min
time scales by about 0.04 % rms, the same in all four subbands. The most likely cause is gain cycling in one chain upstream of
the subband split. This term is fitted as a cubic B-spline in time and projected orthogonal to {1, settling, step, band Q/I,
band U/I}, so it does not change m or the settling curve. The slow part of gamma(t) on time scales of an hour or more is
degenerate with the band-common ellipticity on one track and is left in the settling model.
"""
import numpy as np
from scipy.interpolate import BSpline


def residual_series(o, step_t0, step_s):
    """Leak-cleaned band V/I residual per snapshot and the slow regressors, from a run_chain result o."""
    d = o['d']
    cm = np.ones((4, 32), bool); cm[:, -1] = False; cm[1, 0] = False
    sod = d['toff_sod']; th = (sod - sod.min()) / 3600
    H = (sod >= step_t0).astype(float)
    RR, LL, RLf = o['RR0'], o['LL0'], o['RLf0']
    therm = 1 - np.exp(-th / o['dgfit']['tau_h'])
    K = d['K']
    res = np.zeros((K, 4)); QIs = np.zeros((K, 4)); UIs = np.zeros((K, 4)); Is = np.zeros((K, 4))
    for s in range(4):
        bs = lambda X: X[:, s, :][:, cm[s]].sum(1)
        I = bs(RR) + bs(LL); V = bs(RR) - bs(LL)
        Q = 2 * np.real(RLf[:, s, :][:, cm[s]]).sum(1); U = 2 * np.imag(RLf[:, s, :][:, cm[s]]).sum(1)
        VI, QI, UI = V / I, Q / I, U / I
        QIs[:, s] = QI; UIs[:, s] = UI; Is[:, s] = I
        A = np.c_[np.ones(K), therm, QI, UI]
        cf = np.linalg.lstsq(A, VI - step_s * H, rcond=None)[0]
        res[:, s] = VI - step_s * H - A @ cf
    band = (res * Is).sum(1) / Is.sum(1)
    QIb = (QIs * Is).sum(1) / Is.sum(1); UIb = (UIs * Is).sum(1) / Is.sum(1)
    return dict(th=th, sod=sod, band=band, H=H, therm=therm, QIb=QIb, UIb=UIb)


def _basis(t, dk, T0, T1):
    n = max(int(np.ceil((T1 - T0) / dk)), 1)
    kn = np.r_[[T0] * 3, np.linspace(T0, T1, n + 1), [T1] * 3]
    return BSpline.design_matrix(np.clip(t, T0, T1), kn, 3).toarray()


def choose_knot_spacing(z, candidates_min=(60, 45, 30, 20, 15, 10)):
    """Knot spacing by blocked 10-fold cross-validation (5-min blocks), taking the coarsest spacing within one standard error
    of the minimum."""
    th, band = z['th'], z['band']
    T0, T1 = th.min(), th.max()
    blk = np.floor((th - T0) / (5 / 60)).astype(int); fold = blk % 10
    res = {}
    for dkm in candidates_min:
        dk = dkm / 60; e = np.zeros(len(th))
        for k in range(10):
            tr = fold != k; te = fold == k
            c = np.linalg.lstsq(_basis(th[tr], dk, T0, T1), band[tr], rcond=None)[0]
            e[te] = band[te] - _basis(th[te], dk, T0, T1) @ c
        bm = np.array([np.mean(e[blk == b] ** 2) for b in np.unique(blk)])
        res[dkm] = (float(np.mean(e ** 2)), float(bm.std() / np.sqrt(len(bm))))
    best = min(res, key=lambda k: res[k][0])
    return max(k for k in res if res[k][0] <= res[best][0] + res[best][1]), res


def build_table(z, table_rows, dk_min, step_t0, dt=60.0):
    """Gamma(t) samples with the fast term: the settling-plus-step table times the fast term, renormalised to the same mean over the
    snapshots. table_rows: (n, 2) array (sod, gamma) of the settling-plus-step table. Returns (sod, gamma) samples every 60 s
    plus the step knots."""
    th, sod, band = z['th'], z['sod'], z['band']
    T0, T1 = th.min(), th.max(); S0 = sod.min()
    dk = dk_min / 60
    c = np.linalg.lstsq(_basis(th, dk, T0, T1), band, rcond=None)[0]
    X = np.c_[np.ones(len(th)), z['therm'], z['H'], z['QIb'], z['UIb']]
    proj = np.linalg.lstsq(X, _basis(th, dk, T0, T1) @ c, rcond=None)[0]
    grid = np.arange(np.ceil(S0), sod.max(), dt)
    o = np.argsort(sod)
    interp = lambda x: np.interp(grid, sod[o], x[o])
    Xg = np.c_[np.ones(len(grid)), interp(z['therm']), (grid >= step_t0).astype(float), interp(z['QIb']), interp(z['UIb'])]
    sg = _basis((grid - S0) / 3600, dk, T0, T1) @ c - Xg @ proj
    ok = table_rows
    step_knots = ok[(ok[:, 0] > step_t0 - 1.6) & (ok[:, 0] < step_t0 + 1.4), 0]
    tsamp = np.unique(np.r_[ok[0, 0], grid[(grid > ok[0, 0]) & (grid < ok[-1, 0]) & (np.abs(grid - step_t0) > 1.0)], step_knots, ok[-1, 0]])

    def g_at(t):
        go = np.interp(t, ok[:, 0], ok[:, 1]); VIo = (1 - go ** 2) / (1 + go ** 2)
        s = np.interp(t, grid, sg)
        return go * np.sqrt((1 - VIo - s) / (1 + VIo + s)) / np.sqrt((1 - VIo) / (1 + VIo))
    gn = g_at(tsamp) / (g_at(sod).mean() / np.interp(sod, ok[:, 0], ok[:, 1]).mean())
    return tsamp, gn, sg
