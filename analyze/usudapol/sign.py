"""Sign balance of circular polarization over the V zero-point grid with one symmetric multiplicity (|z| > 2.73 over 8 tests),
the pulse V/I vs γ(t)-shape regression and the beam-squint test on the nebula."""
import numpy as np
from .stats import ci, ceil_to

Z_BONF = 2.73            # p = 0.05/8 two-sided, the same threshold for a detection and for a null
DELTA_GRID = (-0.01, -0.005, 0.0, 0.005, 0.01)


def sigma_V_tot(D):
    return np.sqrt(D['s2V'] + D['s2self']['V'] + (0.004 * D['I']) ** 2)


def crossfit_Vprime(D, Vzp, idx, chi, psi0):
    """V' = V_zp - (b_Q Q + b_U U), (b_Q, b_U) the feed terms of the ellipticity model (feed + sky terms) fitted to V_zp/I,
    cross-fitted on even/odd nseq."""
    I, Q, U = D['I'], D['Q'], D['U']
    z = (Q + 1j * U) / I * np.exp(-2j * chi) * np.exp(-2j * psi0)
    X = np.c_[np.ones_like(I), Q / I, U / I, z.real, z.imag]
    y = Vzp / I
    even = (D['nseq'] % 2 == 0)
    Vp = Vzp.copy()
    bs = []
    for tr, te in ((idx[even[idx]], idx[~even[idx]]), (idx[~even[idx]], idx[even[idx]])):
        b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
        bs.append(b[1:3].tolist())
        Vp[te] = Vzp[te] - b[1] * Q[te] - b[2] * U[te]
    return Vp, bs


def f_test(V, sig, idx, k):
    x = idx[np.abs(V[idx]) > k * sig[idx]] if k > 0 else idx
    n = x.size
    if n == 0:
        return np.nan, 0, np.nan, x
    f = float(np.mean(V[x] < 0))
    return f, n, (f - 0.5) / np.sqrt(0.25 / n), x


def run(D, idx, Vzp, Vprime, rng, B=2000, step_sod=29056.6, step_err=0.0, delta_grid=DELTA_GRID):
    I = D['I']; sig = sigma_V_tot(D)
    after = D['sod'] > step_sod
    grid = []
    for dlt in delta_grid:
        for st in ((-step_err, step_err) if step_err > 0 else (0.0,)):
            shift = dlt * I + st * I * after
            row = {'delta': dlt, 'delta_step': st, 'tests': []}
            for nm, V in (('V_zp', Vzp), ("V'", Vprime)):
                Vs = V - shift
                for k in (2, 3, 4, 5):
                    f, n, z, _ = f_test(Vs, sig, idx, k)
                    row['tests'].append({'V': nm, 'k': k, 'f': f, 'n': n, 'z': z})
            f, n, z, x = f_test(Vzp - shift, sig, idx, 3)
            fb = np.array([np.mean((Vzp - shift)[x[rng.integers(0, n, n)]] < 0) for _ in range(B)])
            row['primary'] = {'f': f, 'n': n, 'z': z, 'CI': ci(fb)}
            grid.append(row)
    sig_pts = [(r['delta'], r['delta_step'], t['V'], t['k'], np.sign(t['z'])) for r in grid for t in r['tests']
               if abs(t['z']) > Z_BONF]
    # detection: at every grid point, some ladder test exceeds with one common sign
    signs_per_point = []
    for r in grid:
        s = {np.sign(t['z']) for t in r['tests'] if abs(t['z']) > Z_BONF}
        signs_per_point.append(s)
    common = set.intersection(*signs_per_point) if all(signs_per_point) else set()
    if common:
        sgn = 'more negative' if -1 in common else 'more positive'
        out = 'a preferred sign is detected (relative to the background V zero point): %s' % sgn
    elif not sig_pts:
        b = max(max(abs(r['primary']['CI'][0] - 0.5), abs(r['primary']['CI'][1] - 0.5)) for r in grid)
        out = 'no preferred sign, |f - 0.5| <= %.2f over the stated zero-point range' % ceil_to(b, 0.01)
    else:
        out = 'depends on the zero point (or ladder step): significant at %s' % [(p[0], p[1], p[2], p[3]) for p in sig_pts]
    prim = [r for r in grid if r['delta'] == 0.0][0]['primary']   # 0 is always a grid point
    allp = {'V_int': float(np.mean(D['V'][idx] < 0)), 'V_zp': float(np.mean(Vzp[idx] < 0)),
            "V'": float(np.mean(Vprime[idx] < 0))}
    return {'grid': grid, 'decision': out, 'primary_delta0': prim, 'all_pulse_fractions': allp}


def offpulse_check(Voff, s2Voff):
    """(a) off-pulse apertures: fraction with |V_off| > 3σ ≈ 0.27 % (background terms only)."""
    fr = np.abs(Voff) > 3 * np.sqrt(s2Voff)
    n = fr.size
    f = fr.mean()
    se = np.sqrt(0.0027 * 0.9973 / n)
    return {'frac': float(f), 'expected': 0.0027, 'se': float(se), 'works': bool(f < 0.0027 + 3 * se)}


def gamma_shape(sod, table_csv):
    """g(t) = (V/I)_bg,raw smooth = (1 - γ²)/(1 + γ²), γ the applied L-voltage factor (L -> L/γ), interpolated in time."""
    t, g = _read_gamma(table_csv)
    gi = np.interp(sod, t, g)
    return (1 - gi ** 2) / (1 + gi ** 2)


def _read_gamma(fn):
    t, g = [], []
    for ln in open(fn):
        if ln.startswith('#') or not ln.strip() or ln[0].isalpha():
            continue
        a, b = ln.split(',')[:2]
        t.append(float(a)); g.append(float(b))
    return np.array(t), np.array(g)


def pulse_zp_regression(D, idx, Vzp, gshape, rng, B=2000):
    """V_zp/I on g(t) with intercept and feed-frame q, u. CI ∋ 0 -> no correction; else the time-resolved zero
    point V_zp - b g(t) I is a definitional variant."""
    X = np.c_[np.ones(idx.size), gshape[idx], D['q'][idx], D['u'][idx]]
    y = (Vzp / D['I'])[idx]
    b0 = np.linalg.lstsq(X, y, rcond=None)[0][1]
    bs = np.array([np.linalg.lstsq(X[s], y[s], rcond=None)[0][1] for s in (rng.integers(0, idx.size, idx.size) for _ in range(B))])
    lo, hi = ci(bs)
    out = {'b': float(b0), 'CI': [lo, hi], 'g_sd': float(np.std(gshape[idx]))}
    out['outcome'] = 'no correction (CI contains 0)' if lo <= 0 <= hi else 'time-resolved zero point enters as a variant'
    if lo > 0:
        out['note'] = 'positive b: unexplained for any instrumental model; enters as the variant'
    return out


def squint_test(vi_pre, g, chi, sod, rng, block=50, nboot=500):
    """Squint (1×χ) test on PRE-γ nebular V/I: V/I = c0 + c1 g(t) + p_c cos χ + p_s sin χ (γ-model basis {1, g(t)}),
    block bootstrap over windows sorted by sod (blocks of 50, 500 draws); 99 % CI; |corr| of (p_c, p_s) with the γ basis > 0.9
    -> uninformative."""
    o = np.argsort(sod)
    y, X = vi_pre[o], np.c_[np.ones(o.size), g[o], np.cos(chi[o]), np.sin(chi[o])]
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    n = o.size; nblk = int(np.ceil(n / block))
    bs = []
    for _ in range(nboot):
        st = rng.integers(0, n - block + 1, nblk)
        ii = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    bs = np.array(bs)
    corr = np.corrcoef(X[:, 1:].T)
    cmax = float(np.max(np.abs(corr[0, 1:])))
    cis = [ci(bs[:, k], 0.5, 99.5) for k in (2, 3)]
    det = any(c[0] > 0 or c[1] < 0 for c in cis)
    out = {'p_c': float(b0[2]), 'p_s': float(b0[3]), 'CI99_p_c': cis[0], 'CI99_p_s': cis[1], 'max_abs_corr_with_gamma_basis': cmax,
           'const': float(b0[0]), 'coef_g': float(b0[1])}
    out['outcome'] = 'uninformative (|corr| > 0.9)' if cmax > 0.9 else ('squint detected' if det else 'squint not detected')
    return out
