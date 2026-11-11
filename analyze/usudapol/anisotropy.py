"""Census and anisotropy <l²>/<v²> with the ellipticity-conditional limits L_R (min over R) and U_R (max over R),
computed inside every bootstrap draw; intervals on a = <v²>/(<l²>+<v²>) mapped to the ratio."""
import numpy as np
from .stats import ci, ratio_ci_from_a, a_to_ratio, floor_to, ceil_to
from .ellipticity import inside_any


def census(D, sel, rng, B=2000, quint=True):
    I, L, V = D['I'], D['L'], D['V']
    idx = np.where(sel)[0]
    n = idx.size
    bi = idx[rng.integers(0, n, (B, n))]
    eli = np.minimum(1, L / I)
    vi = V / I
    sv = D['s2V'] / I ** 2

    def stats_(ix):
        l2m, v2m = D['l2'][ix].mean(-1), D['v2'][ix].mean(-1)
        var = vi[ix].var(-1, ddof=1)
        return dict(ELI=eli[ix].mean(-1), sdVI=np.sqrt(var), sdVI_ns=np.sqrt(np.clip(var - sv[ix].mean(-1), 0, None)),
                    l2=l2m, v2=v2m, a=v2m / (l2m + v2m))
    p = stats_(idx)
    bb = stats_(bi)
    out = {'n': int(n)}
    for k in p:
        out[k] = float(p[k])
        out[k + '_CI'] = ci(bb[k])
    if quint:
        e = np.percentile(I[idx], [20, 40, 60, 80])
        qn = np.digitize(I[idx], e)
        out['a_by_quintile'] = []
        for k in range(5):
            ii = idx[qn == k]
            l2m, v2m = D['l2'][ii].mean(), D['v2'][ii].mean()
            out['a_by_quintile'].append(float(v2m / (l2m + v2m)))
    return out


class RegionRatio:
    """Ratio as a function of the ellipticity offset Δ = m - m_app (band level):
    <v_m²> = <v2> - 2 Δ·<v x> + Δᵀ <x xᵀ> Δ, x = (q, u); <l²> unchanged."""

    def __init__(self, D, m_app, pts, parts):
        self.l2, self.v2, self.v, self.q, self.u = D['l2'], D['v2'], D['v'], D['q'], D['u']
        self.m_app = np.asarray(m_app, float)
        self.dl = pts - self.m_app[None, :]
        self.parts = parts

    def sums(self, ix):
        g = lambda a: a[ix].mean(-1)
        return dict(A=g(self.l2), Bv=g(self.v2), gq=g(self.v * self.q), gu=g(self.v * self.u), hqq=g(self.q ** 2),
                    hqu=g(self.q * self.u), huu=g(self.u ** 2))

    @staticmethod
    def vm(s, d):
        dq, du = d[..., 0], d[..., 1]
        return (s['Bv'] - 2 * (dq * s['gq'] + du * s['gu']) + dq * dq * s['hqq'] + 2 * dq * du * s['hqu'] + du * du * s['huu'])

    def a_at(self, s, m):
        d = np.asarray(m, float) - self.m_app
        v = self.vm(s, d)
        return v / (s['A'] + v)

    def a_extremes(self, s):
        """Returns (a_max over R -> L_R, a_min over R -> U_R) for scalar sums s."""
        v = self.vm(s, self.dl)
        vmax, vmin = v.max(), v.min()
        H = np.array([[s['hqq'], s['hqu']], [s['hqu'], s['huu']]])
        try:
            dstar = np.linalg.solve(H, np.array([s['gq'], s['gu']]))
            if inside_any(self.m_app + dstar, self.parts):
                vmin = min(vmin, float(self.vm(s, dstar)))
        except np.linalg.LinAlgError:
            pass
        return vmax / (s['A'] + vmax), vmin / (s['A'] + vmin)


def ratio_block(D, idx_sample, rr, m_hat, rng, B=2000, edges_fn=None, want_region=True):
    """Ratio at m_hat (primary point) and at m_app (check), with a-mapped CIs; L_R, U_R inside every draw."""
    n = idx_sample.size
    draws = idx_sample[rng.integers(0, n, (B, n))]
    s0 = {k: float(v) for k, v in rr.sums(idx_sample).items()}
    out = {'n': int(n)}
    a_hat = rr.a_at(s0, m_hat)
    a_app = rr.a_at(s0, rr.m_app)
    out['ratio_at_m_hat'] = float(a_to_ratio(a_hat))
    out['ratio_at_m_app'] = float(a_to_ratio(a_app))
    ah, aa, amax, amin = (np.empty(B) for _ in range(4))
    for b in range(B):
        s = {k: float(v) for k, v in rr.sums(draws[b]).items()}
        ah[b] = rr.a_at(s, m_hat)
        aa[b] = rr.a_at(s, rr.m_app)
        if want_region:
            amax[b], amin[b] = rr.a_extremes(s)
    out['ratio_at_m_hat_CI'] = ratio_ci_from_a(ah)
    out['ratio_at_m_app_CI'] = ratio_ci_from_a(aa)
    out['a_at_m_hat'] = float(a_hat)
    out['a_at_m_hat_CI'] = ci(ah)
    if want_region:
        out['L_R'] = float(a_to_ratio(np.percentile(amax, 97.5)))
        out['U_R'] = float(a_to_ratio(np.percentile(amin, 2.5)))
    out['x_iso_at_m_hat'] = out['ratio_at_m_hat'] / 2
    out['_a_hat_draws'] = ah
    return out


def branch_ratio(lo, hi):
    if lo > 2:
        return 'linear exceeds circular beyond the isotropic 2:1'
    if hi < 2:
        return 'circular exceeds the isotropic share (ratio below 2:1)'
    return 'consistent with isotropic'


def decide_overall(res, region_desc):
    L, U = res['L_R'], res['U_R']
    br = branch_ratio(L, U)
    if br.startswith('linear'):
        txt = '>= %.2f, i.e. >= %.1fx isotropic, for ellipticity terms within %s' % (L, floor_to(L / 2, 0.1), region_desc)
    elif br.startswith('circular'):
        txt = '<= %.2f, i.e. <= %.1fx isotropic' % (U, ceil_to(U / 2, 0.1))
    else:
        txt = 'consistent with isotropic, [%.2f, %.2f]' % (L, U)
    h = branch_ratio(*res['ratio_at_m_hat_CI'])
    a = branch_ratio(*res['ratio_at_m_app_CI'])
    return {'branch_R': br, 'text': txt, 'branch_at_m_hat': h, 'branch_at_m_app': a,
            'state_both': (h != br) or (a != br)}


def quintiles(D, idx_sample, rr, m_hat, rng, B=2000):
    """Quintiles of I_int within the sample, edges recomputed in every draw."""
    I = D['I']
    n = idx_sample.size
    e0 = np.percentile(I[idx_sample], [20, 40, 60, 80])
    q0 = np.digitize(I[idx_sample], e0)
    rows = []
    pt = []
    for k in range(5):
        ii = idx_sample[q0 == k]
        s = {kk: float(v) for kk, v in rr.sums(ii).items()}
        pt.append((ii.size, a_to_ratio(rr.a_at(s, m_hat)), a_to_ratio(rr.a_at(s, rr.m_app)), rr.a_at(s, m_hat)))
    ah = np.empty((B, 5)); aa = np.empty((B, 5)); amax = np.empty((B, 5)); amin = np.empty((B, 5))
    for b in range(B):
        dr = idx_sample[rng.integers(0, n, n)]
        e = np.percentile(I[dr], [20, 40, 60, 80])
        qb = np.digitize(I[dr], e)
        for k in range(5):
            s = {kk: float(v) for kk, v in rr.sums(dr[qb == k]).items()}
            ah[b, k] = rr.a_at(s, m_hat)
            aa[b, k] = rr.a_at(s, rr.m_app)
            amax[b, k], amin[b, k] = rr.a_extremes(s)
    for k in range(5):
        rows.append({'Q': k + 1, 'n': int(pt[k][0]), 'ratio_m_hat': float(pt[k][1]), 'ratio_m_hat_CI': ratio_ci_from_a(ah[:, k]),
                     'ratio_m_app': float(pt[k][2]), 'ratio_m_app_CI': ratio_ci_from_a(aa[:, k]),
                     'L_R': float(a_to_ratio(np.percentile(amax[:, k], 97.5))),
                     'U_R': float(a_to_ratio(np.percentile(amin[:, k], 2.5))), 'a': float(pt[k][3])})
    Ls = [r['L_R'] for r in rows]; Us = [r['U_R'] for r in rows]
    if all(x > 2 for x in Ls):
        dec = 'in every brightness quintile (> 2:1 over R)'
    else:
        dec = 'quintiles > 2:1: %s; below 2:1: %s; consistent: %s' % (
            [i + 1 for i in range(5) if Ls[i] > 2], [i + 1 for i in range(5) if Us[i] < 2],
            [i + 1 for i in range(5) if Ls[i] <= 2 <= Us[i]])
    return rows, dec


def piecewise_tertile_LR(D, idx_sample, rr, el, rng, B=2000):
    """Piecewise variant: each El tertile's pulses at their own worst point of R."""
    n = idx_sample.size
    out = np.empty(B)
    for b in range(B):
        dr = idx_sample[rng.integers(0, n, n)]
        e = np.percentile(el[dr], [100 / 3, 200 / 3])
        t = np.digitize(el[dr], e)
        num = 0.0; vtot = 0.0
        for k in range(3):
            ii = dr[t == k]
            s = {kk: float(v) for kk, v in rr.sums(ii).items()}
            v = rr.vm(s, rr.dl).max()
            vtot += v * ii.size; num += s['A'] * ii.size
        out[b] = vtot / (num + vtot)
    return float(a_to_ratio(np.percentile(out, 97.5)))


def matched_mp(D, lf_idx, mp_idx, k=5):
    """Five nearest MP per LFIP pulse in (ln W, ln E), Mahalanobis with the MP covariance."""
    X = np.c_[np.log(D['W']), np.log(np.clip(D['I'], 1e-12, None))]
    C = np.cov(X[mp_idx].T)
    Ci = np.linalg.inv(C)
    nn = []
    for j in lf_idx:
        d = X[mp_idx] - X[j]
        m = np.einsum('ij,jk,ik->i', d, Ci, d)
        nn.append(mp_idx[np.argsort(m)[:k]])
    return np.array(nn)


def lfip_vs_matched(D, lf_idx, nn, rng, B=2000):
    l2, v2 = D['l2'], D['v2']
    r = lambda ix: l2[ix].mean() / v2[ix].mean()
    aL = lambda ix: v2[ix].mean() / (l2[ix].mean() + v2[ix].mean())
    rL, rM = r(lf_idx), r(nn.ravel())
    n = lf_idx.size
    dd = np.empty(B); aLb = np.empty(B); aMb = np.empty(B)
    for b in range(B):
        s = rng.integers(0, n, n)
        aLb[b] = aL(lf_idx[s]); aMb[b] = aL(nn[s].ravel())
        dd[b] = a_to_ratio(aLb[b]) - a_to_ratio(aMb[b])
    lo, hi = ci(dd)
    br = 'consistent (difference within [%.2f, %.2f])' % (lo, hi) if lo <= 0 <= hi else \
        ('differs: LFIP higher' if lo > 0 else 'differs: LFIP lower')
    return {'ratio_LFIP': rL, 'ratio_LFIP_CI': ratio_ci_from_a(aLb), 'ratio_matchedMP': rM,
            'ratio_matchedMP_CI': ratio_ci_from_a(aMb), 'diff': rL - rM, 'diff_CI': [lo, hi], 'branch': br}
