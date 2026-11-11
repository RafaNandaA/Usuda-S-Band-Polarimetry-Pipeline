"""Preferred direction of the PA (R_u, ψ0, Rayleigh, R_L, R_feed vs smearing, rate test r, feed-fixed share) and the LFIP
pulses alone."""
import numpy as np
from .stats import ci


def Ru_psi0(psi):
    z = np.mean(np.exp(2j * psi), axis=-1)
    return np.abs(z), 0.5 * np.angle(z)


def rate_fit(L, PA, chi, grid=np.arange(-0.5, 1.5001, 0.05)):
    """Least squares L e^{2iPA} = A e^{2i r chi} (complex A, real r): grid over r then golden-section refinement."""
    y = L * np.exp(2j * PA)

    def sse(r):
        e = np.exp(2j * r * chi)
        A = np.vdot(e, y) / np.vdot(e, e)
        return float(np.sum(np.abs(y - A * e) ** 2))
    vals = np.array([sse(r) for r in grid])
    k = int(np.argmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = sse(c), sse(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = sse(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = sse(d)
    return 0.5 * (a + b)


def feed_share(L, PA, chi):
    y = L * np.exp(2j * PA)
    X = np.c_[np.exp(2j * chi), np.ones_like(chi)]
    A = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(abs(A[1]) / (abs(A[0]) + abs(A[1])))


def preferred_direction(D, idx, rng, B=2000, chi=None, rate_B=None):
    psi = D['psi']; L = D['L']; PA = D['PA']
    chi = D['chi'] if chi is None else chi
    n = idx.size
    R, psi0 = Ru_psi0(psi[idx])
    Z = n * R ** 2
    dr = idx[rng.integers(0, n, (B, n))]
    Rb, _ = Ru_psi0(psi[dr])
    out = {'n': int(n), 'R_u': float(R), 'R_u_CI': ci(Rb), 'psi0_deg': float(np.degrees(psi0)), 'Rayleigh_Z': float(Z),
           'Rayleigh_p': float(np.exp(-Z)), 'null_E_Ru': float(np.sqrt(np.pi / (4 * n))), '_Rb': Rb}
    w = L[idx]
    zL = np.sum(w * np.exp(2j * psi[idx])) / np.sum(w)
    out['R_L'] = float(abs(zL))
    out['R_L_null'] = float(np.sqrt(np.pi / 4 * np.sum(w ** 2)) / np.sum(w))
    # feed-frame resultant vs smearing prediction
    psif = 0.5 * np.angle(D['Q'] + 1j * D['U'])
    Rf, _ = Ru_psi0(psif[idx])
    sm = abs(np.mean(np.exp(2j * chi[idx])))
    Rfb, _ = Ru_psi0(psif[dr])
    smb = np.abs(np.mean(np.exp(2j * chi[dr]), axis=1))
    dd = Rfb - Rb * smb
    sd = float(np.std(dd))
    pred = R * sm
    out.update({'R_feed': float(Rf), 'R_feed_pred': float(pred), 'abs_mean_e2ichi': float(sm),
                'R_feed_minus_pred_CI': ci(dd), 'feed_check_works': bool(abs(Rf - pred) <= max(0.02, 2 * sd)),
                'chi_span_deg': float(np.degrees(chi[idx].max() - chi[idx].min()))})
    # rate test and feed-fixed share
    r0 = rate_fit(L[idx], PA[idx], chi[idx])
    fs0 = feed_share(L[idx], PA[idx], chi[idx])
    Bt = B if rate_B is None else rate_B
    rb = np.empty(Bt); fsb = np.empty(Bt)
    for b in range(Bt):
        ii = dr[b] if b < B else idx[rng.integers(0, n, n)]
        rb[b] = rate_fit(L[ii], PA[ii], chi[ii])
        fsb[b] = feed_share(L[ii], PA[ii], chi[ii])
    out.update({'r': float(r0), 'r_CI': ci(rb), 'feed_share': fs0, 'feed_share_CI': ci(fsb), '_rb': rb})
    return out


def decide_direction(o):
    if o['Rayleigh_p'] < 1e-3:
        ex = 'a preferred direction exists'
        lo, hi = o['r_CI']
        c1, c0 = lo <= 1 <= hi, lo <= 0 <= hi
        if c1 and not c0:
            fr = 'sky-stationary within errors (feed-fixed part <= %.3f)' % o['feed_share_CI'][1]
        elif c0 and not c1:
            fr = 'fixed to the feed (instrumental)'
        elif not c0 and not c1:
            fr = 'turns with the feed angle at rate r = %.3f [%.3f, %.3f]' % (o['r'], lo, hi)
        else:
            fr = 'frame not decided'
    else:
        ex = 'no preferred direction detected; R_u <= %.3f' % o['R_u_CI'][1]
        fr = None
    return ex, fr


def lfip(D, lf_idx, mp_idx, rng, B=2000, nperm=2000, nn=None):
    psi = D['psi']; chi = D['chi']
    n = lf_idx.size
    R, p0L = Ru_psi0(psi[lf_idx])
    RM, p0M = Ru_psi0(psi[mp_idx])
    Z = n * R ** 2
    Q, U = D['Q'][lf_idx], D['U'][lf_idx]
    ch = chi[lf_idx]
    u_sky = np.sqrt(2.0 / n) * np.sum(np.cos(2 * (psi[lf_idx] - p0M)))
    psif = 0.5 * np.angle(D['Q'] + 1j * D['U'])
    _, p0Mf = Ru_psi0(psif[mp_idx])
    u_feed = np.sqrt(2.0 / n) * np.sum(np.cos(2 * (psif[lf_idx] - p0Mf)))
    sm = abs(np.mean(np.exp(2j * ch)))
    permZ = np.empty(nperm); permu = np.empty(nperm)
    for k in range(nperm):
        cp = ch[rng.permutation(n)]
        ps = 0.5 * np.angle((Q + 1j * U) * np.exp(-2j * cp))
        permZ[k] = n * abs(np.mean(np.exp(2j * ps))) ** 2
        permu[k] = np.sqrt(2.0 / n) * np.sum(np.cos(2 * (ps - p0M)))
    pZ = float((1 + np.sum(permZ >= Z)) / (nperm + 1))
    m = mp_idx.size
    dps = np.empty(B); Rb = np.empty(B)
    for b in range(B):
        il = lf_idx[rng.integers(0, n, n)]; im = mp_idx[rng.integers(0, m, m)]
        r1, a1 = Ru_psi0(psi[il]); _, a2 = Ru_psi0(psi[im])
        dps[b] = np.degrees(np.angle(np.exp(2j * (a1 - a2)))) / 2; Rb[b] = r1
    dp0 = float(np.degrees(np.angle(np.exp(2j * (p0L - p0M)))) / 2)
    out = {'n': int(n), 'R_u': float(R), 'R_u_CI': ci(Rb), 'Rayleigh_Z': float(Z), 'Rayleigh_p': float(np.exp(-Z)),
           'chiperm_p': pZ, 'chiperm_Z_p99': float(np.percentile(permZ, 99)), 'chiperm_u_p99': float(np.percentile(permu, 99)),
           'dpsi_deg': dp0, 'dpsi_CI': ci(dps), 'u_sky': float(u_sky), 'u_feed': float(u_feed),
           'u_feed_smearing_expectation': float(u_sky * sm), 'abs_mean_e2ichi_LFIP': float(sm),
           'route_check_works': bool(Z > np.percentile(permZ, 99) and u_sky > np.percentile(permu, 99))}
    if out['Rayleigh_p'] <= 0.01 and pZ <= 0.01:
        out['existence'] = 'LFIP pulses alone show a preferred direction'
        lo, hi = out['dpsi_CI']
        out['direction'] = ('the same direction as the MP pulses, within +-%.1f deg' % max(abs(lo), abs(hi))
                            if lo <= 0 <= hi else 'offset from the MP direction by %.1f deg' % dp0)
    else:
        out['existence'] = 'no preferred direction detected among LFIP pulses; R_u,LFIP <= %.3f' % out['R_u_CI'][1]
        out['direction'] = None
    if nn is not None:
        dR = np.empty(B)
        for b in range(B):
            s = rng.integers(0, n, n)
            dR[b] = Ru_psi0(psi[lf_idx[s]])[0] - Ru_psi0(psi[nn[s].ravel()])[0]
        out['R_u_matchedMP'] = float(Ru_psi0(psi[nn.ravel()])[0])
        out['R_u_diff_CI'] = ci(dR)
    return out
