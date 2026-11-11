"""Aligned fraction <Q'/I>, LOO version, uniform-angle and chi-permutation nulls, duration slope b_rel, floor."""
import numpy as np
from .stats import ci, one_sig_fig_down


def qp_over_i(zs, I, idx_fit, idx_eval):
    psi = 0.5 * np.angle(zs)
    p0 = 0.5 * np.angle(np.mean(np.exp(2j * psi[idx_fit])))
    return np.real(zs[idx_eval] * np.exp(-2j * p0)) / I[idx_eval]


def loo(zs, I, idx):
    """<Q'_LOO/I>: psi0^(-j) = ½ arg(S - e^{2iψ_j}) (leave-one-out, so a pulse does not set its own reference direction)."""
    e = np.exp(1j * np.angle(zs[idx]))      # e^{2iψ}
    S = e.sum()
    p0j = 0.5 * np.angle(S - e)
    return np.real(zs[idx] * np.exp(-2j * p0j)) / I[idx]


def aligned_fraction(D, idx, rng, nnull=1000):
    zs, I, chi = D['zs'], D['I'], D['chi']
    m = float(np.mean(qp_over_i(zs, I, idx, idx)))
    ml = float(np.mean(loo(zs, I, idx)))
    amp = np.abs(zs[idx])
    nul = np.empty(nnull)
    for k in range(nnull):
        ph = rng.uniform(0, 2 * np.pi, idx.size)
        z = amp * np.exp(1j * ph)
        p0 = 0.5 * np.angle(np.mean(np.exp(1j * ph)))
        nul[k] = np.mean(np.real(z * np.exp(-2j * p0)) / I[idx])
    Qf = D['Q'][idx] + 1j * D['U'][idx]
    perm = np.empty(nnull)
    for k in range(nnull):
        z = Qf * np.exp(-2j * chi[idx][rng.permutation(idx.size)])
        p0 = 0.5 * np.angle(np.mean(np.exp(1j * np.angle(z))))
        perm[k] = np.mean(np.real(z * np.exp(-2j * p0)) / I[idx])
    val = round(min(m, ml), 3)
    out = {'mean_QpI': m, 'mean_QpI_LOO': ml, 'LOO_change': ml - m, 'printed_value': val,
           'uniform_null_mean': float(nul.mean()), 'chiperm_p99': float(np.percentile(perm, 99))}
    out['wording'] = 'carries about 0.1 of the intensity' if 0.05 <= val < 0.15 else 'number alone (%.3f)' % val
    out['print_permuted_beside'] = bool(out['chiperm_p99'] >= val / 2)
    return out


def slope(D, idx, Bcov, rng, B=2000):
    """OLS Q'/I = c0 + b_W ln W + b_B ln B; ψ0 refitted per draw; b_rel = b_W/<Q'/I>."""
    zs, I, W = D['zs'], D['I'], D['W']
    X = np.c_[np.ones(idx.size), np.log(W[idx]), np.log(Bcov[idx])]

    def one(ii, Xi):
        y = qp_over_i(zs, I, ii, ii)
        b = np.linalg.lstsq(Xi, y, rcond=None)[0]
        return b[1], b[1] / y.mean()
    bW, br = one(idx, X)
    bb = np.empty((B, 2))
    for b in range(B):
        s = rng.integers(0, idx.size, idx.size)
        bb[b] = one(idx[s], X[s])
    return {'b_W': float(bW), 'b_W_CI': ci(bb[:, 0]), 'b_rel': float(br), 'b_rel_CI': ci(bb[:, 1]), '_b_rel_draws': bb[:, 1]}


def decide_slope(rA, rB):
    cA, cB = rA['b_rel_CI'], rB['b_rel_CI']
    if cA[1] < 0 and cB[1] < 0:
        return 'the aligned part falls with duration'
    if cA[0] > 0 and cB[0] > 0:
        return 'the aligned part rises with duration'
    if cA[0] <= 0 <= cA[1] and cB[0] <= 0 <= cB[1] and (cA[1] - cA[0]) / 2 <= 0.5 and (cB[1] - cB[0]) / 2 <= 0.5:
        X = max(abs(cA[0]), abs(cA[1]), abs(cB[0]), abs(cB[1]))
        return 'no duration dependence, |b_rel| <= %.2f' % X
    return 'not decided'


def floor(D, idx, rng, B=2000):
    """Floor = (lower 95 % bound of <Q'_LOO/I> in the longest-W quintile of S0_MP)², one significant figure down."""
    W = D['W']
    e = np.percentile(W[idx], 80)
    top = idx[W[idx] >= e]
    ql_all = loo(D['zs'], D['I'], idx)
    pos = np.isin(idx, top)
    v = ql_all[pos]
    bs = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(B)])
    lo, hi = ci(bs)
    out = {'n_longest_quintile': int(v.size), 'mean': float(v.mean()), 'CI': [lo, hi]}
    if lo > 0:
        out['floor'] = one_sig_fig_down(lo ** 2)
        out['keep'] = True
        out['wording'] = ('the aligned fraction in the longest-duration quintile is %.3f > 0; if it stays flat, <l²> cannot '
                          'fall below %.1g' % (v.mean(), out['floor']))
    else:
        out['keep'] = False
        out['wording'] = 'no floor stated; upper bound %.3f' % hi
    return out
