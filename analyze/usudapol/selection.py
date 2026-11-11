"""Sample selection: the Stokes-I S/N cut that keeps the sample complete in V, the completeness tests and the cut ladder,
the false-positive check, the coverage curve, the LFIP phase edges and the quality flags.
The detection lists are per hand, so a pulse that is bright in one hand only can be missed by the other list. A cut on Stokes I
at a level where the fainter-hand list is still complete for any V/I makes the sample independent of the sign and size of V."""
import numpy as np
from scipy import stats as ss
from .stats import ci, boot_idx


def r_quantiles(sn_ours_R, sn_list_R, sn_ours_L, sn_list_L, rng, B=2000):
    """r_j = our per-hand running S/N / list S/N in that hand, pooled over list entries with list S/N >= 10 (both lists).
    r_c = 97.5th percentile, with the median, the 16-84 % range and bootstrap intervals printed."""
    r = []
    for o, l in ((sn_ours_R, sn_list_R), (sn_ours_L, sn_list_L)):
        m = np.isfinite(l) & (l >= 10) & np.isfinite(o)
        r.append(o[m] / l[m])
    r = np.concatenate(r)
    q = lambda x, p: np.percentile(x, p, axis=-1)
    idx = boot_idx(r.size, B, rng)
    rb = r[idx]
    out = {'n': int(r.size), 'r_c': float(q(r, 97.5)), 'r_med': float(q(r, 50)), 'r_16': float(q(r, 16)),
           'r_84': float(q(r, 84)), 'r_c_CI': ci(q(rb, 97.5)), 'r_med_CI': ci(q(rb, 50))}
    return out, r


def I_SN_cut(r_c, sig_R=None, sig_L=None):
    """I_SN_c = 5 r_c (σ_R + σ_L)/σ_I, σ_I² = σ_R² + σ_L²; the equal-noise form 5√2 r_c is used unless the hand-noise
    asymmetry exceeds 10 %."""
    if sig_R is not None and sig_L is not None:
        a = float(np.median(sig_R / sig_L))
        if abs(a - 1) > 0.10:
            sR, sL = float(np.median(sig_R)), float(np.median(sig_L))
            return 5 * r_c * (sR + sL) / np.hypot(sR, sL), 'general (asymmetry %.3f)' % a
        return 5 * np.sqrt(2) * r_c, 'equal-noise (asymmetry %.3f <= 10%%)' % a
    return 5 * np.sqrt(2) * r_c, 'equal-noise (per-hand band sigma not supplied)'


def _degrade(vi_hi, I_hi, s2V_hi, s2I_hi, vi_lo_sig, rng):
    """Degrade higher-bin pulses to the lower bin's per-pulse noise: each higher-bin pulse gets a random lower-bin partner's
    σ(V/I); V and I noise of variance s²σ²_V,full and s²σ²_I,full added with s chosen so that σ(V/I) matches."""
    own = np.sqrt(s2V_hi) / I_hi
    tgt = rng.choice(vi_lo_sig, size=vi_hi.size, replace=True)
    s2 = np.clip(tgt ** 2 - own ** 2, 0, None) / (s2V_hi / I_hi ** 2)
    s = np.sqrt(s2)
    V = vi_hi * I_hi + rng.normal(size=vi_hi.size) * s * np.sqrt(s2V_hi)
    I = I_hi + rng.normal(size=vi_hi.size) * s * np.sqrt(s2I_hi)
    return V / I


def completeness_tests(isn, cut, vi, I, s2V, s2I, rng, ndraw=60):
    """(a) KS of signed V/I, B1 vs degraded B2; (b) central fraction |V/I| < w (w = 30th pct of degraded B2), two-sided
    two-proportion test; (c) the same B2 vs degraded B3 (diagnostic). B1 = [c,2c), B2 = [2c,4c), B3 = [4c, inf).
    The degradation is random, so every p value is the MEDIAN over ndraw >= 60 noise draws."""
    B1 = (isn >= cut) & (isn < 2 * cut)
    B2 = (isn >= 2 * cut) & (isn < 4 * cut)
    B3 = isn >= 4 * cut
    res = {'n_B1': int(B1.sum()), 'n_B2': int(B2.sum()), 'n_B3': int(B3.sum()), 'ndraw': ndraw}

    def one(lo, hi):
        sig_lo = np.sqrt(s2V[lo]) / I[lo]
        vh = _degrade(vi[hi], I[hi], s2V[hi], s2I[hi], sig_lo, rng)
        vl = vi[lo]
        ks = ss.ks_2samp(vl, vh)
        w = np.percentile(np.abs(vh), 30)
        f1, f2 = np.mean(np.abs(vl) < w), np.mean(np.abs(vh) < w)
        n1, n2 = vl.size, vh.size
        pp = (f1 * n1 + f2 * n2) / (n1 + n2)
        z = (f1 - f2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        return ks.statistic, ks.pvalue, w, f1, f2, z, 2 * ss.norm.sf(abs(z))

    def pair(lo, hi, tag):
        if lo.sum() < 10 or hi.sum() < 10:
            return {'tag': tag, 'status': 'too few pulses'}
        r = np.array([one(lo, hi) for _ in range(ndraw)])
        med = np.median(r, axis=0)
        n1, n2 = int(lo.sum()), int(hi.sum())
        Dcrit = 1.628 * np.sqrt((n1 + n2) / (n1 * n2))
        se3 = np.sqrt(0.3 * 0.7 * (1 / n1 + 1 / n2))
        t = 0.3 - 2.58 * se3
        return {'tag': tag, 'KS_D': float(med[0]), 'KS_p': float(med[1]), 'w': float(med[2]), 'frac_lo': float(med[3]),
                'frac_hi': float(med[4]), 'central_z': float(med[5]), 'central_p': float(med[6]),
                'KS_p_range': [float(r[:, 1].min()), float(r[:, 1].max())], 'central_p_range': [float(r[:, 6].min()), float(r[:, 6].max())],
                'KS_Dcrit_p01': float(Dcrit), 'central_min_deficit': float((0.3 - t) / (1 - t)), 'KS_min_central_deficit': float(2 * Dcrit)}
    res['B1vB2'] = pair(B1, B2, 'B1 vs B2')
    res['B2vB3'] = pair(B2, B3, 'B2 vs B3 (diagnostic)')
    a = res['B1vB2']
    res['pass'] = bool('KS_p' in a and a['KS_p'] > 0.01 and a['central_p'] > 0.01)
    if 'central_z' in a:
        res['B1_central_sign'] = 'deficit' if a['central_z'] < 0 else 'excess'
    return res


def calibrated_hand_sigma(base_I, base_V, a):
    """Calibrated per-hand band noise per window from the calibrated per-channel baseline means (one complex sample per
    cell: Var(|R|²) = μ_R², μ_R = (μ_I + μ_V)/2, μ_L = (μ_I - μ_V)/2; band = Σ a_c X_c). Returns σ_R, σ_L, g = (σ_R+σ_L)/σ_I."""
    muR, muL = (base_I + base_V) / 2, (base_I - base_V) / 2
    sR = np.sqrt((a ** 2 * muR ** 2).sum(1))
    sL = np.sqrt((a ** 2 * muL ** 2).sum(1))
    return sR, sL, (sR + sL) / np.hypot(sR, sL)


def lfip_edges(phase, lo=0.38, hi=0.42, span=(0.30, 0.55), bw=0.0025):
    """LFIP phase edges from the phase histogram of the list windows. Background = median count per bin in the sidebands
    [0.30, 0.35) and (0.47, 0.55]; the significant run = contiguous bins around the peak bin with count > bg + 3√bg.
    Primary edges kept ('supported') iff the significant run lies inside [lo, hi] (the distribution has fallen to the
    background within the window) AND >= 95 % of the excess counts in [0.30, 0.55] lie inside [lo, hi]; else the run's edges are
    reported and used."""
    ph = np.mod(phase, 1.0)
    e = np.arange(span[0], span[1] + 1e-9, bw)
    c, _ = np.histogram(ph, e)
    mid = 0.5 * (e[1:] + e[:-1])
    side = (mid < 0.35) | (mid > 0.47)
    bg = float(np.median(c[side]))
    thr = bg + 3 * np.sqrt(max(bg, 1.0))
    k = int(np.argmax(c))
    a = k
    while a > 0 and c[a - 1] > thr:
        a -= 1
    b = k
    while b < c.size - 1 and c[b + 1] > thr:
        b += 1
    run = (float(e[a]), float(e[b + 1]))
    exc = np.clip(c - bg, 0, None)
    inside = (mid >= lo) & (mid <= hi)
    frac = float(exc[inside].sum() / exc.sum()) if exc.sum() > 0 else float('nan')
    keep = bool(frac >= 0.95 and run[0] >= lo - 1e-9 and run[1] <= hi + 1e-9)
    return {'bin_width': bw, 'background_per_bin': bg, 'threshold': thr, 'peak_phase': float(mid[k]), 'run_edges': run,
            'excess_fraction_inside_primary': frac, 'primary_kept': keep, 'edges_used': (lo, hi) if keep else run,
            'counts': c.tolist(), 'bin_edges': e.tolist()}


def run_ladder(isn, base_cut, base_mask, vi, I, s2V, s2I, rng, ndraw=60):
    """isn may be pre-normalised (isn·√2/g_j with the per-window hand noise g_j) so that the scalar base cut 5√2·r_c applies."""
    """Precondition and pre-set ladder ×1, ×1.25, ×1.5, ×2.0; first passing step defines S0; classification if
    none passes."""
    steps = []
    adopted = None
    for f in (1.0, 1.25, 1.5, 2.0):
        cut = base_cut * f
        m = base_mask & np.isfinite(isn)
        r = completeness_tests(isn[m], cut, vi[m], I[m], s2V[m], s2I[m], rng, ndraw)
        r['factor'] = f
        r['cut'] = float(cut)
        steps.append(r)
        if r['pass'] and adopted is None:
            adopted = f
            break
    out = {'steps': steps}
    if adopted is not None:
        out['adopted_factor'] = adopted
        out['classification'] = 'complete (precondition passed at x%.2f)' % adopted
        return out
    last = steps[-1]
    a, d = last['B1vB2'], last['B2vB3']
    sel = ('central_p' in a and a['central_p'] <= 0.01 and a['central_z'] < 0 and
           ('central_p' not in d or d['central_p'] > 0.05 or np.sign(d['central_z']) != np.sign(a['central_z'])))
    intr = ('central_p' in d and d['central_p'] <= 0.01 and 'central_z' in a and np.sign(d['central_z']) == np.sign(a['central_z']))
    if sel:
        cls = 'selection'
    elif intr:
        cls = 'intrinsic brightness dependence'
    else:
        cls = 'selection'   # 'otherwise (neither pattern): as selection'
    out['adopted_factor'] = 2.0
    out['classification'] = cls
    return out


def false_positive_offpulse(profI, peak_idx, mu, sig, cut, npos, span=(0, 500)):
    """I_SN on an off-pulse span of each window with the same number of 10-us boxcar positions."""
    n = profI.shape[0]
    mx = np.zeros(n)
    for j in range(n):
        seg = profI[j, span[0]:span[0] + npos + 9] - mu[j]
        cs = np.convolve(seg, np.ones(10), 'valid')
        mx[j] = cs.max() / (sig[j] * np.sqrt(10))
    nab = int(np.sum(mx >= cut))
    return {'n_above': nab, 'works': nab <= 1, 'broken': nab >= 5, 'max': float(mx.max())}


def coverage_curve(cut, sigI_med, W_grid=(1, 2, 5, 10, 20, 50)):
    """Fluence a pulse needs to pass: W <= 10: I_int >= cut σ_I √10; W > 10 (flat pulse): × W/10."""
    return {int(w): float(cut * sigI_med * np.sqrt(10) * (max(w, 10) / 10.0)) for w in W_grid}


def flag_masks(flag_bits, spike):
    any_flag = (np.asarray(flag_bits).astype(np.int64) != 0)
    if spike is not None:
        any_flag |= np.asarray(spike).astype(bool)
    return any_flag
