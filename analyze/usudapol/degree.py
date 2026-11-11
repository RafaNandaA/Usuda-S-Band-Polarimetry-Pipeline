"""Total degree d and the test of 'not a single polarization state'.
d̂ = max(0, d²)^½, d² = l² + v² (σ_full, 2σ² convention); σ_d² = (Q²σ²_Q + U²σ²_U + V²σ²_V)/(I⁴ d̂²) + d̂² σ²_I/I² (first order);
F_<1 = frac(d̂ + 2σ_d < 1 − ε_inst), F_=1 = frac(|1 − ε_inst − d̂| < 2σ_d), ε_inst = 0.005; strata S/N_int = I/σ_I,full ≥ 20, ≥ 50.
Decision: 'a typical pulse is not a single state' iff lower 95 % bound of F_<1 ≥ 0.5 at S/N ≥ 20; 'most bright pulses' iff only at S/N ≥ 50; opposite branch iff
lower bound of F_=1 ≥ 0.5 at S/N ≥ 20; else 'the data do not decide'. Lin+23 comparison at S/N ≥ 100: ⟨d²⟩^½, median d_V/d."""
import numpy as np
from .stats import ci

EPS_INST = 0.005


def per_pulse(D):
    I, Q, U, V = D['I'], D['Q'], D['U'], D['V']
    d2 = D['l2'] + D['v2']
    dh = np.sqrt(np.clip(d2, 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        sd2 = (Q ** 2 * D['s2Q'] + U ** 2 * D['s2U'] + V ** 2 * D['s2V']) / (I ** 4 * dh ** 2) + dh ** 2 * D['s2I'] / I ** 2
    sd = np.sqrt(np.where(dh > 0, sd2, np.inf))
    return dh, sd, d2


def run(D, idx, rng, B=20000):
    dh, sd, d2 = per_pulse(D)
    snr = D['snr_int']
    lt = (dh + 2 * sd < 1 - EPS_INST).astype(float)
    eq = (np.abs(1 - EPS_INST - dh) < 2 * sd).astype(float)
    out = {}
    for thr in (20, 50):
        ii = idx[snr[idx] >= thr]
        n = ii.size
        if n == 0:
            out['SN>=%d' % thr] = {'n': 0}
            continue
        dr = ii[rng.integers(0, n, (B, n))]
        out['SN>=%d' % thr] = {'n': int(n), 'F_lt1': float(lt[ii].mean()), 'F_lt1_CI': ci(lt[dr].mean(1)),
                               'F_eq1': float(eq[ii].mean()), 'F_eq1_CI': ci(eq[dr].mean(1)), 'median_d': float(np.median(dh[ii]))}
    a, b = out['SN>=20'], out['SN>=50']
    if a.get('n') and a['F_lt1_CI'][0] >= 0.5:
        dec = 'a typical pulse is not a single polarization state over its duration and our band'
    elif b.get('n') and b['F_lt1_CI'][0] >= 0.5:
        dec = 'most bright pulses are not a single polarization state'
    elif a.get('n') and a['F_eq1_CI'][0] >= 0.5:
        dec = 'most pulses are consistent with a single polarization state'
    else:
        dec = 'the data do not decide'
    out['decision'] = dec
    ii = idx[snr[idx] >= 100]
    if ii.size:
        dv = np.sqrt(np.clip(D['v2'][ii], 0, None))
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(dh[ii] > 0, dv / dh[ii], np.nan)
        out['Lin_compare_SN>=100'] = {'n': int(ii.size), 'rms_d': float(np.sqrt(max(0, d2[ii].mean()))),
                                      'median_dV_over_d': float(np.nanmedian(r)),
                                      'label': 'not matched in S/N definition or window (Lin: 8σ in a 16-µs running average); Lin <d²>^½ ≈ 0.45 for S/N > 100'}
    return out
