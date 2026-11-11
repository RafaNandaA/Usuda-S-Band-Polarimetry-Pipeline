"""V zero point. There is no calibrator scan, so system V = 0 is defined on the nebula and the pulse V zero point is only known
within a range:
- δ range = block-bootstrap interval of the residual calibrated nebular V/I (median over windows, relative to the adopted p),
  widened by the time-variable V/I of the gamma(t) ripple and by the observed nebular V/I bound (Wilson & Weiler 1997, as listed
  by Bucciantini & Olmi 2018), scaled to 2258 MHz.
- gain-versus-additive regression on the nebula snapshots, to tell whether the slow L/R drift is a gain change.
- block length from the autocorrelation of the per-window series in time order."""
import numpy as np
from .stats import ci, block_length

NEB_VI_BOUND = 3e-4 * (2258.0 / 610.0) ** -0.5   # = 1.56e-4: observed nebular V/I < 3e-4 at 610 MHz (Wilson & Weiler 1997, via Bucciantini & Olmi 2018), scaled as ν^-1/2
GAMMA_RIPPLE_MAX = 1.3e-3   # time-variable V/I of the fast γ(t) term, max 0.13 % (rms 0.04 %), and its origin (gain or beam) is not known


def zp_grid(vi_neb, sod, p, rng, B=20000, extra=GAMMA_RIPPLE_MAX):
    o = np.argsort(sod)
    x = vi_neb[o]
    L = block_length(x)
    n = x.size
    nblk = int(np.ceil(n / L))
    med = np.empty(B)
    for b in range(B):
        st = rng.integers(0, max(1, n - L + 1), nblk)
        ii = (st[:, None] + np.arange(L)[None, :]).ravel()[:n]
        med[b] = np.median(x[ii])
    lo, hi = ci(med)
    dlo, dhi = lo - p - NEB_VI_BOUND - extra, hi - p + NEB_VI_BOUND + extra
    grid = sorted(set([dlo, dlo / 2, 0.0, dhi / 2, dhi]))
    return {'p': p, 'block_length': int(L), 'median_CI': [lo, hi], 'delta_range': [dlo, dhi], 'grid': grid,
            'bound_added': NEB_VI_BOUND, 'gamma_ripple_added': extra}


def nebula_1a(dump_path, rng, nboot=2000, block_s=60.0):
    """(1a) ln ρ_n = c0 + β ln(P_R/P_L) + η El + κ ln ρ_model on the pre-γ re-dump snapshots (masked coherency).
    ρ_n = mean over subbands of |Σ_c RL| / sqrt(Σ_c RR Σ_c LL) (kept channels; the delay is applied in the dump, so channel sums
    within a subband are coherent; magnitudes avoid φ_s); P_R/P_L = Σ RR / Σ LL; ρ_model = mean over subbands of |A_s e^{2iχ} +
    K_s| from a per-subband linear fit with χ = q − El. Block bootstrap over 60-s blocks of snapshots."""
    import h5py
    f = h5py.File(dump_path, 'r')
    RR, LL = f['coherency/rho_RR'][()], f['coherency/rho_LL'][()]
    RL = f['coherency/rho_RL_re'][()] + 1j * f['coherency/rho_RL_im'][()]
    t = f['geometry/toff_sod'][()]
    el, q = f['geometry/el_deg'][()], f['geometry/q_deg'][()]
    keep = np.ones((4, 32), bool)
    keep[:, 31] = False
    keep[1, 0] = False
    sRR = (RR * keep).sum(2); sLL = (LL * keep).sum(2); sRL = (RL * keep).sum(2)       # (n, 4)
    rho_s = sRL / np.sqrt(sRR * sLL)
    ok = np.all(np.isfinite(rho_s), 1) & np.all(sRR > 0, 1) & np.all(sLL > 0, 1)
    rho_s, sRR, sLL, t, el, q = rho_s[ok], sRR[ok], sLL[ok], t[ok], el[ok], q[ok]
    chi = np.radians(q - el)
    X = np.c_[np.exp(2j * chi), np.ones_like(chi)]
    model = np.zeros(rho_s.shape)
    for s in range(4):
        c = np.linalg.lstsq(X, rho_s[:, s], rcond=None)[0]
        model[:, s] = np.abs(X @ c)
    y = np.log(np.abs(rho_s).mean(1))
    lr = np.log(sRR.sum(1) / sLL.sum(1))
    lm = np.log(model.mean(1))
    D = np.c_[np.ones_like(y), lr, el, lm]
    b0 = np.linalg.lstsq(D, y, rcond=None)[0]
    blk = np.floor((t - t.min()) / block_s).astype(int)
    ub = np.unique(blk)
    idx = [np.where(blk == u)[0] for u in ub]
    bs = np.empty((nboot, 4))
    for k in range(nboot):
        pick = rng.integers(0, ub.size, ub.size)
        ii = np.concatenate([idx[j] for j in pick])
        bs[k] = np.linalg.lstsq(D[ii], y[ii], rcond=None)[0]
    lo, hi = ci(bs[:, 1])
    corr = float(np.corrcoef(lr, lm)[0, 1])
    out = {'n_snap': int(y.size), 'slope': float(b0[1]), 'slope_CI': [lo, hi], 'coef_model': float(b0[3]),
           'coef_model_CI': ci(bs[:, 3]), 'corr_lnPRPL_lnmodel': corr, 'halfwidth': (hi - lo) / 2,
           'lnPRPL_sd': float(np.std(lr))}
    if abs(corr) > 0.9 or out['halfwidth'] > 0.25:
        out['outcome'] = 'uninformative -> not decided (δ(t) carried as the two-sided range [0, -(V/I)_drift(t)])'
    elif lo <= 0 <= hi and not (lo <= -0.5 <= hi) and not (lo <= 0.5 <= hi):
        out['outcome'] = 'γ(t) behaves as a gain'
    elif lo <= -0.5 <= hi and not (lo <= 0 <= hi):
        out['outcome'] = 'the drift is additive in R'
    elif lo <= 0.5 <= hi and not (lo <= 0 <= hi):
        out['outcome'] = 'the drift is additive in L'
    else:
        out['outcome'] = 'not decided (δ(t) carried as the two-sided range [0, -(V/I)_drift(t)])'
    return out
