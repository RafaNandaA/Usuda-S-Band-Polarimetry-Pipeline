"""Apertures and per-pulse error model.

sigma²_X,full = sigma_X² [ W (1 + W/n_b) + 2 I_int/mu_I ]  (band form; off-pulse term + pulse x background cross term)
per-channel cross term: sum_c 2 sigma_X,c² I_int,c / mu_I,c (primary when the calibrated per-channel moments exist)
Gaussian self-noise per 1-MHz x 1-us cell (n = 1): sigma²_X,self = sum_{c,t} (sigma²_X,c/mu_c²) T_X,c,t,
  T_X = a_X + ½ kappa a_P/(2 - kappa/2), a_P = P² - sigma_I,c²(1 + 2P/mu_c), a_X = X² - sigma_X,c²(1 + 2P/mu_c),
  kappa = ½ kappa_max (primary), kappa_max = 1 - d²_int; variants kappa = 0, kappa_max.
"""
import numpy as np


def sigma2_full_band(sig, W, nb, I_int, mu_I, cross=True, scale=1.0):
    s2 = np.asarray(sig, float) ** 2
    off = W * s2 * (1.0 + W / nb)
    cr = 2.0 * s2 * I_int / mu_I if cross else 0.0
    return scale * (off + cr)


def sigma2_cross_perchan(sig2_c, I_c, mu_c, kept):
    """Per-channel cross term (pulse x background); arrays (n, 128); kept channel index array."""
    return np.sum(2.0 * sig2_c[:, kept] * I_c[:, kept] / mu_c[:, kept], axis=1)


def self_noise_cells(P, X, mu_c, sig2_I_c, sig2_X_c, kappa):
    """Gaussian self-noise of one Stokes X for ONE pulse from its cells (arrays chan x time, baseline-subtracted,
    calibrated). Returns sigma²_X,self. kappa scalar in [0, 1]."""
    mu = mu_c[:, None]
    corr = 1.0 + 2.0 * P / mu
    aP = P ** 2 - sig2_I_c[:, None] * corr
    aX = X ** 2 - sig2_X_c[:, None] * corr
    T = aX + 0.5 * kappa * aP / (2.0 - kappa / 2.0)
    return float(np.sum((sig2_X_c[:, None] / mu ** 2) * T))


def self_noise_asym_cells(P, Qp, Up, mu_c, sig2_I_c, sig2_Q_c, sig2_U_c):
    """Exact per-cell self-noise asymmetry Var_self(Q') - Var_self(U') = ς²(a_Q' - a_U') (S_p² cancels), summed with the
    per-cell weights sigma²/mu²."""
    mu = mu_c[:, None]
    corr = 1.0 + 2.0 * P / mu
    s2 = 0.5 * (sig2_Q_c + sig2_U_c)[:, None]
    aQ = Qp ** 2 - s2 * corr
    aU = Up ** 2 - s2 * corr
    return float(np.sum((s2 / mu ** 2) * (aQ - aU)))


def L_debiased(Q, U, s2Q, s2U):
    """L_int = [max(0, Q² + U² - ½(σ²_Q + σ²_U))]^½, zero below 1σ (one σ², Wardle & Kronberg 1974 form)."""
    s2 = 0.5 * (s2Q + s2U)
    L2 = Q ** 2 + U ** 2 - s2
    L = np.sqrt(np.clip(L2, 0, None))
    L[np.sqrt(Q ** 2 + U ** 2) < np.sqrt(s2)] = 0.0
    return L


def moments(I, Q, U, V, s2Q, s2U, s2V):
    """2σ² convention with each component's own full variance."""
    l2 = (Q ** 2 + U ** 2 - s2Q - s2U) / I ** 2
    v2 = (V ** 2 - s2V) / I ** 2
    return l2, v2


def sky_frame(Q, U, chi_rad, mask_psi0):
    """z_s = (Q + iU) e^{-2i chi}; psi0 from MP pulses (mask_psi0), unweighted; Q', U'."""
    zs = (Q + 1j * U) * np.exp(-2j * chi_rad)
    psi = 0.5 * np.angle(zs)
    psi0 = 0.5 * np.angle(np.mean(np.exp(2j * psi[mask_psi0])))
    Qp = np.real(zs * np.exp(-2j * psi0))
    Up = np.imag(zs * np.exp(-2j * psi0))
    return zs, psi, psi0, Qp, Up


def wrap90(deg):
    return (np.asarray(deg) + 90.0) % 180.0 - 90.0


def offpulse_apertures(prof, peak_idx, W, mu_I_cat, start_offset=-700, exclude=100):
    """Off-pulse aperture of W samples per window from the band profiles (±750 us around peak_index), for the off-pulse
    null checks of the debias, the aligned fraction and the sign test. The aperture lies outside the pulse, so every
    statistic computed on it must be consistent with zero. Baseline = mean of the profile samples farther than `exclude` us from the
    peak and outside the aperture (n_b' of them). Returns dict of sums (I,Q,U,V) and n_b' per window.
    The aperture starts at peak + start_offset (default -700 us), and W is clipped to 400 samples."""
    I, Q, U, V = prof
    n, m = I.shape
    t = np.arange(m)
    out = {k: np.zeros(n) for k in 'IQUV'}
    nbp = np.zeros(n)
    Wc = np.clip(np.asarray(W, int), 1, 400)
    for j in range(n):
        p = int(peak_idx[j]) if peak_idx is not None else m // 2
        a0 = max(0, p + start_offset)
        a1 = min(m, a0 + Wc[j])
        offm = (np.abs(t - p) > exclude)
        offm[a0:a1] = False
        nbp[j] = offm.sum()
        for k, arr in zip('IQUV', (I, Q, U, V)):
            row = arr[j]
            out[k][j] = row[a0:a1].sum() - (a1 - a0) * row[offm].mean()
    out['nbp'] = nbp
    out['W'] = Wc.astype(float)
    return out


def rho_pk_profile(profI, peak_idx, lo, hi, mu_I):
    """rho_pk = max over the aperture of (I_t - mu_I)/mu_I from the band profiles; lo/hi are offsets from the
    profile peak index (bins)."""
    n = profI.shape[0]
    out = np.full(n, np.nan)
    for j in range(n):
        p = int(peak_idx[j])
        a0 = max(0, p + int(lo[j]))
        a1 = min(profI.shape[1], p + int(hi[j]) + 1)
        out[j] = (profI[j, a0:a1].max() - mu_I[j]) / mu_I[j]
    return out
