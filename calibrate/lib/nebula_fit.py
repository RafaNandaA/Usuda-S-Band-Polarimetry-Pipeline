"""Calibration fits on the off-pulse Crab Nebula snapshots.

The nebula fills the beam, is linearly polarized at about 2 % and has no measurable circular polarization, so its coherency
over the track gives every instrumental term we can measure without a calibrator scan. The terms are solved in sequence,
because each one needs the previous ones applied:
  bandpass (per hand, per channel) -> R/L delay residual and phase per subband -> differential gain gamma(t) together with
  the ellipticity (Q,U)->V coupling -> D-terms (feed leakage).
Arrays follow the HDF5 layout written by the nebula reader: (K snapshots, n_sub, n_ch).
"""
import time

import h5py
import numpy as np
from scipy.optimize import curve_fit

C_M_PER_S = 2.99792458e8


def channel_mask(n_sub=4, n_ch=32):
    """Channels that enter band sums and band-common fits. Channel 32 of each subband is the half-filled Nyquist bin of the
    64-point FFT and channel 1 of subband 2 fails the same 3-sigma interior test, so both are left out. They stay in every
    per-channel table."""
    m = np.ones((n_sub, n_ch), bool)
    m[:, -1] = False
    m[1, 0] = False
    return m


def load_snapshots(path):
    """Read the masked nebula coherency (listed pulses removed) and the geometry of each snapshot."""
    with h5py.File(path, 'r') as f:
        d = dict(rho_RR=f['coherency/rho_RR'][...], rho_LL=f['coherency/rho_LL'][...],
                 rho_RL=f['coherency/rho_RL_re'][...] + 1j * f['coherency/rho_RL_im'][...],
                 toff_sod=f['geometry/toff_sod'][...], q_deg=f['geometry/q_deg'][...], el_deg=f['geometry/el_deg'][...],
                 az_deg=f['geometry/az_deg'][...], chi_BWG=f['geometry/chi_BWG_deg'][...], I_band=f['geometry/I_band'][...],
                 freq_MHz=f['frequency/freq_MHz'][...], attrs=dict(f.attrs))
    d['K'], d['n_sub'], d['n_ch'] = d['rho_RR'].shape
    return d


# ------------------------------------------------------------------------------------------------ bandpass
def fit_bandpass(data):
    """Amplitude bandpass per hand and channel: B = sqrt(<auto-power>) over the snapshots. Every snapshot holds the same number
    of samples, so the plain mean is the sample-weighted mean. Normalising each hand's nebula power to unit mean also absorbs
    the scalar hand gains, so gain_R = gain_L = 1."""
    return np.sqrt(data['rho_RR'].mean(axis=0)), np.sqrt(data['rho_LL'].mean(axis=0))


def apply_bandpass(data, B_R, B_L):
    return (data['rho_RR'] / (B_R ** 2)[None], data['rho_LL'] / (B_L ** 2)[None], data['rho_RL'] / (B_R * B_L)[None])


# ------------------------------------------------------------------------------------------------ delay and phase
def coherent_delay_search(C, f_Hz, ch_mask, coarse_ns=(-200.0, 200.0, 2.0), fine_half_ns=4.0, fine_step_ns=0.02):
    """Residual R/L delay per subband from the peak of |sum_c C[s,c] exp(-2 pi i f_c tau)|, coarse grid then a 0.02 ns grid."""
    n_sub = C.shape[0]
    tau_s = np.zeros(n_sub)
    for s in range(n_sub):
        m = ch_mask[s]
        Cs, fs_ = C[s, m], f_Hz[s, m]
        taus = np.arange(coarse_ns[0], coarse_ns[1] + coarse_ns[2], coarse_ns[2]) * 1e-9
        F = np.abs((Cs[None, :] * np.exp(-1j * 2 * np.pi * fs_[None, :] * taus[:, None])).sum(axis=1))
        tau_c = taus[np.argmax(F)]
        taus_f = np.arange(tau_c - fine_half_ns * 1e-9, tau_c + fine_half_ns * 1e-9 + fine_step_ns * 1e-9, fine_step_ns * 1e-9)
        F_f = np.abs((Cs[None, :] * np.exp(-1j * 2 * np.pi * fs_[None, :] * taus_f[:, None])).sum(axis=1))
        tau_s[s] = taus_f[np.argmax(F_f)]
    return tau_s


def fit_delay_phase(rho_RL_bp, freq_MHz, q_deg, el_deg, az_deg, feed=(-1.0, 0.0)):
    """Residual delay tau_s and phase phi_s per subband from the nebula cross-hand, with the feed angle
    chi = q + alpha El + gamma Az held at the feed model.

    The per-channel source term is estimated with the leakage floor removed: rho_RL,k = s e^{2i chi_k} + kappa, so
    C = (<rho e^{-2i chi}> - <rho><e^{-2i chi}>) / (1 - |<e^{-2i chi}>|^2). A plain mean of rho e^{-2i chi} is biased by kappa
    whenever the feed-angle coverage is uneven, which is always the case on one track.
    tau_s comes from the coherent peak search, and phi_s from the phase of the delay-corrected coherent sum of each subband.
    RM is held at 0, because over 128 MHz four per-subband phases reproduce a lambda^2 winding to 99.7 %, so RM is degenerate
    with tau_s and phi_s and cannot be measured from this band."""
    alpha, gamma = float(feed[0]), float(feed[1])
    f_Hz = freq_MHz * 1e6
    chi = np.deg2rad(q_deg) + alpha * np.deg2rad(el_deg) + gamma * np.deg2rad(az_deg)
    e_chi = np.exp(-2j * chi)[:, None, None]
    RL_mean = rho_RL_bp.mean(axis=0)
    ec_mean = e_chi.mean()
    C = ((rho_RL_bp * e_chi).mean(axis=0) - RL_mean * ec_mean) / (1.0 - np.abs(ec_mean) ** 2)
    cm = channel_mask(*C.shape)
    tau_s = coherent_delay_search(C, f_Hz, cm)
    expected = 2 * np.pi * f_Hz * tau_s[:, None]
    phi_s = np.angle((np.where(cm, C, 0.0) * np.exp(-1j * expected)).sum(axis=1))
    phi_s = np.mod(phi_s + np.pi, 2 * np.pi) - np.pi
    return dict(alpha=alpha, gamma=gamma, tau_s=tau_s, phi_inst_s=phi_s, RM=0.0, chi_rad=chi)


def apply_delay(rho_RL_bp, freq_MHz, tau_s):
    return rho_RL_bp * np.exp(-1j * 2 * np.pi * (freq_MHz * 1e6)[None] * tau_s[None, :, None])


def apply_inst_phase(rho_RL, phi_s):
    return rho_RL * np.exp(-1j * phi_s[None, :, None])


# ------------------------------------------------------------------------------------------------ gamma(t) and ellipticity
def fit_gain_and_vleak(data, rho_RR_n, rho_LL_n, rho_RL_cal, chm, step_t0, step_s, norm_mask, n_knots=41):
    """Differential gain gamma(t) and the ellipticity coupling m_QV, m_UV, fitted together on the nebular V/I.

    Two instrumental terms live in the nebular V/I(t). The L/R gain ratio drifts in time (thermal settling of the amplifier
    chains, plus the measured R-chain gain step), and V = |R|^2 - |L|^2 takes that drift directly. Receptor ellipticity leaks
    linear polarization into V as m_QV Q + m_UV U, which follows the feed rotation (Heiles 2001, sec. 6.2). On one track both
    change with time, so they are fitted together: per subband V/I = a + b therm(t) + m_QV Q/I + m_UV U/I (step removed first),
    then the leak-cleaned band V/I gives gamma(t) = sqrt((1 - V/I)/(1 + V/I)).
    The step amplitude step_s (in V/I) is fixed from the 1-s snapshots around it, because the 156-ms snapshots do not resolve it.
    gamma is renormalised to mean 1 over norm_mask (the snapshots used for the bandpass), which defines the V zero point as
    system V = 0 over those snapshots."""
    sod = data['toff_sod'].astype(float)
    t_h = (sod - sod.min()) / 3600.0
    K, ns, nc = rho_RR_n.shape
    H = (sod >= step_t0).astype(float)
    s_fix = float(step_s)

    def settle(x, a, b, tau):
        return a + b * (1.0 - np.exp(-x / tau))

    def bsum(X, s):
        return X[:, s, :][:, chm[s]].sum(1)
    RRm = np.array([bsum(rho_RR_n, s) for s in range(ns)]).T
    LLm = np.array([bsum(rho_LL_n, s) for s in range(ns)]).T
    VI_band0 = (RRm - LLm).sum(1) / (RRm + LLm).sum(1)
    try:
        p0, _ = curve_fit(settle, t_h, VI_band0 - s_fix * H, p0=[VI_band0.min(), np.ptp(VI_band0), 2.0], maxfev=40000)
        tau_h = float(p0[2])
    except Exception:
        tau_h = 2.0
    therm = 1.0 - np.exp(-t_h / tau_h)

    mQV = np.zeros(ns); mUV = np.zeros(ns); leak_s = np.zeros((K, ns))
    for s in range(ns):
        I_s = bsum(rho_RR_n, s) + bsum(rho_LL_n, s)
        V_s = bsum(rho_RR_n, s) - bsum(rho_LL_n, s)
        Q_s = 2.0 * np.real(rho_RL_cal[:, s, :][:, chm[s]]).sum(1)
        U_s = 2.0 * np.imag(rho_RL_cal[:, s, :][:, chm[s]]).sum(1)
        VI, QI, UI = V_s / I_s, Q_s / I_s, U_s / I_s
        A = np.c_[np.ones(K), therm, QI, UI]
        cf = np.linalg.lstsq(A, VI - s_fix * H, rcond=None)[0]
        mQV[s], mUV[s] = cf[2], cf[3]
        leak_s[:, s] = cf[2] * QI + cf[3] * UI
        print(f'  subband {s + 1}: m_QV {cf[2]:+.4f}  m_UV {cf[3]:+.4f}')

    I_sub = np.array([bsum(rho_RR_n, s) + bsum(rho_LL_n, s) for s in range(ns)]).T
    VI_clean = ((RRm - LLm).sum(1) - (leak_s * I_sub).sum(1)) / I_sub.sum(1)
    p, _ = curve_fit(settle, t_h, VI_clean - s_fix * H, p0=[VI_clean.min(), np.ptp(VI_clean), tau_h], maxfev=40000)
    VI_s = settle(t_h, *p) + s_fix * H
    tau_h, amp = float(p[2]), float(p[1])
    g_raw = np.sqrt((1.0 - VI_s) / (1.0 + VI_s))
    norm = g_raw[norm_mask].mean()
    gamma_snap = g_raw / norm
    resid = float(np.std(VI_clean - VI_s))
    print(f'  gamma(t): {gamma_snap.min():.5f}..{gamma_snap.max():.5f}, settling time {tau_h:.2f} h, amplitude {amp * 100:+.2f} %, '
          f'gain step {s_fix * 100:+.3f} % in V/I, residual {resid * 100:.3f} %')

    # table knots: evenly spaced, plus two knots 0.5 s either side of the gain step so the step stays sharp
    knot_sod = np.sort(np.r_[np.linspace(sod.min(), sod.max(), n_knots), step_t0 - 0.5, step_t0 + 0.5])
    knot_th = (knot_sod - sod.min()) / 3600.0
    knot_VI = settle(knot_th, *p) + s_fix * (knot_sod >= step_t0).astype(float)
    knot_gamma = np.sqrt((1.0 - knot_VI) / (1.0 + knot_VI)) / norm

    dg_fit = dict(tau_h=tau_h, amp=amp, model='exp_settling+step', resid_rms=resid, vi_rms=float(VI_band0.std()),
                  vi_swing=float(np.ptp(VI_band0)), step=s_fix, step_t0=step_t0, norm=float(norm), n_norm=int(norm_mask.sum()))
    vleak = dict(mQV=mQV, mUV=mUV, band_mUV=float(mUV.mean()))
    return gamma_snap, knot_sod, knot_gamma, dg_fit, np.repeat(mQV, nc), np.repeat(mUV, nc), vleak


def apply_diffgain(rho_LL_n, rho_RL_n, gamma_snap):
    """L -> L/gamma(t): LL -> LL/gamma^2 and RL -> RL/gamma. R is the reference hand."""
    g = gamma_snap[:, None, None]
    return rho_LL_n / g ** 2, rho_RL_n / g


# ------------------------------------------------------------------------------------------------ D-terms
def _source_and_floor(rho_RL, e):
    """Per-channel source term s and cross-hand floor kappa from rho_RL,k = s e_k + kappa, e_k = exp(2i chi_k)."""
    K = e.size
    e_mean = e.mean()
    RL_mean = rho_RL.mean(axis=0)
    s = (np.tensordot(np.conj(e), rho_RL, axes=(0, 0)) / K - RL_mean * np.conj(e).mean()) / (1.0 - np.abs(e_mean) ** 2)
    return s, RL_mean - s * e_mean


def fit_dterms(data, rho_RR, rho_LL, rho_RL_final, n_bootstrap=200, seed=42):
    """Feed leakage from the cross-hand floor kappa: D_R = kappa/I_c and D_L = conj(D_R).

    kappa is the part of the nebula cross-hand that does not rotate with the feed, so it is the first-order leakage of I into
    the cross-hand (the Hermitian part Sigma = D_R + conj(D_L)). The anti-Hermitian part Delta, which mixes linear and circular,
    cannot be measured from a calibrator with V = 0 and is set to 0 (as in Heiles 2001). D_R and D_L separately are not
    identifiable from one source (van Straten 2004, App. B), which is why the auto-power ripple is not used for them.
    Each snapshot is divided by its band intensity (relative to the mean) first, so a common gain drift that follows elevation
    does not leak into kappa. Errors come from a bootstrap over snapshots."""
    keep = data['el_deg'] > 0.0
    K = int(keep.sum())
    chi = np.deg2rad(data['chi_BWG'][keep])
    e = np.exp(2j * chi)
    I_norm = (data['I_band'][keep] / data['I_band'][keep].mean())[:, None, None]
    RR = rho_RR[keep] / I_norm
    LL = rho_LL[keep] / I_norm
    RL = rho_RL_final[keep] / I_norm
    s, kappa = _source_and_floor(RL, e)
    eps = kappa / (RR.mean(axis=0) + LL.mean(axis=0))
    D_R, D_L = eps, np.conj(eps)
    print(f'  D-terms: median |D| = {float(np.nanmedian(np.abs(eps))):.5f}')

    rng = np.random.default_rng(seed)
    n_sub, n_ch = D_R.shape
    boot = np.zeros((n_bootstrap, n_sub, n_ch), dtype=complex)
    t0 = time.time()
    for b in range(n_bootstrap):
        idx = rng.integers(0, K, K)
        _, kappa_b = _source_and_floor(RL[idx], e[idx])
        boot[b] = kappa_b / (RR[idx].mean(axis=0) + LL[idx].mean(axis=0))
    print(f'  D-term bootstrap: {n_bootstrap} draws in {time.time() - t0:.1f} s')
    D_R_err = np.zeros((n_sub, n_ch, 2)); D_L_err = np.zeros((n_sub, n_ch, 2))
    for si in range(n_sub):
        for c in range(n_ch):
            v = np.isfinite(boot[:, si, c])
            if v.sum() < 2:
                D_R_err[si, c] = np.nan; D_L_err[si, c] = np.nan
                continue
            D_R_err[si, c] = boot[v, si, c].real.std(), boot[v, si, c].imag.std()
            D_L_err[si, c] = np.conj(boot[v, si, c]).real.std(), np.conj(boot[v, si, c]).imag.std()
    return dict(s=s, kappa=kappa, D_R=D_R, D_L=D_L, D_R_err=D_R_err, D_L_err=D_L_err, keep_mask=keep, K=K,
                flag=flag_outlier_channels(D_R, D_L))


def flag_outlier_channels(D_R, D_L, win=5, n_mad=5.0, dmax=0.5):
    """Channels whose |D| exceeds 0.5 or stands more than 5 robust sigma above its +-5 channel neighbourhood. Their D-terms
    are written as zero, so a channel with an unconstrained fit is never corrected with it."""
    n_sub, n_ch = D_R.shape
    absR, absL = np.abs(D_R), np.abs(D_L)
    flag = np.zeros((n_sub, n_ch), dtype=int)
    flag |= (absR > dmax) | (absL > dmax)
    for absD in (absR, absL):
        for s in range(n_sub):
            for c in range(n_ch):
                lo, hi = max(0, c - win), min(n_ch, c + win + 1)
                nb = np.delete(absD[s, lo:hi], c - lo)
                med = np.median(nb)
                mad = np.median(np.abs(nb - med)) * 1.4826
                if absD[s, c] - med > n_mad * max(mad, 1e-6):
                    flag[s, c] = 1
    return flag


# ------------------------------------------------------------------------------------------------ table files
def write_tables(out_dir, M_chan, freq_MHz, B_R, B_L, dterm, n_snap, knot_sod, knot_gamma, dg_fit, mQV_ch, mUV_ch, vleak):
    """Write the per-channel tables read by calibrate/apply.jl."""
    n_sub, n_ch = B_R.shape
    with open(out_dir / f'mueller_vleak_M{M_chan}.csv', 'w') as f:
        f.write('# Ellipticity (Q,U)->V coupling, fitted per subband on the nebula (Heiles 2001, sec. 6.2), repeated per channel.\n')
        f.write('# apply: V -> V - m_QV*Q - m_UV*U per channel (feed frame, before the ionospheric rotation).\n')
        f.write(f"# per-subband m_UV={np.array2string(vleak['mUV'], precision=4)} band_m_UV={vleak['band_mUV']:+.4f}\n")
        f.write('subband,channel,m_QV,m_UV\n')
        n_chsub = mQV_ch.size // n_sub
        for c in range(mQV_ch.size):
            f.write(f'{c // n_chsub + 1},{c % n_chsub + 1},{mQV_ch[c]:+.6e},{mUV_ch[c]:+.6e}\n')
    with open(out_dir / f'diffgain_time_M{M_chan}.csv', 'w') as f:
        f.write('# Differential gain gamma(t) from the nebula (system V = 0), mean 1 over the bandpass snapshots. apply L -> L/gamma.\n')
        f.write(f"# model={dg_fit['model']} tau_h={dg_fit['tau_h']:.3f} amp={dg_fit['amp']:+.5f} vi_rms={dg_fit['vi_rms']:.5f} "
                f"vi_swing={dg_fit['vi_swing']:.5f} resid_rms={dg_fit['resid_rms']:.5f}\n")
        f.write('sod_utc_s,gamma\n')
        for sd, g in zip(knot_sod, knot_gamma):
            f.write(f'{sd:.3f},{g:.6f}\n')
    with open(out_dir / f'gainbandpass_cal_M{M_chan}.csv', 'w') as f:
        f.write(f'# Per-channel R/L bandpass amplitudes from the nebula (K={n_snap} snapshots).\n')
        f.write('subband,channel,freq_MHz,bandpass_R,bandpass_L\n')
        for s in range(n_sub):
            for c in range(n_ch):
                f.write(f'{s + 1},{c + 1},{freq_MHz[s, c]:.4f},{B_R[s, c]:.6e},{B_L[s, c]:.6e}\n')
    with open(out_dir / f'dterm_cal_python_M{M_chan}.csv', 'w') as f:
        f.write('# Per-channel D-terms D_R = kappa/I, D_L = conj(D_R) from the nebula cross-hand floor, bootstrap errors.\n')
        f.write(f"# K={dterm['K']} snapshots. flag = 1: outlier channel, D written as 0.\n")
        f.write('subband,channel,freq_MHz,dR_re,dR_im,dL_re,dL_im,absdR,absdL,dR_err,dL_err,floor,floor_err,L_over_I_pct,neb_PA_deg,n_snap,flag\n')
        for s in range(n_sub):
            for c in range(n_ch):
                dR, dL = dterm['D_R'][s, c], dterm['D_L'][s, c]
                dR_err = float(np.hypot(*dterm['D_R_err'][s, c])); dL_err = float(np.hypot(*dterm['D_L_err'][s, c]))
                s_c = dterm['s'][s, c]
                flag = int(dterm['flag'][s, c])
                if flag:
                    dR = dL = 0.0 + 0.0j
                f.write(f'{s + 1},{c + 1},{freq_MHz[s, c]:.4f},{dR.real:+.6e},{dR.imag:+.6e},{dL.real:+.6e},{dL.imag:+.6e},'
                        f'{abs(dR):.6e},{abs(dL):.6e},{dR_err:.6e},{dL_err:.6e},{float(np.abs(2 * dterm["kappa"][s, c])):.6e},0.0,'
                        f'{float(np.abs(s_c) * 100.0):.4f},{float(0.5 * np.degrees(np.angle(s_c))):+.3f},{dterm["K"]},{flag}\n')
