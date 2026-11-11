"""Snapshot selection and the calibration sequence on the nebula snapshots."""
import contextlib
import io

import h5py
import numpy as np

from . import nebula_fit as nf

C_M = 2.99792458e8
SIGMA_ION = -1.0          # pipeline PA = -psi_IAU + const, so the ionospheric rotation enters the cross-hand with this sign
SPIKY_INTERVAL = (40579.0, 40807.0)   # s of day with interference in R subbands 0-1, flagged in every product
SNAPSHOT_S = 0.158        # length of one nebula snapshot in s (1e7 samples at 64 Msps, rounded up)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def flag_mask(path):
    """Interference mask for the nebula snapshots, from the 1-ms monitors stored with them.
    F1: snapshot inside the spiky interval. F2: a 1-ms power bin more than 6 robust sigma (at least 6 x 0.56 %) from the
    snapshot's own median in any stream. F3: extreme 4-bit levels (0 or 15) more than 5 times the track median (at least 0.5 %).
    The same rule flags the per-window products, so the calibrator and the pulses are cleaned the same way."""
    with h5py.File(path, 'r') as f:
        P = f['monitors/P'][...].astype(float)
        f0 = f['monitors/frac_level0'][...].astype(float)
        f15 = f['monitors/frac_level15'][...].astype(float)
        n = f['monitors/n'][...]
        sod = f['geometry/toff_sod'][...]
    full = n == n.max()
    K = P.shape[0]
    Em = np.array([np.median((f0[:, :, st] + f15[:, :, st])[full]) for st in range(8)])
    F2 = np.zeros(K, bool); F3 = np.zeros(K, bool)
    for st in range(8):
        Pw = np.where(full, P[:, :, st], np.nan)
        rel = Pw / np.nanmedian(Pw, axis=1)[:, None] - 1.0
        sg = 1.4826 * np.nanmedian(np.abs(rel - np.nanmedian(rel)))
        F2 |= (np.nan_to_num(np.abs(rel) / (6 * max(sg, 0.0056))) > 1).any(1)
        F3 |= (((f0[:, :, st] + f15[:, :, st]) > max(5 * Em[st], 0.005)) & full).any(1)
    F1 = (sod + SNAPSHOT_S > SPIKY_INTERVAL[0]) & (sod < SPIKY_INTERVAL[1])
    return dict(F1=F1, F2=F2, F3=F3, any=F1 | F2 | F3, sod=sod)


def load_data(nebula_file, ion_track):
    """Masked snapshots with the ionospheric Faraday rotation removed from the cross-hand.
    ion_track: (sod, rm_ion) arrays. The rotation is removed before any fit, so tau_s and phi_s describe the instrument only.
    Every elevation is kept, because the fits on El > 65 deg alone differ from the full track beyond their bootstrap errors and
    the pulses cover the full range."""
    d = nf.load_snapshots(nebula_file)
    fm = flag_mask(nebula_file)
    keep = ~fm['any']
    lam2 = (C_M / (d['freq_MHz'] * 1e6)) ** 2
    d['rm_ion'] = np.interp(d['toff_sod'], ion_track[0], ion_track[1])
    d['rho_RL'] = d['rho_RL'] * np.exp(-2j * SIGMA_ION * d['rm_ion'][:, None, None] * lam2[None])
    for key in ('rho_RR', 'rho_LL', 'rho_RL', 'toff_sod', 'q_deg', 'el_deg', 'az_deg', 'chi_BWG', 'I_band', 'rm_ion'):
        d[key] = d[key][keep]
    d['K'] = int(keep.sum()); d['keep'] = keep
    return d, fm


def run_chain(d, feed, step_t0, step_s, n_boot=None):
    """bandpass -> delay and phase -> gamma(t) with ellipticity -> D-terms, on one snapshot set."""
    cm = nf.channel_mask(d['n_sub'], d['n_ch'])
    B_R, B_L = nf.fit_bandpass(d)
    RR, LL, RL = nf.apply_bandpass(d, B_R, B_L)
    fit = nf.fit_delay_phase(RL, d['freq_MHz'], d['q_deg'], d['el_deg'], d['az_deg'], feed=feed)
    dd = dict(d); dd['chi_BWG'] = np.degrees(fit['chi_rad'])
    RLf = nf.apply_inst_phase(nf.apply_delay(RL, d['freq_MHz'], fit['tau_s']), fit['phi_inst_s'])
    out = dict(d=dd, B_R=B_R, B_L=B_L, fit=fit, RR0=RR, LL0=LL, RLf0=RLf)
    gs, ksod, kg, dgf, mQV, mUV, vl = nf.fit_gain_and_vleak(dd, RR, LL, RLf, cm, step_t0, step_s, np.ones(d['K'], bool))
    LL2, RL2 = nf.apply_diffgain(LL, RL, gs)
    RLf2 = RLf / gs[:, None, None]
    out.update(gamma_snap=gs, knot_sod=ksod, knot_gamma=kg, dgfit=dgf, mQV_ch=mQV, mUV_ch=mUV, vleak=vl, LL=LL2, RLf=RLf2)
    if n_boot is not None:
        out['dterm'] = nf.fit_dterms(dd, RR, LL2, RLf2, n_bootstrap=n_boot)
    return out
