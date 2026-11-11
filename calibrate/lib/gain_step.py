"""Amplitude of the R-chain gain step, measured on 1-s nebula snapshots around it.

The step appears in R only and at the same time in all four subbands, so it is a gain change upstream of the subband split.
The 156-ms snapshots spread over the track do not resolve it, so its amplitude in band V/I is taken from these 1-s snapshots
(mean after minus mean before) with a 5-snapshot block bootstrap for the error.
"""
import h5py
import numpy as np


def measure(path, step_sod, chm, snap_s=1.0, n_boot=4000, block=5, seed=5):
    """Return (dVI, sigma_dVI) of the band V/I step, as fractions."""
    with h5py.File(path, 'r') as f:
        RR = f['coherency/rho_RR'][...]
        LL = f['coherency/rho_LL'][...]
        ft = f['file_time_start_s'][...]
    Rs = np.array([RR[:, s, :][:, chm[s]].sum(1) for s in range(4)]).T.sum(1)
    Ls = np.array([LL[:, s, :][:, chm[s]].sum(1) for s in range(4)]).T.sum(1)
    vi = (Rs - Ls) / (Rs + Ls)
    ip = np.where(ft + snap_s <= step_sod)[0]       # snapshots that end before the step
    iq = np.where(ft >= step_sod)[0]                # snapshots that start after it (the one holding the step is left out)
    rng = np.random.default_rng(seed)

    def resample(idx):
        st = np.arange(0, len(idx), block)
        pick = rng.integers(0, len(st), len(st))
        return np.concatenate([idx[st[p]:st[p] + block] for p in pick])
    d0 = vi[iq].mean() - vi[ip].mean()
    draws = []
    for _ in range(n_boot):
        a, b = resample(ip), resample(iq)
        draws.append(vi[b].mean() - vi[a].mean())
    return float(d0), float(np.std(draws))
