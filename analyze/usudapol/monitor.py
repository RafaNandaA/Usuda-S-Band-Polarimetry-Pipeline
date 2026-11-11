"""Q-guard input: fractional 1-ms raw-power excess per window from the stored monitors (items/cat_N.h5 mon/P, mon/n). The
catalogue column mon_power_excess_max is in flag-threshold units, so the excess is recomputed here:
excess = max over full 1-ms bins (n == max n) and the 8 streams of P/median_bins(P) - 1.
The pulse-sweep bins are not stored per window, so the maximum over all bins is used. It is >= the excess during the sweep, so a
window can only be excluded more often (by interference outside the sweep). A bright pulse raises the 4-bit load, and windows
above 10 % (half of the largest excess seen in bright pulses) are left out of the Q-guard fits."""
import os
import h5py
import numpy as np


def mon_excess(items_dir, n):
    out = np.full(n, np.nan)
    for j in range(n):
        fn = os.path.join(items_dir, 'cat_%d.h5' % j)
        if not os.path.exists(fn):
            continue
        with h5py.File(fn, 'r') as f:
            P = f['mon/P'][()]
            nn = f['mon/n'][()]
        full = nn == nn.max()
        Pf = P[full] if P.shape[0] == nn.size else P[:, full].T
        med = np.median(Pf, axis=0)
        out[j] = float(np.max(Pf / med[None, :] - 1))
    return out
