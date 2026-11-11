"""Read access to the calibrated products. Each quantity used by the analysis has a role name, mapped here to its dataset
in the catalogue H5 (CAT) or the per-channel H5 (FR). Nothing here computes a statistic, and files are opened read-only."""
import h5py
import numpy as np


class MissingColumn(KeyError):
    pass


# role: (product, group, dataset, kind, note)
#   kind: core (loading fails if absent) | opt (the items that need it are reported as not computed)
ROLES = {
    # per-window catalogue, union aperture (primary)
    'I_AU':        ('CAT', 'catalog', 'I_int_union', 'core', 'band I_int over the union aperture'),
    'Q_AU':        ('CAT', 'catalog', 'Q_int_union', 'core', ''),
    'U_AU':        ('CAT', 'catalog', 'U_int_union', 'core', ''),
    'V_AU':        ('CAT', 'catalog', 'V_int_union', 'core', ''),
    'W_AU':        ('CAT', 'catalog', 'width_union_us', 'core', 'span in 1-us bins, gaps included'),
    'union_flag':  ('CAT', 'catalog', 'union_flag', 'core', 'number of 3-sigma runs merged'),
    # 3-sigma run aperture (variant)
    'I_3s':        ('CAT', 'catalog', 'I_int', 'core', ''),
    'Q_3s':        ('CAT', 'catalog', 'Q_int', 'core', ''),
    'U_3s':        ('CAT', 'catalog', 'U_int', 'core', ''),
    'V_3s':        ('CAT', 'catalog', 'V_int', 'core', ''),
    'W_3s':        ('CAT', 'catalog', 'width_us', 'core', ''),
    'W_eq_3s':     ('CAT', 'catalog', 'W_eq_us', 'opt', 'I_int(3 sigma)/I_peak'),
    # noise and baseline
    'mu_I':        ('CAT', 'catalog', 'baseline_mu_I', 'core', 'per-sample off-pulse mean band I'),
    'sigma_I':     ('CAT', 'catalog', 'sigma_I', 'core', 'per-sample band sigma_I'),
    'sigma_Q':     ('CAT', 'catalog', 'sigma_Q', 'core', 'from the calibrated baseline covariance'),
    'sigma_U':     ('CAT', 'catalog', 'sigma_U', 'core', ''),
    'sigma_V':     ('CAT', 'catalog', 'sigma_V', 'core', ''),
    'sigma_QU':    ('CAT', 'catalog', 'sigma_QU', 'opt', '0.5 (sigma_Q + sigma_U)'),
    'n_b':         ('CAT', 'catalog', 'nb', 'core', 'number of baseline samples per window'),
    'sigma_R_band': ('CAT', 'catalog', 'sigma_R_band', 'opt', 'per-hand band sigma, uncalibrated'),
    'sigma_L_band': ('CAT', 'catalog', 'sigma_L_band', 'opt', ''),
    # peaks, apertures, indices
    'I_peak':      ('CAT', 'catalog', 'I_peak', 'core', ''),
    'peak_bin':    ('CAT', 'catalog', 'peak_bin', 'opt', ''),
    'ipk_run':     ('CAT', 'catalog', 'ipk_run', 'opt', ''),
    'union_lo':    ('CAT', 'catalog', 'union_lo', 'opt', 'bin in the full read'),
    'union_hi':    ('CAT', 'catalog', 'union_hi', 'opt', ''),
    'ap3_lo':      ('CAT', 'catalog', 'ap3_lo', 'opt', ''),
    'ap3_hi':      ('CAT', 'catalog', 'ap3_hi', 'opt', ''),
    # timing, phase, geometry
    'phase':       ('CAT', 'catalog', 'phase', 'core', ''),
    'valid':       ('CAT', 'catalog', 'valid', 'core', ''),
    'MP_IP_flag':  ('CAT', 'catalog', 'MP_IP_flag', 'core', ''),
    'nseq':        ('CAT', 'catalog', 'nseq', 'core', ''),
    'sod':         ('CAT', 'catalog', 'sod_utc_s', 'core', ''),
    'el_deg':      ('CAT', 'nebula_diag', 'el_deg', 'core', ''),
    'az_deg':      ('CAT', 'nebula_diag', 'az_deg', 'core', ''),
    'q_deg':       ('CAT', 'nebula_diag', 'q_deg', 'core', 'parallactic angle'),
    'chi_BWG_deg': ('CAT', 'nebula_diag', 'chi_BWG_deg', 'opt', 'feed angle written by the extractor'),
    'ion_pa_shift_deg': ('CAT', 'catalog', 'ion_pa_shift_deg', 'opt', 'PA shift applied to Q,U by the ionosphere correction'),
    'RM_ion':      ('CAT', 'catalog', 'rm_ion', 'opt', 'ionospheric RM per window (rad m^-2)'),
    # selection
    'I_SN_cal':    ('CAT', 'catalog', 'I_SN_cal_run10_pk', 'core', '10-us running S/N of the gain-calibrated band I in the peak window'),
    'I_SN_uncal':  ('CAT', 'catalog', 'I_SN_uncal_run10_pk', 'opt', ''),
    'SN_R_run':    ('CAT', 'catalog', 'SN_R_run10_pk', 'core', 'our per-hand running S/N'),
    'SN_L_run':    ('CAT', 'catalog', 'SN_L_run10_pk', 'core', ''),
    'flag_bits':   ('CAT', 'catalog', 'read_flag_bits', 'opt', 'bit field: 1 interference interval, 2 power, 4 levels, 8 incomplete'),
    # nebula
    'I_neb':       ('CAT', 'nebula_diag', 'I_neb', 'core', ''),
    'L_neb':       ('CAT', 'nebula_diag', 'L_neb', 'core', ''),
    'PA_neb_deg':  ('CAT', 'nebula_diag', 'PA_neb_deg', 'core', ''),
    'V_neb':       ('CAT', 'nebula_diag', 'V_neb', 'core', ''),
    'L_neb_preD':  ('CAT', 'nebula_diag', 'L_neb_preD', 'core', ''),
    'PA_neb_preD_deg': ('CAT', 'nebula_diag', 'PA_neb_preD_deg', 'core', ''),
    'V_neb_preD':  ('CAT', 'nebula_diag', 'V_neb_preD', 'opt', ''),
    'VI_neb_preGamma': ('CAT', 'nebula_diag', 'VI_neb_preGamma', 'opt', 'nebular V/I before the gamma(t) correction'),
    # band profiles (+-750 us)
    'prof_I':      ('CAT', 'raw_profiles', 'I', 'core', ''),
    'prof_Q':      ('CAT', 'raw_profiles', 'Q', 'core', ''),
    'prof_U':      ('CAT', 'raw_profiles', 'U', 'core', ''),
    'prof_V':      ('CAT', 'raw_profiles', 'V', 'core', ''),
    'prof_peak_index': ('CAT', 'raw_profiles', 'peak_index', 'core', ''),
    # per channel, union aperture, before the ellipticity correction
    'fr_I':        ('FR', 'freq_resolved', 'I_chan_union', 'core', ''),
    'fr_Q':        ('FR', 'freq_resolved', 'Q_chan_union', 'core', ''),
    'fr_U':        ('FR', 'freq_resolved', 'U_chan_union', 'core', ''),
    'fr_V':        ('FR', 'freq_resolved', 'V_chan_union', 'core', ''),
    'fr_freq':     ('FR', 'freq_resolved', 'freq_MHz', 'core', ''),
    'fr_base':     ('FR', 'freq_resolved', 'Stokes_base_chan_cal', 'opt', 'calibrated per-channel baseline means (Np, 4, 128)'),
    'fr_mu_I':     ('FR', 'freq_resolved', 'muI_chan_calib', 'opt', 'calibrated baseline mean I per channel'),
    'fr_sig2_Q':   ('FR', 'freq_resolved', 'sig2Q_chan_calib', 'opt', 'calibrated baseline variance per channel'),
    'fr_sig2_U':   ('FR', 'freq_resolved', 'sig2U_chan_calib', 'opt', ''),
    'fr_sig2_V':   ('FR', 'freq_resolved', 'sig2V_chan_calib', 'opt', ''),
    'fr_sig2_I':   ('FR', 'freq_resolved', 'sig2I_chan_calib', 'opt', ''),
}

KEPT_CH = np.array([c for c in range(128) if (c % 32) != 31])   # channel 32 of every subband (0-based 31) is the half-filled edge bin


class Products:
    def __init__(self, cat_path, fr_path, log=print):
        self.log = log
        self.cat = h5py.File(cat_path, 'r')
        self.fr = h5py.File(fr_path, 'r') if fr_path else None
        self.paths = {'CAT': cat_path, 'FR': fr_path}
        self._cache = {}

    def has(self, role):
        try:
            self.get(role)
            return True
        except MissingColumn:
            return False

    def get(self, role):
        if role in self._cache:
            return self._cache[role]
        if role not in ROLES:
            raise MissingColumn('unknown role %s' % role)
        prod, grp, name, kind, note = ROLES[role]
        f = self.cat if prod == 'CAT' else self.fr
        if f is None or grp not in f or name not in f[grp]:
            raise MissingColumn('role %s -> %s/%s not found in %s' % (role, grp, name, self.paths[prod]))
        arr = f[grp][name][()]
        if grp == 'freq_resolved' and arr.ndim == 2 and arr.shape[0] == 128 and arr.shape[1] != 128:
            arr = arr.T                     # per-channel moments are written column-major by Julia as (128, Np)
        if arr.dtype.kind in ('O', 'S', 'U'):
            arr = np.array([float(x.decode() if isinstance(x, bytes) else x) for x in arr])
        self._cache[role] = arr
        return arr

    def require_core(self):
        """Load every core role and raise one MissingColumn that lists all missing ones."""
        missing = []
        for r, (prod, grp, name, kind, note) in ROLES.items():
            if kind != 'core':
                continue
            try:
                self.get(r)
            except MissingColumn as e:
                missing.append(str(e))
        if missing:
            raise MissingColumn('core roles missing:\n  ' + '\n  '.join(missing))

    def band_weights(self):
        """a_c with band = sum_c a_c X_c: a_c = (32/n_kept)/32 on kept channels. Channel 32 of every subband and channel 1 of
        subband 2 are left out of band sums because they fail the interior response test, and the weights keep the /32 scale."""
        a = np.zeros(128)
        keep = np.ones(128, bool)
        keep[[31, 63, 95, 127, 32]] = False
        for s in range(4):
            r = slice(32 * s, 32 * s + 32)
            a[r] = keep[r] * (32.0 / keep[r].sum()) / 32.0
        return a

    def n_b(self):
        """Baseline sample count per window."""
        return np.asarray(self.get('n_b'), float)


def read_detlist(fn):
    """Per-hand detection list: returns {nseq: max S/N} (column 4 = Nseq, column 6 = S/N). A pulse listed twice keeps its
    larger S/N, so it pairs with our window-maximum S/N."""
    d = {}
    for ln in open(fn):
        if ln.startswith('#') or not ln.strip():
            continue
        t = ln.split()
        n = int(round(float(t[3])))
        s = float(t[5])
        d[n] = max(s, d.get(n, -1))
    return d


def chi_model_deg(q, el, az, alpha=-1.0, gamma_az=0.0, rm_ion=None, k=-1.0, lam2=None):
    """Feed rotation chi = q + alpha*El + gamma_az*Az + k*RM_ion*lambda_c^2 (deg), k = -1.
    lambda_c^2 = (c/2258 MHz)^2 = 0.01763 m^2, so 1 rad m^-2 is 1.010 deg of PA."""
    chi = np.asarray(q, float) + alpha * np.asarray(el, float) + gamma_az * np.asarray(az, float)
    if rm_ion is not None:
        if lam2 is None:
            lam2 = (299792458.0 / 2258e6) ** 2
        chi = chi + k * np.degrees(np.asarray(rm_ion, float) * lam2)
    return chi
