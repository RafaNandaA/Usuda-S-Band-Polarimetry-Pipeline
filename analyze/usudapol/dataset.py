"""Per-window quantities for one configuration (the primary one or a definitional variant): phase windows, integrated
Stokes and their variances, feed rotation and the pulse V zero point."""
import numpy as np
from . import errors as E
from .io import chi_model_deg

PRIMARY = dict(aperture='AU', cross=True, noise_scale=1.0, chi=dict(alpha=-1.0, gamma_az=0.0, use_rm=True, k=-1.0),
               zp='median_neb', kappa='mid')


def phase_masks(phase, valid, mpip, lf_edges=(0.38, 0.42)):
    phw = np.where(phase > 0.5, phase - 1.0, phase)
    v = valid > 0
    return phw, v & (np.abs(phw) <= 0.02), v & (phw >= lf_edges[0]) & (phw <= lf_edges[1]), v & (mpip == 1)


def build(P, cfg=None, self_noise=None, log=print):
    """self_noise: optional dict of per-window self-noise sums A_X, B_X and W (from analyze/selfnoise.jl) for the chosen
    aperture. None -> zeros and D['self_ok'] = False."""
    c = dict(PRIMARY)
    c.update(cfg or {})
    ap = c['aperture']
    D = {'cfg': c}
    if ap == 'AU':
        I, Q, U, V, W = (np.asarray(P.get(r), float) for r in ('I_AU', 'Q_AU', 'U_AU', 'V_AU', 'W_AU'))
    else:
        I, Q, U, V, W = (np.asarray(P.get(r), float) for r in ('I_3s', 'Q_3s', 'U_3s', 'V_3s', 'W_3s'))
    mu = np.asarray(P.get('mu_I'), float)
    nb = P.n_b()
    sI, sQ, sU, sV = (np.asarray(P.get(r), float) for r in ('sigma_I', 'sigma_Q', 'sigma_U', 'sigma_V'))
    cr = c['cross']
    sc = c['noise_scale']
    s2Q = E.sigma2_full_band(sQ, W, nb, I, mu, cr, sc)
    s2U = E.sigma2_full_band(sU, W, nb, I, mu, cr, sc)
    s2V = E.sigma2_full_band(sV, W, nb, I, mu, cr, sc)
    s2I = E.sigma2_full_band(sI, W, nb, I, mu, cr, 1.0)
    # per-channel cross term (primary) when the calibrated per-channel moments exist
    D['cross_form'] = 'band'
    if cr and all(P.has(r) for r in ('fr_mu_I', 'fr_sig2_Q', 'fr_sig2_U', 'fr_sig2_V', 'fr_I')):
        # per-channel cross term in band units: Var = sum_c a_c² 2 σ²_c I_c/μ_c (band = sum_c a_c X_c). The pulse adds to the
        # nebula in every cell, so the noise on the pulse grows with I/mu channel by channel.
        a2 = P.band_weights() ** 2
        muc = np.asarray(P.get('fr_mu_I'), float)
        Ic = np.asarray(P.get('fr_I'), float) if ap == 'AU' else None
        if Ic is not None:
            off = lambda s: W * s ** 2 * (1 + W / nb)
            ratio = np.divide(Ic, muc, out=np.zeros_like(Ic), where=(a2[None, :] > 0) & (muc != 0))
            crs = lambda r: np.sum(a2[None, :] * 2.0 * np.nan_to_num(np.asarray(P.get(r), float)) * ratio, axis=1)
            s2Q = sc * (off(sQ) + crs('fr_sig2_Q'))
            s2U = sc * (off(sU) + crs('fr_sig2_U'))
            s2V = sc * (off(sV) + crs('fr_sig2_V'))
            D['cross_form'] = 'per-channel'
    zero = np.zeros_like(I)
    D['self_ok'] = self_noise is not None
    D['s2self'] = {k: zero.copy() for k in ('I', 'Q', 'U', 'V')}
    if self_noise is not None:
        # sigma²_self,X = A_X + c(κ) B_X, c = ½κ/(2 - κ/2). The polarization of one cell is not identifiable, so κ = ½κ_max
        # with κ_max = 1 - d̂²_int.
        tag = 'un' if ap == 'AU' else '3s'
        d2 = np.clip(((Q ** 2 + U ** 2 - s2Q - s2U) + (V ** 2 - s2V)) / I ** 2, 0, 1)
        kmax = 1 - d2
        kap = {'mid': 0.5 * kmax, 'zero': 0 * kmax, 'max': kmax}[c.get('kappa', 'mid')]
        cf = 0.5 * kap / (2 - kap / 2)
        for k in ('I', 'Q', 'U', 'V'):
            D['s2self'][k] = sc * (self_noise['A%s_%s' % (k, tag)] + cf * self_noise['B%s_%s' % (k, tag)])
        D['self_W_check'] = float(np.max(np.abs(self_noise['W_%s' % tag] - W)))
    phase = np.asarray(P.get('phase'), float)
    valid = np.asarray(P.get('valid'))
    phw, MP, LF, LFf = phase_masks(phase, valid, np.asarray(P.get('MP_IP_flag')), c.get('lf_edges', (0.38, 0.42)))
    el, az, qd = (np.asarray(P.get(r), float) for r in ('el_deg', 'az_deg', 'q_deg'))
    ch = c['chi']
    # the ionospheric rotation is already inside the catalogue Q,U (rotated per window), so chi carries only the geometry
    chi_deg = chi_model_deg(qd, el, az, ch['alpha'], ch['gamma_az'], None, ch.get('k', -1.0))
    D['chi_label'] = 'q %+.2f El %+.4f Az' % (ch['alpha'], ch['gamma_az'])
    D['rm_applied'] = False
    if P.has('RM_ion'):
        D['rm_applied'] = 'in the products (catalogue Q,U rotated per window by the GIM correction, not re-applied in chi)'
        D['chi_label'] += ' (RM_ion already applied to catalogue Q,U)'
    chi = np.radians(chi_deg)
    zs, psi, psi0, Qp, Up = E.sky_frame(Q, U, chi, MP)
    l2, v2 = E.moments(I, Q, U, V, s2Q, s2U, s2V)
    # pulse V zero point: p = median calibrated nebular V/I over windows, because system V = 0 is defined on the nebula
    Ineb, Vneb = np.asarray(P.get('I_neb'), float), np.asarray(P.get('V_neb'), float)
    p = float(np.median((Vneb / Ineb)[valid > 0]))
    Vzp = V.copy() if c['zp'] == 'raw' else V - p * I
    Ipk = np.asarray(P.get('I_peak'), float)
    D.update(dict(I=I, Q=Q, U=U, V=V, W=W, mu=mu, nb=nb, sI=sI, sQ=sQ, sU=sU, sV=sV, s2Q=s2Q, s2U=s2U, s2V=s2V, s2I=s2I,
                  phw=phw, MP=MP, LFIP=LF, LFIPflag=LFf, valid=valid > 0, el=el, az=az, qdeg=qd, chi=chi, chi_deg=chi_deg,
                  zs=zs, psi=psi, psi0=psi0, Qp=Qp, Up=Up, l2=l2, v2=v2, q=Q / I, u=U / I, v=V / I,
                  L=E.L_debiased(Q, U, s2Q, s2U), PA=0.5 * np.angle(Q + 1j * U), p_zp=p, Vzp=Vzp, Ipk=Ipk,
                  rho_pk=Ipk / mu, sod=np.asarray(P.get('sod'), float), nseq=np.asarray(P.get('nseq')).astype(np.int64)))
    D['W_eq'] = np.asarray(P.get('W_eq_3s'), float) if (ap == '3s' and P.has('W_eq_3s')) else I / Ipk
    D['snr_int'] = I / np.sqrt(s2I)
    return D


def preell_band(P):
    """Band sums of the per-channel Stokes before the ellipticity and ionosphere corrections, in band units (sum_c a_c X_c)."""
    a = P.band_weights()
    return tuple((np.asarray(P.get(r), float) * a[None, :]).sum(axis=1) for r in ('fr_I', 'fr_Q', 'fr_U', 'fr_V'))


def chi_fr(P, D):
    """chi for the per-channel quantities, which are not corrected for the ionosphere: catalogue PA = per-channel PA -
    ion_pa_shift, so the sky PA of a per-channel quantity is PA_FR - (chi + shift)."""
    if P.has('ion_pa_shift_deg'):
        return D['chi'] + np.radians(np.asarray(P.get('ion_pa_shift_deg'), float))
    return D['chi']
