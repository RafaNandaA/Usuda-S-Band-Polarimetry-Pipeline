"""Figures from the calibrated products, the analysis numbers and the fold. One panel per file.

usage: python3 figures/make_figures.py CONFIG.toml [--cat CAT.h5] [--fr FR.h5] [--numbers NUMBERS.json] [--selfnoise SN.tsv]
                                          [--fold FOLD.h5] [--fold-numbers fold_numbers.json] [--out DIR] [--only 1 2 ... fold]

  1  nebular PA in the feed frame against time, with the feed model chi
  2  nebular PA residual from the feed model
  3  nebular V/I against time
  4  L/I against V/I of the selected MP and LFIP pulses
  5  sky-frame PA histogram of the MP pulses, with the preferred direction psi0
  6  feed-frame PA histogram of the same pulses
  7  rate test: feed-frame PA of chi-binned means against chi, with r = 1 (sky-fixed), the fit and r = 0 (feed-fixed)
  8  <l^2> against duration in tertiles of peak brightness, bright half of the MP pulses
  9  cumulative distribution of the integrated intensity
  fold  mean main-pulse profile I, L, V against rotational phase
Writes fig_<name>.pdf and .png to figures_dir.
"""
import argparse
import csv
import json
import os
import sys
import tomllib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analyze'))
from usudapol import io as cio, dataset, duration as DU, errors as ER   # noqa: E402


def load_config(path):
    cfg = tomllib.load(open(path, 'rb'))
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg.get('paths', {}).items():
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg['paths'][k] = os.path.normpath(os.path.join(base, v))
    return cfg


ap = argparse.ArgumentParser()
ap.add_argument('config')
for k in ('--cat', '--fr', '--numbers', '--selfnoise', '--fold', '--fold-numbers', '--out'):
    ap.add_argument(k)
ap.add_argument('--only', nargs='*')
a = ap.parse_args()
cfg = load_config(a.config)
pa = cfg['paths']
a.cat = a.cat or os.path.join(pa['products_dir'], 'CrabGRP_catalogue.h5')
a.fr = a.fr or os.path.join(pa['products_dir'], 'CrabGRP_FreqResolved.h5')
a.numbers = a.numbers or os.path.join(pa['analysis_dir'], 'NUMBERS.json')
a.selfnoise = a.selfnoise or os.path.join(pa['analysis_dir'], 'selfnoise_sums.tsv')
a.fold = a.fold or os.path.join(pa['products_dir'], 'CrabGRP_fold.h5')
a.fold_numbers = a.fold_numbers or os.path.join(pa['analysis_dir'], 'fold', 'fold_numbers.json')
OUT = a.out or pa['figures_dir']
os.makedirs(OUT, exist_ok=True)
which = set(a.only or list('123456789') + ['fold'])

plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.top': True, 'ytick.right': True,
                     'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
MM = 1 / 25.4; C_MP = '#2a78d6'; C_LF = '#eb6834'; INK = '#0b0b0b'; INK2 = '#52514e'; MUTE = '#898781'
RAMP = ['#86b6ef', '#2a78d6', '#104281']


def newfig(w=84, h=62):
    return plt.subplots(figsize=(w * MM, h * MM))


def save(fig, name, **kw):
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.95, **kw)
    for e in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, 'fig_%s.%s' % (name, e)), dpi=150)
    plt.close(fig); print('saved', name)


wrap90 = lambda x: (np.asarray(x) + 90.0) % 180.0 - 90.0
N = json.load(open(a.numbers))['numbers']

if which & set('123456789'):
    fd = cfg.get('feed', {})
    dataset.PRIMARY['chi'] = dict(alpha=float(fd.get('alpha_el', -1.0)), gamma_az=float(fd.get('gamma_az', 0.0)), use_rm=True, k=-1.0)
    dataset.PRIMARY['lf_edges'] = tuple(N['selection.lfip_edges']['value']['edges_used'])
    P = cio.Products(a.cat, a.fr, print)
    cut = N['selection.adopted_cut']['value']
    rows = list(csv.DictReader(open(a.selfnoise), delimiter='\t'))
    sn = {k: np.full(np.asarray(P.get('phase')).size, np.nan) for k in rows[0] if k != 'cat'}
    for r in rows:
        for k in sn:
            sn[k][int(r['cat'])] = float(r[k])
    D = dataset.build(P, {}, sn, lambda *x: None)
    # the same sample as the analysis: valid, in a phase window, Stokes-I S/N above the adopted cut
    isn = np.asarray(P.get('I_SN_cal'), float)
    S0 = D['valid'] & (D['MP'] | D['LFIP']) & (isn >= cut)
    MPm = S0 & D['MP']; LFm = S0 & D['LFIP']
    print('S0 %d MP %d LFIP %d (analysis %d/%d/%d)' % (S0.sum(), MPm.sum(), LFm.sum(), N['n']['value']['S0'],
                                                     N['n']['value']['S0_MP'], N['n']['value']['S0_LFIP']))
    _, psi, psi0, Qp, Up = ER.sky_frame(D['Q'], D['U'], D['chi'], MPm)
    D.update(psi=psi, psi0=psi0, Qp=Qp, Up=Up)
    neb = P.cat['nebula_diag']; val = D['valid']
    t = (D['sod'] - D['sod'][val].min()) / 3600.0
    PAn = np.asarray(neb['PA_neb_deg'][()], float); chi_deg = D['chi_deg']

if '1' in which or '2' in which:
    res = wrap90(PAn - chi_deg); off = np.median(res[val]); r = res - off; rms = r[val].std()
    print('nebula residual rms %.3f deg, n %d' % (rms, val.sum()))
if '1' in which:
    fig, ax = newfig()
    pd = chi_deg + off + wrap90(PAn - chi_deg - off)
    ax.plot(t[val], pd[val], '.', ms=1.6, mew=0, color=INK2, alpha=0.6, rasterized=True, label='nebula PA')
    ax.plot(t[val][np.argsort(t[val])], (chi_deg + off)[val][np.argsort(t[val])], '-', color=C_MP, lw=1.2, label=r'model $\chi = q - {\rm El}$')
    ax.set_xlabel('time since track start [h]'); ax.set_ylabel('nebular PA, feed frame [deg]'); ax.legend(frameon=False, loc='best', markerscale=4)
    save(fig, '1_nebula_PA_vs_time')
if '2' in which:
    fig, ax = newfig()
    ax.plot(t[val], r[val], '.', ms=1.6, mew=0, color=C_MP, alpha=0.6, rasterized=True); ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlabel('time since track start [h]'); ax.set_ylabel(r'PA $-$ model [deg], rms %.2f$^\circ$' % rms); ax.set_ylim(-8, 8)
    save(fig, '2_nebula_PA_residual_vs_time')
if '3' in which:
    vi = np.asarray(neb['V_neb'][()], float) / np.asarray(neb['I_neb'][()], float)
    fig, ax = newfig()
    ax.plot(t[val], 100 * vi[val], '.', ms=1.6, mew=0, color=C_MP, alpha=0.6, rasterized=True); ax.axhline(0, color=INK2, lw=0.5)
    ax.set_xlabel('time since track start [h]'); ax.set_ylabel(r'nebular $V/I$ [%%], median %.3f' % (100 * np.median(vi[val])))
    save(fig, '3_nebula_VI_vs_time')
if '4' in which:
    LI = D['L'] / D['I']; VI = D['V'] / D['I']
    fig, ax = newfig(84, 76)
    ax.scatter(VI[MPm], LI[MPm], s=2.0, c=C_MP, alpha=0.35, linewidths=0, rasterized=True, label='MP ($n$ = %d)' % MPm.sum())
    ax.scatter(VI[LFm], LI[LFm], s=9, marker='^', facecolors='none', edgecolors=C_LF, linewidths=0.6, label='LFIP ($n$ = %d)' % LFm.sum())
    ax.axvline(0, color=INK2, lw=0.5, ls=':'); ax.set_xlim(-1, 1); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r'$V/I$'); ax.set_ylabel(r'$L/I$ (debiased)')
    leg = ax.legend(loc='upper left', frameon=False, markerscale=2.0)
    for h in leg.legend_handles:
        h.set_alpha(1)
    save(fig, '4_census_LI_VI')
if '5' in which:
    fig, ax = newfig()
    ax.hist(np.degrees(psi[MPm]) % 180, bins=np.arange(0, 181, 6), color=C_MP, edgecolor='white', lw=0.3)
    ax.set_xlabel('sky-frame PA [deg]'); ax.set_ylabel('MP pulses per 6$^\\circ$'); ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    ax.axvline(np.degrees(psi0) % 180, color=INK, ls='--', lw=0.8); save(fig, '5_PA_sky_hist')
if '6' in which:
    fig, ax = newfig()
    ax.hist(np.degrees(D['PA'][MPm]) % 180, bins=np.arange(0, 181, 6), color=MUTE, edgecolor='white', lw=0.3)
    ax.set_xlabel('feed-frame PA [deg]'); ax.set_ylabel('MP pulses per 6$^\\circ$'); ax.set_xlim(0, 180); ax.set_xticks([0, 45, 90, 135, 180])
    save(fig, '6_PA_feed_hist')
if '7' in which:
    dr = N['direction.MP']['value']; r_fit, r_ci = dr['r'], dr['r_CI']
    L = D['L'][MPm]; PA = D['PA'][MPm]; chi = D['chi'][MPm]; z = L * np.exp(2j * PA)
    A_at = lambda r: np.vdot(np.exp(2j * r * chi), z) / np.vdot(np.exp(2j * r * chi), np.exp(2j * r * chi))
    cd = np.degrees(chi); o = np.argsort(cd); edges = np.array_split(o, 12); rng = np.random.default_rng(1)
    xs, ys, lo, hi, xl, xh = [], [], [], [], [], []
    for idx in edges:
        zz = z[idx]; ang = np.degrees(np.angle(zz.mean())) / 2
        bs = np.array([np.degrees(np.angle(zz[rng.integers(0, len(zz), len(zz))].mean())) / 2 for _ in range(300)])
        d = wrap90(bs - ang); xs.append(cd[idx].mean()); ys.append(ang); lo.append(np.percentile(d, 16)); hi.append(np.percentile(d, 84))
        xl.append(cd[idx].min()); xh.append(cd[idx].max())
    xs, ys, lo, hi, xl, xh = map(np.array, (xs, ys, lo, hi, xl, xh))
    ph_fit = np.degrees(np.angle(A_at(r_fit))) / 2
    ys = ph_fit + r_fit * xs + wrap90(ys - (ph_fit + r_fit * xs))
    fig, ax = newfig(); xx = np.linspace(cd.min() - 4, cd.max() + 4, 300)
    for r, ls, c, lab in [(1.0, '--', INK, r'$r$ = 1'), (r_fit, '-', INK, r'fit $r$ = %.3f [%.3f, %.3f]' % (r_fit, *r_ci)), (0.0, ':', MUTE, r'$r$ = 0')]:
        ax.plot(xx, np.degrees(np.angle(A_at(r))) / 2 + r * xx, color=c, ls=ls, lw=1.0, label=lab)
    ax.errorbar(xs, ys, yerr=[-lo, hi], xerr=[xs - xl, xh - xs], fmt='o', ms=3.6, color=C_MP, mec='white', mew=0.4, elinewidth=0.8, zorder=5, label='MP, 12 bins')
    ax.set_xlabel(r'feed angle $\chi$ [deg]'); ax.set_ylabel('feed-frame PA of bin mean [deg]'); ax.legend(frameon=False, loc='upper left', fontsize=6.8)
    ax.set_ylim(ys.min() - 60, ys.max() + 90); save(fig, '7_rate_test')
if '8' in which:
    I = D['I']; H = np.where(MPm & (I > np.median(I[MPm])))[0]
    l2 = (D['Q']**2 + D['U']**2 - (D['s2Q'] + D['s2self']['Q']) - (D['s2U'] + D['s2self']['U'])) / I**2
    W = D['W'][H]; pk = D['rho_pk'][H]; y = l2[H]
    X = DU.design(D, H, 'A'); b = DU.glm_log(X, y); print('GLM model A, bright half: beta_W %.3f' % b[1])
    pe = np.quantile(pk, [0, 1/3, 2/3, 1]); fig, ax = newfig()
    for i in range(3):
        r = (pk >= pe[i]) & (pk <= pe[i+1]); we = np.quantile(W[r], [0, 1/3, 2/3, 1]); xs, ym, ye = [], [], []
        for j in range(3):
            c = r & (W >= we[j]) & ((W <= we[j+1]) if j == 2 else (W < we[j+1]))
            xs.append(np.median(W[c])); ym.append(y[c].mean()); ye.append(y[c].std(ddof=1) / np.sqrt(c.sum()))
        ax.errorbar(xs, ym, ye, fmt=['o', 's', '^'][i], color=RAMP[i], ms=3.5, lw=0.8, ls=['-', '--', ':'][i], label=r'$\rho_{pk}$ tertile %d (median %.1f)' % (i+1, np.median(pk[r])))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'duration $W$ [$\mu$s]'); ax.set_ylabel(r'$\langle l^2\rangle$ (self-noise removed)')
    ax.legend(frameon=False, loc='upper right', fontsize=6.5); save(fig, '8_l2_vs_duration')
if '9' in which:
    E = np.asarray(P.get('I_AU'), float)[D['valid']]; E = np.sort(E[np.isfinite(E) & (E > 0)]); cc = 1 - np.arange(len(E)) / len(E)
    fig, ax = newfig(); ax.loglog(E, cc, drawstyle='steps-post', color=C_MP, lw=1.1, label='valid pulses ($n$ = %d)' % len(E))
    ax.set_xlabel(r'$I_{\rm int}$, union aperture (calibrated units)'); ax.set_ylabel(r"$P(E' \geq E)$"); ax.legend(frameon=False, loc='lower left')
    save(fig, '9_energy_CCDF')

if 'fold' in which:
    # all-rotation sky-frame MP profile in 80-us display bins (5 x 16 us). The per-bin sigma for the L debias and the grey I band
    # is the scatter of the rebinned bins outside the on-pulse box (display only, the fold numbers use the 60-s-block bootstrap).
    d = json.load(open(a.fold_numbers))
    p = d['profile_bins']
    c0 = d['box']['centre_us']; hw = d['box']['half_width_us']; deg_us = 360.0 / d['inputs']['period_us']
    with h5py.File(a.fold, 'r') as h:
        Pf = h['all/MP/profile'][:]                                     # (125, 6): I, Qsky, Usky, V, Qfeed, Ufeed
    R = Pf.reshape(25, 5, 6).sum(1)
    tt = np.array(p['t_us']).reshape(25, 5).mean(1)
    x = (tt - c0) * deg_us
    offm = np.abs(tt - c0) > hw + 40
    sig = R[offm].std(0, ddof=1)
    I, Q, U = R[:, 0], R[:, 1], R[:, 2]
    V = R[:, 3] - N['sign.zero_point_range']['value']['p'] * I
    s2 = 0.5 * (sig[1] ** 2 + sig[2] ** 2)
    L2 = Q ** 2 + U ** 2 - s2
    L = np.where(L2 > s2, np.sqrt(np.clip(L2, 0, None)), 0.0)
    sI = np.full_like(I, sig[0])
    n = I.max()
    with plt.rc_context():
        plt.rcdefaults()             # this panel uses the matplotlib defaults with a larger font
        plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.axvspan(-hw * deg_us, hw * deg_us, color='0.92', lw=0, zorder=0)
        ax.axhline(0, color='0.6', lw=0.5, zorder=1)
        ax.fill_between(x, (I - sI) / n, (I + sI) / n, step='mid', color='0.75', lw=0, zorder=2)
        ax.step(x, I / n, where='mid', color='#1a1a19', lw=1.2, label='$I$', zorder=4)
        ax.step(x, L / n, where='mid', color='#eb6834', lw=1.0, label='$L$ (debiased)', zorder=5)
        ax.step(x, V / n, where='mid', color='#2a78d6', lw=1.0, label='$V$', zorder=3)
        ax.set_xlabel('Rotational phase from the MP centre (deg)')
        ax.set_ylabel('Flux density (normalised to peak $I$)')
        ax.set_xlim(x.min(), x.max())
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(OUT, 'fig_fold.' + ext), dpi=300)
        plt.close(fig)
    print('saved fold')
