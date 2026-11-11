"""Mean main-pulse polarization from the calibrated fold, and its relation to the giant pulses of the catalogue.

usage: python3 fold/fold.py CONFIG.toml [--fold FOLD.h5] [--cat CAT.h5] [--fr FR.h5] [--numbers NUMBERS.json] [--out DIR] [--B N]

Inputs (defaults from the config):
  products_dir/CrabGRP_fold.h5      per-block calibrated feed-frame band Stokes per 16-us bin (MP, OFF and LFIP windows), from
                                    calibrate/apply.jl. Sets: 'all' rotations and 'free' (rotations without a listed pulse).
  work_dir/blocks/block_B.h5        rotation lists of each 60-s block (q, El, flag, listed, list gap)
  work_dir/fold_blocks.tsv, timing_model.json   block grid and rotation period, from extract/extract.jl plan
  products_dir catalogue + FR, analysis_dir/NUMBERS.json   selected MP giant pulses, their preferred direction, V zero point
Writes OUT/fold_numbers.json and OUT/fold_log.txt (OUT = analysis_dir/fold).

Errors come from a bootstrap over the 60-s blocks, with one weight matrix shared by every quantity, so differences between
quantities are paired. The block is the unit because giant pulses and slow gain changes make neighbouring rotations correlated.
"""
import argparse
import json
import os
import sys
import tomllib

import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as CHI2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'analyze'))
from usudapol import io as cio, dataset, selection, stats as S   # noqa: E402


def load_config(path):
    cfg = tomllib.load(open(path, 'rb'))
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg.get('paths', {}).items():
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg['paths'][k] = os.path.normpath(os.path.join(base, v))
    return cfg


ap = argparse.ArgumentParser()
ap.add_argument('config')
ap.add_argument('--fold'); ap.add_argument('--cat'); ap.add_argument('--fr'); ap.add_argument('--numbers')
ap.add_argument('--inputs'); ap.add_argument('--work'); ap.add_argument('--out')
ap.add_argument('--B', type=int, default=2000)
a = ap.parse_args()
cfg = load_config(a.config)
pa = cfg['paths']
a.fold = a.fold or os.path.join(pa['products_dir'], 'CrabGRP_fold.h5')
a.cat = a.cat or os.path.join(pa['products_dir'], 'CrabGRP_catalogue.h5')
a.fr = a.fr or os.path.join(pa['products_dir'], 'CrabGRP_FreqResolved.h5')
a.numbers = a.numbers or os.path.join(pa['analysis_dir'], 'NUMBERS.json')
a.inputs = a.inputs or os.path.join(pa['tables_dir'], 'inputs.json')
a.work = a.work or pa['work_dir']
a.out = a.out or os.path.join(pa['analysis_dir'], 'fold')
os.makedirs(a.out, exist_ok=True)
logf = open(os.path.join(a.out, 'fold_log.txt'), 'w')


def log(*x):
    s = ' '.join(str(v) for v in x); print(s); logf.write(s + '\n'); logf.flush()


B = a.B
rng = np.random.default_rng(S.SEED_BASE + S.ITEM['fold'])
NUM = json.load(open(a.numbers))['numbers']
INP = json.load(open(a.inputs)) if os.path.exists(a.inputs) else {}
P_US = json.load(open(os.path.join(a.work, 'timing_model.json')))['period_us']     # rotation period
DEG_PER_US = 360.0 / P_US
log('fold:', os.path.basename(a.fold), '| B =', B)

# ------------------------------------------------------------------ fold per-block data
F = h5py.File(a.fold, 'r')
blocks = F['blocks'][:]
nb = blocks.size
log('blocks', nb)
X = {}
for s in ('all', 'free'):
    for w in ('MP', 'OFF', 'LFIP'):
        X[s, w] = np.nan_to_num(F[s + '/per_block/' + w][:])          # (nb, 125, 4): I, Q, U, V feed frame, nebula not subtracted
nrot = {'all': F['n_all_per_block'][:], 'free': F['n_free_per_block'][:]}

# rotation lists -> block-mean phasor Z_b = <e^{-2i chi}>, e2sum_b = sum e^{+2i chi}, chi from the config feed model
fd = cfg.get('feed', {})
alpha, gaz = float(fd.get('alpha_el', -1.0)), float(fd.get('gamma_az', 0.0))
Z = {s: np.zeros(nb, complex) for s in ('all', 'free')}
E2 = {s: np.zeros(nb, complex) for s in ('all', 'free')}
rotsets = {'all': [], 'listed': []}
for i, b in enumerate(blocks):
    with h5py.File(os.path.join(a.work, 'blocks', 'block_%d.h5' % b), 'r') as h:
        fl, li, gp = h['rot/flag'][:], h['rot/listed'][:], h['rot/in_list_gap'][:]
        chi_deg = h['rot/q_deg'][:] + alpha * h['rot/el_deg'][:]
        if gaz != 0.0:
            chi_deg = chi_deg + gaz * h['rot/az_deg'][:]
        chi = np.radians(chi_deg)
        rot = h['rot/rotation'][:]
    for s, sel in (('all', fl == 0), ('free', (fl == 0) & (li == 0) & (gp == 0))):
        if sel.sum():
            Z[s][i] = np.mean(np.exp(-2j * chi[sel])); E2[s][i] = np.sum(np.exp(2j * chi[sel]))
        assert sel.sum() == nrot[s][i], (b, s)
    rotsets['all'].append(rot[fl == 0]); rotsets['listed'].append(rot[(fl == 0) & (li == 1)])
rot_all = np.concatenate(rotsets['all']); rot_listed = set(np.concatenate(rotsets['listed']).tolist())
N = {s: int(nrot[s].sum()) for s in nrot}
smear = {s: float(np.abs(E2[s].sum()) / N[s]) for s in N}
log('N_rot all %d free %d listed %d | smear |<e^{2i chi}>| all %.4f free %.4f' % (N['all'], N['free'], len(rot_listed),
                                                                              smear['all'], smear['free']))


def prof_blocks(s, w):
    """per-block profiles with the nebula (mean of the 125 OFF bins of the same block) subtracted, rotated to the sky frame
    with the block-mean phasor. 6 rows: I, Q_sky, U_sky, V, Q_feed, U_feed."""
    off = X[s, 'OFF'].mean(axis=1, keepdims=True)
    P = X[s, w] - off
    zf = P[:, :, 1] + 1j * P[:, :, 2]
    zs = zf * Z[s][:, None]
    return np.stack([P[:, :, 0], zs.real, zs.imag, P[:, :, 3], zf.real, zf.imag], axis=2)   # (nb, 125, 6)


PB = {(s, w): prof_blocks(s, w) for s in ('all', 'free') for w in ('MP', 'LFIP')}

# bootstrap weights over 60-s blocks (shared by every quantity: paired / joint)
Wb = rng.multinomial(nb, np.ones(nb) / nb, size=B).astype(float)     # (B, nb)


def boot(arr):
    """arr (nb, ...) per-block -> (B, ...) bootstrap totals."""
    sh = arr.shape
    return (Wb @ arr.reshape(nb, -1)).reshape((B,) + sh[1:])


tbin = (np.arange(125) + 0.5) * 16.0 - 1000.0                       # us from the window centre
phdeg = tbin * DEG_PER_US


# ------------------------------------------------------------------ Gaussian fit to the I profile -> on-pulse box (±1 FWHM)
def gfit(y, sig):
    g = lambda x, A, c, sg, b: A * np.exp(-0.5 * ((x - c) / sg) ** 2) + b
    p0 = [y.max(), tbin[np.argmax(np.convolve(y, np.ones(9) / 9, 'same'))], 100.0, 0.0]
    p, cv = curve_fit(g, tbin, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=20000)
    return p, np.sqrt(np.diag(cv))


res = {'inputs': {'fold_file': os.path.basename(a.fold), 'B': B, 'seed': S.SEED_BASE + S.ITEM['fold'], 'period_us': P_US,
                  'n_blocks': int(nb)}}
res['n_rot'] = dict(N); res['smear_abs_mean_e2ichi'] = smear

bootMP = {s: boot(PB[s, 'MP']) for s in ('all', 'free')}
bootLF = {s: boot(PB[s, 'LFIP']) for s in ('all', 'free')}
sigbin = {s: bootMP[s].std(0, ddof=1) for s in ('all', 'free')}       # per-bin bootstrap sd (125, 6)
fits = {}
for s in ('all', 'free'):
    y = PB[s, 'MP'].sum(0)[:, 0]
    p, e = gfit(y, sigbin[s][:, 0])
    fits[s] = dict(A=p[0], centre_us=p[1], centre_err_us=e[1], fwhm_us=2.3548 * abs(p[2]), fwhm_err_us=2.3548 * e[2], b=p[3])
    log('Gaussian fit %s: centre %.1f ± %.1f us, FWHM %.1f ± %.1f us' % (s, p[1], e[1], 2.3548 * abs(p[2]), 2.3548 * e[2]))
res['gauss'] = fits
c0, fw = fits['all']['centre_us'], fits['all']['fwhm_us']
box = np.abs(tbin - c0) <= fw                                         # ±1 FWHM of the all-rotation fit, used for every set
nbox = int(box.sum())
log('on-pulse box: centre %.1f us ± %.1f us -> %d bins' % (c0, fw, nbox))
res['box'] = {'centre_us': c0, 'half_width_us': fw, 'n_bins': nbox}

pzp = NUM['sign.zero_point_range']['value']['p']
grid = NUM['sign.zero_point_range']['value']['grid']
dstep = float(INP.get('step_amp_err', 0.0002))   # R-chain gain-step amplitude error, applied at both ends


def integ(P):
    """P (..., 125, 6) -> box sums (..., 6)."""
    return P[..., box, :].sum(-2)


def quantities(T, sQ2, sU2):
    """T (..., 6) box sums. One-σ² debias: L = sqrt(max(0, Q²+U² - ½(σQ²+σU²))), 0 below 1σ."""
    I, Q, U, V = T[..., 0], T[..., 1], T[..., 2], T[..., 3]
    s2 = 0.5 * (sQ2 + sU2)
    L2 = Q ** 2 + U ** 2 - s2
    L = np.where(L2 > s2, np.sqrt(np.clip(L2, 0, None)), 0.0)
    return dict(I=I, L_over_I=L / I, V_over_I=V / I - pzp, PA=0.5 * np.degrees(np.arctan2(U, Q)),
                Lf_over_I=np.hypot(T[..., 4], T[..., 5]) / I)


def wrap90(x):
    return (np.asarray(x) + 90.0) % 180.0 - 90.0


def ci(x, lev=95):
    x = np.asarray(x); x = x[np.isfinite(x)]
    return [float(np.percentile(x, (100 - lev) / 2)), float(np.percentile(x, 100 - (100 - lev) / 2))]


prof = {}
for s in ('all', 'free'):
    T = integ(PB[s, 'MP'].sum(0)); Tb = integ(bootMP[s])
    sQ2, sU2 = Tb[:, 1].var(ddof=1), Tb[:, 2].var(ddof=1)
    q0 = quantities(T, sQ2, sU2); qb = quantities(Tb, sQ2, sU2)
    # S/N_I two ways: bootstrap over blocks, and off-pulse bin scatter × sqrt(m) (bins farther than 37 bins from the centre)
    y = PB[s, 'MP'].sum(0)[:, 0]
    offb = np.abs(np.arange(125) - np.argmin(np.abs(tbin - c0))) > 37
    sn_scatter = T[0] / (y[offb].std(ddof=1) * np.sqrt(nbox))
    sn_boot = T[0] / Tb[:, 0].std(ddof=1)
    pa_d = wrap90(qb['PA'] - q0['PA'])
    vgrid = []
    for d in grid:
        for ds in (-dstep, dstep):
            v = qb['V_over_I'] - d - ds * 1.0      # the gain-step error is applied to the whole profile (an upper bound)
            vgrid.append({'delta': d, 'delta_step': ds, 'V_over_I': float(q0['V_over_I'] - d - ds), 'CI95': ci(v)})
    vsign = 'negative' if all(g['CI95'][1] < 0 for g in vgrid) else ('positive' if all(g['CI95'][0] > 0 for g in vgrid) else
                                                                    'sign not determined at every grid point')
    prof[s] = dict(I_box=float(T[0]), SN_I_boot=float(sn_boot), SN_I_scatter=float(sn_scatter),
                   SN_I_scatter_with_baseline=float(sn_scatter / 1.118),
                   L_over_I=float(q0['L_over_I']), L_over_I_sd=float(qb['L_over_I'].std(ddof=1)), L_over_I_CI95=ci(qb['L_over_I']),
                   L_SN=float(q0['L_over_I'] / qb['L_over_I'].std(ddof=1)),
                   V_over_I=float(q0['V_over_I']), V_over_I_sd=float(qb['V_over_I'].std(ddof=1)), V_over_I_CI95=ci(qb['V_over_I']),
                   V_over_I_grid=vgrid, V_sign_decision=vsign,
                   PA_sky_deg=float(q0['PA']), PA_sd_deg=float(pa_d.std(ddof=1)), PA_CI95_deg=[float(q0['PA'] + c) for c in ci(pa_d)],
                   Lfeed_over_I=float(q0['Lf_over_I']))
    log('MP %s: I_box %.4g | S/N_I boot %.2f, scatter %.2f (with baseline %.2f) | L/I %.3f ± %.3f %s | V/I %.3f ± %.3f | PA %.1f ± %.1f deg'
        % (s, T[0], sn_boot, sn_scatter, sn_scatter / 1.118, q0['L_over_I'], prof[s]['L_over_I_sd'], prof[s]['L_over_I_CI95'],
           q0['V_over_I'], prof[s]['V_over_I_sd'], q0['PA'], prof[s]['PA_sd_deg']))
    log('   V/I sign over the zero-point grid (δ × δ_step): %s' % vsign)
    prof[s]['_boot'] = (Tb, sQ2, sU2, qb)
res['MP'] = {s: {k: v for k, v in prof[s].items() if not k.startswith('_')} for s in prof}

# LFIP: same box width, centred on the LFIP window centre (no Gaussian fit at S/N ~ 1-2)
boxL = np.abs(tbin) <= fw
res['LFIP'] = {}
for s in ('all', 'free'):
    T = PB[s, 'LFIP'].sum(0)[boxL].sum(0); Tb = bootLF[s][:, boxL].sum(1)
    sQ2, sU2 = Tb[:, 1].var(ddof=1), Tb[:, 2].var(ddof=1)
    q0 = quantities(T, sQ2, sU2); qb = quantities(Tb, sQ2, sU2)
    y = PB[s, 'LFIP'].sum(0)[:, 0]
    sn_scatter = T[0] / (y[~boxL].std(ddof=1) * np.sqrt(boxL.sum()))
    res['LFIP'][s] = dict(I_box=float(T[0]), SN_I_boot=float(T[0] / Tb[:, 0].std(ddof=1)), SN_I_scatter=float(sn_scatter),
                          L_over_I_CI95=ci(qb['L_over_I']), V_over_I_CI95=ci(qb['V_over_I']),
                          ratio_to_MP_I=float(T[0] / prof[s]['I_box']))
    log('LFIP %s: S/N_I boot %.2f scatter %.2f | LFIP/MP fluence %.3f | L/I CI %s V/I CI %s' % (
        s, res['LFIP'][s]['SN_I_boot'], sn_scatter, T[0] / prof[s]['I_box'], res['LFIP'][s]['L_over_I_CI95'],
        res['LFIP'][s]['V_over_I_CI95']))

# ------------------------------------------------------------------ feed-frame controls
# A sky-fixed polarization is smeared in the feed frame by |<e^{2i chi}>|. The nebula (OFF window) must show exactly this
# smearing, and so must the pulsar if its PA is fixed on the sky.
ctrl = {}
for s in ('all', 'free'):
    T = integ(PB[s, 'MP'].sum(0)); Tb = integ(bootMP[s]); E2b = Wb @ E2[s]; Nb = Wb @ nrot[s]
    D0 = ((T[4] + 1j * T[5]) - E2[s].sum() / N[s] * (T[1] + 1j * T[2])) / T[0]
    Db = ((Tb[:, 4] + 1j * Tb[:, 5]) - E2b / Nb * (Tb[:, 1] + 1j * Tb[:, 2])) / Tb[:, 0]
    sd = float(np.sqrt(0.5 * (Db.real.var(ddof=1) + Db.imag.var(ddof=1))))
    off = X[s, 'OFF'].mean(axis=1)                                    # (nb, 4), no subtraction
    zf = off[:, 1] + 1j * off[:, 2]; zs = zf * Z[s]
    rneb = float(np.abs(zf.sum()) / np.abs(zs.sum()))
    ctrl[s] = dict(pulsar_absD_over_I=float(abs(D0)), pulsar_sigma=sd, informative=bool(sd <= 0.05),
                   pulsar_Rayleigh_p=float(np.exp(-abs(D0) ** 2 / (2 * sd ** 2))),
                   pulsar_pass=bool(abs(D0) <= max(0.02, 2 * sd)), nebula_Lfeed_over_Lsky=rneb, smear=smear[s],
                   nebula_diff=rneb - smear[s], nebula_pass=bool(abs(rneb - smear[s]) <= 0.01),
                   nebula_L_over_I=float(np.abs(zs.sum()) / off[:, 0].sum()))
    log('feed control %s: pulsar |D|/I %.3f sigma %.3f (%s); nebula |Lf/Ls| %.4f vs smear %.4f (diff %+.4f) -> %s' % (
        s, abs(D0), sd, 'informative' if sd <= 0.05 else 'UNINFORMATIVE', rneb, smear[s], rneb - smear[s],
        'PASS' if abs(rneb - smear[s]) <= 0.01 else 'FAIL'))
res['feed_control'] = ctrl

# OFF flatness (placement check): chi2 of the summed OFF I profile against a constant, per-bin bootstrap sd
offI = X['all', 'OFF'][:, :, 0] - X['all', 'OFF'][:, :, 0].mean(1, keepdims=True)
yo = offI.sum(0); so = boot(offI).std(0, ddof=1)
c2 = float(np.sum((yo / so) ** 2))
res['placement'] = {'MP_centre_us_all': c0, 'MP_centre_us_free': fits['free']['centre_us'], 'centre_within_25us': bool(abs(c0) <= 25),
                    'OFF_chi2': c2, 'OFF_dof': 124, 'OFF_p': float(CHI2.sf(c2, 124))}
log('placement: MP centre %.1f us (within ±25: %s); OFF flat chi2 %.1f/124 p %.3f' % (c0, abs(c0) <= 25, c2, CHI2.sf(c2, 124)))

# units check: fold OFF I per 1-us sample per rotation against the catalogue baseline mu_I
offlevel = X['all', 'OFF'][:, :, 0].sum() / (125 * 16 * N['all'])

# ------------------------------------------------------------------ catalogue: the selected MP giant pulses and their psi0
Pp = cio.Products(a.cat, a.fr, lambda *x: None)
dataset.PRIMARY['chi'] = dict(alpha=alpha, gamma_az=gaz, use_rm=True, k=-1.0)
dataset.PRIMARY['lf_edges'] = tuple(NUM['selection.lfip_edges']['value']['edges_used'])
D = dataset.build(Pp, {}, None, lambda *x: None)
base = D['valid'] & (D['MP'] | D['LFIP'])
isn = np.asarray(Pp.get('I_SN_cal'), float)
fb = np.asarray(Pp.get('fr_base'), float)
_, _, gj = selection.calibrated_hand_sigma(fb[:, 0, :], fb[:, 3, :], Pp.band_weights())
isn = isn * np.sqrt(2) / gj
cut = NUM['selection.adopted_cut']['value']
S0 = base & (isn >= cut)
MPi = np.where(S0 & D['MP'])[0]
zs_p = (D['Q'] + 1j * D['U']) * np.exp(-2j * D['chi'])
psi0 = 0.5 * np.angle(np.mean(np.exp(2j * 0.5 * np.angle(zs_p[MPi]))))
log('catalogue: S0_MP n = %d (analysis %d) | psi0 = %.3f deg (analysis %.3f) | fold OFF mu_I per us per rotation %.4g vs '
    'catalogue median baseline_mu_I %.4g (ratio %.4f)' % (MPi.size, NUM['n']['value']['S0_MP'], np.degrees(psi0),
                                                         NUM['direction.MP']['value']['psi0_deg'], offlevel,
                                                         np.median(D['mu'][D['valid']]), offlevel / np.median(D['mu'][D['valid']])))
res['catalogue'] = {'S0_MP_n': int(MPi.size), 'psi0_deg': float(np.degrees(psi0)),
                    'units_ratio_foldOFF_over_cat_muI': float(offlevel / np.median(D['mu'][D['valid']]))}

# assign pulses to fold blocks (60-s grid in list UTC, which differs from sod by < 0.05 s)
grid_tab = np.genfromtxt(os.path.join(a.work, 'fold_blocks.tsv'), names=True, delimiter='\t')
gstart, gend, gblk = grid_tab['start_list_utc'], grid_tab['end_list_utc'], grid_tab['block'].astype(int)
sod = D['sod']
pos = {int(b): i for i, b in enumerate(blocks)}


def block_of(t):
    j = np.searchsorted(gstart, t, side='right') - 1
    if j >= 0 and t < gend[j] and int(gblk[j]) in pos:
        return pos[int(gblk[j])]
    return pos[int(gblk[np.argmin(np.abs(0.5 * (gstart + gend) - t))])]   # outside the grid (gain-step interval): nearest block


pb = np.array([block_of(t) for t in sod[MPi]])
log('pulses outside a block cell (assigned to nearest): %d' % sum(not (np.any((gstart <= t) & (t < gend))) for t in sod[MPi]))
ew = np.exp(2j * 0.5 * np.angle(zs_p[MPi]))
Ewb = np.zeros(nb, complex); np.add.at(Ewb, pb, ew)                       # per-block sum of e^{2i psi}
psi0_b = 0.5 * np.angle(Wb @ Ewb)                                         # joint bootstrap psi0
# GRP aggregate (selected MP pulses, union aperture, summed Stokes, sky frame)
Ib = np.zeros(nb); Zb = np.zeros(nb, complex); S2b = np.zeros(nb)
np.add.at(Ib, pb, D['I'][MPi]); np.add.at(Zb, pb, zs_p[MPi]); np.add.at(S2b, pb, 0.5 * (D['s2Q'][MPi] + D['s2U'][MPi]))
Vb = np.zeros(nb); np.add.at(Vb, pb, D['Vzp'][MPi])


def aggLI(I, Zc, S2):
    L2 = np.abs(Zc) ** 2 - S2
    return np.sqrt(np.clip(L2, 0, None)) / I


agg0 = aggLI(Ib.sum(), Zb.sum(), S2b.sum()); aggb = aggLI(Wb @ Ib, Wb @ Zb, Wb @ S2b)
aggPA0 = 0.5 * np.degrees(np.angle(Zb.sum()))
log('GRP aggregate S0_MP: L/I %.4f CI %s, PA %.1f deg, V/I %.4f' % (agg0, ci(aggb), aggPA0, Vb.sum() / Ib.sum()))

# ------------------------------------------------------------------ comparisons (a)-(e)
cmp = {}
qa, qf = prof['all']['_boot'][3], prof['free']['_boot'][3]
# (a) free vs all, paired by block
for k in ('L_over_I', 'V_over_I'):
    d0 = res['MP']['free'][k] - res['MP']['all'][k]; db = qf[k] - qa[k]
    cmp['a_free_minus_all_' + k] = {'value': float(d0), 'CI95': ci(db)}
dpa0 = float(wrap90(res['MP']['free']['PA_sky_deg'] - res['MP']['all']['PA_sky_deg']))
dpab = wrap90(qf['PA'] - qa['PA'] - dpa0) + dpa0
cmp['a_free_minus_all_PA_deg'] = {'value': dpa0, 'CI95': ci(dpab)}
# (b) remainder (free) L/I minus GRP aggregate L/I, joint bootstrap
d0 = res['MP']['free']['L_over_I'] - agg0; db = qf['L_over_I'] - aggb
dec_b = ('the remainder is more polarized than the GRP aggregate' if ci(db)[0] > 0 else
         'the remainder is less polarized than the GRP aggregate' if ci(db)[1] < 0 else
         'no difference detected, within [%.3f, %.3f]' % tuple(ci(db)))
cmp['b_remainder_minus_GRPaggregate_L_over_I'] = {'remainder': res['MP']['free']['L_over_I'], 'GRP_aggregate': float(agg0),
                                                   'GRP_aggregate_CI95': ci(aggb), 'value': float(d0), 'CI95': ci(db), 'decision': dec_b}
# (c) profile on-pulse PA (all rotations, free beside) minus psi0, joint bootstrap
for s, qq in (('all', qa), ('free', qf)):
    d0 = float(wrap90(res['MP'][s]['PA_sky_deg'] - np.degrees(psi0)))
    db = wrap90(qq['PA'] - np.degrees(psi0_b) - d0) + d0
    cc = ci(db)
    dec = ('profile PA consistent with the GRP preferred direction, within [%+.1f, %+.1f] deg' % tuple(cc) if cc[0] <= 0 <= cc[1]
           else 'profile PA offset from the GRP preferred direction by %+.1f deg [%+.1f, %+.1f]' % (d0, cc[0], cc[1]))
    cmp['c_PA_minus_psi0_' + s] = {'value_deg': d0, 'sd_deg': float(np.std(db, ddof=1)), 'CI95_deg': cc, 'decision': dec}


# (d) PA slope across the MP: (Q,U)-plane fit z_j = A I_fit,j e^{2i k phi_j} over the box, k in deg/deg
def gmodel(x, A, c, sg, b):
    return A * np.exp(-0.5 * ((x - c) / sg) ** 2)


kgrid = np.linspace(-40, 40, 1601)
Ifit = gmodel(tbin, fits['all']['A'], c0, fits['all']['fwhm_us'] / 2.3548, 0)
phb = (tbin[box] - c0) * DEG_PER_US
Mk = Ifit[box][None, :] * np.exp(-2j * np.radians(kgrid)[:, None] * phb[None, :])   # (nk, nbox)


def slope(zbox):
    """zbox (..., nbox) complex sky L per bin -> k maximising |sum z I e^{-2ik phi}|."""
    Sk = np.abs(np.tensordot(zbox, Mk.T, axes=([-1], [0])))
    return kgrid[np.argmax(Sk, axis=-1)]


sl = {}
for s in ('all', 'free'):
    z0 = PB[s, 'MP'].sum(0)[box][:, 1] + 1j * PB[s, 'MP'].sum(0)[box][:, 2]
    zb = bootMP[s][:, box, 1] + 1j * bootMP[s][:, box, 2]
    k0 = float(slope(z0)); kb = slope(zb)
    edge = float(np.mean(np.abs(kb) >= 39.9))
    sl[s] = {'slope_deg_per_deg': k0, 'sd': float(np.std(kb, ddof=1)), 'CI95': ci(kb), 'frac_boot_at_grid_edge': edge}
    log('PA slope %s: %+.2f deg/deg, sd %.2f, CI %s (grid-edge fraction %.3f)' % (s, k0, sl[s]['sd'], sl[s]['CI95'], edge))
cmp['d_PA_slope'] = sl
cmp['d_PA_slope']['decision'] = ('flat within errors (CI contains 0)' if sl['all']['CI95'][0] <= 0 <= sl['all']['CI95'][1]
                                 else 'slope detected (CI excludes 0)')
cmp['d_PA_slope']['Slowikowska15_MP'] = [-0.16, 0.20]
# (e) GRP share of the MP flux: fold (1 - <I>_free/<I>_all, paired) and catalogue (listed GRPs inside fold rotations)
Ia = integ(PB['all', 'MP'])[:, 0]; If = integ(PB['free', 'MP'])[:, 0]
sh0 = 1 - (If.sum() / N['free']) / (Ia.sum() / N['all'])
shb = 1 - (Wb @ If / (Wb @ nrot['free'])) / (Wb @ Ia / (Wb @ nrot['all']))
rotc = np.rint(D['nseq'] + np.asarray(Pp.get('phase'), float)).astype(np.int64)   # a listed pulse belongs to rotation round(Nseq + Phase)
inF = np.isin(rotc, rot_all)
catMP = D['valid'] & D['MP'] & inF
share_cat_ap = D['I'][catMP].sum() / Ia.sum()          # union-aperture fluence (misses the sub-3σ wings)
# same integration region as the fold: raw_profiles (±750 us, calibrated, centred on the window peak) summed over ±1 FWHM minus μ_I
with h5py.File(a.cat, 'r') as hc:
    ic = np.sort(np.where(catMP)[0]); hw = int(round(fw))
    rp = hc['raw_profiles/I'][ic]
    boxflu = rp[:, 750 - hw:750 + hw].sum(1) - D['mu'][ic] * (2 * hw)
share_cat = boxflu.sum() / Ia.sum()
log('catalogue fluence share: union aperture %.3f, ±%d-us box %.3f (box/aperture %.3f)' % (share_cat_ap, hw, share_cat,
                                                                                         boxflu.sum() / D['I'][ic].sum()))
cmp['e_GRP_share'] = {'fold_share': float(sh0), 'fold_CI95': ci(shb), 'catalogue_share': float(share_cat),
                      'catalogue_share_union_aperture': float(share_cat_ap),
                      'n_listed_rotations_with_catalogue_window': int(np.isin(np.array(sorted(rot_listed)), rotc).sum()),
                      'n_cat_MP_in_fold_rotations': int(catMP.sum()), 'n_listed_rotations_in_fold': len(rot_listed),
                      'catalogue_share_frac_of_fold': float(share_cat / sh0),
                      'units_check_foldOFF_over_cat_muI': res['catalogue']['units_ratio_foldOFF_over_cat_muI'],
                      'decision': ('fold share equals the catalogue fluence share within errors' if ci(shb)[0] <= share_cat <= ci(shb)[1]
                                   else 'fold share differs from the catalogue fluence share')}
log('GRP share: fold %.3f CI %s | catalogue %.3f (n=%d valid MP windows in fold rotations; listed rotations %d)' % (
    sh0, ci(shb), share_cat, catMP.sum(), len(rot_listed)))
res['comparisons'] = cmp

# per-bin profile for the figure (all rotations)
Pall = PB['all', 'MP'].sum(0); sbin = sigbin['all']
Lraw2 = Pall[:, 1] ** 2 + Pall[:, 2] ** 2; s2L = 0.5 * (sbin[:, 1] ** 2 + sbin[:, 2] ** 2)
Lbin = np.where(Lraw2 - s2L > s2L, np.sqrt(np.clip(Lraw2 - s2L, 0, None)), 0.0)
res['profile_bins'] = {'t_us': tbin.tolist(), 'phase_deg': phdeg.tolist(), 'I': Pall[:, 0].tolist(), 'L': Lbin.tolist(),
                       'V': (Pall[:, 3] - pzp * Pall[:, 0]).tolist(), 'sigma_I': sbin[:, 0].tolist(),
                       'sigma_L': np.sqrt(s2L).tolist(), 'sigma_V': sbin[:, 3].tolist(),
                       'PA_sky_deg': (0.5 * np.degrees(np.arctan2(Pall[:, 2], Pall[:, 1]))).tolist(),
                       'note': 'all rotations, OFF-subtracted per block, sky frame; sigma = 60-s-block bootstrap sd per bin'}
json.dump(res, open(os.path.join(a.out, 'fold_numbers.json'), 'w'), indent=1, default=float)
log('wrote', os.path.join(a.out, 'fold_numbers.json'))
