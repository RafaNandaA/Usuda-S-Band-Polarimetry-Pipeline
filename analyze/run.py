"""Polarization statistics of the calibrated giant-pulse catalogue.

usage: python3 analyze/run.py CONFIG.toml [--cat CAT.h5] [--fr FR.h5] [--inputs inputs.json] [--selfnoise SN.tsv]
                                 [--items DIR] [--nebula NEB.h5] [--out DIR] [--B N]

Every path defaults to the config (products_dir, tables_dir/inputs.json, work_dir/items, paths.nebula_file, analysis_dir). If the
self-noise sums are missing they are computed first with analyze/selfnoise.jl (Julia). Writes OUT/NUMBERS.json and
OUT/run_log.txt, and prints the outcome of every decision rule. Seeds are fixed per statistic (usudapol/stats.py).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # -I leaves the script directory off sys.path

import numpy as np
from usudapol import io as cio, stats as S, dataset, selection, ellipticity, anisotropy as AN, direction as DR, aligned as AL
from usudapol import duration as DU, element as EL, sign as SG, errors as ER, monitor, zeropoint as ZP, degree as DG, energy as EN

# number of variants checked per statement, used for the multiplicity of the variant test
K_C = {'ratio': 9, 'R_u': 11, 'aligned': 7, 'sign': 7, 'duration': 12, 'LFIP': 4}


def seed(name):
    return S.SEED_BASE + S.ITEM[name]


def load_config(path):
    cfg = tomllib.load(open(path, 'rb'))
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg.get('paths', {}).items():
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg['paths'][k] = os.path.normpath(os.path.join(base, v))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--cat'); ap.add_argument('--fr'); ap.add_argument('--inputs'); ap.add_argument('--out')
    ap.add_argument('--B', type=int)
    ap.add_argument('--selfnoise', help='self-noise sums (TSV from analyze/selfnoise.jl)')
    ap.add_argument('--items', help='directory of the stored windows (1-ms monitors for the Q-guard)')
    ap.add_argument('--nebula', help='nebula snapshot file (gain-versus-additive test)')
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    pa = cfg['paths']
    a.cat = a.cat or os.path.join(pa['products_dir'], 'CrabGRP_catalogue.h5')
    a.fr = a.fr or os.path.join(pa['products_dir'], 'CrabGRP_FreqResolved.h5')
    a.inputs = a.inputs or os.path.join(pa['tables_dir'], 'inputs.json')
    a.out = a.out or pa['analysis_dir']
    a.B = a.B or int(cfg.get('analysis', {}).get('bootstrap', S.B_DEFAULT))
    a.items = a.items or os.path.join(pa['work_dir'], 'items')
    a.nebula = a.nebula or pa.get('nebula_file')
    a.selfnoise = a.selfnoise or os.path.join(a.out, 'selfnoise_sums.tsv')
    os.makedirs(a.out, exist_ok=True)
    if not os.path.exists(a.selfnoise):
        subprocess.run(['julia', '-t', 'auto', os.path.join(HERE, 'selfnoise.jl'), os.path.abspath(a.config), a.selfnoise], check=True)
    logf = open(os.path.join(a.out, 'run_log.txt'), 'w')

    def log(*x):
        s = ' '.join(str(v) for v in x)
        print(s); logf.write(s + '\n'); logf.flush()
    t0 = time.time()
    B = a.B
    R = S.Recorder(log)
    inp = json.load(open(a.inputs)) if os.path.exists(a.inputs) else {}
    log('cat:', a.cat, '| fr:', a.fr, '| B =', B)
    log('seeds:', json.dumps({k: seed(k) for k in S.ITEM}))
    fd = cfg.get('feed', {})        # feed model chi = q + alpha*El + gamma_az*Az, the same as in calibrate/
    dataset.PRIMARY['chi'] = dict(alpha=float(fd.get('alpha_el', -1.0)), gamma_az=float(fd.get('gamma_az', 0.0)), use_rm=True, k=-1.0)
    P = cio.Products(a.cat, a.fr, log)
    P.require_core()

    rows = list(csv.DictReader(open(a.selfnoise), delimiter='\t'))
    ncat = np.asarray(P.get('phase')).size
    sn = {k: np.full(ncat, np.nan) for k in rows[0] if k != 'cat'}
    for r in rows:
        j = int(r['cat'])
        for k in sn:
            sn[k][j] = float(r[k])
    log('self-noise sums loaded for %d windows (cat_N -> catalogue row N)' % len(rows))
    # ---------- LFIP phase edges from the list phase histogram ----------
    lfe = selection.lfip_edges(np.asarray(P.get('phase'), float))
    dataset.PRIMARY['lf_edges'] = tuple(lfe['edges_used'])
    R.put('selection.lfip_edges', {k: v for k, v in lfe.items() if k not in ('counts', 'bin_edges')}, counts=lfe['counts'])
    R.decide('LFIP edges', '0.38-0.42 kept iff the significant run lies inside it and >= 95 % of the excess',
             'edges 0.38-0.42 kept' if lfe['primary_kept'] else 'edges replaced by %s' % (lfe['edges_used'],))
    # ---------- selection: Stokes-I S/N cut complete in V ----------
    rng = np.random.default_rng(seed('selection'))
    D0 = dataset.build(P, {}, sn, log)          # needed for V/I and σ in the completeness tests
    base = D0['valid'] & (D0['MP'] | D0['LFIP'])
    S0mask = base.copy()
    sel_ok = all(P.has(r) for r in ('I_SN_cal', 'SN_R_run', 'SN_L_run')) and ('detlist_R' in inp)
    if sel_ok:
        Rl, Ll = cio.read_detlist(inp['detlist_R']), cio.read_detlist(inp['detlist_L'])
        nseq = D0['nseq']
        lR = np.array([Rl.get(n, np.nan) for n in nseq]); lL = np.array([Ll.get(n, np.nan) for n in nseq])
        if inp.get('r_c') is not None:
            rq = {'r_c': inp['r_c'], 'source': 'given in inputs.json'}
        else:
            rq, _ = selection.r_quantiles(np.asarray(P.get('SN_R_run'), float), lR, np.asarray(P.get('SN_L_run'), float), lL, rng, B)
        # sigma_R_band/sigma_L_band are uncalibrated per-hand powers, so their ratio carries the hand gain. In calibrated
        # (hand-equalised) units the background noise per hand is equal by construction, and the per-window hand noise below
        # is taken from the calibrated per-channel baselines. The uncalibrated ratio is printed.
        if P.has('sigma_R_band'):
            R.put('selection.uncal_hand_sigma_ratio_median', float(np.median(np.asarray(P.get('sigma_R_band'), float) / np.asarray(P.get('sigma_L_band'), float))))
        cut, form = selection.I_SN_cut(rq['r_c'], None, None)
        isn = np.asarray(P.get('I_SN_cal'), float)
        if P.has('fr_base'):
            # calibrated per-hand noise -> per-window I_SN_c,j = 5 r_c g_j, g_j = (σ_R + σ_L)/σ_I, applied as isn·√2/g_j
            # against the scalar 5√2 r_c
            fb = np.asarray(P.get('fr_base'), float)
            sRc, sLc, gj = selection.calibrated_hand_sigma(fb[:, 0, :], fb[:, 3, :], P.band_weights())
            isn = isn * np.sqrt(2) / gj
            form = 'per-window 5 r_c (σ_R+σ_L)/σ_I from the calibrated per-hand noise'
            R.put('selection.hand_noise', {'sR_over_sL_median': float(np.median((sRc / sLc)[base])),
                                           'sR_over_sL_p1_p99': np.percentile((sRc / sLc)[base], [1, 99]).tolist(),
                                           'g_over_sqrt2_median': float(np.median(gj[base]) / np.sqrt(2)),
                                           'g_over_sqrt2_range': [float(gj[base].min() / np.sqrt(2)), float(gj[base].max() / np.sqrt(2))]})
        lad = selection.run_ladder(isn, cut, base, D0['Vzp'] / D0['I'], D0['I'], D0['s2V'], D0['s2I'], rng)
        cutA = cut * lad['adopted_factor']
        S0mask = base & (isn >= cutA)
        R.put('selection.r', rq); R.put('selection.I_SN_c', cut, form=form); R.put('selection.ladder', lad)
        R.put('selection.adopted_cut', cutA)
        R.decide('selection', 'precondition (KS p>0.01 and central p>0.01) / ladder / classification', lad['classification'],
                 adopted_factor=lad['adopted_factor'])
        R.put('selection.coverage_curve', selection.coverage_curve(cutA, float(np.median(D0['sI']))))
        sel_class = lad['classification']
    else:
        R.nc('selection', 'I_SN_cal, per-hand running S/N or detection lists not available -> S0 = valid & phase mask (no I_SN '
             'cut); completeness classification unknown')
        sel_class = 'not computed'
    D = dataset.build(P, {}, sn, log)
    if D['self_ok']:
        log('self-noise: max |W_selfnoise - W_AU| = %g' % D['self_W_check'])
    log('cross term form:', D['cross_form'], '| chi:', D['chi_label'], '| zero point p = %.5f' % D['p_zp'])
    MPi = np.where(S0mask & D['MP'])[0]; LFi = np.where(S0mask & D['LFIP'])[0]
    S0i = np.where(S0mask)[0]
    R.put('n', {'S0': int(S0i.size), 'S0_MP': int(MPi.size), 'S0_LFIP': int(LFi.size)})
    log('S0: %d (MP %d, LFIP %d)' % (S0i.size, MPi.size, LFi.size))
    # psi0 from the selected MP pulses
    zs, psi, psi0, Qp, Up = ER.sky_frame(D['Q'], D['U'], D['chi'], S0mask & D['MP'])
    D.update(zs=zs, psi=psi, psi0=psi0, Qp=Qp, Up=Up)
    medI = np.median(D['I'][MPi])
    H = S0i[D['I'][S0i] > medI]; HMP = MPi[D['I'][MPi] > medI]
    qguard = None
    if os.path.isdir(a.items):
        mex = monitor.mon_excess(a.items, D['I'].size)
        qguard = mex > 0.10
        R.put('q_guard', {'n_excluded_S0': int(np.sum(qguard[S0i])), 'median_excess': float(np.nanmedian(mex[S0i])),
                          'rule': 'max over full 1-ms bins and streams of P/median - 1 > 0.10 (all bins; sweep bins not stored)'})
    else:
        R.nc('Q-guard', 'window monitors not available -> Q-guard fits and variants not computed')

    # cross term: per-channel form (primary) against the band form, as a check (median per-pulse relative difference)
    offV = D['W'] * D['sV'] ** 2 * (1 + D['W'] / D['nb'])
    band = 2 * D['sV'] ** 2 * D['I'] / D['mu']
    pc = D['s2V'] - offV
    okc = S0i[band[S0i] > 0]
    R.put('check.cross_term_perchan_vs_band', {'median_rel_diff': float(np.median(np.abs(pc[okc] / band[okc] - 1))),
                                               'form_used': D['cross_form']})
    # ---------- off-pulse debias null ----------
    prof = tuple(np.asarray(P.get(r), float) for r in ('prof_I', 'prof_Q', 'prof_U', 'prof_V'))
    pk = np.asarray(P.get('prof_peak_index'))
    off = ER.offpulse_apertures(prof, pk, D['W'], D['mu'])
    s2off = lambda s: off['W'] * s ** 2 * (1 + off['W'] / off['nbp'])
    nq = (off['Q'] ** 2 + off['U'] ** 2 - s2off(D['sQ']) - s2off(D['sU']))[S0i] / s2off(D['sV'])[S0i]
    nv = (off['V'] ** 2 - s2off(D['sV']))[S0i] / s2off(D['sV'])[S0i]
    chk = {'QU_null_mean_over_sig2': float(nq.mean()), 'QU_SE': float(nq.std() / np.sqrt(nq.size)),
           'V_null_mean_over_sig2': float(nv.mean()), 'V_SE': float(nv.std() / np.sqrt(nv.size))}
    chk['works'] = bool(abs(chk['QU_null_mean_over_sig2']) < 3 * chk['QU_SE'] and abs(chk['V_null_mean_over_sig2']) < 3 * chk['V_SE'])
    R.put('check.offpulse_debias_null', chk)
    R.decide('off-pulse debias null', 'within 3 SE', 'works' if chk['works'] else 'BROKEN (offset)')

    # ---------- receptor ellipticity from the pulses ----------
    rng = np.random.default_rng(seed('ellipticity'))
    Ipe, Qpe, Upe, Vpe = dataset.preell_band(P)
    if 'm_app_subband' not in inp:
        raise cio.MissingColumn('m_app_subband (applied m per subband) must be given in inputs.json')
    msub = np.asarray(inp['m_app_subband'], float)
    Ifr = np.asarray(P.get('fr_I'), float) * P.band_weights()[None, :]
    wsb = np.array([Ifr[MPi][:, 32 * s:32 * s + 32].sum() for s in range(4)])
    wsb = wsb / wsb.sum()
    m_app = (wsb[:, None] * msub).sum(0)
    R.put('ellipticity.m_app_band', m_app.tolist(), weights=wsb.tolist())
    tt = (D['sod'] - np.median(D['sod'])) / 3600.0
    gsh = SG.gamma_shape(D['sod'], inp['gamma_table_csv']) if 'gamma_table_csv' in inp else None
    lab = np.sqrt(np.clip(Qpe ** 2 + Upe ** 2, 0, None)) / Ipe
    chiF = dataset.chi_fr(P, D)
    pr = np.angle((Qpe + 1j * Upe) * np.exp(-2j * chiF) * np.exp(-2j * psi0))
    extra = [lab, lab * np.cos(pr), lab * np.sin(pr), np.cos(2 * pr), np.sin(2 * pr), np.cos(D['chi']), np.sin(D['chi']), tt]
    if gsh is not None:
        extra.append(gsh)
    extra = np.c_[tuple(extra)]
    er = ellipticity.run(Ipe[MPi], Qpe[MPi], Upe[MPi], Vpe[MPi], D['s2I'][MPi], D['s2Q'][MPi], D['s2U'][MPi], D['s2V'][MPi],
                         chiF[MPi], psi0, D['el'][MPi], D['sod'][MPi], rng, B, m_app, extra[MPi])
    neb = None
    if all(k in inp for k in ('m_neb_sigma19', 'm_neb_sigma95')):
        # band-mean σ from the per-subband σ with the same intensity weights (subbands independent), nebula m = m_app
        bs = lambda d: np.array([np.sqrt(np.sum(wsb ** 2 * np.asarray(d[c]) ** 2)) for c in ('QV', 'UV')])
        s19, s95 = bs(inp['m_neb_sigma19']), bs(inp['m_neb_sigma95'])
        conv = bool(np.all(s95 <= 1.25 * s19))
        R.put('ellipticity.nebula_band_sigma', {'sigma19': s19.tolist(), 'sigma95': s95.tolist(), 'converged(<=1.25)': conv,
                                                'note': 'block-bootstrap sigmas of the nebula m from the config'})
        if conv:
            neb = (m_app.tolist(), np.diag(s95 ** 2))
        else:
            R.decide('nebula ellipticity region', 'σ95 <= 1.25 σ19 in both components', 'not converged -> nebula region not included (not a bound)')
    else:
        R.nc('nebula ellipticity region', 'nebula m with block σ19/σ95 not supplied -> R = pulse-measured region only')
    fallback = (not er['E1_pass']) or er['fallback_trigger_sd>0.05']
    parts, pts, rdesc = ellipticity.region(er['ellipses'], m_app, neb, fallback)
    m_hat = np.asarray(er['m_hat'])
    er_print = {k: v for k, v in er.items() if k not in ('ellipses', 'boot_s')}
    R.put('ellipticity.result', er_print, region=rdesc)
    R.decide('ellipticity E1', '|corr feed vs sky regressors| < 0.9', 'pass' if er['E1_pass'] else 'FAIL -> F-E')
    R.decide('ellipticity E3', 'tertile consistency chi2 p > 0.01', 'pass' if er['E3_pass'] else 'fail: the on-axis coupling varies with elevation')
    R.decide('ellipticity region', 'R = pulse region U nebula region (if converged) / F-E', rdesc)
    rr = AN.RegionRatio(D, m_app, pts, parts)

    # ---------- census ----------
    rng = np.random.default_rng(seed('census'))
    R.put('census.MP', AN.census(D, S0mask & D['MP'], rng, B))
    R.put('census.LFIP', AN.census(D, S0mask & D['LFIP'], rng, B, quint=False))
    R.put('census.E_LI_all_valid_MP', float(np.mean(np.minimum(1, D['L'] / D['I'])[base & D['MP']])))
    R.put('census.residual_L_scale', {'ratio_and_l2_factor_range': [0.9976, 1.0159], 'tau_resid_loss_c_max': 0.002},
          source='cross-hand coherence of synthetic 4-bit data with the R/L delay applied before channelisation, relative to an '
                 'ideal reference: 0.9988-1.0079')

    # ---------- anisotropy <l²>/<v²> ----------
    rng = np.random.default_rng(seed('anisotropy'))
    ov = AN.ratio_block(D, MPi, rr, m_hat, rng, B)
    dec = AN.decide_overall(ov, rdesc)
    R.put('anisotropy.overall', {k: v for k, v in ov.items() if not k.startswith('_')}, decision=dec)
    R.decide('anisotropy overall', 'L_R > 2 / U_R < 2 / consistent; at m_hat and m_app beside', dec['text'], **dec)
    qrows, qdec = AN.quintiles(D, MPi, rr, m_hat, rng, B)
    R.put('anisotropy.quintiles', qrows); R.decide('anisotropy quintiles', 'every L_R,k > 2 / U_R,k < 2', qdec)
    R.put('anisotropy.piecewise_El_tertile_L_R', AN.piecewise_tertile_LR(D, MPi, rr, D['el'], rng, B))
    lf = AN.ratio_block(D, LFi, rr, m_hat, rng, B)
    R.put('anisotropy.LFIP', {k: v for k, v in lf.items() if not k.startswith('_')}, branch=AN.branch_ratio(lf['L_R'], lf['U_R']))
    R.decide('anisotropy LFIP', 'same three branches on S0_LFIP', AN.branch_ratio(lf['L_R'], lf['U_R']))
    nn = AN.matched_mp(D, LFi, MPi)
    mm = AN.lfip_vs_matched(D, LFi, nn, rng, B)
    R.put('anisotropy.LFIP_vs_matchedMP', mm); R.decide('anisotropy LFIP vs matched MP', 'CI of difference vs 0', mm['branch'])
    # noise-scale variants (paired: the same draws as the primary)
    for nm, sc in (('noise_x0.9', 0.9), ('noise_x1.1', 1.1)):
        Dv = dataset.build(P, {'noise_scale': sc}, sn, log)
        rv = AN.RegionRatio(Dv, m_app, pts, parts)
        rngp = np.random.default_rng(seed('anisotropy'))
        vb = AN.ratio_block(Dv, MPi, rv, m_hat, rngp, B, want_region=False)
        rngp = np.random.default_rng(seed('anisotropy'))
        pb = AN.ratio_block(D, MPi, rr, m_hat, rngp, B, want_region=False)
        diff = S.a_to_ratio(vb['_a_hat_draws']) - S.a_to_ratio(pb['_a_hat_draws'])
        R.put('anisotropy.variant.' + nm, S.definitional_variant(diff, vb['ratio_at_m_hat'], pb['ratio_at_m_hat'],
                                                                 lambda x: x > 2, K_C['ratio']))

    # ---------- preferred direction ----------
    rng = np.random.default_rng(seed('direction'))
    pdr = DR.preferred_direction(D, MPi, rng, B)
    ex, fr = DR.decide_direction(pdr)
    R.put('direction.MP', {k: v for k, v in pdr.items() if not k.startswith('_')})
    R.decide('direction existence', 'Rayleigh p < 1e-3', ex); R.decide('direction frame', 'r CI vs 0 and 1', str(fr))
    R.decide('direction feed-fixed check', 'R_feed within max(0.02, 2σ) of smearing prediction', 'works' if pdr['feed_check_works'] else 'BROKEN (excess)')
    neb_r = DR.rate_fit(np.asarray(P.get('L_neb'), float)[S0i], np.radians(np.asarray(P.get('PA_neb_deg'), float))[S0i], D['chi'][S0i])
    neb_r_pre = DR.rate_fit(np.asarray(P.get('L_neb_preD'), float)[S0i], np.radians(np.asarray(P.get('PA_neb_preD_deg'), float))[S0i], D['chi'][S0i])
    R.put('direction.nebula_rate', {'post-D (PA_neb_deg)': neb_r, 'pre-D (PA_neb_preD_deg)': neb_r_pre})
    chivar = {'alpha=-0.9': dict(alpha=-0.9, gamma_az=0.0), 'alpha=-1.1': dict(alpha=-1.1, gamma_az=0.0),
              'sensitivity(-1,+0.0335)': dict(alpha=-1.0, gamma_az=0.0335)}
    if 'gamma_Az_interval' in inp:
        for g in inp['gamma_Az_interval']:
            chivar['gamma_Az=%+.4f' % g] = dict(alpha=-1.0, gamma_az=g)
    if 'gamma_Az_joint' in inp:
        for g in inp['gamma_Az_joint']['interval_2sd']:
            chivar['gamma_Az=%+.5f' % g] = dict(alpha=-1.0, gamma_az=g)
    for nm, cv in chivar.items():
        cv['use_rm'] = True
        Dv = dataset.build(P, {'chi': cv}, sn, log)
        psv = 0.5 * np.angle(Dv['zs'])
        rngp = np.random.default_rng(seed('direction'))
        drw = MPi[rngp.integers(0, MPi.size, (B, MPi.size))]
        Rv = np.abs(np.mean(np.exp(2j * psv[drw]), 1)); Rp = np.abs(np.mean(np.exp(2j * D['psi'][drw]), 1))
        Rpt = abs(np.mean(np.exp(2j * psv[MPi])))
        res = S.definitional_variant(Rv - Rp, Rpt, pdr['R_u'], lambda x: x > 0, K_C['R_u'])
        res['R_u'] = Rpt
        R.put('direction.chi_variant.' + nm, res)

    # ---------- LFIP alone ----------
    rng = np.random.default_rng(seed('lfip'))
    lr = DR.lfip(D, LFi, MPi, rng, B, 2000, nn)
    R.put('direction.LFIP', lr); R.decide('LFIP existence', 'Rayleigh p <= 0.01 AND chi-perm p <= 0.01', lr['existence'])
    R.decide('LFIP direction', 'Δψ CI contains 0', str(lr['direction']))
    LFf = np.where(S0mask & D['LFIPflag'])[0]
    R.put('direction.LFIP_flagmask_variant', {k: lr2 for k, lr2 in DR.lfip(D, LFf, MPi, np.random.default_rng(seed('lfip')), B, 2000).items()})

    # ---------- aligned fraction ----------
    rng = np.random.default_rng(seed('aligned'))
    af = AL.aligned_fraction(D, MPi, rng, 1000)
    R.put('aligned.value', af); R.decide('aligned value', 'printed = min(<Q\'/I>, LOO); about 0.1 iff in [0.05, 0.15)', af['wording'])
    slopes = {}
    for sname, ii in (('S0_MP', MPi), ('H_MP', HMP)):
        Bs = S.B_ALIGNED_SLOPE_S0MP if (sname == 'S0_MP' and B >= S.B_DEFAULT) else B
        rA = AL.slope(D, ii, D['rho_pk'], rng, Bs); rB = AL.slope(D, ii, D['I'], rng, Bs)
        slopes[sname] = AL.decide_slope(rA, rB)
        R.put('aligned.slope.' + sname, {'A': {k: v for k, v in rA.items() if not k.startswith('_')},
                                         'B': {k: v for k, v in rB.items() if not k.startswith('_')}, 'branch': slopes[sname]})
        R.decide('aligned slope ' + sname, 'b_rel CIs (models A, B); B = %d' % Bs,
                 slopes[sname] + (' (marginal)' if sname == 'S0_MP' else ''))
    fl = AL.floor(D, MPi, rng, B)
    R.put('aligned.floor', fl); R.decide('aligned floor', 'lower bound of <Q\'_LOO/I> in longest-W quintile > 0', fl['wording'])
    qoff = np.real((off['Q'] + 1j * off['U']) * np.exp(-2j * D['chi']) * np.exp(-2j * psi0))[MPi]
    R.put('aligned.check.offpulse_Qp', {'mean_over_SE': float(qoff.mean() / (qoff.std() / np.sqrt(qoff.size)))})

    # ---------- duration at fixed brightness ----------
    rng = np.random.default_rng(seed('duration'))
    du = DU.run(D, {'S0': S0i, 'H': H}, rng, B, qguard=qguard, moments=('l2',))
    txt, samp, br, notes = DU.statement(du, D['self_ok'], qguard is not None)
    R.put('duration.primary', du, notes=notes)
    R.decide('duration', 'four l² fits per sample, S0 then H; self-noise removed', txt, sample=samp, raw_branch=br, notes=notes)
    for nm, kw in (('noise_x0.75', dict(noise_scale=0.75)), ('noise_x1.25', dict(noise_scale=1.25)), ('W_eq', dict(Wkey='W_eq')),
                   ('decile_FE', dict(decile_fe=True))):
        R.put('duration.variant.' + nm, DU.run(D, {'H': H}, np.random.default_rng(seed('duration')), B, qguard=None, moments=('l2',), **kw))

    # ---------- element bound ----------
    rng = np.random.default_rng(seed('element'))
    eb = EL.bound(D, MPi, rr, rng, B, rho2=None)
    R.put('element.bound', eb); R.decide('element bound', 'printed iff L_el > 2', eb['printed'])

    # ---------- sign balance of V ----------
    rng = np.random.default_rng(seed('sign'))
    if sel_class == 'selection':
        R.decide('sign balance', 'precondition', 'not computed (the sample is not complete in V)')
    else:
        Vp, bs = SG.crossfit_Vprime(D, D['Vzp'], MPi, D['chi'], psi0)
        vin = (np.asarray(P.get('V_neb'), float) / np.asarray(P.get('I_neb'), float))[S0i]
        zg = ZP.zp_grid(vin, D['sod'][S0i], D['p_zp'], np.random.default_rng(seed('zero_point')), B)
        R.put('sign.zero_point_range', zg)
        R.decide('V zero-point range', 'block-bootstrap interval of the residual nebular V/I + observed nebular bound 1.56e-4 '
                 '(Wilson & Weiler 1997 via Bucciantini & Olmi 2018, scaled ν^-1/2) + γ(t) ripple ±1.3e-3',
                 'δ ∈ [%.5f, %.5f] (grid %s)' % (zg['delta_range'][0], zg['delta_range'][1], [round(x, 6) for x in zg['grid']]))
        sg = SG.run(D, MPi, D['Vzp'], Vp, rng, B, inp.get('step_sod', 29056.6), inp.get('step_amp_err', 0.0), zg['grid'])
        if 'step_amp_err' not in inp:
            R.nc('δ_step', 'gain-step amplitude error not supplied -> δ_step = 0')
        R.put('sign.balance', sg, crossfit_b=bs, precondition=sel_class)
        R.decide('sign balance', '|z| > 2.73 over 8 tests at every grid point (one multiplicity)', sg['decision'])
        R.put('sign.check.offpulse', SG.offpulse_check(off['V'][MPi], s2off(D['sV'])[MPi]))
        if gsh is not None:
            zr = SG.pulse_zp_regression(D, MPi, D['Vzp'], gsh, rng, B)
            R.put('zero_point.pulse_regression', zr); R.decide('pulse V/I vs γ(t) shape', 'b CI vs 0', zr['outcome'])
        if P.has('VI_neb_preGamma') and gsh is not None:
            sq = SG.squint_test(np.asarray(P.get('VI_neb_preGamma'), float)[S0i], gsh[S0i], D['chi'][S0i], D['sod'][S0i],
                                np.random.default_rng(seed('zero_point')))
            R.put('zero_point.squint', sq)
            R.decide('squint', '99 % block CI of (p_c, p_s) excludes 0', sq['outcome'] +
                     ('; pulse zero point p = median calibrated nebular V/I' if sq['outcome'] != 'squint detected'
                      else '; p should be the χ-independent part of the joint fit and is NOT applied in this run'))
        else:
            R.nc('squint', 'nebular V/I before the γ(t) correction not supplied')
        if a.nebula and os.path.exists(a.nebula):
            r1a = ZP.nebula_1a(a.nebula, np.random.default_rng(seed('nebula_gain')), B)
            R.put('zero_point.nebula_gain_vs_additive', r1a, nebula_file=os.path.basename(a.nebula))
            R.decide('nebula gain vs additive', 'slope CI vs 0 and ±½ (uninformative if |corr| > 0.9 or half-width > 0.25)', r1a['outcome'])
        else:
            R.nc('nebula gain vs additive', 'nebula snapshot file not found')

    # ---------- robustness: track halves (subsample) and time-block bootstrap ----------
    rng = np.random.default_rng(seed('halves'))
    t_med = np.median(D['sod'][MPi])
    half = D['sod'][MPi] <= t_med
    rat = lambda ii: D['l2'][MPi][ii].mean() / D['v2'][MPi][ii].mean()
    Ru = lambda ii: abs(np.mean(np.exp(2j * D['psi'][MPi][ii])))
    QpI = lambda ii: float(np.mean(AL.qp_over_i(D['zs'], D['I'], MPi[ii], MPi[ii])))
    for nm, fn, k in (('ratio', rat, K_C['ratio']), ('R_u', Ru, K_C['R_u']), ('aligned', QpI, K_C['aligned'])):
        R.put('robustness.halves.' + nm, S.subsample_variant(S.hetero_draws(fn, half, B, rng), k))
    rng = np.random.default_rng(seed('block'))
    order = MPi[np.argsort(D['sod'][MPi])]
    # block length from the autocorrelation of each statistic's per-pulse series in time order
    cpsi = np.cos(2 * (D['psi'] - psi0)); spsi = np.sin(2 * (D['psi'] - psi0))
    Lb = {'ratio': max(S.block_length(D['l2'][order]), S.block_length(D['v2'][order])),
          'R_u': max(S.block_length(cpsi[order]), S.block_length(spsi[order]))}
    Lb['r'] = Lb['R_u']
    sigt = SG.sigma_V_tot(D)
    selV = order[np.abs(D['Vzp'][order]) > 3 * sigt[order]]
    Lb['f'] = S.block_length((D['Vzp'][selV] < 0).astype(float))
    blk = {'block_lengths': Lb}
    bi = S.block_boot_idx(order, B, rng, Lb['ratio'])
    blk['ratio_block_CI'] = S.ci(D['l2'][bi].mean(1) / D['v2'][bi].mean(1)); blk['ratio_pulse_CI'] = ov['ratio_at_m_app_CI']
    bi = S.block_boot_idx(order, B, rng, Lb['R_u'])
    blk['R_u_block_CI'] = S.ci(np.abs(np.mean(np.exp(2j * D['psi'][bi]), 1))); blk['R_u_pulse_CI'] = pdr['R_u_CI']
    rr_b = np.array([DR.rate_fit(D['L'][ii], D['PA'][ii], D['chi'][ii]) for ii in bi])
    blk['r_block_CI'] = S.ci(rr_b); blk['r_pulse_CI'] = pdr['r_CI']
    bf = S.block_boot_idx(selV, B, rng, Lb['f'])
    blk['f_block_CI'] = S.ci((D['Vzp'][bf] < 0).mean(1))
    blk['f_pulse_CI'] = S.ci((D['Vzp'][selV[rng.integers(0, selV.size, (B, selV.size))]] < 0).mean(1))
    for k in ('ratio', 'R_u', 'r', 'f'):
        wb = np.diff(blk[k + '_block_CI'])[0]; wp = np.diff(blk[k + '_pulse_CI'])[0]
        blk[k + '_block_replaces'] = bool(wb > 1.2 * wp)
        R.decide('block bootstrap ' + k, 'block width > 1.2 × pulse width -> block interval replaces (L = %d)' % Lb[k],
                 'block interval replaces the pulse interval' if blk[k + '_block_replaces'] else 'pulse interval stands')
    R.put('robustness.block_bootstrap', blk)
    R.put('robustness.expected_chance_conditionals', 1.2)

    # ---------- total degree ----------
    rng = np.random.default_rng(seed('degree'))
    dg = DG.run(D, MPi, rng, B)
    R.put('degree.MP', dg); R.decide('not a single state', 'lower 95 % bound of F_<1 >= 0.5 at S/N >= 20 / 50; mirror F_=1', dg['decision'])
    # ---------- energy distribution ----------
    rng = np.random.default_rng(seed('energy'))
    if sel_ok:
        W90 = np.percentile(D['W'][MPi], 90)
        efloor = cutA * float(np.median(D['sI'][MPi])) * np.sqrt(10) * max(W90, 10) / 10
    else:
        efloor = 0.0
    en = EN.run(D['I'][MPi], D['W'][MPi], rng, efloor, B, 1000)
    en['E_min_coverage_floor'] = efloor
    R.put('energy.MP', en); R.decide('energy', 'GoF p > 0.1 (Clauset+09) then CI of difference to Enoto+21', en['decision'])
    enL = EN.fit(D['I'][LFi], efloor)
    if enL and enL[2] >= 50:
        R.put('energy.LFIP', {'alpha': enL[0], 'E_min': enL[3], 'n_above': enL[2]})
    else:
        R.nc('energy LFIP', 'n above E_min < 50')
    R.dump(os.path.join(a.out, 'NUMBERS.json'), {'cat': os.path.basename(a.cat), 'fr': os.path.basename(a.fr), 'B': B,
                                                 'chi': D['chi_label'], 'cross_form': D['cross_form'],
                                                 'self_noise': D['self_ok'], 'runtime_s': time.time() - t0})
    log('done in %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
