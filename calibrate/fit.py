"""Fit the calibration tables from the off-pulse Crab Nebula.

usage: python3 calibrate/fit.py CONFIG.toml [--ion-track CSV]

Reads the nebula snapshots (paths.nebula_file), the 1-s snapshots around the R-chain gain step (paths.gain_step_file) and the
global ionosphere map (paths.gim_file), and writes into paths.tables_dir:
  gainbandpass_cal_M64.csv   amplitude bandpass per hand and channel
  diffgain_time_M64.csv      differential gain gamma(t): settling + R-chain gain step + band-common fast term
  dterm_cal_python_M64.csv   D-terms (kappa form) with bootstrap errors
  mueller_vleak_M64.csv      ellipticity (Q,U)->V coupling per subband, repeated per channel
  rm_ion_track.csv           ionospheric RM_ion(t)
  scalars.txt                R/L delay ramp, residual delay, R/L phase per subband, RM, feed model, dropped channels
  inputs.json                calibration quantities used by analyze/run.py
The full ionosphere track is also written to paths.work_dir/ionosphere_track.csv (with --ion-track an existing track is used).
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import tomllib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)          # -I leaves the script directory off sys.path

from lib import chain, nebula_fit as nf, ionosphere as ion, gain_step, fast_gamma   # noqa: E402

M_CHAN = 64


def load_config(path):
    cfg = tomllib.load(open(path, 'rb'))
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg.get('paths', {}).items():
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg['paths'][k] = os.path.normpath(os.path.join(base, v))
    return cfg


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Fit the calibration tables from the nebula snapshots.')
    ap.add_argument('config')
    ap.add_argument('--ion-track', default=None, help='use this RM_ion track CSV instead of computing it from the GIM file')
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    P, cal, feed_cfg = cfg['paths'], cfg['calibration'], cfg['feed']
    out = pathlib.Path(P['tables_dir']); out.mkdir(parents=True, exist_ok=True)
    os.makedirs(P['work_dir'], exist_ok=True)
    feed = (float(feed_cfg['alpha_el']), float(feed_cfg['gamma_az']))
    t_step = float(cal['step_sod'])
    ramp = np.array(cfg['delay']['rl_delay_s'], float)

    # ionospheric rotation, computed once from the GIM along the track
    if a.ion_track:
        track_csv = a.ion_track
    else:
        d0 = nf.load_snapshots(P['nebula_file'])
        obs = cfg['observation']
        R = ion.track(P['gim_file'], obs['date'], obs['site_lat_deg'], obs['site_lon_deg'], d0['toff_sod'], d0['az_deg'], d0['el_deg'])
        track_csv = os.path.join(P['work_dir'], 'ionosphere_track.csv')
        ion.write_track(track_csv, R)
    ion_t = ion.read_track(track_csv)

    d, fm = chain.load_data(P['nebula_file'], ion_t)
    print(f"snapshots: {d['K']} kept of {fm['any'].size} (interference mask: {int(fm['any'].sum())})")
    step_s, step_err = gain_step.measure(P['gain_step_file'], t_step, nf.channel_mask())
    print(f'R-chain gain step: {step_s * 100:+.3f} +- {step_err * 100:.3f} % in V/I')

    o = chain.run_chain(d, feed, t_step, step_s, n_boot=int(cal.get('n_bootstrap', 200)))
    f = o['fit']
    nf.write_tables(out, M_CHAN, d['freq_MHz'], o['B_R'], o['B_L'], o['dterm'], d['K'], o['knot_sod'], o['knot_gamma'],
                    o['dgfit'], o['mQV_ch'], o['mUV_ch'], o['vleak'])

    # band-common fast term on top of the settling + step table
    dg_csv = out / f'diffgain_time_M{M_CHAN}.csv'
    lines = open(dg_csv).read().split('\n')
    rows = np.array([r.split(',') for r in lines if r[:1].isdigit()], float)
    z = fast_gamma.residual_series(o, t_step, step_s)
    dk_cv, cv = fast_gamma.choose_knot_spacing(z)
    dk = float(cal.get('fast_gamma_knot_min', dk_cv))
    print(f'fast gamma term: knot spacing {dk:g} min (cross-validation choice {dk_cv} min)')
    tsamp, gn, _ = fast_gamma.build_table(z, rows, dk, t_step)
    head = [l for l in lines if l.startswith('#')]
    head.insert(1, f'# gamma(t) = settling + R-chain gain step, times a band-common fast term (cubic B-spline, knots every {dk:g} min), sampled every 60 s.')
    with open(dg_csv, 'w') as fh:
        fh.write('\n'.join(head) + '\nsod_utc_s,gamma\n')
        for t, g in zip(tsamp, gn):
            fh.write('%.3f,%.9f\n' % (t, g))

    T = np.loadtxt(track_csv, delimiter=',', skiprows=2)
    with open(out / 'rm_ion_track.csv', 'w') as fh:
        fh.write(f'# RM_ion(sod) [rad m^-2, IAU sign] from {os.path.basename(P["gim_file"])}, IGRF, thin shell 450 km\nsod,rm_ion\n')
        for t, r in zip(T[:, 0], T[:, 8]):
            fh.write('%.1f,%.6f\n' % (t, r))

    tau_res = f['tau_s']
    with open(out / 'scalars.txt', 'w') as fh:
        fh.write('# phase_offset_RL = phi_s [rad] at f = 0, in the frame after the read-time R/L delay ramp delay_RL. '
                 'The per-channel phase on L is phi_s + 2 pi f_ch tau_resid_RL.\n')
        fh.write('phase_offset_RL = ' + ','.join('%.17g' % x for x in f['phi_inst_s']) + '\n')
        fh.write('delay_RL = ' + ','.join('%.17g' % x for x in ramp) + '\n')
        fh.write('RM = 0\n')            # degenerate with tau_s and phi_s over 128 MHz, so it is not fitted
        fh.write('gain_R = 1,1,1,1\ngain_L = 1,1,1,1\n')   # scalar hand gains are inside the bandpass
        fh.write('bwg_alpha = %.17g\nbwg_gamma = %.17g\n' % feed)
        fh.write('tau_resid_RL = ' + ','.join('%.17g' % x for x in tau_res) + '\n')
        fh.write('ion_sigma = -1\n')
        fh.write('drop_channels = 1:32,2:1,2:32,3:32,4:32\n')

    # inputs for the analysis
    m = {}
    for r in open(out / f'mueller_vleak_M{M_CHAN}.csv'):
        if r[:1].isdigit():
            s, c, qv, uv = r.split(',')
            m.setdefault(int(s), []).append((float(qv), float(uv)))
    ns = cfg.get('nebula_systematics', {})
    g, gsd = ns.get('gamma_az_free'), ns.get('gamma_az_free_sd')
    inputs = {
        'm_app_subband': [[sum(x[0] for x in m[s]) / len(m[s]), sum(x[1] for x in m[s]) / len(m[s])] for s in sorted(m)],
        'm_neb_sigma19': {'QV': ns.get('m_sigma_block19_QV'), 'UV': ns.get('m_sigma_block19_UV')},
        'm_neb_sigma95': {'QV': ns.get('m_sigma_block95_QV'), 'UV': ns.get('m_sigma_block95_UV')},
        'gamma_table_csv': str(dg_csv),
        'step_sod': t_step,
        'step_amp_VI_fraction': round(100 * step_s, 3) / 100,     # rounded to 0.001 % of I
        'step_amp_err': round(100 * step_err, 3) / 100,
        'gamma_Az_joint': None if g is None else {'value': g, 'sd': gsd, 'interval_2sd': [g - 2 * gsd, g + 2 * gsd]},
        'detlist_R': P['detlist_R'], 'detlist_L': P['detlist_L'],
        'pk_window_len': None, 'tables_dir': str(out) + '/', 'r_c': None, 'I_SN_c': None,
    }
    if inputs['m_neb_sigma19']['QV'] is None:
        del inputs['m_neb_sigma19'], inputs['m_neb_sigma95']
    if inputs['gamma_Az_joint'] is None:
        del inputs['gamma_Az_joint']
    json.dump(inputs, open(out / 'inputs.json', 'w'), indent=1)
    print('tau_resid [ns]', np.round(tau_res * 1e9, 3), '| phi_s [rad]', np.round(f['phi_inst_s'], 4),
          '| m_QV', np.round(o['vleak']['mQV'], 4), '| m_UV', np.round(o['vleak']['mUV'], 4))
    print('tables written to', out, '| nebula file md5', md5(P['nebula_file']))


if __name__ == '__main__':
    main()
