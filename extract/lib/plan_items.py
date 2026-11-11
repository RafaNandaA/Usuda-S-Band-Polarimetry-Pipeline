"""Item list for the extract stage (no raw-data read).

    python3 extract/lib/plan_items.py <config.toml>        (normally run by: julia extract/extract.jl <config> plan)

Writes <work_dir>/items.tsv, fold_blocks.tsv (60-s fold blocks) and timing_model.json. items.tsv has one row per read item: item, kind, offset_bytes, length_bytes, block, meta (k=v;...).
  catwin    one 20-ms window per listed pulse kept by the S/N cut (catalogue row = rank of Nseq)
  foldrot   one fold rotation: three 7.5-ms windows (main pulse, off-pulse, interpulse) in one read
The nebula snapshots are planned by extract/lib/nebula.jl.
The timing model is fitted to the detection lists themselves (rotation number + phase against barycentric time), so no
ephemeris is needed.
"""
import glob, os, re, sys, tomllib
import numpy as np

FS = 64e6; WB = 4                     # samples per second, bytes per word
K = 4.148808e3                        # dispersion constant, s MHz^2 / (pc cm^-3)
SOD_X = 28799.9686                    # file clock of the extractor: list time minus 31.4 ms
LWIN = 1_280_000                      # 20-ms catalogue window
T0 = 37691.25                         # reference time of the timing polynomial (s of day, TDB)


def load_config(path):
    cfg = tomllib.load(open(path, 'rb'))
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg.get('paths', {}).items():
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg['paths'][k] = os.path.normpath(os.path.join(base, v))
    return cfg


BAD = re.compile(r'^([+-]?\d*\.?\d+)[eE]([+-])(\d{3,})$')
def tok(s):
    """list phases sometimes carry a three-digit exponent, read them as two digits"""
    m = BAD.match(s)
    return float(f"{m.group(1)}e{m.group(2)}{m.group(3)[:2]}") if m else float(s)


def read_lists(files):
    rows = []
    for f in files:
        for ln in open(f):
            if ln.startswith('#') or not ln.strip():
                continue
            p = ln.split()
            if len(p) < 6:
                continue
            rows.append((float(p[1]), float(p[2]), int(float(p[3])), tok(p[4]), float(p[5]), p))
    return rows


def main(cfgpath):
    cfg = load_config(cfgpath); P = cfg['paths']
    files = sorted([P['detlist_R'], P['detlist_L']])
    DM = float(cfg['dedispersion']['dm']); FREF = float(cfg['dedispersion']['ref_freq_MHz'])
    FSUB = [float(x) for x in cfg['observation']['subband_lower_MHz']]; FLO = FSUB[0]
    C0 = float(cfg['timing']['arrival_offset_ms'])
    rows = read_lists(files)
    lutc = np.array([r[0] for r in rows]); ltdb = np.array([r[1] for r in rows]); lnseq = np.array([r[2] for r in rows])
    lph = np.array([r[3] for r in rows])

    # ---- timing model: rotation number + phase against TDB, quadratic with 5-sigma clipping, and UTC -> TDB as a polynomial
    x = ltdb - T0; PHI = lnseq + lph; keep = np.ones(len(x), bool)
    for _ in range(8):
        c = np.polyfit(x[keep], PHI[keep], 2); r = PHI - np.polyval(c, x)
        med = np.median(r[keep]); s = 1.4826 * np.median(np.abs(r[keep] - med)) or 1e-12
        inside = np.abs(r - med) < 5 * s
        if (inside == keep).all():
            break
        keep = inside
    c2, c1, c0 = c; Prot = 1.0 / c1
    um = lutc.mean(); cu = np.polyfit(lutc - um, ltdb - lutc, 5)

    def tdb_at_phase(PH):
        xx = (PH - c0) / c1
        for _ in range(4):
            xx = xx - (c2 * xx * xx + c1 * xx + c0 - PH) / (2 * c2 * xx + c1)
        return T0 + xx

    def utc_from_tdb(tdb):
        u = tdb - np.polyval(cu, tdb - um)
        for _ in range(3):
            u = tdb - np.polyval(cu, u - um)
        return u
    t_utc = lambda PH: utc_from_tdb(tdb_at_phase(np.asarray(PH, float)))

    items = []
    # ---- catalogue windows: lists merged by Nseq, larger S/N of the two hands, kept at S/N >= min_det_snr
    cut = float(cfg['selection']['min_det_snr']); det = {}
    for f in files:
        for r in read_lists([f]):
            sod, nseq, phase, snr = r[0], r[2], float(r[5][4]), r[4]
            if snr < cut:
                continue
            if nseq not in det or snr > det[nseq][2]:
                det[nseq] = (sod, phase, snr)
    for i, n in enumerate(sorted(det)):
        sod, phase, snr = det[n]
        w0 = int(np.rint((sod - SOD_X) * FS))
        items.append((f'cat:{i}', 'catwin', w0 * WB, LWIN * WB, -1, f'cat={i};nseq={n};sod={sod!r};phase={phase!r};detsnr={snr!r}'))

    # ---- fold rotations: 60-s blocks over the list span, none across the gain step, rotations in proportion to length
    F = cfg['fold']; lo, hi = float(lutc.min()), float(lutc.max()); GAP = tuple(F['gain_step_gap_s']); BS = float(F['block_s'])
    edges = sorted(set([lo, hi] + [GAP[0] - BS * k for k in range(0, 10) if GAP[0] - BS * k > lo] +
                       [GAP[1] + BS * k for k in range(0, 400) if GAP[1] + BS * k < hi]))
    blocks = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if not (a == GAP[0] and b == GAP[1])]
    dur = np.array([b - a for a, b in blocks]); NROT = int(F['n_rotations'])
    nb_rot = np.floor(NROT * dur / dur.sum()).astype(int)
    for i in np.argsort(-(NROT * dur / dur.sum() - nb_rot))[:NROT - nb_rot.sum()]:
        nb_rot[i] += 1
    rots, rblk = [], []
    for bi, ((a, b), n) in enumerate(zip(blocks, nb_rot)):
        mid = 0.5 * (a + b); tdbm = mid + np.polyval(cu, mid - um)
        n0 = int(np.floor(c2 * (tdbm - T0)**2 + c1 * (tdbm - T0) + c0)) - n // 2
        rr = np.arange(n0, n0 + n); tr = t_utc(rr.astype(float))
        assert tr.min() >= a and tr.max() + Prot <= b, (bi, a, b)
        rots += rr.tolist(); rblk += [bi] * n
    rots = np.array(rots); rblk = np.array(rblk)
    listed = set(np.rint(lnseq + lph).astype(np.int64).tolist())        # a listed pulse belongs to rotation round(Nseq + Phase)
    is_listed = np.array([n in listed for n in rots])
    su = np.sort(lutc); g = np.where(np.diff(su) > float(F['list_gap_s']))[0]; lgaps = [(float(su[i]), float(su[i + 1])) for i in g]
    tr_all = t_utc(rots.astype(float))
    in_lgap = np.zeros(len(rots), bool)
    for a, b in lgaps:
        in_lgap |= (tr_all > a) & (tr_all < b)
    SP = cfg['flags']['interference_interval_s']
    in_spiky = (tr_all + 0.0436 > SP[0]) & (tr_all + 0.0436 < SP[1])
    phi_MP = (float(F['mp_centre_ms']) - C0) * 1e-3 / Prot
    phi = dict(MP=phi_MP, OFF=phi_MP + float(F['off_offset_turn']), LFIP=float(F['lfip_phase_turn']))
    WREAD = int(round(7.5e-3 * FS)); PRE = int(round(1.10e-3 * FS)); C0W = int(round(C0 * 1e-3 * FS))
    assert (WREAD - PRE - int(1.0e-3 * FS)) / FS >= K * (DM + 0.1) * (1.0 / FLO**2 - 1.0 / FREF**2)   # tail margin covers the sweep
    # utc_rot (rotation time, list clock) is stored to 1 us
    for rot, blk, tr0, li, gp, sk in zip(rots, rblk, tr_all, is_listed, in_lgap, in_spiky):
        s = {cmp: int(np.rint((float(t_utc(np.array([rot + ph]))[0]) - SOD_X) * FS)) + C0W for cmp, ph in phi.items()}
        st = {cmp: s[cmp] - PRE for cmp in s}
        w0 = min(st.values()); w1 = max(v + WREAD for v in st.values())
        meta = ';'.join([f'rot={rot}', 'starts=' + ','.join(str(st[cmp] - w0 + 1) for cmp in ('MP', 'OFF', 'LFIP')),
                         'sX=' + ','.join(str(s[cmp]) for cmp in ('MP', 'OFF', 'LFIP')),
                         f'listed={int(li)}', f'in_list_gap={int(gp)}', f'in_spiky={int(sk)}', f'utc_rot={float("%.6f" % tr0)!r}'])
        items.append((f'rot:{rot}', 'foldrot', w0 * WB, (w1 - w0) * WB, int(blk), meta))

    raw = P.get('raw_file', '')
    if raw and os.path.isfile(raw):
        size = os.path.getsize(raw)
        bad = [it[0] for it in items if it[2] < 0 or it[2] + it[3] > size]
        if bad:
            print('items outside the raw file (dropped):', len(bad)); items = [it for it in items if it[0] not in set(bad)]
    items.sort(key=lambda it: (it[2], it[0]))
    os.makedirs(P['work_dir'], exist_ok=True)
    out = os.path.join(P['work_dir'], 'items.tsv')
    with open(out, 'w') as fh:
        fh.write('item\tkind\toffset_bytes\tlength_bytes\tblock\tmeta\n')
        for it in items:
            fh.write('\t'.join(map(str, it)) + '\n')
    # fold block grid and timing model, used by fold/fold.py to place each catalogue pulse in its block
    with open(os.path.join(P['work_dir'], 'fold_blocks.tsv'), 'w') as fh:
        fh.write('block\tstart_list_utc\tend_list_utc\tn_rot\n')
        for bi, ((a, b), n) in enumerate(zip(blocks, nb_rot)):
            fh.write('%d\t%.3f\t%.3f\t%d\n' % (bi, a, b, n))
    with open(os.path.join(P['work_dir'], 'timing_model.json'), 'w') as fh:
        fh.write('{"T0_tdb_sod": %r, "F0_Hz": %r, "F1_Hz_per_s": %r, "period_us": %r}\n' % (float(T0), float(c1), float(2 * c2), float(1e6 / c1)))
    nk = {k: sum(1 for it in items if it[1] == k) for k in ('catwin', 'foldrot')}
    print(f'{out}: {nk}; timing model F0 = {c1:.12f} Hz, F1 = {2 * c2:.4e} Hz/s, {int(keep.sum())} of {len(x)} list rows used')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python3 plan_items.py <config.toml>')
    main(sys.argv[1])
