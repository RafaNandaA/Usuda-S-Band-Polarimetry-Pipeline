"""Ionospheric Faraday rotation RM_ion(t) along the line of sight, from a global ionosphere map (IONEX) and IGRF.

Thin-shell model at 450 km (the IONEX shell height), slant factor 1/cos(z') at the pierce point, IONEX maps interpolated in
time with the sun-fixed longitude rotation (15 deg/h), IGRF field at the pierce point.
RM [rad m^-2] = 2.631e-13 STEC [m^-2] B_par [T], with B_par along the propagation (source to observer), so RM > 0 when the
field points toward the observer (IAU sign). Typical GIM errors are 0.1-0.3 rad m^-2 (Sotomayor-Beltran et al. 2013).
"""
import datetime
import gzip

import numpy as np

RE, H_SHELL = 6371.0, 450.0
LATS = np.arange(87.5, -87.5 - 1e-9, -2.5)
LONS = np.arange(-180, 180 + 1e-9, 5.0)


def read_ionex(path):
    """TEC maps (TECU) and their epochs (s of day) from a gzipped IONEX file."""
    txt = gzip.open(path, 'rt').read().split('\n')
    maps = {}
    i = 0
    while i < len(txt):
        if 'START OF TEC MAP' in txt[i]:
            k = int(txt[i].split()[0]); i += 1
            ep = [int(x) for x in txt[i].split()[:6]]; i += 1
            M = np.zeros((len(LATS), len(LONS)))
            while 'END OF TEC MAP' not in txt[i]:
                if 'LAT/LON1/LON2/DLON/H' in txt[i]:
                    la = float(txt[i][:8]); j = int(round((87.5 - la) / 2.5)); i += 1
                    vals = []
                    while len(vals) < len(LONS):
                        vals += [int(x) for x in txt[i].split()]; i += 1
                    M[j] = np.array(vals[:len(LONS)]) * 0.1
                    continue
                i += 1
            maps[k] = (ep[3] * 3600.0, M)
        i += 1
    tk = np.array([maps[k][0] for k in sorted(maps)])
    if tk[-1] == 0:
        tk[-1] = 86400.0          # the last map of the day is stamped 00:00 of the next day
    return tk, np.array([maps[k][1] for k in sorted(maps)])


def _bilinear(M, la, lo):
    lo = (lo + 180.0) % 360.0 - 180.0
    fi = (87.5 - la) / 2.5; fj = (lo + 180.0) / 5.0
    i0 = int(np.clip(np.floor(fi), 0, len(LATS) - 2)); j0 = int(np.floor(fj)) % (len(LONS) - 1); j1 = j0 + 1
    a = fi - i0; b = fj - np.floor(fj)
    return (1 - a) * ((1 - b) * M[i0, j0] + b * M[i0, j1]) + a * ((1 - b) * M[i0 + 1, j0] + b * M[i0 + 1, j1])


def _vtec(tk, Ms, t, la, lo):
    k = int(np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2)); w = (t - tk[k]) / (tk[k + 1] - tk[k])
    e1 = _bilinear(Ms[k], la, lo + (t - tk[k]) * 15.0 / 3600.0)
    e2 = _bilinear(Ms[k + 1], la, lo + (t - tk[k + 1]) * 15.0 / 3600.0)
    return (1 - w) * e1 + w * e2


def _enu(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    e = np.array([-np.sin(lo), np.cos(lo), 0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return e, n, u


def rm_ion(tk, Ms, date, lat, lon, t, az, el):
    """RM_ion (rad m^-2) toward (az, el) at time t (s of day, UTC) from the site (lat, lon)."""
    import ppigrf
    z = np.radians(90 - el); zp = np.arcsin(RE / (RE + H_SHELL) * np.sin(z)); psi = z - zp
    la0, lo0, A = np.radians(lat), np.radians(lon), np.radians(az)
    lap = np.arcsin(np.sin(la0) * np.cos(psi) + np.cos(la0) * np.sin(psi) * np.cos(A))
    lop = lo0 + np.arctan2(np.sin(A) * np.sin(psi) * np.cos(la0), np.cos(psi) - np.sin(la0) * np.sin(lap))
    lap_d, lop_d = np.degrees(lap), np.degrees(lop)
    v = _vtec(tk, Ms, t, lap_d, lop_d); stec = v / np.cos(zp)
    e, n, u = _enu(lat, lon)
    s = np.cos(np.radians(el)) * (np.sin(A) * e + np.cos(A) * n) + np.sin(np.radians(el)) * u      # toward the source
    when = datetime.datetime(date.year, date.month, date.day, int(t // 3600), int((t % 3600) // 60))
    Be, Bn, Bu = [float(x[0]) for x in ppigrf.igrf(lop_d, lap_d, H_SHELL, when)]
    e2, n2, u2 = _enu(lap_d, lop_d)
    Bpar = -np.dot((Be * e2 + Bn * n2 + Bu * u2) * 1e-9, s)
    return 2.631e-13 * stec * 1e16 * Bpar, v, stec, Bpar * 1e9, lap_d, lop_d


def track(gim_file, date, lat, lon, sod, az, el, t0=28800.0, t1=46800.0, dt=60.0):
    """RM_ion on a 60-s grid over the track, with az and el interpolated from the snapshot geometry.
    Returns an (n, 9) array: sod, az, el, VTEC, STEC, B_par (nT), pierce lat, pierce lon, RM_ion."""
    tk, Ms = read_ionex(gim_file)
    o = np.argsort(sod)
    grid = np.arange(t0, t1 + 0.1, dt)
    azg = np.interp(grid, sod[o], az[o]); elg = np.interp(grid, sod[o], el[o])
    rows = []
    for t, a, e in zip(grid, azg, elg):
        r, v, st, Bp, la, lo = rm_ion(tk, Ms, date, lat, lon, t, a, e)
        rows.append((t, a, e, v, st, Bp, la, lo, r))
    return np.array(rows)


def write_track(path, R):
    """Full track, written with six significant digits (well below the GIM error)."""
    hdr = 'RM_ion(t), IAU sign, thin shell 450 km, IGRF at the pierce point\nsod,az_deg,el_deg,vtec_tecu,stec_tecu,bpar_nT,ipp_lat,ipp_lon,rm_ion'
    np.savetxt(path, R, delimiter=',', header=hdr, comments='', fmt='%.6g')


def read_track(path):
    T = np.loadtxt(path, delimiter=',', skiprows=2)
    return T[:, 0], T[:, 8]
