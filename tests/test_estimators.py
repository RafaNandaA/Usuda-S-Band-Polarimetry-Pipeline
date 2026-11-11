"""Unit tests of the estimators with known answers on synthetic inputs. No data are needed.

    python3 tests/test_estimators.py

Seeds are fixed (7001+), so every run gives the same numbers."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analyze'))
from usudapol import errors as ER, direction as DR, aligned as AL, ellipticity as EP, duration as DU, sign as SG
from usudapol import anisotropy as AN, selection as SE, stats as S

RES = []


def check(name, ok, detail):
    RES.append((name, bool(ok), detail))
    print('%-4s %-60s %s' % ('PASS' if ok else 'FAIL', name, detail))


def iso_pop(rng, n, sig, d=0.5):
    I = rng.lognormal(3, 0.6, n)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    Q, U, V = (d * I[:, None] * u).T
    s = np.full(n, sig)
    return I, Q + rng.normal(0, sig, n), U + rng.normal(0, sig, n), V + rng.normal(0, sig, n), s


def t_isotropic_ratio():
    rng = np.random.default_rng(7001)
    I, Q, U, V, s = iso_pop(rng, 200000, 3.0)
    l2, v2 = ER.moments(I, Q, U, V, s ** 2, s ** 2, s ** 2)
    r = l2.mean() / v2.mean()
    check('isotropic population -> ratio 2 (2σ² convention)', abs(r - 2) < 0.05, 'ratio %.3f' % r)
    l2b = (Q ** 2 + U ** 2 - s ** 2) / I ** 2          # 1σ² debias of l² (catalogue L convention) is biased
    rb = l2b.mean() / v2.mean()
    check('debias convention: 1σ² on l² departs from 2 (known bias)', rb > 2.05, 'ratio %.3f' % rb)


def t_uniform_R2():
    rng = np.random.default_rng(7002)
    n = 300
    R2 = np.array([DR.Ru_psi0(rng.uniform(-np.pi / 2, np.pi / 2, n))[0] ** 2 for _ in range(4000)])
    check('uniform angles -> E[R_u²] = 1/n', abs(R2.mean() * n - 1) < 0.05, 'n E[R²] = %.3f' % (R2.mean() * n))


def planted_direction(rng, n=4000, psi0=np.radians(30), f=0.3, feed_fixed=False):
    chi = np.radians(rng.uniform(-60, 60, n))
    al = rng.random(n) < f
    psi = np.where(al, psi0 + rng.normal(0, np.radians(10), n), rng.uniform(-np.pi / 2, np.pi / 2, n))
    L = rng.lognormal(0, 0.5, n)
    PA = psi + (0 if feed_fixed else chi)
    Q, U = L * np.cos(2 * PA), L * np.sin(2 * PA)
    return Q, U, chi, L


def t_planted_direction():
    rng = np.random.default_rng(7003)
    Q, U, chi, L = planted_direction(rng)
    zs, psi, p0, Qp, Up = ER.sky_frame(Q, U, chi, np.ones(Q.size, bool))
    R, _ = DR.Ru_psi0(psi)
    exp = 0.3 * np.exp(-2 * np.radians(10) ** 2)
    check('planted preferred direction recovered (ψ0 = 30°)', abs(np.degrees(p0) - 30) < 3 and abs(R - exp) < 0.03,
          'ψ0 %.1f°, R_u %.3f (expected %.3f)' % (np.degrees(p0), R, exp))
    r = DR.rate_fit(L, 0.5 * np.angle(Q + 1j * U), chi)
    check('rate test: sky-fixed direction -> r ≈ 1', abs(r - 1) < 0.1, 'r = %.3f' % r)
    Q2, U2, chi2, L2 = planted_direction(rng, feed_fixed=True)
    r2 = DR.rate_fit(L2, 0.5 * np.angle(Q2 + 1j * U2), chi2)
    check('rate test: feed-fixed direction -> r ≈ 0', abs(r2) < 0.1, 'r = %.3f' % r2)


def t_aligned_loo():
    rng = np.random.default_rng(7004)
    n = 3000
    I = np.ones(n)
    zs = 0.3 * np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    ix = np.arange(n)
    m = AL.qp_over_i(zs, I, ix, ix).mean(); ml = AL.loo(zs, I, ix).mean()
    check('aligned fraction, uniform angles: LOO ≈ 0, plain mean > 0 (self-selection)', abs(ml) < 0.012 and m > ml,
          'mean %.4f, LOO %.4f (expected plain ≈ 0.3·√(π/4n) = %.4f)' % (m, ml, 0.3 * np.sqrt(np.pi / (4 * n))))
    zs2 = 0.3 * np.exp(1j * np.where(rng.random(n) < 0.4, 0.2, rng.uniform(0, 2 * np.pi, n)))
    ml2 = AL.loo(zs2, I, ix).mean()
    check('aligned fraction, 40% at one angle: <Q\'/I> = 0.3·0.4 = 0.12', abs(ml2 - 0.12) < 0.012, 'LOO %.4f' % ml2)


def t_eiv():
    rng = np.random.default_rng(7005)
    n = 200000   # band 0.005 sized from the bootstrap SE at this n (≈ 0.0015 per slope with 4 partly collinear regressors)
    I = np.full(n, 100.0)
    q0 = rng.normal(0, 0.25, n); u0 = rng.normal(0, 0.25, n)
    chi = np.radians(rng.uniform(-60, 60, n))
    mq, mu = 0.05, -0.07
    v0 = mq * q0 + mu * u0 + rng.normal(0, 0.1, n)
    s = 15.0
    Q = I * q0 + rng.normal(0, s, n); U = I * u0 + rng.normal(0, s, n); V = I * v0 + rng.normal(0, s, n)
    z = np.zeros(n)
    X, v = EP.design(I, Q, U, V, chi, 0.0)
    Sm, Suv = EP.err_cov(I, Q, U, V, z, np.full(n, s ** 2), np.full(n, s ** 2), np.full(n, s ** 2), chi, 0.0)
    b = EP.eiv_fit(X, v, Sm, Suv); bo = EP.ols_fit(X, v)
    check('EIV regression recovers planted (m_Q, m_U) = (0.05, -0.07); OLS attenuated',
          abs(b[1] - mq) < 0.005 and abs(b[2] - mu) < 0.005 and abs(bo[2]) < abs(mu) - 0.01,
          'EIV (%.4f, %.4f), OLS (%.4f, %.4f), λ = %.2f' % (b[1], b[2], bo[1], bo[2], 0.0625 / (0.0625 + 0.0225)))


def t_glm():
    rng = np.random.default_rng(7006)
    n = 5000
    W = rng.lognormal(1.5, 0.6, n); Bc = rng.lognormal(1, 0.8, n)
    mu = np.exp(-1 - 0.8 * np.log(W) + 0.3 * np.log(Bc))
    y = mu + rng.normal(0, 0.02, n)      # additive noise: negative y allowed
    X = np.c_[np.ones(n), np.log(W), np.log(Bc)]
    b = DU.glm_log(X, y)
    check('log-link quasi-score GLM recovers β_W = -0.8, β_B = 0.3', abs(b[1] + 0.8) < 0.05 and abs(b[2] - 0.3) < 0.03,
          'β = %s, %.1f%% of y < 0' % (np.round(b, 3), 100 * np.mean(y < 0)))


def t_region_closed_form():
    rng = np.random.default_rng(7007)
    n = 500
    D = {'l2': rng.random(n) * 0.2, 'q': rng.normal(0, .3, n), 'u': rng.normal(0, .3, n), 'v': rng.normal(0, .1, n)}
    sv = 0.001
    D['v2'] = D['v'] ** 2 - sv
    m_app = np.array([0.02, -0.06]); m = np.array([0.1, 0.05])
    rr = AN.RegionRatio(D, m_app, np.array([m]), [])
    s = {k: float(x) for k, x in rr.sums(np.arange(n)).items()}
    vm_direct = np.mean((D['v'] + (m_app[0] - m[0]) * D['q'] + (m_app[1] - m[1]) * D['u']) ** 2 - sv)
    vm_cf = rr.vm(s, m - m_app)
    check('ellipticity offset: closed-form <v_m²> = direct recomputation', abs(vm_cf - vm_direct) < 1e-12,
          'diff %.2e' % (vm_cf - vm_direct))


def t_sigma_full_and_selfnoise():
    """Monte Carlo of dual-circular complex voltages: Var(Q̂_int) = σ²_full + σ²_self (n_b -> ∞) (van Straten 2009, Eq. 28)."""
    rng = np.random.default_rng(7008)
    mu, W, ntr = 1.0, 8, 40000
    P = np.array([0.0, 0.5, 1.5, 3.0, 2.0, 1.0, 0.4, 0.1])
    pol = np.array([0.4, 0.2, 0.3])   # q, u, v fractions of the pulse (degree 0.54)
    Ip = mu + P; Qp, Up, Vp = (P[:, None] * pol).T
    # coherency: RR = (I+V)/2, LL = (I-V)/2, <R L*> = (Q + iU)/2
    Qs = np.empty(ntr)
    for k in range(W):
        C = np.array([[(Ip[k] + Vp[k]) / 2, (Qp[k] + 1j * Up[k]) / 2], [(Qp[k] - 1j * Up[k]) / 2, (Ip[k] - Vp[k]) / 2]])
        Lc = np.linalg.cholesky(C)
        x = (rng.normal(size=(ntr, 2)) + 1j * rng.normal(size=(ntr, 2))) / np.sqrt(2)
        RL = x @ Lc.T
        Qs = Qs + 2 * np.real(RL[:, 0] * np.conj(RL[:, 1])) if k else 2 * np.real(RL[:, 0] * np.conj(RL[:, 1]))
    var_mc = Qs.var()
    s2 = 0.5 * mu ** 2
    full = ER.sigma2_full_band(np.sqrt(s2), W, 1e12, P.sum(), mu, True)
    self_ = np.sum(s2 * (P ** 2 + Qp ** 2 - Up ** 2 - Vp ** 2) / mu ** 2)
    check('σ²_full + σ²_self matches Monte Carlo Var(Q̂_int)', abs(var_mc / (full + self_) - 1) < 0.03,
          'MC %.3f vs model %.3f (off %.2f, cross %.2f, self %.2f)' % (var_mc, full + self_, W * s2, full - W * s2, self_))
    check('omitting the cross term underestimates (catalogue σ)', var_mc / (W * s2 + self_) > 1.3,
          'ratio %.2f' % (var_mc / (W * s2 + self_)))


def t_self_noise_estimator():
    """Per-cell self-noise estimator (κ known = true S_p²/P²) is unbiased for σ²_self of Q."""
    rng = np.random.default_rng(7009)
    nc, nt, ntr = 16, 6, 3000
    mu = np.ones(nc)
    P = rng.uniform(0.5, 3, (nc, nt)); pol = np.array([0.5, 0.1, 0.2])
    d2 = np.sum(pol ** 2); kappa = 1 - d2
    est = []
    for _ in range(ntr):
        Pm = P + rng.normal(0, np.sqrt(0.5 * (1 + P) ** 2 * 0 + 0.5 * (mu[:, None] ** 2 + 2 * mu[:, None] * P + P ** 2 * (1 + d2) / 1)), P.shape) * 0
        # draw cell Stokes from voltages
        Ip = mu[:, None] + P; Qp, Up, Vp = (P[..., None] * pol).transpose(2, 0, 1)
        RR = (Ip + Vp) / 2; LL = (Ip - Vp) / 2; RLc = (Qp + 1j * Up) / 2
        a = np.sqrt(RR); b = RLc.conj() / a; c = np.sqrt(np.clip(LL - np.abs(b) ** 2, 0, None))
        x = (rng.normal(size=(2, nc, nt)) + 1j * rng.normal(size=(2, nc, nt))) / np.sqrt(2)
        R = a * x[0]; L = b * x[0] + c * x[1]
        Ih = np.abs(R) ** 2 + np.abs(L) ** 2 - mu[:, None]; Qh = 2 * np.real(R * np.conj(L))
        est.append(ER.self_noise_cells(Ih, Qh, mu, 0.5 * mu ** 2, 0.5 * mu ** 2, kappa))
    true = np.sum(0.5 * (P ** 2 + (P * pol[0]) ** 2 - (P * pol[1]) ** 2 - (P * pol[2]) ** 2))
    m = np.mean(est); se = np.std(est) / np.sqrt(ntr)
    check('self-noise cell estimator unbiased (κ = 1 - d², true)', abs(m - true) < 4 * se, 'mean %.2f ± %.2f vs true %.2f' % (m, se, true))


def t_sign():
    rng = np.random.default_rng(7010)
    n = 4000
    V = np.where(rng.random(n) < 0.6, -1, 1) * rng.uniform(5, 10, n)
    f, k, z, _ = SG.f_test(V, np.ones(n), np.arange(n), 3)
    check('sign f_test recovers planted f = 0.6', abs(f - 0.6) < 0.02 and z > 10, 'f %.3f z %.1f' % (f, z))


def t_completeness():
    rng = np.random.default_rng(7011)
    n = 20000
    isn = 7 * (1 + rng.pareto(1.6, n))
    I = isn * 10.0
    v = np.clip(rng.normal(0, 0.15, n), -1, 1)
    s2V = np.full(n, 25.0); s2I = s2V.copy()
    vi = v + rng.normal(0, 5 / I)
    r = SE.completeness_tests(isn, 7.0, vi, I, s2V, s2I, rng)
    check('completeness tests pass on a complete synthetic sample', r['pass'], 'KS p %.3f, central p %.3f' % (r['B1vB2']['KS_p'], r['B1vB2']['central_p']))
    keep = ~((isn < 14) & (np.abs(v) < 0.1) & (rng.random(n) < 0.3))
    r2 = SE.completeness_tests(isn[keep], 7.0, vi[keep], I[keep], s2V[keep], s2I[keep], rng)
    check('central-fraction test fails on a planted B1 central deficit', not r2['pass'] and r2['B1_central_sign'] == 'deficit',
          'central p %.2g' % r2['B1vB2']['central_p'])


def t_element_ratio():
    """Independent elements of fixed ellipticity ε0, random PA and V sign: ratio = cot² 2ε0 for any N."""
    rng = np.random.default_rng(7012)
    eps0 = np.radians(12); target = 1 / np.tan(2 * eps0) ** 2
    out = []
    for N in (3, 20):
        n = 40000
        Ik = rng.exponential(1, (n, N))
        ph = rng.uniform(0, 2 * np.pi, (n, N)); sg = rng.choice([-1, 1], (n, N))
        Q = (Ik * np.cos(2 * eps0) * np.cos(ph)).sum(1); U = (Ik * np.cos(2 * eps0) * np.sin(ph)).sum(1)
        V = (Ik * sg * np.sin(2 * eps0)).sum(1); I = Ik.sum(1)
        out.append(np.mean((Q ** 2 + U ** 2) / I ** 2) / np.mean(V ** 2 / I ** 2))
    check('element ratio = cot² 2ε0 independent of N', all(abs(o / target - 1) < 0.05 for o in out),
          'N=3: %.2f, N=20: %.2f, target %.2f' % (out[0], out[1], target))


def t_amap_and_block():
    a = np.array([0.1, 0.2, 0.3])
    check('a -> ratio mapping (1-a)/a', np.allclose(S.a_to_ratio(a), [9, 4, 7 / 3]), str(S.a_to_ratio(a)))
    rng = np.random.default_rng(7013)
    o = np.arange(1000)
    bi = S.block_boot_idx(o, 10, rng, 50)
    contig = np.all(np.diff(bi[:, :50], axis=1) == 1)
    check('block bootstrap: shape and contiguous blocks of 50', bi.shape == (10, 1000) and contig, str(bi.shape))


def t_degree():
    from usudapol import degree as DG
    rng = np.random.default_rng(7014)
    n = 2000
    I = np.full(n, 1000.0); sig = 10.0
    out = {}
    for d in (1.0 - DG.EPS_INST, 0.5):
        u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        Q, U, V = (d * I[:, None] * u).T + rng.normal(0, sig, (3, n))
        s2 = np.full(n, sig ** 2)
        D = {'I': I, 'Q': Q, 'U': U, 'V': V, 's2Q': s2, 's2U': s2, 's2V': s2, 's2I': s2, 'snr_int': I / sig}
        D['l2'], D['v2'] = ER.moments(I, Q, U, V, s2, s2, s2)
        out[d] = DG.run(D, np.arange(n), rng, 500)['SN>=20']
    a, b = out[1.0 - DG.EPS_INST], out[0.5]
    check('degree: single-state pulses -> F_=1 high; d = 0.5 -> F_<1 = 1', a['F_eq1'] > 0.9 and b['F_lt1'] > 0.99,
          'F_=1 %.3f (single state), F_<1 %.3f (d = 0.5)' % (a['F_eq1'], b['F_lt1']))


def t_energy():
    from usudapol import energy as EN
    rng = np.random.default_rng(7015)
    E = 10 * (1 - rng.random(3000)) ** (-1 / 1.6)
    a, D, n, em = EN.fit(E)
    # GoF power: a lognormal far tail can pass at this n (Clauset+09 §5), so the rejection check uses an exponential tail
    E2 = 1 + rng.exponential(10, 2100)
    r = EN.run(E2, rng.lognormal(1, 0.5, E2.size), rng, 0, B=20, ngof=200)
    check('energy: MLE recovers α = 2.6; GoF rejects an exponential', abs(a - 2.6) < 0.08 and not r['gof_pass'],
          'α %.3f (E_min %.1f); exponential gof p %.3f' % (a, em, r['gof_p']))


def t_block_length():
    rng = np.random.default_rng(7016)
    e = rng.normal(size=20000)
    for i in range(1, e.size):
        e[i] += 0.9 * e[i - 1]
    L = S.block_length(e); L0 = S.block_length(rng.normal(size=20000))
    check('block length: AR(0.9) -> 1/e lag ≈ 9.5; white -> 1', L in (9, 10, 11) and L0 == 1, 'AR %d, white %d' % (L, L0))


if __name__ == '__main__':
    for t in (t_isotropic_ratio, t_uniform_R2, t_planted_direction, t_aligned_loo, t_eiv, t_glm, t_region_closed_form,
              t_sigma_full_and_selfnoise, t_self_noise_estimator, t_sign, t_completeness, t_element_ratio, t_amap_and_block, t_degree, t_energy, t_block_length):
        t()
    nf = sum(1 for r in RES if not r[1])
    print('\n%d tests, %d failed' % (len(RES), nf))
    sys.exit(1 if nf else 0)
