"""Element bound (primary form, c_L = c_V = 0): L_el = (<l²> - X_up)/<v²> minimised over R inside every draw;
lower limit = 2.5th percentile via a_el = <v²>/(<l²> - X_up + <v²>). X_up = max(<Q'/I>², <(Q'/I)²> - <(U'/I)²>) with the
noise difference removed. The cross terms between pulse halves are set to zero."""
import numpy as np
from .stats import a_to_ratio, ci


def xup_parts(D, ix, psi0):
    I = D['I']
    zs = D['zs'][ix]
    Qp = np.real(zs * np.exp(-2j * psi0)) / I[ix]
    Up = np.imag(zs * np.exp(-2j * psi0)) / I[ix]
    th = 2 * (D['chi'][ix] + psi0)            # rotation feed -> aligned frame
    c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
    s2Qp = (D['s2Q'][ix] * c2 + D['s2U'][ix] * s2) / I[ix] ** 2
    s2Up = (D['s2Q'][ix] * s2 + D['s2U'][ix] * c2) / I[ix] ** 2
    diff = np.mean(Qp ** 2 - Up ** 2 - (s2Qp - s2Up))
    return max(np.mean(Qp) ** 2, diff), np.mean(Qp) ** 2, diff


def bound(D, idx, rr, rng, B=2000, rho2=None):
    psi = D['psi']
    n = idx.size

    def one(ix):
        p0 = 0.5 * np.angle(np.mean(np.exp(2j * psi[ix])))
        X, _, _ = xup_parts(D, ix, p0)
        s = {k: float(v) for k, v in rr.sums(ix).items()}
        if rho2 is not None and rho2 < 0:
            X = X + abs(rho2) * s['A']
        vmax = rr.vm(s, rr.dl).max()
        return X, s['A'], vmax
    X0, A0, V0 = one(idx)
    ab = np.empty(B)
    for b in range(B):
        X, A, V = one(idx[rng.integers(0, n, n)])
        ab[b] = V / (A - X + V)
    Lel_pt = (A0 - X0) / V0
    Lel = float(a_to_ratio(np.percentile(ab, 97.5)))
    _, qp2, diff = xup_parts(D, idx, D['psi0'])
    out = {'X_up': float(X0), 'meanQp_sq': float(qp2), 'Qp2_minus_Up2': float(diff), 'L_el_point_worst_m': float(Lel_pt),
           'L_el_lower': Lel, 'rho2_used': rho2}
    if Lel > 2:
        t = np.sqrt(1 / Lel)
        eps = 0.5 * np.degrees(np.arctan(t))
        out.update({'tan2eps_max': float(t), 'eps_max_deg': float(eps), 'V_over_P_max': float(np.sin(np.radians(2 * eps))),
                    'printed': '|tan 2eps| <~ %.2f (|eps| <~ %.1f deg, |V|/P <= %.2f)' % (np.ceil(t * 100) / 100, eps,
                                                                                         np.sin(np.radians(2 * eps))),
                    'group_size_relation': 'n >~ %.2f tan^2 2eps0' % Lel})
    else:
        out['printed'] = 'no bound stated (L_el <= 2: elements could be isotropic)'
    out['premise'] = ('conditional on independence of the elements at all separations (c_L = c_V = 0); between-half cross '
                      'terms not measured')
    return out
