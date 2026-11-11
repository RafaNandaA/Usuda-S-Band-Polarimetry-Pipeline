"""Shared statistics: seeds, bootstrap index matrices, percentile intervals, the a <-> ratio mapping, the robustness rule for
variants, and the result recorder (NUMBERS.json)."""
import json
import math
import numpy as np

B_DEFAULT = 20000           # bootstrap draws for every interval that decides a statement, so the 2.5 % ends are stable
B_ALIGNED_SLOPE_S0MP = 30000  # more draws for the aligned-fraction slope of the full MP sample, whose interval end lies close to 0
SEED_BASE = 2601            # every statistic has its own fixed seed, SEED_BASE + ITEM[name], so results are reproducible
ITEM = {
    'selection': 33, 'ellipticity': 51, 'zero_point': 53, 'census': 61, 'degree': 62, 'anisotropy': 63, 'duration': 64,
    'aligned': 65, 'direction': 66, 'lfip': 67, 'sign': 68, 'element': 69, 'energy': 611, 'fold': 91, 'halves': 46,
    'block': 47, 'nebula_gain': 54,
}


def rng_for(item):
    seed = SEED_BASE + ITEM[item]
    return np.random.default_rng(seed), seed


def boot_idx(n, B, rng):
    return rng.integers(0, n, size=(B, n))


def block_boot_idx(order, B, rng, block=50):
    """Time-block bootstrap: blocks of `block` consecutive pulses (order = indices sorted by time), for pulses that are not
    independent in time."""
    n = order.size
    nblk = int(math.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(B, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return order[idx]


def ci(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def a_to_ratio(a):
    a = np.asarray(a, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (1 - a) / a


def ratio_ci_from_a(a_draws):
    """CI on a = <v²>/(<l²>+<v²>), mapped to ratio = (1-a)/a (monotone decreasing)."""
    lo_a, hi_a = ci(a_draws)
    return [float(a_to_ratio(hi_a)), float(a_to_ratio(lo_a))]


def floor_to(x, step):
    return math.floor(x / step + 1e-12) * step


def ceil_to(x, step):
    return math.ceil(x / step - 1e-12) * step


def one_sig_fig_down(x):
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x))
    return math.floor(x / 10**e) * 10**e


# ---------------- robustness rule for variants ----------------
def definitional_variant(diff_draws, v_point, p_point, threshold_side, k_c):
    """Definitional variant: a statement becomes conditional iff (i) paired-bootstrap CI of variant - primary excludes 0 at level
    0.05/k_c AND (ii) the variant's point estimate lies on the other side of the decision threshold.
    threshold_side: function value -> branch label; (ii) <=> branch(v_point) != branch(p_point)."""
    a = 0.05 / max(k_c, 1)
    lo, hi = ci(diff_draws, 100 * a / 2, 100 * (1 - a / 2))
    excl = (lo > 0) or (hi < 0)
    other = threshold_side(v_point) != threshold_side(p_point)
    return {'diff_CI': [lo, hi], 'level': a, 'excludes0': bool(excl), 'other_side': bool(other),
            'conditional': bool(excl and other), 'variant_point': float(v_point)}


def subsample_variant(diff_draws, k_c):
    """Subsample variant: a statement becomes conditional iff the CI of (subsample - complement) excludes 0 at level 0.05/k_c."""
    a = 0.05 / max(k_c, 1)
    lo, hi = ci(diff_draws, 100 * a / 2, 100 * (1 - a / 2))
    excl = (lo > 0) or (hi < 0)
    return {'diff_CI': [lo, hi], 'level': a, 'conditional': bool(excl)}


def hetero_draws(stat, sub_mask, B, rng):
    """Bootstrap draws of stat(subsample) - stat(complement); stat takes an index array (into the sample)."""
    s = np.where(sub_mask)[0]
    c = np.where(~sub_mask)[0]
    out = np.empty(B)
    for b in range(B):
        out[b] = stat(s[rng.integers(0, s.size, s.size)]) - stat(c[rng.integers(0, c.size, c.size)])
    return out


# ---------------- recorder ----------------
class Recorder:
    def __init__(self, log):
        self.numbers = {}
        self.decisions = []
        self.log = log
        self.not_computed = []

    def put(self, key, value, **meta):
        self.numbers[key] = _clean(dict(value=value, **meta))

    def decide(self, item, rule, outcome, **detail):
        d = _clean(dict(item=item, rule=rule, outcome=outcome, **detail))
        self.decisions.append(d)
        self.log('DECISION %s: %s -> %s' % (item, rule, outcome))

    def nc(self, item, why):
        self.not_computed.append({'item': item, 'why': why})
        self.log('NOT COMPUTED %s: %s' % (item, why))

    def dump(self, path, header):
        json.dump(_clean({'header': header, 'numbers': self.numbers, 'decisions': self.decisions,
                          'not_computed': self.not_computed}), open(path, 'w'), indent=1)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else str(f)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def block_length(x, maxlag=None):
    """Block length = first lag at which the autocorrelation of the per-item series (time order) drops below 1/e or to
    <= 0 (whichever comes first), minimum 1."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    if n < 3 or np.allclose(x, 0):
        return 1
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    maxlag = n // 4 if maxlag is None else maxlag
    for k in range(1, maxlag):
        if ac[k] < 1 / np.e or ac[k] <= 0:
            return max(1, k)
    return maxlag
