"""
stats.py — Statistical inference over the experimental campaign.

The design uses COMMON RANDOM NUMBERS: every policy runs on identical
realizations within each seed.  The pertinent statistic is therefore the
per-seed PAIRED DIFFERENCE, whose variance is far below that of independent
means.  Percentile bootstrap confidence intervals on the paired difference,
the Wilcoxon signed-rank test (which requires no normality), Cohen's effect
size for paired samples, and the probability of superiority are reported.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def paired_summary(x: np.ndarray, y: np.ndarray, *, n_boot: int = 20000,
                   alpha: float = 0.05, seed: int = 20260726) -> dict:
    """Compare policy `x` against reference `y`, paired by seed.

    Returns means, absolute and relative mean difference, bootstrap confidence
    interval, Wilcoxon test, effect size, and probability of superiority
    (fraction of seeds with x < y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("las muestras pareadas deben tener igual longitud")
    d = x - y
    n = d.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    if np.allclose(d, 0.0):
        p_w = 1.0
    else:
        try:
            p_w = float(sps.wilcoxon(x, y, zero_method="zsplit").pvalue)
        except ValueError:
            p_w = 1.0
    sd = d.std(ddof=1) if n > 1 else 0.0
    base = y.mean()
    return dict(
        n=n,
        mean_x=float(x.mean()), sd_x=float(x.std(ddof=1) if n > 1 else 0.0),
        mean_y=float(base), sd_y=float(y.std(ddof=1) if n > 1 else 0.0),
        diff_mean=float(d.mean()),
        diff_ci_low=float(lo), diff_ci_high=float(hi),
        diff_rel_pct=float(100.0 * d.mean() / base) if base != 0 else float("nan"),
        diff_rel_ci_low_pct=float(100.0 * lo / base) if base != 0 else float("nan"),
        diff_rel_ci_high_pct=float(100.0 * hi / base) if base != 0 else float("nan"),
        p_wilcoxon=p_w,
        cohen_dz=float(d.mean() / sd) if sd > 0 else float("nan"),
        prob_superiority=float(np.mean(d < 0)),
        significant=bool((lo > 0) or (hi < 0)),
    )


def mean_ci(x, *, n_boot: int = 20000, alpha: float = 0.05,
            seed: int = 20260726) -> dict:
    """Mean and bootstrap confidence interval of one sample."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(mean=float(x.mean()),
                sd=float(x.std(ddof=1)) if n > 1 else 0.0,
                median=float(np.median(x)),
                q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                ci_low=float(lo), ci_high=float(hi), n=int(n))


def convergence_curve(x, *, alpha: float = 0.05, step: int = 5,
                      n_boot: int = 4000, seed: int = 20260726) -> dict:
    """Relative half-width of the confidence interval as a function of the
    number of replicates, used to justify the size of the campaign."""
    x = np.asarray(x, dtype=float)
    ns, half = [], []
    for m in range(step, x.size + 1, step):
        s = mean_ci(x[:m], n_boot=n_boot, alpha=alpha, seed=seed)
        ns.append(m)
        h = 0.5 * (s["ci_high"] - s["ci_low"])
        half.append(100.0 * h / abs(s["mean"]) if s["mean"] != 0 else np.nan)
    return dict(n=ns, half_width_rel_pct=half)


def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm–Bonferroni correction for multiple tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = dict(p_raw=float(p), p_holm=float(adj), reject=bool(adj < alpha))
    return out
