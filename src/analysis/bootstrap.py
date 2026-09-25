"""Cluster bootstrap over prompts, with the bias-corrected and accelerated interval and the
percentile interval it is read against.

A prompt's draws are resampled together, and the statistic is recomputed in full on each
resample -- including which prefill level is best -- so the interval carries the variance
of having selected.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from .counts import Counts, all_prompts

CHUNK = 1_000


def replicates(stat, counts: Counts, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """``resamples`` draws of the prompts with replacement, the statistic on each."""
    n = len(counts.prompt_ids)
    out = []
    for start in range(0, resamples, CHUNK):
        idx = rng.integers(0, n, size=(min(CHUNK, resamples - start), n))
        out.append(stat(counts, idx))
    return np.concatenate(out)


def jackknife(stat, counts: Counts) -> np.ndarray:
    """The statistic with each prompt left out in turn."""
    n = len(counts.prompt_ids)
    idx = np.array([np.delete(np.arange(n), i) for i in range(n)])
    return stat(counts, idx)


def bca_interval(theta_hat: float, thetas: np.ndarray, jack: np.ndarray, alpha: float) -> tuple[float, float]:
    """Efron's BCa endpoints: percentiles of ``thetas`` shifted by the bias z0 and the
    acceleration from the jackknife. Replicates tied with the estimate count half toward
    z0, so a statistic on a coarse grid is not biased by exact ties."""
    normal = NormalDist()
    below = ((thetas < theta_hat).sum() + (thetas <= theta_hat).sum()) / (2 * len(thetas))
    z0 = normal.inv_cdf(below)

    d = jack.mean() - jack
    spread = (d**2).sum()
    a = (d**3).sum() / (6 * spread**1.5) if spread else 0.0

    def endpoint(q: float) -> float:
        z = z0 + normal.inv_cdf(q)
        return float(np.percentile(thetas, 100 * normal.cdf(z0 + z / (1 - a * z))))

    return endpoint(alpha / 2), endpoint(1 - alpha / 2)


def percentile_interval(thetas: np.ndarray, alpha: float) -> tuple[float, float]:
    """The replicates' own quantiles, uncorrected -- the robustness check BCa is read against."""
    lo, hi = np.percentile(thetas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def intervals(
    stat, counts: Counts, resamples: int, alpha: float, rng: np.random.Generator
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(bca, percentile)`` -- both read off one set of replicates, so the robustness
    check is the same draws rather than a second bootstrap."""
    theta_hat = float(stat(counts, all_prompts(counts))[0])
    thetas = replicates(stat, counts, resamples, rng)
    return (
        bca_interval(theta_hat, thetas, jackknife(stat, counts), alpha),
        percentile_interval(thetas, alpha),
    )
