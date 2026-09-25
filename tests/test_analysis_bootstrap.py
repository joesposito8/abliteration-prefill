"""The BCa endpoints, pinned by hand rather than against a library the study does not carry."""

from __future__ import annotations

import numpy as np
import pytest
from analysis.bootstrap import (
    bca_interval,
    intervals,
    jackknife,
    percentile_interval,
    replicates,
)
from analysis.counts import Counts, all_prompts


def test_an_unbiased_symmetric_bootstrap_is_the_percentile_interval():
    """The estimate sits at the replicates' median (z0 = 0) and the jackknife is
    symmetric (a = 0), so nothing shifts."""
    thetas = np.arange(1, 101, dtype=float)

    lo, hi = bca_interval(50.5, thetas, np.array([1.0, 2.0, 3.0]), alpha=0.1)

    assert (lo, hi) == pytest.approx(np.percentile(thetas, [5, 95]))


def test_bias_and_acceleration_shift_the_endpoints():
    """By hand: 29 replicates below 30 and one tied, so z0 = inv_cdf(0.295) = -0.539; the
    jackknife [0, 0, 3] gives a = -6 / (6 * 6^1.5) = -0.068. Then the lower endpoint is
    the 0.096th percentile of 1..100 and the upper the 68.8th."""
    thetas = np.arange(1, 101, dtype=float)

    lo, hi = bca_interval(30.0, thetas, np.array([0.0, 0.0, 3.0]), alpha=0.1)

    assert lo == pytest.approx(1.095, abs=0.01)
    assert hi == pytest.approx(69.10, abs=0.01)


def test_a_flat_jackknife_carries_no_acceleration():
    thetas = np.arange(1, 101, dtype=float)

    assert bca_interval(50.5, thetas, np.zeros(3), 0.1) == pytest.approx(np.percentile(thetas, [5, 95]))


def counts(n_prompts: int) -> Counts:
    u = np.arange(n_prompts)[:, None, None] * np.ones((1, 2, 8), dtype=int)
    return Counts(np.arange(n_prompts), u, np.full_like(u, 20), np.zeros_like(u))


def mean_u(held: Counts, idx: np.ndarray) -> np.ndarray:
    return held.u[idx, 0, 0].mean(axis=1)


def test_replicates_are_seeded_and_chunked():
    held = counts(7)

    a = replicates(mean_u, held, 2_500, np.random.default_rng(1))
    b = replicates(mean_u, held, 2_500, np.random.default_rng(1))

    assert a.shape == (2_500,)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() <= 6


def test_the_jackknife_leaves_each_prompt_out_once():
    held = counts(4)  # u = 0, 1, 2, 3

    assert jackknife(mean_u, held).tolist() == pytest.approx([2.0, 5 / 3, 4 / 3, 1.0])


def test_the_percentile_interval_is_the_replicates_own_quantiles():
    thetas = np.arange(1, 101, dtype=float)

    assert percentile_interval(thetas, 0.05) == pytest.approx(np.percentile(thetas, [2.5, 97.5]))
    assert percentile_interval(thetas, 0.025) == pytest.approx(np.percentile(thetas, [1.25, 98.75]))


def test_both_intervals_come_off_one_set_of_replicates():
    """The percentile endpoints are the quantiles of exactly the draws BCa corrected, so
    re-running the replicates with the same seed reproduces them."""
    held = counts(7)
    rng = np.random.default_rng(1)

    corrected, plain = intervals(mean_u, held, 2_500, 0.05, rng)
    thetas = replicates(mean_u, held, 2_500, np.random.default_rng(1))

    assert plain == pytest.approx(percentile_interval(thetas, 0.05))
    assert corrected == pytest.approx(
        bca_interval(float(mean_u(held, all_prompts(held))[0]), thetas, jackknife(mean_u, held), 0.05)
    )
