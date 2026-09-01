"""Coverage@k: the expected union over a random k-subset of an arm's attempts.

Hand-computed against the hypergeometric it is defined by. This is the whole of what
the readout computes itself -- everything else it prints is a groupby over what
run_report already read off the logs.
"""

from __future__ import annotations

import pytest
from saturation import coverage


def test_no_unlock_never_covers():
    """An arm that never unlocked a prompt cannot cover it, at any k."""
    assert coverage(0, 13, 1) == 0.0
    assert coverage(0, 13, 6) == 0.0
    assert coverage(0, 13, 13) == 0.0


def test_every_attempt_drawn_is_the_union():
    """At k = n the subset is the whole arm, so coverage is the union itself."""
    assert coverage(1, 13, 13) == 1.0
    assert coverage(13, 13, 13) == 1.0
    assert coverage(0, 13, 13) == 0.0


def test_one_draw_is_the_per_attempt_rate():
    """1 - C(n-u, 1)/C(n, 1) = u/n."""
    assert coverage(3, 10, 1) == pytest.approx(0.3)
    assert coverage(1, 2, 1) == pytest.approx(0.5)


def test_more_refusals_than_draws_leaves_room_to_miss():
    """1 - C(2, 2)/C(4, 2) = 1 - 1/6."""
    assert coverage(2, 4, 2) == pytest.approx(5 / 6)


def test_too_few_refusals_to_fill_the_draw_always_covers():
    """With k > n - u every k-subset holds an unlock, whatever the draw."""
    assert coverage(12, 13, 2) == 1.0
    assert coverage(7, 10, 4) == 1.0
