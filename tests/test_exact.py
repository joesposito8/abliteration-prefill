"""Fisher's exact test, against an external reference and the boundary the floor sits on.

The reported detection floor for the third metric is read off where this function falls
relative to Benjamini-Hochberg's rank-1 threshold, so the pair of tables that straddles
that threshold is pinned here rather than left to a readout.
"""

from __future__ import annotations

import pytest
from study.exact import fisher_two_sided

# 313 queries at a 5% false discovery rate. A test at or under this is rejected whatever
# the other 312 do, so it is the threshold the floor is defined against.
RANK_ONE = 0.05 / 313


def test_the_tea_tasting_table():
    """The published 3/1/1/3 result. Probabilities are C(4,k)^2/70, so the tables no more
    likely than the observed one sum to (1+16+16+1)/70 -- which is the two-sided
    convention, and the one thing a wrong one would not reproduce."""
    assert fisher_two_sided(3, 1, 4) == pytest.approx(17 / 35)


def test_eleven_against_none_clears_the_rank_one_threshold():
    """11 of 20 against 0 of 20 -- a 0.55 gap -- is the smallest table that clears."""
    assert fisher_two_sided(11, 0, 20) == pytest.approx(1.4530659691950015e-04)
    assert fisher_two_sided(11, 0, 20) <= RANK_ONE


def test_ten_against_none_does_not():
    """A 0.50 gap at the same margins misses, which is what puts the floor at 0.55."""
    assert fisher_two_sided(10, 0, 20) == pytest.approx(4.3591979075850045e-04)
    assert fisher_two_sided(10, 0, 20) > RANK_ONE
