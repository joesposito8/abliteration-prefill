"""The sampling estimator, checked against exact expectations rather than simulated.

Summing over every outcome makes the assertions equalities. The last check is the shape
the reporting decision was made for: queries that do not differ at all, where the estimate
of the query-to-query term is negative and is reported that way.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest
from abliteration.selection import BASE_CONDITION
from harness.dataset import CONTROL
from variance_decomposition import Arms, best_family, decompose, sampling_variance


def binomial(u: int, n: int, p: float) -> float:
    return comb(n, u) * p**u * (1 - p) ** (n - u)


def estimate(u: list[int], n: list[int]) -> float:
    return sampling_variance(np.array([u], dtype=float), np.array([n], dtype=float))[0]


def test_one_slot_recovers_the_binomial_variance():
    """Summed over every outcome, u(n-u)/(n^2(n-1)) is exactly p(1-p)/n. At 20 draws and
    a rate of 0.9, which is where the abliterated arm actually sits."""
    n, p = 20, 0.9
    expected = sum(binomial(u, n, p) * estimate([u], [n]) for u in range(n + 1))
    assert expected == pytest.approx(p * (1 - p) / n)


def test_two_variants_recover_the_pooled_variance_even_when_they_differ():
    """A family is two variants that need not share a rate, so their sum is a mixture and
    not one binomial. Estimated slot by slot and then weighted, the term is right anyway
    -- which is the whole reason it is not estimated on the pooled count."""
    n, p1, p2 = 10, 0.9, 0.3
    expected = sum(
        binomial(u1, n, p1) * binomial(u2, n, p2) * estimate([u1, u2], [n, n])
        for u1 in range(n + 1)
        for u2 in range(n + 1)
    )
    assert expected == pytest.approx((p1 * (1 - p1) + p2 * (1 - p2)) / (4 * n))


def test_queries_that_do_not_differ_leave_a_negative_remainder():
    """With no observed spread there is still a sampling term to subtract, so the
    query-to-query estimate goes below zero. It is reported unclipped: the estimator is
    centred on zero when the queries are alike, and clipping would hide that."""
    held = Arms(
        a=np.full(6, 0.9),
        p=np.full(6, 0.5),
        within_a=np.full(6, 0.01),
        within_p=np.full(6, 0.02),
        n_a=10,
        n_p=20,
        family="persona_switch",
    )
    stats = decompose(held, np.arange(6))

    assert stats["total"] == pytest.approx(0.0)
    assert stats["hetero"] == pytest.approx(-0.03)


def cell_rows(slot: str, n: int, unlocked: int) -> list[dict]:
    return [
        {"condition": BASE_CONDITION, "slot": slot, "unlocked": int(i < unlocked)}
        for i in range(n)
    ]


def test_the_best_prefill_is_a_family_rate_not_a_slot_and_not_a_total():
    """Three ways to get this wrong, all of them live on the collected pilot.

    The baseline should win here at 0.9. Ranking slots rather than families would take
    persona_switch:0, which unlocks everything but whose sibling variant does not.
    Ranking by summed unlocks would take role_chaining, which is behind on rate and ahead
    only because a family holds two slots against the baseline's one. Leaving the
    unprefilled control in the running would take the control.
    """
    rows = (
        cell_rows("static_baseline", 10, 9)
        + cell_rows("persona_switch:0", 10, 10)
        + cell_rows("persona_switch:1", 10, 2)
        + cell_rows("role_chaining:0", 10, 8)
        + cell_rows("role_chaining:1", 10, 8)
        + cell_rows(CONTROL, 10, 10)
    )
    assert best_family(pd.DataFrame(rows)) == "static_baseline"
