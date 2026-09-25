"""Fisher's exact test on a 2x2 table of unlocked against refused generations.

The third metric compares two arms on one query at a time. Conditional on both margins,
the count in the first row is hypergeometric, so the test needs no distributional
assumption and no normal approximation -- which matters here because both arms saturate,
and a query where one arm unlocks every draw is exactly where an approximation fails.

Exact in ``math.comb`` rather than through scipy: scipy is not a dependency, and adding
one is a study decision.
"""

from __future__ import annotations

from math import comb

def _hypergeometric(k: int, unlocked: int, n: int) -> float:
    """P(the first row holds ``k`` of the ``unlocked``) with both rows at ``n`` draws."""
    return comb(n, k) * comb(n, unlocked - k) / comb(2 * n, unlocked)


def fisher_two_sided(x: int, y: int, n: int) -> float:
    """Two-sided p-value for ``x`` unlocks in one arm against ``y`` in the other.

    Both arms carry ``n`` draws. The p-value is the total probability of every table no
    more likely than the observed one, which is the two-sided convention scipy's
    ``fisher_exact`` uses; with equal row totals the null is symmetric, so it comes to
    twice the tail except where the observed table sits at the centre.
    """
    unlocked = x + y
    observed = _hypergeometric(x, unlocked, n)
    lo, hi = max(0, unlocked - n), min(n, unlocked)
    return sum(
        p for k in range(lo, hi + 1)
        if (p := _hypergeometric(k, unlocked, n)) <= observed
    )
