"""The rule that picks a batch width, which decides how the GPU budget is spent.

Both sweeps read it, at different ceilings: the target's peak is a few GiB beside its
weights, the helper's weights alone are most of the card.
"""

from __future__ import annotations

import pytest

from batch_sweep import choose_width


def measured(width: int, prompts_per_second: float, peak_vram_gb: float) -> dict:
    return {
        "width": width,
        "prompts_per_second": prompts_per_second,
        "peak_vram_gb": peak_vram_gb,
    }


def test_the_fastest_that_fits_wins():
    widths = [measured(8, 1.0, 10), measured(16, 1.8, 12), measured(32, 2.5, 15)]

    assert choose_width(widths, ceiling=60) == (32, "fastest within the memory ceiling")


def test_a_smaller_width_wins_when_it_costs_almost_nothing():
    """A failed batch re-runs that many samples, so narrower is preferred at equal speed."""
    widths = [measured(8, 2.4, 10), measured(16, 2.45, 12), measured(32, 2.5, 15)]

    assert choose_width(widths, tolerance=0.10) == (8, "smallest within 10% of the fastest")


def test_stepping_down_stops_at_the_first_width_that_costs_too_much():
    widths = [measured(8, 1.0, 10), measured(16, 2.4, 12), measured(32, 2.5, 15)]

    assert choose_width(widths, tolerance=0.10)[0] == 16


def test_a_width_over_the_ceiling_cannot_win_however_fast_it_is():
    widths = [measured(8, 1.0, 10), measured(32, 9.9, 80)]

    assert choose_width(widths, ceiling=60)[0] == 8


def test_nothing_fitting_the_ceiling_is_refused_rather_than_guessed():
    with pytest.raises(SystemExit, match="every width exceeded 60"):
        choose_width([measured(8, 1.0, 70), measured(16, 2.0, 75)], ceiling=60)


def test_the_ceiling_is_what_a_27b_helper_needs_it_to_be():
    """Its weights alone are ~54 GiB, so the target's 60 GiB ceiling would reject
    every width before a single KV entry was counted."""
    widths = [measured(8, 1.0, 61), measured(16, 1.9, 65), measured(32, 2.6, 73)]

    assert choose_width(widths, ceiling=72)[0] == 16  # 32 is over even this ceiling
    with pytest.raises(SystemExit):
        choose_width(widths, ceiling=60)


def test_each_width_records_what_it_gave_up_against_the_fastest():
    widths = [measured(8, 1.25, 10), measured(32, 2.5, 15)]

    choose_width(widths)

    assert [measurement["under_fastest"] for measurement in widths] == [0.5, 0.0]
