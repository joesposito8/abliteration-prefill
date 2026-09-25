"""The rule that picks a model's batch width, which decides how the GPU budget is spent.

Every sweep reads it, at different bounds: the 4B target's peak is a few GiB beside its
weights, the 27B helper's weights alone are most of the card.
"""

from __future__ import annotations

import json

import pytest
from harness.width import WidthReport, select_width
from study.datasets import model_dir


def measured(width: int, prompts_per_second: float, peak_gb: float) -> WidthReport:
    return WidthReport(width, prompts_per_second, peak_gb)


def test_the_fastest_that_fits_wins():
    widths = [measured(8, 1.0, 10), measured(16, 1.8, 12), measured(32, 2.5, 15)]

    choice = select_width(widths, ceiling_gb=60, tolerance=0.10)

    assert choice.width == 32
    assert choice.rule_path == ("fastest_within_ceiling",)


def test_a_smaller_width_wins_when_it_costs_almost_nothing():
    """A failed batch re-runs that many samples, so narrower is preferred at equal speed."""
    widths = [measured(8, 2.4, 10), measured(16, 2.45, 12), measured(32, 2.5, 15)]

    choice = select_width(widths, ceiling_gb=60, tolerance=0.10)

    assert choice.width == 8
    assert choice.rule_path == ("fastest_within_ceiling", "within_tolerance")


def test_stepping_down_stops_at_the_first_width_that_costs_too_much():
    widths = [measured(8, 1.0, 10), measured(16, 2.4, 12), measured(32, 2.5, 15)]

    assert select_width(widths, ceiling_gb=60, tolerance=0.10).width == 16


def test_a_width_over_the_ceiling_cannot_win_however_fast_it_is():
    """The whole ceiling rests on this one."""
    widths = [measured(8, 1.0, 10), measured(32, 9.9, 80)]

    choice = select_width(widths, ceiling_gb=60, tolerance=0.10)

    assert choice.width == 8
    assert choice.rejected == (32,)


def test_nothing_fitting_the_ceiling_is_refused_rather_than_guessed():
    widths = [measured(8, 1.0, 70), measured(16, 2.0, 75)]

    with pytest.raises(ValueError, match="every width exceeded 60"):
        select_width(widths, ceiling_gb=60, tolerance=0.10)


def test_the_ceiling_is_what_a_27b_helper_needs_it_to_be():
    """Its weights alone are ~54 GiB, so a 4B target's 60 GiB ceiling would reject
    every width before a single KV entry was counted."""
    widths = [measured(8, 1.0, 61), measured(16, 1.9, 65), measured(32, 2.6, 73)]

    assert select_width(widths, ceiling_gb=72, tolerance=0.10).width == 16
    with pytest.raises(ValueError):
        select_width(widths, ceiling_gb=60, tolerance=0.10)


def test_the_chosen_width_records_what_it_gave_up_against_the_fastest():
    widths = [measured(8, 1.25, 10), measured(32, 2.5, 15)]

    choice = select_width(widths, ceiling_gb=60, tolerance=0.60)

    assert (choice.fastest, choice.width, choice.under_fastest) == (32, 8, 0.5)


def test_every_recorded_curve_implies_the_width_it_chose():
    """At each artifact's OWN recorded bounds rather than today's: a sweep is only
    self-consistent if its ceiling and tolerance still reach its choice."""
    from generation import TARGETS

    for module in TARGETS:
        artifact = model_dir(module.MODEL_ID) / "batch_sweep.json"
        if not artifact.exists():
            continue
        sweep = json.loads(artifact.read_text())
        reports = [
            WidthReport(w["width"], w["prompts_per_second"], w["peak_vram_gb"])
            for w in sweep["widths"]
        ]

        choice = select_width(
            reports, ceiling_gb=sweep["ceiling_gb"], tolerance=sweep["tolerance"]
        )

        assert (choice.width, list(choice.rule_path)) == (
            sweep["chosen"],
            sweep["rule_path"],
        )


def test_a_declared_width_matches_the_sweep_that_chose_it():
    """Applies to whichever models have been swept, so it tightens as they are."""
    from generation import TARGETS

    for module in TARGETS:
        artifact = model_dir(module.MODEL_ID) / "batch_sweep.json"
        if not artifact.exists():
            continue
        assert module.BATCH == json.loads(artifact.read_text())["chosen"]
