"""Offline tests for the primary-selection rule (no GPU, no API)."""

from __future__ import annotations

import pytest

from abliteration.selection import (
    NEAR_TIE_PROMPTS,
    LayerReport,
    condition_id,
    near_tie_band,
    select_primary,
    selectable,
)

N = 72


def _report(layer, unlocked, *, quality=0.5, n=N):
    return LayerReport(
        condition=condition_id(layer), layer=layer, n_prompts=n,
        n_unlocked=unlocked, breadth=unlocked / n, quality=quality,
        n_malformed=0, malformed_rate=0.0, n_degenerate=0, degenerate_rate=0.0,
        quality_unlocked=quality,
    )


# --- the near-tie band ---------------------------------------------------------


def test_band_boundary_is_three_prompts_inclusive():
    reports = [_report(1, 70), _report(2, 67), _report(3, 66), _report(4, 60)]
    assert {r.layer for r in near_tie_band(reports)} == {1, 2}  # 70-3=67 in, 66 out
    assert NEAR_TIE_PROMPTS == 3


def test_band_never_contains_the_base_row():
    reports = [_report(None, 72), _report(1, 40), _report(2, 39)]
    assert "base" not in {r.condition for r in near_tie_band(reports)}
    assert [r.layer for r in selectable(reports)] == [1, 2]


def test_band_raises_when_there_is_no_layer():
    with pytest.raises(ValueError, match="no layer"):
        near_tie_band([_report(None, 10)])


# --- selection -----------------------------------------------------------------


def test_single_best_resolves_on_breadth_alone():
    reports = [_report(1, 70), _report(2, 60), _report(3, 59)]
    selection = select_primary(reports)
    assert (selection.layer, selection.rule_path) == (1, ("breadth",))
    assert selection.runner_up == "layer_02"


def test_quality_breaks_a_breadth_tie_inside_the_band():
    reports = [_report(1, 70, quality=0.50), _report(2, 69, quality=0.90)]
    selection = select_primary(reports)
    assert selection.layer == 2  # fewer unlocks, but inside the band and better quality
    assert selection.rule_path == ("breadth", "quality")


def test_lowest_layer_index_is_the_final_fallback():
    reports = [_report(5, 70, quality=0.7), _report(2, 70, quality=0.7)]
    selection = select_primary(reports)
    assert selection.layer == 2
    assert selection.rule_path == ("breadth", "layer_index")


def test_rule_path_records_only_rules_that_narrowed_the_field():
    # Quality is identical across the band, so it decided nothing and must not be claimed.
    reports = [_report(3, 70, quality=0.6), _report(4, 69, quality=0.6)]
    assert "quality" not in select_primary(reports).rule_path


def test_base_row_is_never_selected_even_at_the_highest_breadth():
    reports = [_report(None, 72, quality=0.99), _report(1, 40), _report(2, 39)]
    selection = select_primary(reports)
    assert selection.layer == 1
    assert selection.runner_up == "layer_02"


def test_band_is_reported_in_rank_order():
    reports = [_report(1, 68, quality=0.4), _report(2, 70, quality=0.9), _report(3, 69, quality=0.5)]
    assert select_primary(reports).band == ("layer_02", "layer_03", "layer_01")


# --- condition naming ----------------------------------------------------------


def test_condition_id_is_zero_padded():
    assert (condition_id(7), condition_id(35), condition_id(None)) == ("layer_07", "layer_35", "base")
