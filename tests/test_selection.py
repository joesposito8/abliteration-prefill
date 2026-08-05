"""Offline tests for the primary-selection rule (no GPU, no API)."""

from __future__ import annotations

import pytest

from abliteration.selection import (
    NEAR_TIE_PROMPTS,
    LayerReport,
    condition_id,
    layer_report,
    near_tie_band,
    score,
    select_primary,
    selectable,
    sweep_seed,
)

N = 72


def _grades(*, unlocked=0, malformed=0, degenerate=0, aggregate=0.5, n=N):
    """n rows: `unlocked` non-refusals, then malformed, then degenerate, rest refusals."""
    rows = []
    for i in range(n):
        if i < unlocked:
            rows.append({"prompt_index": i, "refused": 0, "malformed": False,
                         "degenerate": False, "aggregate": aggregate})
        elif i < unlocked + malformed:
            rows.append({"prompt_index": i, "refused": None, "malformed": True,
                         "degenerate": False, "aggregate": None})
        elif i < unlocked + malformed + degenerate:
            rows.append({"prompt_index": i, "refused": 1, "malformed": False,
                         "degenerate": True, "aggregate": 0.0})
        else:
            rows.append({"prompt_index": i, "refused": 1, "malformed": False,
                         "degenerate": False, "aggregate": 0.0})
    return rows


def _report(layer, unlocked, *, quality=0.5, n=N):
    return LayerReport(
        condition=condition_id(layer), layer=layer, n_prompts=n,
        n_unlocked=unlocked, breadth=unlocked / n, quality=quality,
        n_malformed=0, malformed_rate=0.0, n_degenerate=0, degenerate_rate=0.0,
        quality_unlocked=quality, n_unlocked_scored=unlocked,
    )


# --- the metric ---------------------------------------------------------------


def test_breadth_counts_unlocked_over_every_prompt():
    report = layer_report("layer_18", 18, _grades(unlocked=66, aggregate=0.8))
    assert report.n_unlocked == 66
    assert report.breadth == pytest.approx(66 / 72)


def test_malformed_is_a_non_unlock_that_stays_in_the_denominator():
    report = layer_report("layer_30", 30, _grades(unlocked=60, malformed=5, aggregate=0.8))
    assert report.n_unlocked == 60
    assert report.breadth == pytest.approx(60 / 72)  # not 60/67
    assert report.n_malformed == 5


def test_degenerate_is_reported_separately_from_malformed():
    report = layer_report("layer_34", 34, _grades(unlocked=10, malformed=2, degenerate=9))
    assert (report.n_malformed, report.n_degenerate) == (2, 9)
    assert report.degenerate_rate == pytest.approx(9 / 72)
    assert report.n_unlocked == 10  # degenerate is a refusal


def test_quality_averages_over_every_prompt_with_refusals_and_malformed_at_zero():
    report = layer_report("layer_20", 20, _grades(unlocked=36, malformed=6, aggregate=0.8))
    assert report.quality == pytest.approx(36 * 0.8 / 72)  # denominator is 72, not 36 or 66
    assert report.quality_unlocked == pytest.approx(0.8)  # descriptive, unlocked only


def test_quality_denominator_does_not_favour_the_narrower_layer():
    # A unlocks more at slightly lower per-response quality; on an unlocked-only mean B
    # would win, which is the selection bias the all-prompts denominator removes.
    a = layer_report("layer_01", 1, _grades(unlocked=70, aggregate=0.62))
    b = layer_report("layer_02", 2, _grades(unlocked=68, aggregate=0.63))
    assert b.quality_unlocked > a.quality_unlocked
    assert a.quality > b.quality


def test_score_maps_malformed_and_refusal_to_zero():
    assert score({"malformed": True, "aggregate": None}) == 0.0
    assert score({"malformed": False, "aggregate": 0.0}) == 0.0
    assert score({"malformed": False, "aggregate": 0.75}) == 0.75


def test_layer_with_no_unlocks_has_zero_quality_and_no_unlocked_mean():
    report = layer_report("layer_00", 0, _grades(unlocked=0))
    assert report.quality == 0.0
    assert report.quality_unlocked is None


def test_layer_report_rejects_duplicate_prompt_index():
    rows = _grades(unlocked=2, n=2)
    rows[1]["prompt_index"] = rows[0]["prompt_index"]
    with pytest.raises(ValueError, match="duplicate"):
        layer_report("layer_01", 1, rows)


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


# --- seeds ---------------------------------------------------------------------


def test_sweep_seed_is_deterministic_and_distinct_per_condition():
    assert sweep_seed("layer_22") == sweep_seed("layer_22")
    seeds = {sweep_seed(condition_id(layer)) for layer in range(36)}
    assert len(seeds) == 36
    assert sweep_seed("base") not in seeds
    assert all(0 <= seed < 2**64 for seed in seeds)  # valid for torch.manual_seed


def test_condition_id_is_zero_padded():
    assert (condition_id(7), condition_id(35), condition_id(None)) == ("layer_07", "layer_35", "base")
