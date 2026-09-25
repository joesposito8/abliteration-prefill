"""The reported metrics as an Inspect ``EvalResults``: what Inspect records, and where."""

from __future__ import annotations

import pytest
from analysis import metrics
from analysis.results import results, values
from conftest import analysis_scores
from inspect_ai.log import EvalResults

FAMILIES = {"base_best_prefill": "role_chaining", "abliterated_best_prefill": "persona_switch"}


@pytest.fixture(scope="module")
def held() -> EvalResults:
    scores = analysis_scores(unlocked={("abliterated", "none"): [18, 14, 10],
                                       ("base", "role_chaining"): [12, 8, 4]})
    return results(scores, metrics.headline() + metrics.descriptives(), FAMILIES)


def test_the_results_carry_one_score_over_every_sample(held):
    assert held.total_samples == held.completed_samples == 960
    assert len(held.scores) == 1
    assert held.scores[0].name == held.scores[0].scorer == "strongreject"


def test_each_metric_key_is_tagged_with_the_function_that_produced_it(held):
    """``group`` is Inspect's own record of which metric a key came from, so it is what
    separates the confirmatory, ceiling and descriptive readings."""
    groups = {m.group for m in held.scores[0].metrics.values()}

    assert groups == {"abliteration_vs_prefill", "composition_ceiling", "same_queries", "grouped"}
    assert set(values(held, {"abliteration_vs_prefill"})) == {
        "d1", "d1_bca_lo", "d1_bca_hi", "d1_percentile_lo", "d1_percentile_hi",
        "d1_manski_lo", "d1_manski_hi",
    }
    assert len(values(held, {"grouped"})) == 44  # (16 levels + 6 portfolio) x rate, malformed


def test_the_selected_families_travel_on_the_score_rather_than_as_metrics(held):
    """``EvalMetric.value`` is built with ``float()``, so a level name cannot be one."""
    assert held.scores[0].metadata == FAMILIES


def test_the_reported_parameters_are_recorded_beside_their_metric(held):
    params = {m.name: m.params for m in held.scores[0].metrics.values()}

    assert params["d1"]["alpha"] == metrics.ALPHA_D1
    assert params["d2"]["alpha"] == metrics.ALPHA_D2
    assert params["d1"]["resamples"] == params["d2"]["resamples"] == metrics.RESAMPLES
    assert params["overlap_tests"]["fdr"] == metrics.FDR


def test_the_results_load_back_through_inspects_own_types(held):
    again = EvalResults.model_validate_json(held.model_dump_json())

    assert again.scores[0].metrics["d1"].value == held.scores[0].metrics["d1"].value
    assert again.scores[0].metadata == FAMILIES
