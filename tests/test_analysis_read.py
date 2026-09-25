"""The metadata a metric reads, and the one way it silently stops arriving."""

from __future__ import annotations

import pytest
from analysis.read import derived, portfolio_of
from conftest import analysis_scores
from inspect_ai._eval.task.results import ScorerInfo, eval_results
from inspect_ai.scorer import Metric, SampleScore, Value, grouped, metric


def test_a_level_is_grouped_by_where_its_prefill_came_from():
    assert portfolio_of("none") == "unprefilled"
    assert portfolio_of("static_baseline") == "static"
    assert portfolio_of("role_chaining") == "generated"


def test_the_portfolio_classes_share_no_name_with_a_level():
    """A key built from both groupings lands in one namespace, and Inspect renames a
    collision rather than refusing it."""
    from analysis.counts import LEVELS

    assert not {portfolio_of(level) for level in LEVELS} & set(LEVELS)


def test_the_derived_keys_are_functions_of_the_metadata_already_there():
    held = derived({"prompt_id": 7, "prefill_slot": "role_chaining:1"}, "abliterated")

    assert held["prompt_id"] == 7  # nothing the sample carried is dropped
    assert held["level"] == "role_chaining"
    assert held["cell"] == "abliterated/role_chaining"
    assert held["portfolio_cell"] == "abliterated/generated"


def test_a_sample_id_carries_its_condition():
    """The two conditions reuse ids, and a collision would be read as two epochs of one
    sample rather than two generations."""
    scores = analysis_scores()

    assert len({s.sample_id for s in scores}) == len(scores)
    assert {s.sample_id.split("/")[0] for s in scores} == {"base", "layer_23"}


@metric
def annotated() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return float(len([s for s in scores if s.sample_metadata]))

    return compute


@metric
def unannotated() -> Metric:
    def compute(scores):
        return float(len([s for s in scores if getattr(s, "sample_metadata", None)]))

    return compute


@pytest.mark.parametrize(
    "held, expected", [(annotated(), 6.0), (unannotated(), 0.0)], ids=["annotated", "unannotated"]
)
def test_a_metric_only_receives_sample_metadata_when_its_parameter_is_annotated(held, expected):
    """Inspect reads the type hints to tell the current protocol from the deprecated
    ``list[Score]`` one, and the deprecated path hands over bare Scores. An unannotated
    metric therefore computes on something else rather than failing, which is why every
    metric in this package annotates and why ``grouped`` would otherwise raise."""
    scores = [{"strongreject": s} for s in analysis_scores()[:6]]

    results, _ = eval_results(
        samples=len(scores), scores=scores, reducers=[],
        scorers=[ScorerInfo(name="strongreject", metrics=[held])], metrics=None,
    )

    assert next(iter(results.scores[0].metrics.values())).value == expected


def test_grouping_needs_the_derived_key_and_says_so_when_it_is_missing():
    scores = [SampleScore(score=s.score, sample_id=s.sample_id, sample_metadata={})
              for s in analysis_scores()[:2]]

    with pytest.raises(ValueError, match="no cell metadata"):
        grouped(annotated(), "cell", all=False)(scores)
