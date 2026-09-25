"""The reported metrics as an Inspect ``EvalResults``.

Inspect turns a dict-returning metric into one ``EvalMetric`` per key, tagged with the
metric function's name as ``group``. Letting it do that -- rather than assembling the
records here -- is what keeps the output the shape a reader already knows how to load.

``eval_results`` is private to ``inspect_ai``. The version is pinned with ``==``, so the
import cannot drift under the study.
"""

from __future__ import annotations

from inspect_ai._eval.task.results import ScorerInfo, eval_results
from inspect_ai.log import EvalResults
from inspect_ai.scorer import Metric, SampleScore

from .read import SCORER


def results(scores: list[SampleScore], metrics: list[Metric], metadata: dict) -> EvalResults:
    """``metadata`` carries what a metric cannot: ``EvalMetric.value`` is built with
    ``float()``, so the selected prefill families travel on the score instead."""
    held, _ = eval_results(
        samples=len(scores),
        scores=[{SCORER: score} for score in scores],
        reducers=[],
        scorers=[ScorerInfo(name=SCORER, metrics=metrics, metadata=metadata)],
        metrics=None,
    )
    return held


def values(held: EvalResults, groups: set[str]) -> dict[str, float]:
    """The metrics of the named groups, by name. ``group`` is how Inspect records which
    metric function produced a key, so it is what separates the reported classes."""
    return {
        name: m.value for name, m in held.scores[0].metrics.items() if m.group in groups
    }
