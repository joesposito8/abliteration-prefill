"""The eval task for one condition."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from .provider import FROZEN_CONFIG


@task
def refusal_unlock(dataset: Dataset, seed: int, layer: int | None = None) -> Task:
    """One condition's eval.

    The caller builds ``dataset``: every parameter is recorded verbatim in
    ``EvalSpec.task_args``, and a Dataset records as its name where the prefill mapping
    behind it would record as megabytes of attack text per log.

    ``layer`` is recorded and never read. Nothing else in the log holds it — the
    condition id is opaque to the harness by design, so the log would otherwise not say
    which layer produced it.

    No scorer: generation and grading are separate passes and ``score()`` takes the
    scorer at grading time, so nothing here needs an API key.
    """
    if isinstance(dataset, str):
        raise TypeError(f"dataset must be a Dataset, not the name {dataset!r}")
    return Task(
        dataset=dataset,
        solver=generate(),
        config=FROZEN_CONFIG.merge(GenerateConfig(seed=seed)),
    )
