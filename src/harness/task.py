"""The eval task for one condition."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from generation import target

from .meta import run_metadata
from .provider import frozen_config


@task
def refusal_unlock(dataset: Dataset, seed: int, model: str) -> Task:
    """One condition's eval, against the target ``model`` names.

    The caller builds ``dataset``: every parameter is recorded verbatim in
    ``EvalSpec.task_args``, and a Dataset records as its name where the prefill mapping
    behind it would record as megabytes of attack text per log. That name is the prompt
    set, which is what the header hashes. ``model`` is a slug for the same reason: a
    module does not survive being written to the log.

    No scorer: generation and grading are separate passes and ``score()`` takes the
    scorer at grading time, so nothing here needs an API key.
    """
    if isinstance(dataset, str):
        raise TypeError(f"dataset must be a Dataset, not the name {dataset!r}")
    return Task(
        dataset=dataset,
        solver=generate(),
        config=frozen_config(target(model)).merge(GenerateConfig(seed=seed)),
        metadata=run_metadata(dataset.name, model),
    )
