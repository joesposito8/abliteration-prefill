"""The eval task for one condition."""

from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from generation.qwen import DECODING

from .batching import IN_FLIGHT


def decoding_kwargs() -> dict[str, Any]:
    """``qwen.DECODING`` as ``GenerateConfig`` keyword arguments.

    ``provider.require_frozen_decoding`` reads the same values back out of a config and
    is the inverse of this. A change to ``DECODING``'s shape has to move both, so they
    are written once each and never restated.
    """
    return {
        "temperature": DECODING["temperature"],
        "top_p": DECODING["top_p"],
        "top_k": DECODING["top_k"],
        # GenerateConfig names the length cap differently from transformers.
        "max_tokens": DECODING["max_new_tokens"],
        # Neither has a GenerateConfig field, and unknown fields are rejected outright.
        "extra_body": {"min_p": DECODING["min_p"], "do_sample": DECODING["do_sample"]},
    }


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
    return Task(
        dataset=dataset,
        solver=generate(),
        config=GenerateConfig(
            seed=seed, max_connections=IN_FLIGHT, **decoding_kwargs()
        ),
    )
