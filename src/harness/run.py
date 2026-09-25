"""Run one task, or a whole sweep of conditions, picking up whatever an earlier attempt
already finished.

``eval_set`` recognises an earlier log and re-runs only the samples missing from it,
whether the earlier attempt ended in an error or was killed outright. Its retries are
off: they are the one part of it that would decide unattended to spend GPU time, and a
failure here is usually systematic rather than transient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from abliteration import orthogonalize_, restore_targets, snapshot_targets
from abliteration.edit import target_matrices
from generation import target
from inspect_ai import Task, eval_set
from inspect_ai.dataset import Dataset
from inspect_ai.log import EvalLog

from .conditions import Condition
from .dataset import build_dataset
from .task import refusal_unlock


def run_eval_set(
    task: Task,
    *,
    log_dir: str,
    model: str,
    module: Any,
    tokenizer: Any,
    label: str,
    log_buffer: int,
) -> EvalLog:
    """Run one task against a loaded model, into its own log directory.

    ``max_tasks=1`` is a correctness rule and not a throughput choice: a provider holds
    one module and one open batch, and two tasks in flight write into both.
    """
    success, logs = eval_set(
        tasks=[task],
        log_dir=log_dir,
        model=model,
        model_args={"module": module, "tokenizer": tokenizer},
        score=False,
        max_tasks=1,
        retry_attempts=0,
        retry_cleanup=False,
        log_buffer=log_buffer,
    )

    if not success:
        raise RuntimeError(
            f"{label} did not finish: {[log.status for log in logs]}. The log is kept, "
            "and re-running resumes from what it holds."
        )
    return logs[0]


def run_condition(
    condition: Condition,
    dataset: Dataset,
    *,
    module: Any,
    tokenizer: Any,
    root: Path,
) -> EvalLog:
    """Run every sample of one condition under whatever weights ``module`` holds.

    Two conditions in flight would both generate under whichever edit was applied last,
    with each log still naming its own.
    """
    return run_eval_set(
        refusal_unlock(dataset, seed=condition.seed, model=condition.model),
        log_dir=condition.log_dir(root),
        model=condition.model_name,
        module=module,
        tokenizer=tokenizer,
        label=condition.id,
        log_buffer=target(condition.model).BATCH,
    )


def restore_and_verify(module, base, condition_id: str) -> None:
    """Put the base weights back, then prove they went back."""
    import torch

    restore_targets(module, base)
    for (weight, _), saved in zip(target_matrices(module), base):
        if not torch.equal(weight.data, saved):
            raise RuntimeError(f"{condition_id}: restore left weights modified")


def run_sweep(
    conditions: Sequence[Condition],
    prefills: Mapping[tuple[int, str], str],
    *,
    module: Any,
    tokenizer: Any,
    directions: Any,
    root: Path,
) -> list[EvalLog]:
    """Run each condition under its own edit, against one loaded model.

    The base weights are cloned once for the whole sweep, which is what ``edit.py``
    documents for this loop; ``abliterated`` re-clones every target matrix per use.

    ``condition.layer`` indexes ``directions``; ``None`` means no edit, and layer 0 is a
    layer like any other.
    """
    base = snapshot_targets(module)
    logs = []
    for condition in conditions:
        try:
            if condition.layer is not None:
                orthogonalize_(module, directions[condition.layer])
            logs.append(
                run_condition(
                    condition,
                    build_dataset(condition, prefills),
                    module=module,
                    tokenizer=tokenizer,
                    root=root,
                )
            )
        finally:
            restore_and_verify(module, base, condition.id)
    return logs
