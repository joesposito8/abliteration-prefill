"""Generate one condition, picking up whatever an earlier attempt already finished.

``eval_set`` recognises a condition's earlier log and re-runs only the samples missing
from it, whether the earlier attempt ended in an error or was killed outright. Its
retries are off: they are the one part of it that would decide unattended to spend GPU
time, and a failure here is usually systematic rather than transient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai import eval_set
from inspect_ai.dataset import Dataset
from inspect_ai.log import EvalLog

from .batching import BATCH
from .conditions import Condition
from .task import refusal_unlock


def run_condition(
    condition: Condition,
    dataset: Dataset,
    *,
    module: Any,
    tokenizer: Any,
    root: Path,
) -> EvalLog:
    """Run every sample of one condition under whatever weights ``module`` holds.

    ``max_tasks=1`` is the study's rule and not a throughput choice: the module is one
    mutable object, so two conditions in flight at once both generate under whichever
    edit was applied last, and each log still names its own.
    """
    success, logs = eval_set(
        tasks=[refusal_unlock(dataset, seed=condition.seed)],
        log_dir=condition.log_dir(root),
        model=condition.model_name,
        model_args={"module": module, "tokenizer": tokenizer},
        score=False,
        max_tasks=1,
        retry_attempts=0,
        retry_cleanup=False,
        log_buffer=BATCH,
    )

    if not success:
        raise RuntimeError(
            f"{condition.id} did not finish: {[log.status for log in logs]}. The log is "
            "kept, and re-running this condition resumes from what it holds."
        )
    return logs[0]
