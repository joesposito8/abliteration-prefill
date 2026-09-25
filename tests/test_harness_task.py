"""What the task puts in the log header, and what it must keep out of it."""

from __future__ import annotations

import pytest
from harness.conditions import Condition
from harness.dataset import build_dataset
from generation import qwen3_4b
from harness.provider import frozen_config, require_frozen_config
from harness.task import refusal_unlock
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset, Sample

SEED = 20260803
MODEL = "qwen3-4b"


def condition() -> Condition:
    return Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30)


def tiny() -> MemoryDataset:
    """Task rejects an empty dataset, and these tests only read back the config."""
    return MemoryDataset([Sample(id="000/none", input="q", target="")], name="strongreject")


def run(prefills, tmp_path, **eval_kwargs):
    """Routed through ``Condition.log_dir``, so the driver's own call site is covered."""
    c = condition()
    return eval(
        refusal_unlock(build_dataset(c, prefills), seed=SEED, model=c.model),
        model="mockllm/model",
        sample_id=["005/none/00"],
        log_dir=c.log_dir(tmp_path),
        score=False,
        **eval_kwargs,
    )[0]


# --- the decoding mapping is written once ----------------------------------


def test_the_config_the_task_builds_passes_the_provider_check():
    """Otherwise the suite passes while every real eval is rejected at generation."""
    require_frozen_config(
        refusal_unlock(tiny(), seed=SEED, model=MODEL).config, frozen_config(qwen3_4b)
    )




def test_the_seed_is_whatever_the_caller_supplied():
    """The task must not know how a condition's seed was derived."""
    assert refusal_unlock(tiny(), seed=7, model=MODEL).config.seed == 7





# --- what the log header records -------------------------------------------


def test_rebuilding_the_task_from_its_recorded_args_is_refused(prefills, tmp_path):
    """eval_retry rebuilds this way, and Task reads a name as one sample per character."""
    recorded = run(prefills, tmp_path).eval.task_args

    with pytest.raises(TypeError, match="must be a Dataset"):
        refusal_unlock(**recorded)


def test_task_args_name_the_dataset_rather_than_its_contents(prefills, tmp_path):
    """Every parameter is recorded verbatim; the prefills behind it are megabytes."""
    log = run(prefills, tmp_path)

    assert log.eval.task_args == {"dataset": "strongreject", "seed": SEED, "model": MODEL}
    assert "[system_simulation:0 for 5]" not in repr(log.eval.task_args)


def test_the_task_runs_from_inside_the_repo_so_a_real_run_records_the_commit():
    """`EvalSpec.revision` is git context read from the task's run directory.

    Inspect skips that lookup entirely while ``PYTEST_CURRENT_TEST`` is set
    (``_util/git.py:32``), so no test can observe the recorded commit. This asserts the
    input to the lookup instead: that the directory it will run in is inside the repo.
    """
    import subprocess

    from inspect_ai._eval.task.util import task_run_dir

    run_dir = task_run_dir(refusal_unlock(tiny(), seed=SEED, model=MODEL))
    top_level = subprocess.run(
        ["git", "-C", run_dir, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )

    assert top_level.returncode == 0, top_level.stderr


def test_the_seed_survives_into_the_log(prefills, tmp_path):
    """An eval kwarg beats Task.config, so what was declared and what ran can differ."""
    assert run(prefills, tmp_path).plan.config.seed == SEED
