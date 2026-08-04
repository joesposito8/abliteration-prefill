"""What the task puts in the log header, and what it must keep out of it."""

from __future__ import annotations

from pathlib import Path

from generation.qwen import DECODING
from harness.batching import BATCH, IN_FLIGHT
from harness.conditions import Condition
from harness.dataset import build_dataset
from harness.provider import require_frozen_decoding
from harness.task import refusal_unlock
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset, Sample

SEED = 20260803


def condition(log_root: Path = Path("/tmp/unused")) -> Condition:
    return Condition(
        id="layer_22",
        seed=SEED,
        log_root=log_root,
        layer=22,
        prompt_set="pilot",
    )


def tiny() -> MemoryDataset:
    """Task rejects an empty dataset, and these tests only read back the config."""
    return MemoryDataset([Sample(id="000/none", input="q", target="")])


def run(prefills, tmp_path, **eval_kwargs):
    """Routed through ``Condition.log_dir``, so the driver's own call site is covered."""
    c = condition(tmp_path)
    return eval(
        refusal_unlock(build_dataset(c, prefills), seed=SEED, layer=22),
        model="mockllm/model",
        sample_id=["005/none"],
        log_dir=c.log_dir,
        score=False,
        **eval_kwargs,
    )[0]


# --- the decoding mapping is written once ----------------------------------


def test_the_config_the_task_builds_passes_the_provider_check():
    """Otherwise the suite passes while every real eval is rejected at generation."""
    require_frozen_decoding(refusal_unlock(tiny(), seed=SEED).config)


def test_the_config_carries_the_whole_frozen_set():
    config = refusal_unlock(tiny(), seed=SEED).config

    assert config.temperature == DECODING["temperature"]
    assert config.top_p == DECODING["top_p"]
    assert config.top_k == DECODING["top_k"]
    assert config.max_tokens == DECODING["max_new_tokens"]
    assert config.extra_body == {
        "min_p": DECODING["min_p"],
        "do_sample": DECODING["do_sample"],
    }


def test_the_seed_is_whatever_the_caller_supplied():
    """The task must not know how a condition's seed was derived."""
    assert refusal_unlock(tiny(), seed=7).config.seed == 7


def test_the_pipeline_is_deep_enough_for_a_second_batch():
    """At 1x the running batch owns every permit and the next cannot assemble."""
    depth = refusal_unlock(tiny(), seed=SEED).config.max_connections

    assert depth == IN_FLIGHT == 2 * BATCH


def test_no_batch_api_and_no_cache_are_requested():
    config = refusal_unlock(tiny(), seed=SEED).config

    assert config.batch is None  # Inspect's provider batch API, not our coalescer
    assert config.cache_prompt is None


# --- what the log header records -------------------------------------------


def test_task_args_name_the_dataset_rather_than_its_contents(prefills, tmp_path):
    """Every parameter is recorded verbatim; the prefills behind it are megabytes.

    ``layer`` is in here because nothing else in the log holds it — the harness never
    parses a condition id, so the layer is not recoverable from the model name.
    """
    log = run(prefills, tmp_path)

    assert log.eval.task_args == {"dataset": "pilot", "seed": SEED, "layer": 22}
    assert "[system_simulation:0 for 5]" not in repr(log.eval.task_args)


def test_the_task_runs_from_inside_the_repo_so_a_real_run_records_the_commit():
    """`EvalSpec.revision` is git context read from the task's run directory.

    Inspect skips that lookup entirely while ``PYTEST_CURRENT_TEST`` is set
    (``_util/git.py:32``), so no test can observe the recorded commit. This asserts the
    input to the lookup instead: that the directory it will run in is inside the repo.
    """
    import subprocess

    from inspect_ai._eval.task.util import task_run_dir

    run_dir = task_run_dir(refusal_unlock(tiny(), seed=SEED))
    top_level = subprocess.run(
        ["git", "-C", run_dir, "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )

    assert top_level.returncode == 0, top_level.stderr


def test_the_merged_config_that_ran_is_the_frozen_one(prefills, tmp_path):
    log = run(prefills, tmp_path)

    assert log.plan.config.seed == SEED
    assert log.plan.config.temperature == DECODING["temperature"]
    assert log.plan.config.max_tokens == DECODING["max_new_tokens"]
