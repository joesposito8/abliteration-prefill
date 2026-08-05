"""What a log records about the run beyond what Inspect records for itself."""

from __future__ import annotations

import hashlib

from generation.qwen import MODEL_ID, REVISION
from harness.conditions import Condition
from harness.dataset import EVAL_SETS, build_dataset
from harness.meta import run_metadata
from harness.task import refusal_unlock
from inspect_ai import eval
from inspect_ai.log import read_eval_log

SEED = 20260803


def run(prefills, tmp_path):
    c = Condition(id="layer_22", seed=SEED, layer=22, prompt_set="pilot")
    return eval(
        refusal_unlock(build_dataset(c, prefills), seed=SEED),
        model="mockllm/model",
        sample_id=["005/none/00"],
        log_dir=c.log_dir(tmp_path),
        score=False,
    )[0]


def test_the_metadata_reaches_the_written_header(prefills, tmp_path):
    """On the task rather than the eval call, so a driver cannot leave it off."""
    written = read_eval_log(run(prefills, tmp_path).location, header_only=True)

    assert written.eval.metadata == run_metadata("pilot")


def test_the_weights_the_condition_started_from_are_named(prefills, tmp_path):
    """``eval.model`` is the condition's name and says nothing about the checkpoint."""
    log = run(prefills, tmp_path)

    assert log.eval.model == "mockllm/model"
    assert log.eval.metadata["target_model"] == MODEL_ID
    assert log.eval.metadata["target_revision"] == REVISION


def test_every_prompt_set_is_hashed_from_a_file_that_exists():
    for prompt_set in EVAL_SETS:
        assert run_metadata(prompt_set)["prompt_set_sha256"]


def test_the_prompts_are_hashed_as_they_were_read(prefills, tmp_path):
    """``eval.dataset`` carries a name and a count, never a content hash."""
    log = run(prefills, tmp_path)

    assert log.eval.dataset.name == "pilot"
    assert (
        log.eval.metadata["prompt_set_sha256"]
        == hashlib.sha256(EVAL_SETS["pilot"].csv.read_bytes()).hexdigest()
    )

