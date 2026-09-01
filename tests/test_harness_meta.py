"""What a log records about the run beyond what Inspect records for itself."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

from generation.qwen3_4b import MODEL_ID, REVISION
from harness.conditions import Condition
from harness.dataset import EVAL_SETS, build_dataset
from harness.meta import run_metadata
from harness.task import refusal_unlock
from inspect_ai import eval
from inspect_ai.log import read_eval_log

MODEL = "qwen3-4b"

SEED = 20260803


def run(prefills, tmp_path):
    c = Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30)
    return eval(
        refusal_unlock(build_dataset(c, prefills), seed=SEED, model=c.model),
        model="mockllm/model",
        sample_id=["005/none/00"],
        log_dir=c.log_dir(tmp_path),
        score=False,
    )[0]


def test_the_metadata_reaches_the_written_header(prefills, tmp_path):
    """On the task rather than the eval call, so a driver cannot leave it off."""
    written = read_eval_log(run(prefills, tmp_path).location, header_only=True)

    assert written.eval.metadata == run_metadata("strongreject", MODEL)


def test_the_weights_the_condition_started_from_are_named(prefills, tmp_path):
    """``eval.model`` is the condition's name and says nothing about the checkpoint."""
    log = run(prefills, tmp_path)

    assert log.eval.model == "mockllm/model"
    assert log.eval.metadata["target_model"] == MODEL_ID
    assert log.eval.metadata["target_revision"] == REVISION


def test_the_environment_that_produced_the_run_reaches_the_header(prefills, tmp_path):
    """``eval.packages`` holds the framework version and nothing that generated tokens."""
    written = read_eval_log(run(prefills, tmp_path).location, header_only=True)

    assert list(written.eval.packages) == ["inspect_ai"]
    assert set(written.eval.metadata["environment"]) == {
        "torch",
        "transformers",
        "gpu",
    }


def test_the_versions_and_the_card_are_read_from_the_running_process(monkeypatch):
    """What the pod records. The suite runs where there is no torch and no GPU."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.11.0+cu128",
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda index: "NVIDIA A100 80GB PCIe",
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(__version__="5.14.1")
    )

    assert run_metadata("strongreject", MODEL)["environment"] == {
        "torch": "2.11.0+cu128",
        "transformers": "5.14.1",
        "gpu": "NVIDIA A100 80GB PCIe",
    }


def test_a_missing_torch_is_recorded_rather_than_raised(monkeypatch):
    """Grading and analysis build the same metadata with no GPU extra installed."""
    monkeypatch.setitem(sys.modules, "torch", None)

    environment = run_metadata("strongreject", MODEL)["environment"]

    assert environment["torch"] is None
    assert environment["gpu"] is None


def test_every_prompt_set_is_hashed_from_a_file_that_exists():
    for prompt_set in EVAL_SETS:
        assert run_metadata(prompt_set, MODEL)["prompt_set_sha256"]


def test_the_prompts_are_hashed_as_they_were_read(prefills, tmp_path):
    """``eval.dataset`` carries a name and a count, never a content hash."""
    log = run(prefills, tmp_path)

    assert log.eval.dataset.name == "strongreject"
    assert (
        log.eval.metadata["prompt_set_sha256"]
        == hashlib.sha256(EVAL_SETS["strongreject"].csv.read_bytes()).hexdigest()
    )

