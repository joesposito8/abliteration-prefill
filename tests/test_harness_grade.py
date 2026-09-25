"""Grading a generated sweep: what the scored log records and what it leaves alone."""

from __future__ import annotations

import hashlib
from pathlib import Path

import grade
import pytest
from harness.conditions import Condition
from harness.dataset import build_dataset
from harness.run import run_condition
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import list_eval_logs, read_eval_log

MODEL = "qwen3-4b"

SEED = 20260803
SAMPLES = 6

REFUSAL = "1.b 0\n2.b 1\n3.b 1"


def generate(prefills, tokenizer, tmp_path, *, fail_first=False, injected=None):
    """One condition's generation logs, small enough to judge every sample."""
    c = Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30)
    rows = list(build_dataset(c, prefills))[:SAMPLES]

    if fail_first:
        injected.fail_after = 0
        with pytest.raises(RuntimeError):
            run_condition(
                c,
                MemoryDataset(rows, name="strongreject"),
                module=object(),
                tokenizer=tokenizer,
                root=tmp_path / "generated",
            )
        injected.fail_after = None

    run_condition(
        c,
        MemoryDataset(rows, name="strongreject"),
        module=object(),
        tokenizer=tokenizer,
        root=tmp_path / "generated",
    )
    return tmp_path / "generated", tmp_path / "scored"


@pytest.fixture
def judged_as_refusal(monkeypatch, fake_judge):
    """Replaces the pin with a local judge, through the header the real one uses."""
    judge = fake_judge(lambda _: REFUSAL)
    monkeypatch.setattr(grade, "GRADER", judge.role)
    return judge


def graded(source, output):
    return grade.grade_sweep(source, output)


# --- what the pin is, and where it is written ------------------------------


def test_the_scored_log_names_the_judge_that_produced_it(
    prefills, tokenizer, tmp_path, fake_generate_prompts, judged_as_refusal
):
    """``score`` reads roles from the header and never writes them back, so a log graded
    without this can only be asked which judge it used one sample at a time."""
    source, output = generate(prefills, tokenizer, tmp_path)

    [log] = graded(source, output)
    written = read_eval_log(log.location, header_only=True)

    assert written.eval.model_roles["grader"].model == "mockllm/model"


def test_the_judge_is_resolved_from_that_header(
    prefills, tokenizer, tmp_path, fake_generate_prompts, judged_as_refusal
):
    """Nothing passes the role to the scorer; the header is the only declaration."""
    source, output = generate(prefills, tokenizer, tmp_path)

    graded(source, output)

    assert len(judged_as_refusal.judged) == SAMPLES


# --- what grading reads and writes -----------------------------------------


def test_every_sample_is_scored(
    prefills, tokenizer, tmp_path, fake_generate_prompts, judged_as_refusal
):
    source, output = generate(prefills, tokenizer, tmp_path)

    [log] = graded(source, output)
    scored = read_eval_log(log.location)

    assert len(scored.samples) == SAMPLES
    assert all("strongreject" in sample.scores for sample in scored.samples)


def test_the_generated_log_is_left_byte_identical(
    prefills, tokenizer, tmp_path, fake_generate_prompts, judged_as_refusal
):
    """It is what the GPU produced and cost the money; grading may be re-run."""
    source, output = generate(prefills, tokenizer, tmp_path)
    [info] = list_eval_logs(str(source))
    generated = grade._local(info.name)
    before = hashlib.sha256(generated.read_bytes()).hexdigest()

    graded(source, output)

    assert hashlib.sha256(generated.read_bytes()).hexdigest() == before


def test_a_regrade_keeps_the_grades_it_found(
    prefills, tokenizer, tmp_path, fake_generate_prompts, judged_as_refusal
):
    """The judge is the study's instrument, so two of its readings are compared rather
    than one replacing the other."""
    source, output = generate(prefills, tokenizer, tmp_path)
    [first] = graded(source, output)

    again = grade.grade_log(Path(first.location), tmp_path / "regraded.eval")

    assert [list(sample.scores) for sample in again.samples] == (
        [["strongreject", "strongreject1"]] * SAMPLES
    )


def test_only_the_finished_log_of_a_condition_is_graded(
    prefills, tokenizer, tmp_path, fake_generate_prompts, judged_as_refusal
):
    """A failed attempt stays in the directory, holding a fraction of the samples."""
    source, output = generate(
        prefills, tokenizer, tmp_path, fail_first=True, injected=fake_generate_prompts
    )
    assert len(list_eval_logs(str(source))) == 2

    logs = graded(source, output)

    assert len(logs) == 1
    assert logs[0].results.total_samples == SAMPLES
