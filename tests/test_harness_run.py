"""Resuming a condition, and the weights each condition of a sweep generates under.

Every resume property here belongs to Inspect rather than to us, which is why they are
asserted: a release that quietly stopped reusing samples would be discovered as a doubled
GPU bill partway through a metered run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

import pytest
from generation.qwen3_4b import BATCH
from harness.conditions import Condition, condition_of
from harness.dataset import CONTROL, build_dataset
from harness.run import run_condition, run_sweep
from inspect_ai.log import (
    list_eval_logs,
    read_eval_log,
    read_eval_log_sample_summaries,
)
from study import draws

MODEL = "qwen3-4b"
SEED = 20260803
PILOT = 30 * draws(CONTROL)


def condition() -> Condition:
    return Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30)


def run(prefills, tokenizer, tmp_path):
    c = condition()
    return run_condition(
        c,
        build_dataset(c, prefills),
        module=object(),
        tokenizer=tokenizer,
        root=tmp_path,
    )


def settled(tmp_path) -> list[str]:
    """Every sample id generated for the condition, across whatever logs it has."""
    return [
        str(summary.id)
        for info in list_eval_logs(condition().log_dir(tmp_path))
        for summary in read_eval_log_sample_summaries(info)
        if summary.error is None and summary.limit is None
    ]


def statuses(tmp_path) -> list[str]:
    return [
        read_eval_log(info, header_only=True).status
        for info in list_eval_logs(condition().log_dir(tmp_path))
    ]


def test_two_prompt_sets_do_not_share_a_log_dir(tmp_path):
    """A dataset hashes into task identity as its name alone, so sharing a dir would
    make the smaller set's rows read as progress on the larger."""
    other = Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject")

    assert condition().log_dir(tmp_path) != other.log_dir(tmp_path)


def test_two_targets_do_not_share_a_log_dir_or_a_task_identity(tmp_path):
    """Condition ids carry no model, so the same layer of two targets is the same id,
    and a shared provider would make ``eval_set`` resume one as the other."""
    other = Condition(model="phi-4", id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30)

    assert condition().model_name != other.model_name
    assert condition().log_dir(tmp_path) != other.log_dir(tmp_path)


def test_the_condition_is_recoverable_from_a_log_model_name():
    """The freeze step reads it back out of ``log.eval.model``, over a tree that holds
    both the three-segment names written now and the two-segment archived ones."""
    assert condition_of(condition().model_name) == condition().id
    assert condition_of("qwen-local/layer_09") == "layer_09"


def test_one_condition_generates_at_a_time(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    """The module is one mutable object, so two conditions in flight would both
    generate under whichever edit was applied last, each log still naming its own."""
    log = run(prefills, tokenizer, tmp_path)

    assert log.eval.config.max_tasks == 1
    assert log.eval.config.epochs == 1
    assert log.eval.config.log_buffer == BATCH


# --- what a second run costs -----------------------------------------------


def test_a_finished_condition_is_not_generated_again(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    run(prefills, tokenizer, tmp_path)
    first = fake_generate_prompts.rows

    run(prefills, tokenizer, tmp_path)

    assert first == PILOT
    assert fake_generate_prompts.rows == PILOT


def test_an_interrupted_condition_generates_only_what_is_missing(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    fake_generate_prompts.fail_after = 2
    with pytest.raises(RuntimeError, match="did not finish"):
        run(prefills, tokenizer, tmp_path)

    done = len(settled(tmp_path))
    assert 0 < done < PILOT

    attempted = fake_generate_prompts.rows
    fake_generate_prompts.fail_after = None
    run(prefills, tokenizer, tmp_path)

    assert fake_generate_prompts.rows - attempted == PILOT - done


def test_the_finished_condition_holds_each_sample_exactly_once(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    """What lets analysis pool a resumed condition without choosing between rows."""
    fake_generate_prompts.fail_after = 2
    with pytest.raises(RuntimeError, match="did not finish"):
        run(prefills, tokenizer, tmp_path)
    fake_generate_prompts.fail_after = None

    log = run(prefills, tokenizer, tmp_path)
    ids = [str(s.id) for s in read_eval_log_sample_summaries(log.location)]

    assert log.results.total_samples == PILOT
    assert sorted(ids) == sorted({*ids}) and len(ids) == PILOT


# --- what an interrupted run leaves behind ---------------------------------


def test_a_failed_run_keeps_its_log_to_be_diagnosed_from(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    """The prompt behind a memory error is in that log and nowhere else."""
    fake_generate_prompts.fail_after = 2

    with pytest.raises(RuntimeError, match="did not finish"):
        run(prefills, tokenizer, tmp_path)

    assert statuses(tmp_path) == ["error"]


def test_exactly_one_log_reports_success(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    """Grading runs later and finds the condition's result by that status."""
    fake_generate_prompts.fail_after = 2
    with pytest.raises(RuntimeError, match="did not finish"):
        run(prefills, tokenizer, tmp_path)
    fake_generate_prompts.fail_after = None
    run(prefills, tokenizer, tmp_path)

    assert sorted(statuses(tmp_path)) == ["error", "success"]


# --- which weights each condition of a sweep generates under ---------------

# Indexed by layer, so an edit recorded during generation names the layer that applied it.
DIRECTIONS = list(range(40))



@pytest.fixture
def edits_when_generating(
    monkeypatch, weights, fake_generate_prompts
) -> list[list[int]]:
    """What the module held at each forward pass, since the log cannot record it."""
    seen: list[list[int]] = []

    def recording(model, tok, prompts, *, seed, decoding):
        seen.append(list(model.edits))
        return fake_generate_prompts(model, tok, prompts, seed=seed, decoding=decoding)

    monkeypatch.setattr("harness.provider.generate_prompts", recording)
    return seen


def sweep(conditions, prefills, tokenizer, weights, tmp_path):
    return run_sweep(
        conditions,
        prefills,
        module=weights,
        tokenizer=tokenizer,
        directions=DIRECTIONS,
        root=tmp_path,
    )


def test_each_condition_generates_under_its_own_edit_and_nothing_else(
    prefills, tokenizer, tmp_path, weights, edits_when_generating
):
    """An edit left in place would have every later condition generate under weights
    that compound, with each log still naming its own layer."""
    conditions = [
        Condition(model=MODEL, id="base", seed=SEED, layer=None, prompt_set="strongreject", num_prompts=30),
        Condition(model=MODEL, id="layer_02", seed=SEED, layer=2, prompt_set="strongreject", num_prompts=30),
        Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30),
    ]

    logs = sweep(conditions, prefills, tokenizer, weights, tmp_path)

    assert [log.eval.model for log in logs] == [c.model_name for c in conditions]
    assert {tuple(e) for e in edits_when_generating} == {(), (2,), (22,)}
    assert weights.edits == []


def test_layer_zero_is_edited_like_every_other_layer(
    prefills, tokenizer, tmp_path, weights, edits_when_generating
):
    """Only ``None`` means unedited; layer 0 is a layer."""
    sweep(
        [Condition(model=MODEL, id="layer_00", seed=SEED, layer=0, prompt_set="strongreject", num_prompts=30)],
        prefills,
        tokenizer,
        weights,
        tmp_path,
    )

    assert {tuple(e) for e in edits_when_generating} == {(0,)}


def test_a_failed_condition_leaves_the_base_weights_loaded(
    prefills, tokenizer, tmp_path, weights, fake_generate_prompts
):
    """The process survives the raise, so an edit left behind would reach whatever the
    operator runs next in it."""
    fake_generate_prompts.fail_after = 2

    with pytest.raises(RuntimeError, match="did not finish"):
        sweep(
            [Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30)],
            prefills,
            tokenizer,
            weights,
            tmp_path,
        )

    assert weights.edits == []


def test_a_restore_that_did_not_take_stops_the_sweep(
    monkeypatch, prefills, tokenizer, tmp_path, weights, fake_generate_prompts
):
    """The failure this catches is silent otherwise: the log names the layer it was
    asked for either way, so every later condition would carry this one's edit."""
    monkeypatch.setattr("harness.run.restore_targets", lambda m, base: None)
    conditions = [
        Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30),
        Condition(model=MODEL, id="layer_23", seed=SEED, layer=23, prompt_set="strongreject", num_prompts=30),
    ]

    with pytest.raises(RuntimeError, match="layer_22: restore left weights modified"):
        sweep(conditions, prefills, tokenizer, weights, tmp_path)

    assert not list_eval_logs(conditions[1].log_dir(tmp_path))


def test_the_sweep_stops_at_a_failed_condition(
    prefills, tokenizer, tmp_path, weights, fake_generate_prompts
):
    """Continuing would spend GPU hours on the conditions after a systematic failure."""
    fake_generate_prompts.fail_after = 2
    conditions = [
        Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="strongreject", num_prompts=30),
        Condition(model=MODEL, id="layer_23", seed=SEED, layer=23, prompt_set="strongreject", num_prompts=30),
    ]

    with pytest.raises(RuntimeError, match="layer_22 did not finish"):
        sweep(conditions, prefills, tokenizer, weights, tmp_path)

    assert not list_eval_logs(conditions[1].log_dir(tmp_path))


# --- the interruption that actually happens on a pod -----------------------

CHILD = """
import sys, time
from pathlib import Path
sys.path[:0] = ["src", "tests"]
from conftest import Weights, FakeTokenizer
from generation.batched import Continuation
from generation.qwen3_4b import BATCH
from harness.conditions import Condition, condition_of
from harness.dataset import CONTROL, build_dataset
from harness.run import run_condition
import harness.provider as provider

MODEL = "qwen3-4b"

calls = {"n": 0}


def fake(model, tok, prompts, *, seed, decoding):
    calls["n"] += 1
    if calls["n"] > 2:
        Path(sys.argv[1], "READY").write_text("x")
        time.sleep(300)
    return [
        Continuation(continuation=" c", raw_continuation=" c",
                     prompt_tokens=3, new_tokens=4)
        for _ in prompts
    ], 0.01


provider.generate_prompts = fake


class Prefills(dict):
    def __missing__(self, key):
        return f"[{key[1]} for {key[0]}]"


c = Condition(model=MODEL, id="layer_22", seed=20260803, layer=22, prompt_set="strongreject", num_prompts=30)
run_condition(c, build_dataset(c, Prefills()), module=object(),
              tokenizer=FakeTokenizer(), root=Path(sys.argv[1]))
"""


def test_a_killed_run_resumes_from_what_reached_the_log(
    prefills, tokenizer, tmp_path, fake_generate_prompts
):
    """A killed process leaves no finished log at all, so this recovers from the
    journal rather than from a header — a different path than a failed run takes."""
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(CHILD))

    # The child signals through the filesystem: Inspect takes over stdout for its
    # display, so a pipe is not a reliable channel out of a run.
    ready = tmp_path / "READY"
    output = tmp_path / "child.out"
    with output.open("w") as sink:
        child = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path)], stdout=sink, stderr=sink
        )
        try:
            deadline = time.monotonic() + 120
            while not ready.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    pytest.fail(f"child never generated: {output.read_text()[-2000:]}")
                time.sleep(0.2)
        finally:
            child.kill()
            child.wait(timeout=30)

    flushed = len(settled(tmp_path))
    assert 0 < flushed < PILOT

    log = run(prefills, tokenizer, tmp_path)

    assert log.results.total_samples == PILOT
    # At most, because recovery reads the sample buffer as well as the log, so it keeps
    # samples that finished after the last flush and would otherwise be regenerated.
    assert 0 < fake_generate_prompts.rows <= PILOT - flushed
