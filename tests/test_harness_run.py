"""Resuming a condition: what an interrupted run leaves behind, and what it costs.

Every property here belongs to Inspect rather than to us, which is why they are asserted:
a release that quietly stopped reusing samples would be discovered as a doubled GPU bill
partway through a metered run.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from harness.batching import BATCH
from harness.conditions import Condition
from harness.dataset import build_dataset
from harness.run import run_condition
from inspect_ai.log import (
    list_eval_logs,
    read_eval_log,
    read_eval_log_sample_summaries,
)

SEED = 20260803
PILOT = 30 * 13


def condition() -> Condition:
    return Condition(id="layer_22", seed=SEED, layer=22, prompt_set="pilot")


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
    other = Condition(id="layer_22", seed=SEED, layer=22, prompt_set="strongreject")

    assert condition().log_dir(tmp_path) != other.log_dir(tmp_path)


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


# --- the interruption that actually happens on a pod -----------------------

CHILD = """
import sys, time
from pathlib import Path
sys.path[:0] = ["src", "tests"]
from conftest import FakeTokenizer
from generation.qwen import Continuation
from harness.conditions import Condition
from harness.dataset import build_dataset
from harness.run import run_condition
import harness.batching as batching

calls = {"n": 0}


def fake(model, tok, prompts, *, seed):
    calls["n"] += 1
    if calls["n"] > 2:
        print("READY", flush=True)
        time.sleep(120)
    return [
        Continuation(continuation=" c", raw_continuation=" c",
                     prompt_tokens=3, new_tokens=4)
        for _ in prompts
    ], 0.01


batching.generate_prompts = fake


class Prefills(dict):
    def __missing__(self, key):
        return f"[{key[1]} for {key[0]}]"


c = Condition(id="layer_22", seed=20260803, layer=22, prompt_set="pilot")
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

    child = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for line in child.stdout:
            if line.strip() == "READY":
                break
        else:
            child.kill()
            pytest.fail(f"child never generated: {child.stderr.read()}")
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
