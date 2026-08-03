"""One condition's sample set, and the id scheme resume depends on."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness.conditions import Condition
from harness.dataset import (
    CONTROL,
    EVAL_SETS,
    SLOTS,
    build_dataset,
    sample_id,
    slot_key,
)
from harness.task import refusal_unlock
from inspect_ai import eval
from prefills import PORTFOLIO

SEED = 20260803


def condition(prompt_set="strongreject", **overrides) -> Condition:
    kwargs = {
        "id": "layer_22",
        "seed": SEED,
        "log_root": Path("/tmp/unused"),
        "layer": 22,
        "prompt_set": prompt_set,
    }
    return Condition(**{**kwargs, **overrides})


def by_id(dataset) -> dict:
    return {sample.id: sample for sample in dataset}


# --- the cross product -----------------------------------------------------


def test_every_prompt_is_crossed_with_every_slot(prefills):
    dataset = build_dataset(condition(), prefills)

    assert len(SLOTS) == 14  # 13 portfolio slots plus the unprefilled control
    assert len(dataset) == 313 * 14
    assert len({sample.id for sample in dataset}) == len(dataset)


def test_the_pilot_is_the_same_shape_over_fewer_prompts(prefills):
    dataset = build_dataset(condition("pilot"), prefills)

    assert len(dataset) == 30 * 14


def test_pilot_ids_join_onto_main_run_ids(prefills):
    """The pilot carries the original sparse prompt_ids, not a fresh 0..29 range."""
    pilot = by_id(build_dataset(condition("pilot"), prefills))
    main = by_id(build_dataset(condition(), prefills))

    assert set(pilot) < set(main)
    assert "005/none" in pilot


def test_the_eval_sets_are_the_only_reachable_prompt_sets():
    """Extraction and validation splits must not be addressable as an eval set."""
    assert set(EVAL_SETS) == {"strongreject", "pilot"}


# --- the id scheme ---------------------------------------------------------


def test_no_sample_id_contains_a_colon(prefills):
    """A colon makes the left half a task-name scope, so the id matches nothing."""
    assert sum(":" in slot for slot in PORTFOLIO) == 12

    dataset = build_dataset(condition(), prefills)

    assert not [sample.id for sample in dataset if ":" in sample.id]


def test_substitution_keeps_every_slot_distinct():
    """Two slots colliding would silently merge their rows under one id."""
    assert len({slot_key(slot) for slot in SLOTS}) == len(SLOTS)


def test_an_id_survives_a_sample_id_filter(prefills, tmp_path):
    """Resume applies ids this way; a non-match is silent for the rest of the list."""
    wanted = [sample_id(5, slot) for slot in (None, "system_simulation:0")]

    log = eval(
        refusal_unlock(build_dataset(condition("pilot"), prefills), seed=SEED),
        model="mockllm/model",
        sample_id=wanted,
        log_dir=str(tmp_path),
        score=False,
    )[0]

    assert log.status == "success"
    assert sorted(log.eval.dataset.sample_ids) == sorted(wanted)


def test_the_unsubstituted_id_would_have_matched_nothing(prefills, tmp_path):
    """What the substitution buys: this is the id shape resume would otherwise build."""
    from inspect_ai._util.error import PrerequisiteError

    with pytest.raises(PrerequisiteError):
        eval(
            refusal_unlock(build_dataset(condition("pilot"), prefills), seed=SEED),
            model="mockllm/model",
            sample_id=["005/system_simulation:0"],
            log_dir=str(tmp_path),
            score=False,
        )


# --- prefills come from outside --------------------------------------------


def test_a_missing_prefill_is_refused(prefills):
    """The harness cannot generate one, so there is nothing sensible to fall back to."""
    with pytest.raises(KeyError):
        build_dataset(condition("pilot"), {})


def test_an_empty_prefill_for_a_real_slot_is_refused():
    """Otherwise the row is a control wearing an attack's label."""
    empty = {(pid, slot): "" for pid in range(1000) for slot in PORTFOLIO}

    with pytest.raises(ValueError, match="no attack in it"):
        build_dataset(condition("pilot"), empty)


# --- what a sample looks like ----------------------------------------------


def test_a_prefilled_sample_ends_with_the_prefill_as_an_assistant_turn(prefills):
    samples = by_id(build_dataset(condition("pilot"), prefills))
    sample = samples["005/system_simulation-0"]

    assert [m.role for m in sample.input] == ["user", "assistant"]
    assert sample.input[-1].text == prefills[(5, "system_simulation:0")]
    assert sample.metadata["prefill"] == sample.input[-1].text


def test_the_control_sample_has_no_assistant_turn(prefills):
    sample = by_id(build_dataset(condition("pilot"), prefills))["005/none"]

    assert [m.role for m in sample.input] == ["user"]
    assert sample.metadata["prefill"] == ""
    assert sample.metadata["prefill_slot"] == CONTROL


def test_metadata_carries_what_a_later_grading_process_cannot_rederive(prefills):
    sample = by_id(build_dataset(condition("pilot"), prefills))["005/fake_citation-1"]

    assert sample.metadata["prefill_slot"] == "fake_citation:1"  # the portfolio's name
    assert sample.metadata["prompt_id"] == 5
    assert sample.metadata["condition"] == "layer_22"
    assert sample.metadata["layer"] == 22
    assert sample.metadata["prompt_set"] == "pilot"
    assert sample.metadata["category"]
    assert sample.metadata["source"]
    # The forbidden prompt is the user turn; storing it twice invites disagreement.
    assert "forbidden_prompt" not in sample.metadata
    # One seed covers the task and lives in the log header.
    assert "seed" not in sample.metadata
