"""One condition's sample set, and the id scheme resume depends on."""

from __future__ import annotations

import pytest
from harness.conditions import Condition
from harness.dataset import (
    CONTROL,
    EVAL_SETS,
    build_dataset,
    sample_id,
    slot_key,
)
from harness.task import refusal_unlock
from inspect_ai import eval
from prefills import PORTFOLIO

SEED = 20260803
PROMPTS = 313


def condition(prompt_set="strongreject", **overrides) -> Condition:
    kwargs = {
        "id": "layer_22",
        "seed": SEED,
        "layer": 22,
        "prompt_set": prompt_set,
    }
    return Condition(**{**kwargs, **overrides})


def by_id(dataset) -> dict:
    return {sample.id: sample for sample in dataset}


# --- the two arms ----------------------------------------------------------


def test_an_unprefilled_condition_is_replicates_of_the_bare_prompt(prefills):
    """35 of the 37 layers are unprefilled breadth controls."""
    dataset = build_dataset(condition(), prefills)

    assert len(dataset) == PROMPTS * EVAL_SETS["strongreject"].attempts
    assert {s.metadata["prefill_slot"] for s in dataset} == {CONTROL}


def test_a_prefilled_condition_adds_one_sample_per_portfolio_slot(prefills):
    """Only the aligned base and the composition primary carry the portfolio."""
    dataset = build_dataset(condition(prefilled=True), prefills)

    assert len(dataset) == PROMPTS * (EVAL_SETS["strongreject"].attempts + len(PORTFOLIO))
    assert {s.metadata["prefill_slot"] for s in dataset} == {CONTROL, *PORTFOLIO}


def test_both_arms_are_the_same_width(prefills):
    """The matched budget is the study's control; unequal arms would break it."""
    assert EVAL_SETS["strongreject"].attempts == len(PORTFOLIO) == 13


def test_the_sweep_sums_to_the_preregistered_workload(prefills):
    """313 x 507: 481 unprefilled (13 x 37) + 13 prefill-only + 13 composed."""
    controls = len(build_dataset(condition(), prefills))
    both_arms = len(build_dataset(condition(prefilled=True), prefills))

    assert 35 * controls + 2 * both_arms == PROMPTS * 507


def test_every_sample_id_is_unique(prefills):
    dataset = build_dataset(condition(prefilled=True), prefills)

    assert len({s.id for s in dataset}) == len(dataset)


def test_the_pilot_is_the_same_shape_over_fewer_prompts(prefills):
    dataset = build_dataset(condition("pilot", prefilled=True), prefills)

    assert len(dataset) == 30 * (EVAL_SETS["pilot"].attempts + len(PORTFOLIO))


def test_the_layer_sweep_is_one_attempt_per_validation_prompt(prefills):
    """Selection ranks layers on breadth at k=1 over the 72 held-out behaviors."""
    dataset = build_dataset(condition("validation"), prefills)

    assert len(dataset) == 72
    assert {s.metadata["prefill_slot"] for s in dataset} == {CONTROL}
    assert {s.metadata["replicate"] for s in dataset} == {0}


def test_the_validation_set_is_disjoint_from_the_eval_set(prefills):
    """Selection never sees a prompt the study is scored on."""
    validation = {s.input[0].text for s in build_dataset(condition("validation"), prefills)}
    evaluation = {s.input[0].text for s in build_dataset(condition(), prefills)}

    assert validation & evaluation == set()


def test_pilot_ids_join_onto_main_run_ids(prefills):
    """The pilot carries the original sparse prompt_ids, not a fresh 0..29 range."""
    pilot = by_id(build_dataset(condition("pilot"), prefills))
    main = by_id(build_dataset(condition(), prefills))

    assert set(pilot) < set(main)
    assert "005/none/00" in pilot


# --- the id scheme ---------------------------------------------------------


def test_no_sample_id_contains_a_colon(prefills):
    """A colon makes the left half a task-name scope, so the id matches nothing."""
    assert sum(":" in slot for slot in PORTFOLIO) == 12

    dataset = build_dataset(condition(prefilled=True), prefills)

    assert not [s.id for s in dataset if ":" in s.id]


def test_substitution_keeps_every_slot_distinct():
    """Two slots colliding would silently merge their rows under one id."""
    slots = (CONTROL, *PORTFOLIO)

    assert len({slot_key(slot) for slot in slots}) == len(slots)


def test_replicates_of_one_prompt_differ_only_by_their_index(prefills):
    dataset = build_dataset(condition(), prefills)
    ids = sorted(s.id for s in dataset if s.id.startswith("005/"))

    assert ids == [f"005/none/{n:02d}" for n in range(EVAL_SETS["strongreject"].attempts)]


def test_an_id_survives_a_sample_id_filter(prefills, tmp_path):
    """Resume applies ids this way; a non-match is silent for the rest of the list."""
    wanted = [sample_id(5, CONTROL, 7), sample_id(5, "system_simulation:0", 0)]

    log = eval(
        refusal_unlock(
            build_dataset(condition("pilot", prefilled=True), prefills), seed=SEED
        ),
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
            refusal_unlock(
                build_dataset(condition("pilot", prefilled=True), prefills), seed=SEED
            ),
            model="mockllm/model",
            sample_id=["005/system_simulation:0/00"],
            log_dir=str(tmp_path),
            score=False,
        )


# --- prefills come from outside --------------------------------------------


def test_a_missing_prefill_is_refused(prefills):
    """The harness cannot generate one, so there is nothing sensible to fall back to."""
    with pytest.raises(KeyError):
        build_dataset(condition("pilot", prefilled=True), {})


def test_an_unprefilled_condition_needs_no_prefills():
    """The 35 breadth controls must not depend on a helper run that has not happened."""
    assert len(build_dataset(condition("pilot"), {})) == 30 * EVAL_SETS["pilot"].attempts


def test_an_empty_prefill_for_a_real_slot_is_refused():
    """Otherwise the row is a control wearing an attack's label."""
    empty = {(pid, slot): "" for pid in range(1000) for slot in PORTFOLIO}

    with pytest.raises(ValueError, match="no attack in it"):
        build_dataset(condition("pilot", prefilled=True), empty)


# --- what a sample looks like ----------------------------------------------


def test_a_prefilled_sample_ends_with_the_prefill_as_an_assistant_turn(prefills):
    samples = by_id(build_dataset(condition("pilot", prefilled=True), prefills))
    sample = samples["005/system_simulation-0/00"]

    assert [m.role for m in sample.input] == ["user", "assistant"]
    assert sample.input[-1].text == prefills[(5, "system_simulation:0")]
    assert sample.metadata["prefill"] == sample.input[-1].text


def test_a_control_sample_has_no_assistant_turn(prefills):
    sample = by_id(build_dataset(condition("pilot"), prefills))["005/none/03"]

    assert [m.role for m in sample.input] == ["user"]
    assert sample.metadata["prefill"] == ""
    assert sample.metadata["prefill_slot"] == CONTROL
    assert sample.metadata["replicate"] == 3


def test_metadata_carries_what_a_later_grading_process_cannot_rederive(prefills):
    samples = by_id(build_dataset(condition("pilot", prefilled=True), prefills))
    sample = samples["005/fake_citation-1/00"]

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
