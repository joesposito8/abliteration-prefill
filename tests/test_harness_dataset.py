"""One condition's sample set, and the id scheme resume depends on."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

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
from prefills import FAMILIES, PORTFOLIO, family_of
from prefills.families import STATIC_SLOT_ID
from study import draws

MODEL = "qwen3-4b"

SEED = 20260803
PROMPTS = 313


def condition(prompt_set="strongreject", **overrides) -> Condition:
    kwargs = {
        "id": "layer_22",
        "seed": SEED,
        "layer": 22,
        "prompt_set": prompt_set,
    }
    return Condition(model=MODEL, **{**kwargs, **overrides})


def by_id(dataset) -> dict:
    return {sample.id: sample for sample in dataset}


# --- the grid ---------------------------------------------------------------


def test_an_unprefilled_condition_is_the_controls_draws_of_the_bare_prompt(prefills):
    dataset = build_dataset(condition(), prefills)

    assert len(dataset) == PROMPTS * draws(CONTROL) == 6_260
    assert {s.metadata["prefill_slot"] for s in dataset} == {CONTROL}


def test_a_prefilled_condition_draws_every_prompt_at_every_portfolio_slot(prefills):
    """Both model states carry all 14 levels; the other layers are not run on this set."""
    dataset = build_dataset(condition(prefilled=True), prefills)

    assert len(dataset) == PROMPTS * sum(draws(s) for s in (CONTROL, *PORTFOLIO))
    assert len(dataset) == PROMPTS * 160
    assert {s.metadata["prefill_slot"] for s in dataset} == {CONTROL, *PORTFOLIO}


@pytest.mark.parametrize(
    "schedule,default,slots", [({CONTROL: 3}, 1, 1), ({CONTROL: 2, "slot_3": 7}, 2, 5)]
)
def test_the_schedule_and_the_portfolio_size_are_independent(
    monkeypatch, prefills, schedule, default, slots
):
    """The schedule's per-slot draws and the portfolio's size are different numbers. v1
    tied them to one r; the grid is now the schedule summed over the slots present, under
    any schedule and any portfolio size. Counted per slot rather than only in total,
    because a right total over a wrong spread is the failure that would still ship."""
    portfolio = tuple(f"slot_{i}" for i in range(slots))
    monkeypatch.setattr("harness.dataset.PORTFOLIO", portfolio)
    monkeypatch.setitem(
        EVAL_SETS,
        "strongreject",
        replace(
            EVAL_SETS["strongreject"], draws=lambda slot: schedule.get(slot, default)
        ),
    )

    dataset = build_dataset(condition(num_prompts=30, prefilled=True), prefills)

    per_slot = Counter(s.metadata["prefill_slot"] for s in dataset)
    assert per_slot == {
        slot: 30 * schedule.get(slot, default) for slot in (CONTROL, *portfolio)
    }
    assert len(dataset) == 30 * sum(
        schedule.get(slot, default) for slot in (CONTROL, *portfolio)
    )


def test_the_control_and_the_static_baseline_draw_twenty(prefills):
    """The frozen schedule itself, per slot: 20, 20, and 10 for the other twelve."""
    per_slot = Counter(
        s.metadata["prefill_slot"]
        for s in build_dataset(condition(num_prompts=30, prefilled=True), prefills)
    )

    assert per_slot[CONTROL] == per_slot[STATIC_SLOT_ID] == 30 * 20
    assert {per_slot[s] for s in PORTFOLIO if s != STATIC_SLOT_ID} == {30 * 10}


def test_every_family_gets_the_same_draws_of_every_prompt(prefills):
    """Why those three numbers: n is counted per family, not per slot. The control, the
    static baseline and each of the six generated families get 20 draws of every prompt,
    the generated ones splitting theirs over two variants."""
    per_family = Counter(
        family_of(s.metadata["prefill_slot"])
        for s in build_dataset(condition(num_prompts=30, prefilled=True), prefills)
    )

    assert set(per_family) == {CONTROL, STATIC_SLOT_ID, *FAMILIES}
    assert set(per_family.values()) == {30 * 20}


def test_a_prefilled_condition_is_the_preregistered_workload(prefills):
    """313 x 160 per model state, and the two states are the preregistered 313 x 320."""
    prefilled = len(build_dataset(condition(prefilled=True), prefills))

    assert prefilled == 50_080
    assert 2 * prefilled == PROMPTS * 320 == 100_160


def test_no_cell_is_enumerated_twice(prefills):
    """What the run manifest was for: a malformed grid, caught before a GPU is rented."""
    cells = [
        (s.metadata["prompt_id"], s.metadata["prefill_slot"], s.metadata["replicate"])
        for s in build_dataset(condition(prefilled=True), prefills)
    ]

    assert len(set(cells)) == len(cells) == 50_080


def test_every_sample_id_is_unique(prefills):
    dataset = build_dataset(condition(prefilled=True), prefills)

    assert len({s.id for s in dataset}) == len(dataset)


def test_a_subset_run_is_the_same_shape_over_fewer_prompts(prefills):
    dataset = build_dataset(condition(num_prompts=30, prefilled=True), prefills)

    assert len(dataset) == 30 * 160 == 4_800


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


def test_subset_ids_join_onto_full_run_ids(prefills):
    """A subset run carries the set's own sparse prompt_ids, not a fresh 0..29 range, so
    its rows join onto a full run's rather than colliding with different prompts."""
    subset = by_id(build_dataset(condition(num_prompts=30), prefills))
    full = by_id(build_dataset(condition(), prefills))

    assert set(subset) < set(full)
    assert "005/none/00" in subset


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

    assert ids == [f"005/none/{n:02d}" for n in range(draws(CONTROL))]


def test_an_id_survives_a_sample_id_filter(prefills, tmp_path):
    """Resume applies ids this way; a non-match is silent for the rest of the list."""
    wanted = [sample_id(5, CONTROL, 7), sample_id(5, "system_simulation:0", 0)]

    log = eval(
        refusal_unlock(
            build_dataset(condition(num_prompts=30, prefilled=True), prefills),
            seed=SEED,
            model=MODEL,
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
                build_dataset(condition(num_prompts=30, prefilled=True), prefills),
                seed=SEED,
                model=MODEL,
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
        build_dataset(condition(num_prompts=30, prefilled=True), {})


def test_an_unprefilled_condition_needs_no_prefills():
    """The selection sweep must not depend on a helper run that has not happened."""
    assert len(build_dataset(condition(num_prompts=30), {})) == 30 * draws(CONTROL) == 600


def test_an_empty_prefill_for_a_real_slot_is_refused():
    """Otherwise the row is a control wearing an attack's label."""
    empty = {(pid, slot): "" for pid in range(1000) for slot in PORTFOLIO}

    with pytest.raises(ValueError, match="no attack in it"):
        build_dataset(condition(num_prompts=30, prefilled=True), empty)


# --- what a sample looks like ----------------------------------------------


def test_a_prefilled_sample_ends_with_the_prefill_as_an_assistant_turn(prefills):
    samples = by_id(build_dataset(condition(num_prompts=30, prefilled=True), prefills))
    sample = samples["005/system_simulation-0/00"]

    assert [m.role for m in sample.input] == ["user", "assistant"]
    assert sample.input[-1].text == prefills[(5, "system_simulation:0")]
    assert sample.metadata["prefill"] == sample.input[-1].text


def test_a_control_sample_has_no_assistant_turn(prefills):
    sample = by_id(build_dataset(condition(num_prompts=30), prefills))["005/none/03"]

    assert [m.role for m in sample.input] == ["user"]
    assert sample.metadata["prefill"] == ""
    assert sample.metadata["prefill_slot"] == CONTROL
    assert sample.metadata["replicate"] == 3


def test_metadata_carries_what_a_later_grading_process_cannot_rederive(prefills):
    samples = by_id(build_dataset(condition(num_prompts=30, prefilled=True), prefills))
    sample = samples["005/fake_citation-1/00"]

    assert sample.metadata["prefill_slot"] == "fake_citation:1"  # the portfolio's name
    assert sample.metadata["prompt_id"] == 5
    assert sample.metadata["condition"] == "layer_22"
    assert sample.metadata["layer"] == 22
    assert sample.metadata["prompt_set"] == "strongreject"
    assert sample.metadata["category"]
    assert sample.metadata["source"]
    # The forbidden prompt is the user turn; storing it twice invites disagreement.
    assert "forbidden_prompt" not in sample.metadata
    # One seed covers the task and lives in the log header.
    assert "seed" not in sample.metadata
