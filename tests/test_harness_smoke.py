"""The smoke run's dataset: which of its prefills are real and which stand in."""

from __future__ import annotations

import harness_smoke as smoke
from prefills import STATIC_BASELINE, validate
from prefills.families import STATIC_SLOT_ID

PILOT = 30
ARM = 13


def test_it_is_the_full_cross_product_a_prefilled_condition_runs():
    """The point is to exercise the real shape, not a reduced one."""
    dataset = smoke.smoke_dataset()

    assert len(dataset) == PILOT * ARM * 2


def test_the_static_baseline_carries_its_real_text():
    """It is the one slot needing no helper model, so it is not stood in for."""
    prefills = smoke.smoke_prefills()

    assert {v for (_, slot), v in prefills.items() if slot == STATIC_SLOT_ID} == {
        STATIC_BASELINE
    }


def test_every_synthetic_prefill_could_have_come_from_the_helper():
    """A shape the validators would have rejected exercises nothing real."""
    for (prompt_id, slot), prefill in smoke.smoke_prefills().items():
        assert validate(prefill) is None, (prompt_id, slot, prefill)


def test_the_synthetic_prefills_vary_by_slot_and_by_prompt():
    """Prefills of one length pad evenly and hide what a mixed batch costs."""
    prefills = smoke.smoke_prefills()
    first = smoke.HELPER_SLOTS[0]

    assert len({prefills[(5, slot)] for slot in smoke.HELPER_SLOTS}) == len(
        smoke.SYNTHETIC
    )
    assert prefills[(5, first)] != prefills[(9, first)]


def test_every_sample_says_whether_its_prefill_was_real():
    """This log is plumbing evidence, and nothing in it may read as study data."""
    synthetic = {
        sample.metadata["prefill_slot"]: sample.metadata["synthetic_prefill"]
        for sample in smoke.smoke_dataset()
    }

    assert synthetic["none"] is False
    assert synthetic[STATIC_SLOT_ID] is False
    assert all(synthetic[slot] for slot in smoke.HELPER_SLOTS)


def test_the_condition_is_unedited_and_named_for_what_it_is():
    assert smoke.CONDITION.layer is None
    assert smoke.CONDITION.model_name == "qwen-local/smoke_base"
