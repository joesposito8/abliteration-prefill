"""The two ways the smoke run fails without saying so."""

from __future__ import annotations

import harness_smoke as smoke
from prefills.families import STATIC_SLOT_ID


def test_the_dataset_builds_and_says_which_prefills_were_real():
    """A signature change here surfaces on pod day at an hourly rate, and an unlabelled
    log reads as study data. Everything else about this run is visible in its output."""
    synthetic = {
        sample.metadata["prefill_slot"]: sample.metadata["synthetic_prefill"]
        for sample in smoke.smoke_dataset()
    }

    assert synthetic["none"] is False
    assert synthetic[STATIC_SLOT_ID] is False
    assert all(synthetic[slot] for slot in smoke.HELPER_SLOTS)
