"""The sweep's condition list. Everything else in the script needs a GPU."""

from __future__ import annotations

from pathlib import Path

import generate
from abliteration.selection import BASE_CONDITION
from generation.qwen import N_LAYERS
from study import SEED


def test_the_sweep_is_the_base_row_then_every_layer():
    conditions = generate.sweep_conditions()

    assert len(conditions) == 37
    assert (conditions[0].id, conditions[0].layer) == (BASE_CONDITION, None)
    assert [c.layer for c in conditions[1:]] == list(range(N_LAYERS))


def test_every_condition_runs_the_validation_set_unprefilled_under_one_seed():
    for condition in generate.sweep_conditions():
        assert (condition.seed, condition.prompt_set, condition.prefilled) == (
            SEED, "validation", False
        )


def test_each_condition_resumes_from_its_own_directory():
    """Two conditions sharing a log dir would read each other's samples as progress."""
    dirs = [c.log_dir(Path("/sweep")) for c in generate.sweep_conditions()]

    assert len(set(dirs)) == len(dirs)
