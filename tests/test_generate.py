"""The sweep's condition list. Everything else in the script needs a GPU."""

from __future__ import annotations

from pathlib import Path

import freeze_abliteration as freeze
import generate
from abliteration.selection import BASE_CONDITION, condition_id
from study import SEED

N_LAYERS = 36


def test_the_sweep_is_the_base_row_then_every_layer():
    conditions = generate.sweep_conditions(N_LAYERS)

    assert len(conditions) == 37
    assert (conditions[0].id, conditions[0].layer) == (BASE_CONDITION, None)
    assert [c.layer for c in conditions[1:]] == list(range(N_LAYERS))


def test_every_condition_runs_the_validation_set_unprefilled_under_one_seed():
    for condition in generate.sweep_conditions(N_LAYERS):
        assert (condition.seed, condition.prompt_set, condition.prefilled) == (
            SEED, "validation", False
        )


def test_the_ids_are_the_ones_the_freeze_step_looks_for():
    """The two scripts agree on the condition set only by both calling condition_id;
    a change to either layer count would otherwise surface as a missing condition."""
    generated = {c.id for c in generate.sweep_conditions(N_LAYERS)}
    expected = {condition_id(layer) for layer in [None, *range(freeze.N_LAYERS)]}

    assert generated == expected


def test_each_condition_resumes_from_its_own_directory():
    """Two conditions sharing a log dir would read each other's samples as progress."""
    dirs = [c.log_dir(Path("/sweep")) for c in generate.sweep_conditions(N_LAYERS)]

    assert len(set(dirs)) == len(dirs)
