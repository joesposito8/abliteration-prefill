"""The run's condition list, on both paths. Everything else in the script needs a GPU."""

from __future__ import annotations

from pathlib import Path

import generate
import pytest
from abliteration.selection import BASE_CONDITION
from generation import phi4, qwen3_4b
from study import SEED

QWEN = "qwen3-4b"
PHI = "phi-4"


def refuse(*args):
    raise AssertionError("read a frozen artifact this run does not have")


def test_the_selection_sweep_is_the_base_row_then_every_layer():
    conditions = generate.eval_conditions("validation", qwen3_4b, None)

    assert len(conditions) == 37
    assert (conditions[0].id, conditions[0].layer) == (BASE_CONDITION, None)
    assert [c.layer for c in conditions[1:]] == list(range(qwen3_4b.N_LAYERS))


def test_an_evaluation_run_is_the_two_model_states():
    """The factorial is base against the primary; the other 35 layers are not run here."""
    conditions = generate.eval_conditions("strongreject", qwen3_4b, 23)

    assert [(c.id, c.layer) for c in conditions] == [
        (BASE_CONDITION, None),
        ("layer_23", 23),
    ]


def test_the_run_follows_the_target_it_is_given():
    """One condition per layer plus the base, so a deeper model gets a longer run."""
    conditions = generate.eval_conditions("validation", phi4, None)

    assert len(conditions) == phi4.N_LAYERS + 1 == 41
    assert {c.model for c in conditions} == {PHI}
    assert [c.layer for c in conditions[1:]] == list(range(phi4.N_LAYERS))


def test_every_condition_runs_the_named_set_under_one_seed():
    for condition in generate.eval_conditions("strongreject", qwen3_4b, 23):
        assert (condition.seed, condition.prompt_set) == (SEED, "strongreject")


def test_both_model_states_carry_every_prefill_level():
    """"none" is the control level of the prefill factor, not a condition of its own."""
    conditions = generate.eval_conditions("strongreject", qwen3_4b, 23)

    assert all(c.prefilled for c in conditions)


def test_the_primary_follows_whichever_layer_the_freeze_selected():
    """A layer index means nothing except against the extraction that produced it."""
    conditions = generate.eval_conditions("strongreject", qwen3_4b, 7)

    assert {c.id for c in conditions if c.prefilled} == {BASE_CONDITION, "layer_07"}


def test_each_condition_resumes_from_its_own_directory():
    """Two conditions sharing a log dir would read each other's samples as progress."""
    dirs = [c.log_dir(Path("/sweep")) for c in generate.eval_conditions("strongreject", qwen3_4b, 23)]

    assert len(set(dirs)) == len(dirs)


def test_a_prompt_limit_reaches_every_condition():
    conditions = generate.eval_conditions("strongreject", qwen3_4b, 23, 30)

    assert {c.num_prompts for c in conditions} == {30}
    assert {c.prompt_set for c in conditions} == {"strongreject"}


def test_a_partial_run_does_not_resume_into_the_full_one():
    """``eval_set`` calls a log complete on sample count against the dataset it was handed,
    so a 30-prompt log in the full run's directory would satisfy it and skip the rest."""
    partial = generate.eval_conditions("strongreject", qwen3_4b, 23, 30)
    full = generate.eval_conditions("strongreject", qwen3_4b, 23)

    assert {c.log_dir(Path("/run")) for c in partial}.isdisjoint(
        c.log_dir(Path("/run")) for c in full
    )


def test_the_selection_sweep_reads_neither_a_primary_nor_the_prefills(monkeypatch):
    """Chicken and egg: this run is what writes the manifest a primary is read from, so
    on a first target neither it nor the prefills it would compose with exist yet."""
    monkeypatch.setattr(generate, "primary_layer", refuse)
    monkeypatch.setattr(generate, "load_prefills", refuse)

    conditions, prefills = generate.plan("validation", phi4)

    assert prefills == {}
    assert not any(c.prefilled for c in conditions)
    for condition in conditions:
        assert (condition.seed, condition.prompt_set) == (SEED, "validation")


def test_an_evaluation_run_reads_the_primary_and_the_frozen_prefills(monkeypatch):
    monkeypatch.setattr(generate, "primary_layer", lambda module: 23)
    monkeypatch.setattr(generate, "load_prefills", lambda: {(0, "static_baseline"): "Sure,"})

    conditions, prefills = generate.plan("strongreject", qwen3_4b)

    assert prefills == {(0, "static_baseline"): "Sure,"}
    assert {c.id for c in conditions if c.prefilled} == {BASE_CONDITION, "layer_23"}


def test_a_run_without_a_prompt_set_refuses_to_start():
    """The three sets differ by orders of magnitude in GPU time, so a default would let a
    mistyped command spend the eval set's budget on a run that meant the sweep."""
    with pytest.raises(SystemExit):
        generate.main(["results/sweep/generated", QWEN])
