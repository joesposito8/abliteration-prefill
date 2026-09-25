"""The count table: one cell per (prompt, state, level), draws counted rather than assumed."""

from __future__ import annotations

import numpy as np
import pytest
from analysis.counts import ABL, BASE, LEVELS, NONE, Counts, resolve
from analysis.read import derived
from harness.dataset import CONTROL
from inspect_ai.scorer import SampleScore, Score
from prefills import PORTFOLIO
from study import draws

STATES = {"base": "base", "layer_23": "abliterated"}


def scores(prompt_ids=(5, 9)) -> list[SampleScore]:
    """Every cell at its scheduled draws, through the reader's own derivation.
    persona_switch unlocks on one variant only, and one abliterated control draw is
    malformed."""
    out = []
    for condition, state in STATES.items():
        for prompt_id in prompt_ids:
            for slot in (CONTROL, *PORTFOLIO):
                for draw in range(draws(slot)):
                    malformed = int(condition == "layer_23" and slot == CONTROL and draw == 0)
                    value = {
                        "unlocked": int(slot == "persona_switch:0" and not malformed),
                        "malformed": malformed,
                    }
                    out.append(
                        SampleScore(
                            score=Score(value=value),
                            sample_id=f"{condition}/{prompt_id:03d}/{slot}/{draw:02d}",
                            sample_metadata=derived(
                                {"prompt_id": prompt_id, "prefill_slot": slot}, state
                            ),
                            scorer="strongreject",
                        )
                    )
    return out


def test_a_family_pools_its_two_variants_and_draws_are_counted():
    counts = Counts.from_scores(scores())
    persona = LEVELS.index("persona_switch")

    assert counts.prompt_ids.tolist() == [5, 9]
    assert counts.u.shape == (2, 2, 8)
    assert (counts.n == 20).all()
    assert counts.u[:, BASE, persona].tolist() == [10, 10]
    assert counts.u[:, ABL, NONE].tolist() == [0, 0]
    assert counts.malformed[:, ABL, NONE].tolist() == [1, 1]
    assert counts.malformed.sum() == 2


def test_a_cell_with_no_rows_is_refused():
    held = [
        s for s in scores()
        if not (s.sample_metadata["state"] == "base"
                and s.sample_metadata["prompt_id"] == 9
                and s.sample_metadata["level"] == CONTROL)
    ]

    with pytest.raises(ValueError, match="no rows"):
        Counts.from_scores(held)


def test_a_cell_short_of_its_draws_is_refused():
    with pytest.raises(ValueError, match="draw schedule"):
        Counts.from_scores(scores()[1:])


def test_resolving_adds_the_malformed_rows_inside_the_mask_only():
    counts = Counts.from_scores(scores())

    corner = resolve(counts, ABL, [NONE])

    assert corner.u[:, ABL, NONE].tolist() == [1, 1]
    assert np.array_equal(np.delete(corner.u, NONE, axis=2), np.delete(counts.u, NONE, axis=2))
    assert corner.u[:, BASE, NONE].tolist() == [0, 0]
    assert counts.u[:, ABL, NONE].tolist() == [0, 0]  # the original is untouched
