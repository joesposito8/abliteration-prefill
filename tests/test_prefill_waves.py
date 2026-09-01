"""What each wave asks for, replayed against draws a helper never had to make.

The whole convergence argument is decidable offline: ``helper_seed`` is pure, so the
draws the rules request against a given set of outcomes are fixed, and a stand-in for
the helper's text is enough to exercise every path the real run can take.
"""

from __future__ import annotations

from harness.prefill_task import HelperDraw
from prefills import (
    FAMILIES,
    MAX_ATTEMPTS,
    STATIC_BASELINE,
    VARIANTS_PER_FAMILY,
    helper_seed,
)
from study.datasets import load_strongreject_prompts

from produce_prefills import replay, waves

PROMPTS = 313
CELLS = PROMPTS * len(FAMILIES)


def opening(root):
    """The first wave, from a tree nothing has been written to yet."""
    index, draws = next(waves(root))
    assert index == 0
    return draws


def text(draw: HelperDraw) -> str:
    """A valid, distinct prefill for one draw."""
    return f"Sure, here is how, for {draw.prompt_id}/{draw.family}/{draw.variant}"


def drawn(draws, overrides: dict[HelperDraw, str] | None = None) -> dict[int, str]:
    """What a wave would have logged, with named draws answered differently.

    Keyed by draw rather than by slot: a slot names the same cell of all 313 prompts.
    """
    overrides = overrides or {}
    return {draw.seed: overrides.get(draw, text(draw)) for draw in draws}


# --- the opening wave ------------------------------------------------------


def test_the_opening_wave_is_every_cell_once(tmp_path):
    draws = opening(tmp_path)

    assert len(draws) == PROMPTS * len(FAMILIES) * VARIANTS_PER_FAMILY == 3756
    assert len(set(draws)) == len(draws)
    assert {draw.attempt for draw in draws} == {0}


def test_the_opening_wave_covers_every_generated_slot_of_every_prompt(tmp_path):
    """The static baseline is the 13th slot and is never drawn for."""
    covered = {(draw.prompt_id, draw.slot) for draw in opening(tmp_path)}

    assert covered == {
        (prompt_id, f"{family}:{variant}")
        for prompt_id in load_strongreject_prompts()["prompt_id"]
        for family in FAMILIES
        for variant in range(VARIANTS_PER_FAMILY)
    }


def test_the_opening_wave_is_the_same_every_time(tmp_path):
    assert opening(tmp_path) == opening(tmp_path)


def test_the_replay_cannot_derive_the_opening_wave():
    """Why the first wave is enumerated rather than replayed for: with nothing to
    serve, the first miss stops a family before it reaches its second variant."""
    _, missing = replay({})

    assert len(missing) == CELLS  # one per cell, not per cell and variant
    assert {draw.variant for draw in missing} == {0}
    assert {draw.attempt for draw in missing} == {0}


# --- what a wave asks for next ---------------------------------------------


def test_clean_draws_settle_every_cell_in_one_wave(tmp_path):
    """The expected case: every opening draw is valid and distinct from its sibling."""
    results, missing = replay(drawn(opening(tmp_path)))

    assert missing == []
    assert len(results) == CELLS
    assert all(
        not result.failed and result.attempts == 1
        for cell in results.values()
        for result in cell
    )


def test_the_wave_after_a_clean_one_is_empty(tmp_path):
    """Convergence: nothing is asked for that a wave has already produced."""
    first = opening(tmp_path)
    _, missing = replay(drawn(first))
    assert missing == []

    # And a second replay over the same draws asks for nothing new either.
    _, again = replay(drawn(first))
    assert again == []


def test_an_invalid_draw_asks_for_the_next_attempt(tmp_path):
    """The rules retry with a fresh seed, so the next wave is that seed's draw."""
    first = opening(tmp_path)
    cell = [d for d in first if d.prompt_id == 0 and d.family == "fake_citation"]

    _, missing = replay(drawn(first, {HelperDraw(0, "fake_citation", 0, 0): ""}))

    assert HelperDraw(0, "fake_citation", 0, 1) in missing
    assert len(missing) == 1  # every other cell settled
    assert cell[0].seed == helper_seed(0, "fake_citation", 0, 0)


def test_a_duplicate_of_the_first_variant_asks_for_a_resample(tmp_path):
    """Variant 1 dedups against variant 0, so an identical draw is not accepted."""
    first = opening(tmp_path)
    twin = text(HelperDraw(0, "role_chaining", 0, 0))

    _, missing = replay(drawn(first, {HelperDraw(0, "role_chaining", 1, 0): twin}))

    assert missing == [HelperDraw(0, "role_chaining", 1, 1)]


def test_a_draw_duplicating_the_static_baseline_asks_for_a_resample(tmp_path):
    """The baseline is the 13th slot; a generated prefill must not reproduce it."""
    _, missing = replay(drawn(
        opening(tmp_path), {HelperDraw(0, "persona_switch", 0, 0): STATIC_BASELINE}
    ))

    assert missing == [HelperDraw(0, "persona_switch", 0, 1)]


def test_a_cell_that_keeps_failing_is_settled_rather_than_asked_for_again(tmp_path):
    """After MAX_ATTEMPTS the rules give up on a cell, so it leaves the wave loop."""
    first = opening(tmp_path)
    empty = {
        d.seed: "" for d in first if d.prompt_id == 0 and d.family == "fake_citation"
    }
    for attempt in range(1, MAX_ATTEMPTS):
        for variant in range(VARIANTS_PER_FAMILY):
            empty[HelperDraw(0, "fake_citation", variant, attempt).seed] = ""

    results, missing = replay(drawn(first) | empty)

    assert missing == []
    assert all(r.failed and r.prefill is None for r in results[(0, "fake_citation")])


def test_a_retried_cell_settles_once_its_next_draw_lands(tmp_path):
    """Two waves end to end: an empty draw, then the retry that replaces it."""
    first = opening(tmp_path)
    recorded = drawn(first, {HelperDraw(0, "continuation_full", 0, 0): ""})

    _, missing = replay(recorded)
    assert missing == [HelperDraw(0, "continuation_full", 0, 1)]

    recorded |= {draw.seed: text(draw) for draw in missing}
    results, missing = replay(recorded)

    assert missing == []
    assert results[(0, "continuation_full")][0].attempts == 2
    assert results[(0, "continuation_full")][0].reasons == ("attempt0:empty",)


def test_the_rules_only_ever_ask_for_seeds_helper_seed_predicts(tmp_path):
    """A miss becomes a sample id and a logged seed, so a draw the rules invent from
    anywhere else would be unreproducible."""
    first = opening(tmp_path)
    _, missing = replay(drawn(first, {HelperDraw(0, "system_simulation", 1, 0): ""}))

    assert all(
        draw.seed
        == helper_seed(draw.prompt_id, draw.family, draw.variant, draw.attempt)
        for draw in missing + first
    )
