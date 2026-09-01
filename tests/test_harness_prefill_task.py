"""What a wave asks the helper for — no GPU, no weights.

The draws are a pure function of the frozen prompt set and the frozen portfolio, so the
whole enumeration is decidable here.
"""

from __future__ import annotations

import pytest
from generation.gemma3_27b import build_prompt
from harness.helper_provider import HELPER_FROZEN_CONFIG
from harness.prefill_task import HelperDraw, produce_prefills, wave_dataset
from harness.provider import require_frozen_config
from inspect_ai import eval
from prefills import (
    FAMILIES,
    MAX_ATTEMPTS,
    MAX_DEDUP_RESAMPLES,
    PORTFOLIO,
    VARIANTS_PER_FAMILY,
    fill_prompt,
    helper_seed,
    load_prompt,
)
from study.datasets import load_strongreject_prompts

SEED = 20260724

# One draw per generated slot, which is where a colon could reach an id from.
EVERY_SLOT = [
    HelperDraw(prompt_id=0, family=family, variant=variant, attempt=0)
    for family in FAMILIES
    for variant in range(VARIANTS_PER_FAMILY)
]


def run_wave(tmp_path, tokenizer, draws):
    """One wave through the real task and provider, returning the written log."""
    return eval(
        produce_prefills(wave_dataset(draws), seed=SEED),
        model="gemma-helper/prefills",
        model_args={"module": object(), "tokenizer": tokenizer},
        log_dir=str(tmp_path),
        score=False,
    )[0]


# --- what a draw is -------------------------------------------------------


def test_a_draw_carries_the_frozen_seed_for_its_coordinates():
    draw = HelperDraw(prompt_id=5, family="fake_citation", variant=1, attempt=2)

    assert draw.seed == helper_seed(5, "fake_citation", 1, 2)
    assert draw.slot == "fake_citation:1"


def test_every_draw_a_cell_could_need_has_its_own_seed():
    """Attempt N is only a fresh draw if its seed differs from attempt N-1's."""
    budget = MAX_ATTEMPTS + MAX_DEDUP_RESAMPLES
    seeds = {
        HelperDraw(7, family, variant, attempt).seed
        for family in FAMILIES
        for variant in range(VARIANTS_PER_FAMILY)
        for attempt in range(budget)
    }

    assert len(seeds) == len(FAMILIES) * VARIANTS_PER_FAMILY * budget


# --- what the samples carry ------------------------------------------------


def test_a_draws_slot_is_one_the_portfolio_names():
    """The slot keys the frozen CSV and is what build_dataset looks a prefill up by."""
    assert {draw.slot for draw in EVERY_SLOT} == {s for s in PORTFOLIO if ":" in s}


def test_no_sample_id_carries_a_colon():
    """``eval(sample_id=…)`` splits an id on its first colon and reads the left half as
    a task name, so a slot's colon drops the sample from a resume with the run still
    reporting success. 12 of the 13 slots contain one."""
    ids = [sample.id for sample in wave_dataset(EVERY_SLOT).samples]

    assert not any(":" in sample_id for sample_id in ids)
    assert len(set(ids)) == len(ids)


def test_a_sample_carries_the_coordinates_and_the_seed():
    draw = HelperDraw(prompt_id=9, family="persona_switch", variant=0, attempt=1)

    [sample] = wave_dataset([draw]).samples

    assert sample.metadata == {
        "prompt_id": 9,
        "family": "persona_switch",
        "variant": 0,
        "attempt": 1,
        "helper_seed": draw.seed,
    }


def test_a_sample_asks_for_its_own_prompt_under_its_own_family():
    draw = HelperDraw(prompt_id=11, family="role_chaining", variant=0, attempt=0)
    forbidden = load_strongreject_prompts().set_index("prompt_id").loc[11]

    [sample] = wave_dataset([draw]).samples

    assert sample.input == fill_prompt(
        load_prompt("role_chaining"), forbidden["forbidden_prompt"]
    )


# --- the task ---------------------------------------------------------------


def test_the_config_the_task_builds_passes_the_providers_check():
    """Otherwise the suite passes while every real wave is rejected at generation."""
    task = produce_prefills(wave_dataset(EVERY_SLOT[:1]), seed=SEED)

    require_frozen_config(task.config, HELPER_FROZEN_CONFIG)
    assert task.config.seed == SEED


def test_a_dataset_name_is_refused():
    """``Task`` reads a name as one sample per character, and eval_retry rebuilds the
    task from its recorded arguments."""
    with pytest.raises(TypeError, match="must be a Dataset"):
        produce_prefills("prefills", seed=SEED)


def test_the_task_records_what_the_wave_was_produced_under():
    task = produce_prefills(wave_dataset(EVERY_SLOT[:1]), seed=SEED)

    assert task.metadata["helper_model"] == "mlabonne/gemma-3-27b-it-abliterated"
    assert task.metadata["helper_revision"] == "eaa815dffdf0"
    assert len(task.metadata["prompt_set_sha256"]) == 64


def test_the_helper_prompt_reaches_the_model_byte_for_byte(
    tmp_path, helper_tokenizer, fake_helper_prompts
):
    """A byte altered on the way to the GPU means the prefills were produced under a
    prompt that is not the authored one, with nothing in the log to say so."""
    draw = HelperDraw(prompt_id=5, family="system_simulation", variant=0, attempt=0)
    forbidden = load_strongreject_prompts().set_index("prompt_id").loc[5]

    log = run_wave(tmp_path, helper_tokenizer, [draw])

    [rendered] = fake_helper_prompts.calls[0]["prompts"]
    assert rendered == build_prompt(
        helper_tokenizer,
        fill_prompt(load_prompt("system_simulation"), forbidden["forbidden_prompt"]),
    )
    assert log.status == "success"


def test_a_wave_logs_every_draws_seed(tmp_path, helper_tokenizer, fake_helper_prompts):
    """One seed reaches the provider for the whole wave, so the frozen per-draw seeds
    are only recoverable from the samples."""
    draws = [HelperDraw(3, "continuation_full", v, attempt=0) for v in (0, 1)]

    log = run_wave(tmp_path, helper_tokenizer, draws)

    assert log.plan.config.seed == SEED
    assert {s.metadata["helper_seed"] for s in log.samples} == {d.seed for d in draws}
