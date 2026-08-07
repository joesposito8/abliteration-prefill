"""The frozen work list: its shape, its batches, and its agreement with the harness.

The manifest re-derives the cross-product that ``harness.dataset`` builds samples from,
so the two enumerating one grid rather than two is a property these tests hold, not
something shared code guarantees.
"""

from __future__ import annotations

import json
from math import ceil

import build_run_manifest as manifest
import pandas as pd
import pytest
from abliteration.selection import condition_id, sweep_layers
from generation.qwen import DECODING, MODEL_ID, N_LAYERS, REVISION
from harness.batching import BATCH
from harness.conditions import Condition
from harness.dataset import CONTROL, EVAL_SETS, build_dataset, sample_id
from prefills import PORTFOLIO
from study import SEED
from study.datasets import (
    DATA_DIR,
    RUN_MANIFEST_CSV,
    RUN_MANIFEST_JSON,
    load_strongreject_prompts,
)
from study.manifest import rollup_sha256, sha256_file

PROMPTS = 313
GENERATIONS_PER_PROMPT = 507
BATCHES = 2498


@pytest.fixture(scope="module")
def primary() -> int:
    return manifest.primary_layer()


@pytest.fixture(scope="module")
def table(primary) -> pd.DataFrame:
    return manifest.build_table(load_strongreject_prompts()["prompt_id"].tolist(), primary)


@pytest.fixture(scope="module")
def frozen() -> dict:
    return json.loads(RUN_MANIFEST_JSON.read_text())


def base_dataset(prefills):
    """The harness's own enumeration of the base condition over the eval set."""
    return build_dataset(Condition(id="base", seed=SEED, prefilled=True), prefills)


# --- the preregistered grid ------------------------------------------------


def test_every_prompt_gets_the_same_five_hundred_and_seven_generations(table):
    """The matched budget is the study's control; an uneven prompt would break it."""
    assert table.groupby("prompt_id").size().unique().tolist() == [GENERATIONS_PER_PROMPT]
    assert len(table) == PROMPTS * GENERATIONS_PER_PROMPT == 158_691


def test_the_unprefilled_arm_covers_the_base_row_and_every_layer(table):
    unprefilled = table[table.prefill_id == CONTROL]

    assert unprefilled.condition.unique().tolist() == [
        condition_id(layer) for layer in sweep_layers(N_LAYERS)
    ]
    assert unprefilled.groupby(["condition", "prompt_id"]).size().unique().tolist() == [
        EVAL_SETS["strongreject"].attempts
    ]


def test_only_the_base_and_the_primary_carry_a_prefill_arm(table, primary):
    """Composition runs on one checkpoint, so every prefilled row names it or the base."""
    prefilled = table[table.prefill_id != CONTROL]

    assert sorted(prefilled.condition.unique()) == sorted(
        {"base", condition_id(primary)}
    )
    assert set(prefilled.loc[prefilled.condition != "base", "layer_id"]) == {primary}


def test_a_prefilled_row_is_a_single_replicate(table):
    """One sample per prefill: replicating the portfolio would explode the budget."""
    prefilled = table[table.prefill_id != CONTROL]

    assert set(prefilled.replicate_id) == {0}
    assert prefilled.groupby(["condition", "prompt_id"]).size().unique().tolist() == [
        len(PORTFOLIO)
    ]


def test_the_base_rows_name_no_layer(table):
    assert table.loc[table.condition == "base", "layer_id"].isna().all()
    assert table.loc[table.condition != "base", "layer_id"].notna().all()


# --- the primary comes from the freeze, never from here --------------------


def test_the_primary_is_read_from_the_abliteration_freeze(monkeypatch, tmp_path):
    """A layer index means nothing except against the extraction that produced it."""
    elsewhere = tmp_path / "abliteration_manifest.json"
    elsewhere.write_text(json.dumps({"primary": {"condition": "layer_07", "layer": 7}}))
    monkeypatch.setattr(manifest, "ABLITERATION_MANIFEST_JSON", elsewhere)

    assert manifest.primary_layer() == 7


def test_the_prefilled_arm_follows_whichever_layer_was_selected():
    prefilled = {condition_id(layer) for layer, slot, _ in manifest.groups(7) if slot != CONTROL}

    assert prefilled == {"base", "layer_07"}


# --- agreement with the harness --------------------------------------------


def test_the_unprefilled_arm_matches_the_datasets_own_order(table, prefills):
    """Row for row: the manifest and the harness must enumerate one grid, not two."""
    dataset = base_dataset(prefills)
    control = [s.id for s in dataset if s.metadata["prefill_slot"] == CONTROL]
    rows = table[(table.condition == "base") & (table.prefill_id == CONTROL)]

    assert [
        sample_id(row.prompt_id, row.prefill_id, row.replicate_id)
        for row in rows.itertuples()
    ] == control


def test_the_prefilled_arm_holds_the_same_samples_as_the_dataset(table, prefills):
    """Same cells, different order — the manifest is slot-major so a batch is one prefill."""
    dataset = base_dataset(prefills)
    attacks = {s.id for s in dataset if s.metadata["prefill_slot"] != CONTROL}
    rows = table[(table.condition == "base") & (table.prefill_id != CONTROL)]

    assert {
        sample_id(row.prompt_id, row.prefill_id, row.replicate_id)
        for row in rows.itertuples()
    } == attacks


def test_the_prefilled_arm_is_ordered_so_one_batch_carries_one_prefill(table):
    rows = table[(table.condition == "base") & (table.prefill_id != CONTROL)]

    assert rows.prefill_id.head(PROMPTS).unique().tolist() == [PORTFOLIO[0]]


# --- the declared partition ------------------------------------------------


def test_no_batch_spans_a_condition_or_a_prefill(table):
    """What the plan declares. The harness serialises conditions but not prefills."""
    spans = table.groupby("batch_id")[["condition", "prefill_id"]].nunique()

    assert spans.max().max() == 1


def test_no_batch_is_wider_than_the_frozen_batch_width(table):
    """Width is a frozen study parameter: generated text is not width-invariant."""
    assert table.groupby("batch_id").size().max() == BATCH
    assert table.batch_size.between(1, BATCH).all()


def test_batch_size_is_the_batch_s_own_row_count(table):
    counted = table.groupby("batch_id").size()

    assert (table.groupby("batch_id").batch_size.first() == counted).all()


def test_only_the_last_batch_of_a_group_is_partial(table):
    counted = table.groupby("batch_id").size()
    last = set(table.groupby(["condition", "prefill_id"], sort=False).batch_id.max())

    assert set(counted[counted < BATCH].index) <= last


def test_the_batch_count_is_what_the_partition_arithmetic_predicts(table, primary):
    predicted = sum(
        ceil(PROMPTS * replicates / BATCH) for *_, replicates in manifest.groups(primary)
    )

    assert table.batch_id.nunique() == predicted == BATCHES


# --- the gate --------------------------------------------------------------


def test_a_workload_that_is_not_the_preregistered_one_is_refused(table):
    """The count is the whole point of building this before any GPU time is bought."""
    with pytest.raises(SystemExit, match="the study's shape moved"):
        manifest.assert_workload(table.iloc[:-1])


def test_a_cell_enumerated_twice_is_refused(table):
    """Right row count, wrong grid — what the count alone cannot catch."""
    doubled = pd.concat([table.iloc[:-1], table.iloc[[0]]])

    with pytest.raises(SystemExit, match="repeat a cell"):
        manifest.assert_workload(doubled)


# --- seed and generation config --------------------------------------------


def test_every_row_carries_the_one_project_wide_seed(table):
    """The batcher opens a batch on one seed and refuses a member that asks for another."""
    assert table.seed.unique().tolist() == [SEED]


def test_the_generation_config_hash_covers_the_model_the_revision_and_the_decoding(table):
    digest = table.gen_config_hash.unique()

    assert digest.tolist() == [
        rollup_sha256(
            {"model": MODEL_ID, "revision": REVISION, "decoding": dict(DECODING)}
        )
    ]
    warmer = {**manifest.gen_config(), "decoding": {**DECODING, "temperature": 0.8}}
    assert rollup_sha256(warmer) != digest[0]


def test_the_generation_config_is_the_one_the_batch_width_was_chosen_under():
    """Width 64 was measured under these settings; under others it is not the answer."""
    sweep = json.loads((DATA_DIR / "batch_sweep.json").read_text())
    config = manifest.gen_config()

    assert sweep["chosen"] == BATCH
    assert (sweep["model"], sweep["revision"]) == (config["model"], config["revision"])
    assert sweep["decoding"] == config["decoding"]


# --- the committed artifact ------------------------------------------------


def test_the_committed_table_hashes_to_what_the_manifest_pins(frozen):
    assert sha256_file(RUN_MANIFEST_CSV) == frozen["table"]["sha256"]


def test_the_committed_manifest_is_what_the_builder_produces_today(table, primary, frozen):
    rebuilt = manifest.build_manifest(table, frozen["table"]["sha256"], primary)

    assert rebuilt == frozen


def test_a_rebuild_that_changed_the_table_refuses_to_re_freeze():
    with pytest.raises(SystemExit, match="REBUILD CHANGED A FROZEN ARTIFACT"):
        manifest.assert_frozen_table_unchanged("0" * 64)
