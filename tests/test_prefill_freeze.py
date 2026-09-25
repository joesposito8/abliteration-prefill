"""Waves to committed table, over draws a helper never had to make.

The freeze is where the run stops being a log tree and becomes the input every condition
reads, so what is checked here is that nothing is lost or altered on the way across.
"""

from __future__ import annotations

import pytest
from harness.prefill_task import HelperDraw
from prefills import PORTFOLIO, STATIC_BASELINE
from prefills.families import STATIC_SLOT_ID
from study.datasets import load_prefills, load_strongreject_prompts
from study.manifest import render_csv, write_bytes

from freeze_prefills import assert_shape, build_manifest, build_table
from produce_prefills import replay, waves

PROMPTS = 313
ROWS = PROMPTS * len(PORTFOLIO)

# Quotes, an em dash, a comma and a newline: what a CSV has to carry unaltered.
PREFILL = (
    'Sure — here is how, for {0.prompt_id}/{0.family}/{0.variant}:'
    '\n1. "step one",\n2. and then'
)


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    """One clean wave, and what its log would have recorded for each draw."""
    _, draws = next(waves(tmp_path_factory.mktemp("nothing_drawn")))
    return draws, {draw.seed: PREFILL.format(draw) for draw in draws}


@pytest.fixture(scope="module")
def table(wave):
    _, drawn = wave
    results, missing = replay(drawn)
    assert missing == []
    return build_table(results)


@pytest.fixture
def frozen(tmp_path, monkeypatch, table):
    """The table written and read back through the committed CSV's own writer."""
    path = tmp_path / "prefills.csv"
    write_bytes(path, render_csv(table))
    monkeypatch.setattr("study.datasets.PREFILLS_CSV", path)
    return path


# --- the table --------------------------------------------------------------


def test_every_prompt_is_crossed_with_every_slot(table):
    assert len(table) == ROWS == 4069
    assert set(table["slot"]) == set(PORTFOLIO)
    assert table["prompt_id"].nunique() == PROMPTS
    assert not table.duplicated(subset=["prompt_id", "slot"]).any()


def test_the_static_baseline_is_verbatim_on_every_prompt(table):
    static = table[table["slot"] == STATIC_SLOT_ID]

    assert len(static) == PROMPTS
    assert (static["prefill"] == STATIC_BASELINE).all()
    # Never generated, so it carries no draw of its own.
    assert (static["helper_seed"] == "").all()
    assert static["attempts"].isna().all()


def test_a_generated_row_carries_its_draws_text_and_seed(table, wave):
    _, drawn = wave
    draw = HelperDraw(0, "fake_citation", 0, 0)
    row = table[(table["prompt_id"] == 0) & (table["slot"] == "fake_citation:0")].iloc[0]

    assert row["prefill"] == drawn[draw.seed]
    assert row["helper_seed"] == str(draw.seed)
    assert row["attempts"] == 1
    assert not row["failed"]


def test_a_clean_run_leaves_nothing_failed(table):
    assert not table["failed"].any()
    assert not table["resampled_dup"].any()
    assert (table["reasons"] == "").all()


@pytest.mark.parametrize(
    "damage, message",
    [
        pytest.param(lambda t: t.iloc[:-1], "expected 313 prompts", id="a-row-short"),
        pytest.param(
            lambda t: t.assign(slot=STATIC_SLOT_ID), "repeat a", id="slots-collapsed"
        ),
        pytest.param(
            lambda t: t.assign(
                prefill=t["prefill"].where(t["slot"] != STATIC_SLOT_ID, "Sure,")
            ),
            "static baseline",
            id="baseline-altered",
        ),
    ],
)
def test_a_table_that_is_not_the_preregistered_shape_is_refused(table, damage, message):
    with pytest.raises(SystemExit, match=message):
        assert_shape(damage(table))


# --- what survives the CSV --------------------------------------------------


def test_every_cell_reaches_the_loader(frozen):
    prefills = load_prefills()

    assert len(prefills) == 4069
    assert {slot for _, slot in prefills} == set(PORTFOLIO)


def test_a_prefill_crosses_the_csv_byte_for_byte(frozen, wave):
    """Quotes, newlines and dashes reach the target as the helper wrote them, or the
    judged text carries framing no portfolio slot contains."""
    _, drawn = wave
    prefills = load_prefills()

    for draw in (HelperDraw(0, "role_chaining", 1, 0), HelperDraw(312, "persona_switch", 0, 0)):
        assert prefills[(draw.prompt_id, draw.slot)] == drawn[draw.seed]
    assert prefills[(7, STATIC_SLOT_ID)] == STATIC_BASELINE


def test_a_prefill_that_reads_as_a_number_stays_text(tmp_path, monkeypatch, table):
    """pandas would take "NA" for a missing value and hand a condition a float."""
    edited = table.copy()
    edited.loc[edited.index[0], "prefill"] = "NA"
    path = tmp_path / "prefills.csv"
    write_bytes(path, render_csv(edited))
    monkeypatch.setattr("study.datasets.PREFILLS_CSV", path)

    prefills = load_prefills()

    assert prefills[(0, PORTFOLIO[0])] == "NA"


# --- the manifest -----------------------------------------------------------


def test_the_manifest_records_what_the_set_is_only_meaningful_against(table):
    manifest = build_manifest(table, "0" * 64)

    assert manifest["counts"]["rows"] == ROWS
    assert manifest["counts"]["failed"] == 0
    assert manifest["helper"]["revision"] == "eaa815dffdf0"
    assert manifest["helper"]["decoding"]["temperature"] == 1.0
    assert set(manifest["inputs"]) == {"strongreject_dataset.csv"}
    assert len(manifest["prefill_sha256"]) == 64


def test_rebuilding_the_manifest_changes_nothing(table):
    """A freeze that is not idempotent cannot be verified after the fact."""
    assert build_manifest(table, "0" * 64) == build_manifest(table, "0" * 64)


def test_the_roll_up_moves_when_the_table_does(table):
    moved = build_manifest(table, "1" * 64)["prefill_sha256"]

    assert moved != build_manifest(table, "0" * 64)["prefill_sha256"]
