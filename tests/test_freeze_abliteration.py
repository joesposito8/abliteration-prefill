"""Freezing the graded sweep: what the rule is handed, and what gets committed.

The reader is exercised against a real scored log, because its job is to know where
Inspect puts things. Everything downstream takes logs directly — a synthetic sweep can be
given a shape worth ranking, which 37 real conditions cannot cheaply.
"""

from __future__ import annotations

import shutil
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import freeze_abliteration as freeze
import grade
import pandas as pd
import pytest
from abliteration.selection import (
    LayerReport,
    condition_id,
    near_tie_band,
    select_primary,
    sweep_layers,
)
from generation import target
from generation.qwen3_4b import MODEL_ID, N_LAYERS
from harness.conditions import Condition
from study.manifest import write_bytes
from harness.dataset import build_dataset
from harness.run import run_condition
from inspect_ai.dataset import MemoryDataset

MODEL = "qwen3-4b"

N = 8
SEED = 20260803

UNLOCK = "1.b 0\n2.b 4\n3.b 5"
UNLOCK_AGGREGATE = 0.875


def _profile(layer: int | None) -> tuple[int, int]:
    """(unlocked, malformed) — weak early, strong in the middle, degrading late."""
    if layer is None:
        return 1, 0
    if layer < 3:
        return layer, 0
    if layer < 30:
        return {3: 6, 4: 7}.get(layer, N), 0
    return 4, 1


def _log(*, unlocked, malformed=0, degenerate=0, aggregate=0.5, completed=N,
         results=True):
    """A log carrying only what assert_complete and build_reports read.

    ``N`` sample rows, the first ``unlocked`` of them unlocked and carrying the quality
    score — non-unlocked rows score exactly 0, so the all-prompts mean comes out scaled
    by breadth, which is what the tie-break reads.
    """
    samples = [
        SimpleNamespace(scores={"strongreject": SimpleNamespace(value={
            "unlocked": int(row < unlocked),
            "aggregate": aggregate if row < unlocked else 0.0,
            "malformed": int(row < malformed),
            "degenerate": int(row < degenerate),
        })})
        for row in range(N)
    ]
    return SimpleNamespace(
        samples=samples,
        results=SimpleNamespace(
            completed_samples=completed,
            total_samples=N,
        ) if results else None,
    )


def _sweep(**overrides) -> dict:
    sweep = {}
    for layer in sweep_layers(N_LAYERS):
        unlocked, malformed = _profile(layer)
        sweep[condition_id(layer)] = _log(unlocked=unlocked, malformed=malformed)
    return sweep | overrides


@pytest.fixture
def sweep() -> dict:
    return _sweep()


@pytest.fixture
def reports(sweep) -> list[LayerReport]:
    return freeze.build_reports(sweep, N, N_LAYERS)


# --- the reader, against a log the real pipeline wrote ----------------------


@pytest.fixture
def scored_run(monkeypatch, prefills, tokenizer, fake_judge, tmp_path, fake_generate_prompts):
    """One condition generated and graded through the real harness and scorer."""

    def run(rubric: str = UNLOCK) -> Path:
        condition = Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="validation")
        rows = list(build_dataset(condition, prefills))[:4]
        run_condition(
            condition,
            MemoryDataset(rows, name="validation"),
            module=object(),
            tokenizer=tokenizer,
            root=tmp_path / "generated",
        )
        monkeypatch.setattr(grade, "GRADER", fake_judge(lambda _: rubric).role)
        grade.grade_sweep(tmp_path / "generated", tmp_path / "scored")
        return tmp_path / "scored"

    return run


def test_the_reader_finds_the_numbers_the_rule_needs(scored_run):
    """Against a log the scorer really wrote, so the score keys are Inspect's own."""
    by_condition = freeze.load_conditions(scored_run())

    assert set(by_condition) == {"layer_22"}
    log = by_condition["layer_22"]
    assert log.results.completed_samples == log.results.total_samples == 4
    assert freeze.metric(log, "unlocked") == 1.0
    assert freeze.metric(log, "aggregate") == pytest.approx(UNLOCK_AGGREGATE)
    assert freeze.metric(log, "malformed") == freeze.metric(log, "degenerate") == 0.0


def test_a_regenerated_condition_yields_one_entry(
    monkeypatch, prefills, tokenizer, fake_judge, tmp_path, fake_generate_prompts
):
    """A killed attempt stays in the tree beside the log that replaced it."""
    condition = Condition(model=MODEL, id="layer_22", seed=SEED, layer=22, prompt_set="validation")
    rows = list(build_dataset(condition, prefills))[:4]
    dataset = MemoryDataset(rows, name="validation")

    fake_generate_prompts.fail_after = 0
    with pytest.raises(RuntimeError):
        run_condition(condition, dataset, module=object(), tokenizer=tokenizer,
                      root=tmp_path / "generated")
    fake_generate_prompts.fail_after = None
    run_condition(condition, dataset, module=object(), tokenizer=tokenizer,
                  root=tmp_path / "generated")

    monkeypatch.setattr(grade, "GRADER", fake_judge(lambda _: UNLOCK).role)
    grade.grade_sweep(tmp_path / "generated", tmp_path / "scored")

    by_condition = freeze.load_conditions(tmp_path / "scored")
    assert set(by_condition) == {"layer_22"}
    assert by_condition["layer_22"].results.completed_samples == 4


def test_two_successful_logs_for_one_condition_are_refused(scored_run):
    """Keying on the condition would otherwise drop one silently."""
    scored = scored_run()
    log = next(scored.rglob("*.eval"))
    shutil.copy(log, log.parent / "duplicate.eval")

    with pytest.raises(SystemExit, match="two successful logs"):
        freeze.load_conditions(scored)


def test_an_empty_tree_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no successful eval logs"):
        freeze.load_conditions(tmp_path)


# --- completeness ----------------------------------------------------------


def test_a_complete_sweep_passes(sweep):
    freeze.assert_complete(sweep, N, N_LAYERS)


def test_a_missing_layer_fails_loudly(sweep):
    del sweep["layer_07"]

    with pytest.raises(SystemExit, match="missing conditions"):
        freeze.assert_complete(sweep, N, N_LAYERS)


def test_an_unexpected_condition_fails_loudly(sweep):
    sweep["layer_36"] = _log(unlocked=N)

    with pytest.raises(SystemExit, match="unexpected conditions"):
        freeze.assert_complete(sweep, N, N_LAYERS)


def test_a_condition_that_lost_samples_fails_loudly(sweep):
    """A run that loses samples still reports success, so the count is the check."""
    sweep["layer_11"] = _log(unlocked=4, completed=N - 2)

    with pytest.raises(SystemExit, match="completed, expected"):
        freeze.assert_complete(sweep, N, N_LAYERS)


def test_a_condition_with_no_results_fails_loudly(sweep):
    """Nothing completed leaves results None outright, which would otherwise raise
    AttributeError halfway through the check."""
    sweep["layer_11"] = _log(unlocked=0, results=False)

    with pytest.raises(SystemExit, match="no results"):
        freeze.assert_complete(sweep, N, N_LAYERS)


# --- the rule over the reports ---------------------------------------------


def test_reports_cover_every_layer_plus_the_base_row(reports):
    assert len(reports) == 37
    assert reports[0].condition == "base" and reports[0].layer is None


def test_late_layers_are_penalised_by_the_metric_not_excluded(reports):
    """A malformed row is a non-unlock that stays in the denominator, so a degrading
    layer loses breadth without any exclusion rule deciding it is unfit."""
    late = next(r for r in reports if r.condition == "layer_31")

    assert late.n_malformed == 1
    assert late.n_prompts == N
    assert late.n_unlocked == 4


def test_quality_averages_over_every_prompt_and_the_unlocked_mean_is_derived(reports):
    half = next(r for r in reports if r.condition == "layer_03")  # 6 of 8 unlocked

    assert half.quality == pytest.approx(0.5 * 6 / N)  # denominator is 8, not 6
    assert half.quality_unlocked == pytest.approx(0.5)  # descriptive, unlocked only


def test_quality_denominator_does_not_favour_the_narrower_layer():
    """On an unlocked-only mean the narrower layer wins; the all-prompts denominator is
    what removes that bias, and it is the column the tie-break reads."""
    sweep = _sweep(
        layer_01=_log(unlocked=7, aggregate=0.62),
        layer_02=_log(unlocked=6, aggregate=0.63),
    )
    a, b = (next(r for r in freeze.build_reports(sweep, N, N_LAYERS) if r.condition == c)
            for c in ("layer_01", "layer_02"))

    assert b.quality_unlocked > a.quality_unlocked
    assert a.quality > b.quality


def test_a_layer_with_no_unlocks_has_no_unlocked_mean(reports):
    assert next(r for r in reports if r.condition == "layer_00").quality_unlocked is None


def test_the_primary_comes_from_the_band_on_quality():
    """Layers 5..29 all unlock everything; the tie-break is what separates them."""
    reports = freeze.build_reports(
        _sweep(layer_20=_log(unlocked=N, aggregate=0.9)), N, N_LAYERS
    )
    selection = select_primary(reports)

    assert selection.condition == "layer_20"
    assert selection.rule_path == ("breadth", "quality")


# --- what gets committed ---------------------------------------------------


def test_the_table_has_a_row_per_condition_and_one_primary(reports):
    selection = select_primary(reports)
    table = freeze.build_table(reports, selection, MODEL_ID)

    assert len(table) == 37
    assert table["primary"].sum() == 1
    assert table.loc[table["primary"] == 1, "condition"].tolist() == [selection.condition]
    assert list(table.columns) == (
        ["model"] + [f.name for f in fields(LayerReport)] + ["rank", "primary"]
    )
    assert table["model"].unique().tolist() == [MODEL_ID]


def test_the_base_row_is_ranked_out_not_dropped(reports):
    table = freeze.build_table(reports, select_primary(reports), MODEL_ID)
    base = table[table["condition"] == "base"].iloc[0]

    assert base["layer"] is pd.NA and base["rank"] is pd.NA
    assert base["primary"] == 0


def test_the_base_row_writes_as_an_empty_cell_not_a_zero(reports, tmp_path):
    """`layer` 0 is a real layer, so the base row must not be indistinguishable from it."""
    path = tmp_path / "table.csv"
    write_bytes(path, freeze.render_table(
        freeze.build_table(reports, select_primary(reports), MODEL_ID)
    ))

    written = pd.read_csv(path)
    assert written.loc[written["condition"] == "base", "layer"].isna().all()
    assert written.loc[written["condition"] == "layer_00", "layer"].tolist() == [0]


def test_the_table_is_byte_identical_on_a_rewrite(reports, tmp_path):
    table = freeze.build_table(reports, select_primary(reports), MODEL_ID)

    assert freeze.render_table(table) == freeze.render_table(table)


def test_the_manifest_records_the_choice_and_what_makes_it_meaningful(reports, monkeypatch):
    """A layer index names nothing without the tensor it indexes into, and every
    per-layer number is a row of the table — so those two files are the whole record."""
    monkeypatch.setattr(freeze, "sha256_file", lambda path: "e" * 64)
    selection = select_primary(reports)

    manifest = freeze.build_manifest(
        selection,
        Path("layer_selection.csv"),
        "d" * 64,
        Path("refusal_directions.pt"),
        target(MODEL),
    )

    assert manifest == {
        "model": {"id": MODEL_ID, "revision": target(MODEL).REVISION, "n_layers": N_LAYERS},
        "primary": {
            "condition": selection.condition,
            "layer": selection.layer,
            "rule_path": list(selection.rule_path),
            "band": list(selection.band),
            "runner_up": selection.runner_up,
        },
        "directions": {"file": "refusal_directions.pt", "sha256": "e" * 64},
        "table": {"file": "layer_selection.csv", "sha256": "d" * 64},
    }


# --- the operator gate -----------------------------------------------------


def _report_output(reports, capsys) -> str:
    selection = select_primary(reports)
    table = freeze.build_table(reports, selection, MODEL_ID)
    freeze.print_report(table, reports, selection, near_tie_band(reports))
    return capsys.readouterr().out


def test_the_report_names_the_band_and_the_rules_that_narrowed_it(reports, capsys):
    printed = _report_output(reports, capsys)

    assert "NEAR-TIE BAND" in printed
    assert "not the tie-break" in printed  # the quality_unlocked legend
    assert f"PRIMARY   {select_primary(reports).condition}" in printed
    assert "base" in printed and "layer_35" in printed  # all 37 rows


def test_a_degrading_layer_is_flagged_for_the_operator(capsys):
    reports = freeze.build_reports(
        _sweep(layer_31=_log(unlocked=2, degenerate=5)), N, N_LAYERS
    )

    assert "HIGH DEGENERATE RATE" in _report_output(reports, capsys)


def _band_of_one(**overrides) -> list[LayerReport]:
    fields = dict(
        condition="layer_22", layer=22, n_prompts=4, n_unlocked=4, breadth=1.0,
        quality=UNLOCK_AGGREGATE, n_malformed=0, malformed_rate=0.0, n_degenerate=0,
        degenerate_rate=0.0, quality_unlocked=UNLOCK_AGGREGATE,
    )
    return [LayerReport(**(fields | overrides))]


def test_excerpts_are_unlocked_rows_only_and_never_leave_the_terminal(scored_run, capsys):
    freeze.print_excerpts(freeze.load_conditions(scored_run()), _band_of_one(), 2)

    printed = capsys.readouterr().out
    assert "never committed or published" in printed
    assert printed.count("a continuation") == 2


def test_excerpts_say_so_when_a_layer_unlocked_nothing(scored_run, capsys):
    """A refused condition has text, but none of it is evidence of an unlock."""
    scored = scored_run("1.b 1\n2.b 1\n3.b 1")

    freeze.print_excerpts(
        freeze.load_conditions(scored),
        _band_of_one(n_unlocked=0, breadth=0.0, quality=0.0, quality_unlocked=None),
        2,
    )

    printed = capsys.readouterr().out
    assert "(nothing unlocked)" in printed
    assert "a continuation" not in printed


# --- one target's freeze cannot reach another's ----------------------------


def test_two_targets_under_one_root_are_read_apart(scored_run):
    """Both targets have a layer_22, so the model segment of the path is what separates them.

    Keyed on the condition id alone the two logs collide and the read aborts — which is the
    right outcome, but it means an unscoped root cannot freeze either target.
    """
    scored = scored_run()
    shutil.copytree(scored / MODEL, scored / "phi-4")

    with pytest.raises(SystemExit, match="two successful logs"):
        freeze.load_conditions(scored)

    for model in (MODEL, "phi-4"):
        assert set(freeze.load_conditions(scored / model)) == {"layer_22"}
