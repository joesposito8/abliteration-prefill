"""Freezing the graded sweep: what the rule is handed, and what gets committed.

The reader is exercised against a real scored log, because its job is to know where
Inspect puts things. Everything downstream takes rows directly — a synthetic sweep can
be given a shape worth ranking, which 37 real conditions cannot cheaply.
"""

from __future__ import annotations

import freeze_abliteration as freeze
import grade
import pytest
from abliteration.selection import condition_id, near_tie_band, select_primary
from harness.conditions import Condition
from harness.dataset import build_dataset
from harness.run import run_condition
from inspect_ai.dataset import MemoryDataset

N = 8
SEED = 20260803

UNLOCK = "1.b 0\n2.b 4\n3.b 5"

RUN_META = {
    "shape": {"n_prompts": N, "seed": SEED},
    "contrast_sets": {"extraction_harmful.csv": "a" * 64},
    "directions": {"file": "refusal_directions.pt", "sha256": "b" * 64},
    "extraction_batch_size": 32,
    "structure": {"n_layers": 36, "d_model": 2560, "n_target_matrices": 73},
    "environment": {"torch": "2.11.0"},
    "peak_vram_gb": 20.5,
}


def _profile(layer: int | None) -> tuple[int, int]:
    """(unlocked, malformed) — weak early, strong in the middle, degrading late."""
    if layer is None:
        return 1, 0
    if layer < 3:
        return layer, 0
    if layer < 30:
        return {3: 6, 4: 7}.get(layer, N), 0
    return 4, 1


def _rows(layer: int | None, *, aggregate: float = 0.5) -> list[dict]:
    unlocked, malformed = _profile(layer)
    rows = []
    for index in range(N):
        is_malformed = index < malformed
        is_unlocked = not is_malformed and index < unlocked
        rows.append({
            "condition": condition_id(layer),
            "prompt_index": index,
            "refused": 0 if is_unlocked else 1,
            "aggregate": aggregate if is_unlocked else 0.0,
            "malformed": is_malformed,
            "degenerate": False,
            "truncated": False,
            "thinking_leak": False,
            "continuation": f"continuation for {condition_id(layer)} prompt {index}",
        })
    return rows


@pytest.fixture
def sweep_rows() -> list[dict]:
    return [row for layer in [None, *range(36)] for row in _rows(layer)]


# --- the reader, against a log the real pipeline wrote ----------------------


@pytest.fixture
def scored_run(monkeypatch, prefills, tokenizer, fake_judge, tmp_path, fake_generate_prompts):
    """One condition generated and graded through the real harness and scorer."""
    condition = Condition(id="layer_22", seed=SEED, layer=22, prompt_set="validation")
    rows = list(build_dataset(condition, prefills))[:4]
    run_condition(
        condition,
        MemoryDataset(rows, name="validation"),
        module=object(),
        tokenizer=tokenizer,
        root=tmp_path / "generated",
    )
    monkeypatch.setattr(grade, "GRADER", fake_judge(lambda _: UNLOCK).role)
    grade.grade_sweep(tmp_path / "generated", tmp_path / "scored")
    return tmp_path / "scored"


def test_the_reader_finds_every_field_the_rule_needs(scored_run):
    rows = freeze.load_rows(scored_run)

    assert len(rows) == 4
    assert {row["condition"] for row in rows} == {"layer_22"}
    assert sorted(row["prompt_index"] for row in rows) == [0, 3, 4, 6]
    assert all(row["refused"] == 0 and not row["malformed"] for row in rows)
    assert all(row["aggregate"] == pytest.approx(0.875) for row in rows)
    assert all(row["continuation"] and not row["thinking_leak"] for row in rows)


def test_a_regenerated_condition_is_not_counted_twice(
    monkeypatch, prefills, tokenizer, fake_judge, tmp_path, fake_generate_prompts
):
    """A killed attempt stays in the tree, and samples_df keys on the run rather than
    the sample, so reading the directory indiscriminately doubles the condition."""
    condition = Condition(id="layer_22", seed=SEED, layer=22, prompt_set="validation")
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

    assert len(freeze.load_rows(tmp_path / "scored")) == 4


def test_an_empty_tree_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no successful eval logs"):
        freeze.load_rows(tmp_path)


# --- completeness ----------------------------------------------------------


def test_a_complete_sweep_passes(sweep_rows):
    freeze.assert_complete(sweep_rows, N)


def test_a_missing_prompt_fails_loudly(sweep_rows):
    del sweep_rows[5]

    with pytest.raises(SystemExit, match="missing prompts"):
        freeze.assert_complete(sweep_rows, N)


def test_a_missing_layer_fails_loudly(sweep_rows):
    remaining = [row for row in sweep_rows if row["condition"] != "layer_07"]

    with pytest.raises(SystemExit, match="missing conditions"):
        freeze.assert_complete(remaining, N)


def test_a_duplicate_cell_fails_loudly(sweep_rows):
    sweep_rows.append(dict(sweep_rows[0]))

    with pytest.raises(SystemExit, match="duplicate cells"):
        freeze.assert_complete(sweep_rows, N)


# --- the rule over the reports ---------------------------------------------


def test_reports_cover_every_layer_plus_the_base_row(sweep_rows):
    reports, _ = freeze.build_reports(sweep_rows)

    assert len(reports) == 37
    assert reports[0].condition == "base" and reports[0].layer is None


def test_late_layers_are_penalised_by_the_metric_not_excluded(sweep_rows):
    """A malformed row is a non-unlock that stays in the denominator, so a degrading
    layer loses breadth without any exclusion rule deciding it is unfit."""
    reports, _ = freeze.build_reports(sweep_rows)
    late = next(r for r in reports if r.condition == "layer_31")

    assert late.n_malformed == 1
    assert late.n_prompts == N
    assert late.n_unlocked == 3


def test_the_primary_comes_from_the_band_on_quality(sweep_rows):
    """Layers 5..29 all unlock everything; the tie-break is what separates them."""
    high = [row for row in _rows(20, aggregate=0.9)]
    sweep_rows = [row for row in sweep_rows if row["condition"] != "layer_20"] + high

    reports, _ = freeze.build_reports(sweep_rows)
    selection = select_primary(reports)

    assert selection.condition == "layer_20"
    assert selection.rule_path == ("breadth", "quality")


def test_excerpts_show_a_spread_of_unlocked_rows(sweep_rows, capsys):
    reports, _ = freeze.build_reports(sweep_rows)
    freeze.print_excerpts(sweep_rows, near_tie_band(reports)[:2], 2)

    printed = capsys.readouterr().out
    assert "never committed or published" in printed
    assert printed.count("continuation for") == 4


# --- what gets committed ---------------------------------------------------


def test_the_table_has_a_row_per_condition_and_one_primary(sweep_rows, tmp_path):
    reports, diagnostics = freeze.build_reports(sweep_rows)
    selection = select_primary(reports)
    rows = freeze.build_table(reports, diagnostics, selection)

    assert len(rows) == 37
    assert sum(row["primary"] for row in rows) == 1
    assert [row["condition"] for row in rows if row["primary"]] == [selection.condition]
    assert list(rows[0]) == freeze.COLUMNS


def test_the_table_is_byte_identical_on_a_rewrite(sweep_rows, tmp_path):
    reports, diagnostics = freeze.build_reports(sweep_rows)
    rows = freeze.build_table(reports, diagnostics, select_primary(reports))

    first = freeze.write_table(tmp_path / "a.csv", rows)
    second = freeze.write_table(tmp_path / "b.csv", rows)

    assert first == second


def test_the_manifest_is_reproducible_and_change_sensitive(sweep_rows):
    reports, _ = freeze.build_reports(sweep_rows)
    selection = select_primary(reports)
    hashes = {"samples": "c" * 64, "n_samples": len(sweep_rows), "code": {}, "commit": "abc"}
    table = {"file": "layer_selection.csv", "sha256": "d" * 64}

    def manifest(**overrides):
        return freeze.build_manifest(
            reports=reports, selection=selection, table=table,
            run_meta=RUN_META, hashes={**hashes, **overrides},
        )

    assert manifest() == manifest()
    assert manifest()["abliteration_sha256"] != manifest(commit="def")["abliteration_sha256"]
    assert manifest()["primary"]["condition"] == selection.condition


def test_the_manifest_records_the_resolved_inspect_install(sweep_rows):
    """The log header carries the version but never the path, so a local checkout and
    the pinned wheel are indistinguishable from it."""
    reports, _ = freeze.build_reports(sweep_rows)
    spec = freeze.build_manifest(
        reports=reports, selection=select_primary(reports),
        table={"file": "t.csv", "sha256": "d" * 64}, run_meta=RUN_META,
        hashes={"samples": "c" * 64, "n_samples": 1, "code": {}, "commit": "abc"},
    )

    assert spec["harness"]["inspect_ai"] == "0.3.251"
    assert spec["harness"]["inspect_ai_path"].endswith("inspect_ai")


def test_the_manifest_describes_the_batching_that_actually_ran(sweep_rows):
    from harness.batching import BATCH

    reports, _ = freeze.build_reports(sweep_rows)
    spec = freeze.build_manifest(
        reports=reports, selection=select_primary(reports),
        table={"file": "t.csv", "sha256": "d" * 64}, run_meta=RUN_META,
        hashes={"samples": "c" * 64, "n_samples": 1, "code": {}, "commit": "abc"},
    )

    assert spec["generation"]["batch_size"] == BATCH
    assert spec["generation"]["k"] == 1
    assert spec["decoding"]["max_new_tokens"] == 1024
    assert "arrival-dependent" in spec["generation"]["batch_composition"]


def test_rows_sha256_does_not_depend_on_row_order(sweep_rows):
    assert freeze.rows_sha256(sweep_rows) == freeze.rows_sha256(list(reversed(sweep_rows)))
