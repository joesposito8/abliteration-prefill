"""The sweep end to end with no GPU and no API: edit, generate, grade, select, freeze.

The pieces are covered apart — the loop in test_harness_run, grading in test_harness_grade,
the reader and the rule in test_freeze_abliteration. What only this can show is that a
multi-condition tree comes back out keyed by condition, base row included, and that the two
committed artifacts are what the operator is asked to approve.

Only the two things that need hardware are faked: the forward pass and the judge.
"""

from __future__ import annotations

import json

import freeze_abliteration as freeze
import grade
import pytest
from abliteration.selection import condition_id, sweep_layers
from harness.conditions import Condition
from harness.run import run_sweep
from inspect_ai.dataset import MemoryDataset

SEED = 20260803
N_LAYERS = 2
PROMPTS = 3
UNLOCK = "1.b 0\n2.b 4\n3.b 5"

DIRECTIONS = list(range(N_LAYERS))


@pytest.fixture
def small_sweep(monkeypatch, tmp_path, prefills):
    """Two layers instead of 36 and three prompts instead of 72, everything else real."""
    monkeypatch.setattr(freeze, "N_LAYERS", N_LAYERS)
    (tmp_path / "refusal_directions.pt").write_bytes(b"stands in for the tensor pinned")

    from harness.dataset import build_dataset

    monkeypatch.setattr(
        "harness.run.build_dataset",
        lambda condition, _: MemoryDataset(
            list(build_dataset(condition, prefills))[:PROMPTS], name=condition.prompt_set
        ),
    )
    return tmp_path


@pytest.fixture
def frozen(monkeypatch, small_sweep, weights, tokenizer, fake_judge, fake_generate_prompts):
    """Generate every condition, grade it, then apply the rule — the operator's path."""
    conditions = [
        Condition(id=condition_id(layer), seed=SEED, layer=layer, prompt_set="validation")
        for layer in sweep_layers(N_LAYERS)
    ]
    run_sweep(
        conditions, {}, module=weights, tokenizer=tokenizer,
        directions=DIRECTIONS, root=small_sweep / "generated",
    )

    monkeypatch.setattr(grade, "GRADER", fake_judge(lambda _: UNLOCK).role)
    grade.grade_sweep(small_sweep / "generated", small_sweep / "scored")

    by_condition = freeze.load_conditions(small_sweep / "scored")
    freeze.assert_complete(by_condition, PROMPTS)
    reports = freeze.build_reports(by_condition, PROMPTS)
    return reports, freeze.select_primary(reports)


def test_every_condition_comes_back_keyed_by_its_own_name(frozen):
    """Three separate evals, one tree, and the base row named by the same rule as a layer."""
    reports, _ = frozen

    assert [r.condition for r in reports] == ["base", "layer_00", "layer_01"]
    assert all(r.n_prompts == PROMPTS and r.n_unlocked == PROMPTS for r in reports)


def test_the_rule_selects_a_primary_from_the_graded_sweep(frozen):
    """Every layer unlocks everything here, so the tie-break runs to the last rule."""
    _, selection = frozen

    assert selection.condition == "layer_00"  # never the base row, then lowest index
    assert selection.rule_path == ("breadth", "layer_index")


def test_the_committed_artifacts_are_written_and_agree(frozen, small_sweep):
    reports, selection = frozen
    table_path = small_sweep / "layer_selection.csv"
    manifest_path = small_sweep / "abliteration_manifest.json"

    table_sha = freeze.write_table(table_path, freeze.build_table(reports, selection))
    freeze.write_manifest(manifest_path, freeze.build_manifest(
        selection, table_path, table_sha, small_sweep / "refusal_directions.pt"
    ))
    manifest = json.loads(manifest_path.read_text())

    assert manifest["primary"]["condition"] == selection.condition
    assert manifest["table"]["sha256"] == table_sha
    assert len(table_path.read_text().splitlines()) == 1 + len(reports)
