"""The sweep end to end with no GPU and no API: edit, generate, grade, select, freeze.

The pieces are covered apart — the loop in test_harness_run, grading in test_harness_grade,
the reader and the rule in test_freeze_abliteration. What only this can show is that a
multi-condition tree comes back out keyed by condition, base row included, and that the two
committed artifacts are what the operator is asked to approve.

Only the two things that need hardware are faked: the forward pass and the judge.
"""

from __future__ import annotations

import json
import operator
import sys
from types import SimpleNamespace

import freeze_abliteration as freeze
import grade
import pytest
from abliteration.selection import condition_id
from harness.conditions import Condition
from harness.run import run_sweep
from inspect_ai.dataset import MemoryDataset

SEED = 20260803
N_LAYERS = 2
PROMPTS = 3
UNLOCK = "1.b 0\n2.b 4\n3.b 5"

DIRECTIONS = list(range(N_LAYERS))
MATRICES = 3  # the real edit spans 73; a snapshot and a verify have to cover all of them


class Weights:
    """A module whose whole content is the edits currently applied to it."""

    def __init__(self) -> None:
        self.edits: list[int] = []


@pytest.fixture
def weights(monkeypatch) -> Weights:
    """Stands in for the tensor operations, which need torch and a real model."""
    module = Weights()
    monkeypatch.setattr("harness.run.snapshot_targets", lambda m: [tuple(m.edits)] * MATRICES)
    monkeypatch.setattr("harness.run.orthogonalize_", lambda m, r: m.edits.append(r))
    monkeypatch.setattr(
        "harness.run.restore_targets",
        lambda m, base: m.edits.__setitem__(slice(None), base[0]),
    )
    monkeypatch.setattr(
        "harness.run.target_matrices",
        lambda m: [(SimpleNamespace(data=tuple(m.edits)), 0)] * MATRICES,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(equal=operator.eq))
    return module


@pytest.fixture
def small_sweep(monkeypatch, tmp_path, prefills):
    """Two layers instead of 36 and three prompts instead of 72, everything else real."""
    monkeypatch.setattr(freeze, "N_LAYERS", N_LAYERS)
    monkeypatch.setattr(freeze, "TABLE_PATH", tmp_path / "layer_selection.csv")
    monkeypatch.setattr(freeze, "MANIFEST_PATH", tmp_path / "abliteration_manifest.json")
    directions_file = tmp_path / "refusal_directions.pt"
    directions_file.write_bytes(b"stands in for the tensor the manifest pins")
    monkeypatch.setattr(freeze, "DIRECTIONS_PATH", directions_file)

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
        for layer in [None, *range(N_LAYERS)]
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


def test_the_committed_artifacts_are_written_and_agree(frozen):
    reports, selection = frozen
    table = freeze.build_table(reports, selection)
    table_sha = freeze.write_table(freeze.TABLE_PATH, table)
    freeze.write_manifest(freeze.MANIFEST_PATH, freeze.build_manifest(selection, table_sha))

    manifest = json.loads(freeze.MANIFEST_PATH.read_text())

    assert manifest["primary"]["condition"] == selection.condition
    assert manifest["table"]["sha256"] == table_sha
    assert len(freeze.TABLE_PATH.read_text().splitlines()) == 1 + len(reports)
