"""The preflight around direction extraction.

The corpus and structure checks are plain Python and run everywhere. The tensor path is
gated on torch, which is in the ``gpu`` extra rather than ``dev``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import extract_directions as extract
import pytest
from generation.qwen import N_LAYERS
from study.datasets import DATA_DIR

from conftest import needs_torch

try:
    import torch
except ImportError:  # every test that uses it is skipped
    pass


def _model(targets=1 + 2 * N_LAYERS):
    """Only what check_structure reads: how many matrices target_matrices yields."""
    return SimpleNamespace(_targets=[object()] * targets)


@pytest.fixture(autouse=True)
def stub_target_matrices(monkeypatch):
    monkeypatch.setattr(extract, "target_matrices", lambda model: model._targets)


# --- the corpora the run must have used ------------------------------------


def test_the_committed_corpora_pass():
    extract.check_datasets()


def test_a_drifted_corpus_is_refused(monkeypatch, tmp_path):
    """A working copy of the contrast set would change every direction silently."""
    manifest = json.loads((DATA_DIR / "freeze_manifest.json").read_text())
    manifest["artifacts"]["extraction_harmful.csv"]["sha256"] = "0" * 64
    drifted = tmp_path / "freeze_manifest.json"
    drifted.write_text(json.dumps(manifest))
    monkeypatch.setattr(extract, "FREEZE_MANIFEST_JSON", drifted)

    with pytest.raises(SystemExit, match="extraction_harmful.csv has drifted"):
        extract.check_datasets()


# --- the structure the unit tests only see on a stub -----------------------


def test_the_real_matrix_count_is_one_embedding_plus_two_per_layer():
    extract.check_structure(_model())


def test_a_model_with_the_wrong_matrix_count_is_refused():
    """The existing unit test makes this assertion against a 3-layer stub, so nothing
    else checks the config the sweep will actually edit."""
    with pytest.raises(SystemExit, match="expected 73 target matrices, found 71"):
        extract.check_structure(_model(targets=71))


# --- the tensor path -------------------------------------------------------


@needs_torch
def test_nan_directions_are_refused():
    directions = torch.ones(4, 8)
    directions[2, 0] = float("nan")

    with pytest.raises(SystemExit, match="NaN"):
        extract.check_directions(directions)


@needs_torch
def test_two_layers_with_an_identical_direction_are_refused():
    """Adjacent duplicates mean the extraction collapsed, not that the layers agree."""
    directions = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    directions[2] = directions[1]

    with pytest.raises(SystemExit, match="identical direction"):
        extract.check_directions(directions)


@needs_torch
def test_distinct_directions_pass():
    extract.check_directions(torch.nn.functional.normalize(torch.randn(4, 8), dim=-1))
