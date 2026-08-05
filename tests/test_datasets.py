"""Frozen-dataset invariants — fully offline (no network, no paid API).

Checks the committed artifacts match their manifest hashes and reproduce exactly
from ``SEED`` via the same ``study.datasets`` helpers the build used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json

import pandas as pd
import pytest

from study import SEED
from study.datasets import (
    EVAL_SET_COLUMNS,
    EXTRACTION_HARMFUL_CSV,
    EXTRACTION_HARMLESS_CSV,
    FREEZE_MANIFEST_JSON,
    HARMBENCH_STANDARD_CSV,
    PILOT_PROMPTS_CSV,
    VALIDATION_HARMFUL_CSV,
    load_extraction_harmful,
    load_extraction_harmless,
    load_harmbench_standard,
    load_pilot_prompts,
    load_strongreject_prompts,
    load_validation_prompts,
    sample_indices,
    split_indices,
    verbatim_overlap,
)

EVAL_SET_LOADERS = (
    load_strongreject_prompts,
    load_pilot_prompts,
    load_validation_prompts,
)

MANIFEST = json.loads(FREEZE_MANIFEST_JSON.read_text())
REPO = FREEZE_MANIFEST_JSON.parents[1]


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts():
    assert len(load_extraction_harmful()) == 128
    assert len(load_validation_prompts()) == 72
    assert len(load_extraction_harmless()) == 128
    assert len(load_pilot_prompts()) == 30
    assert len(load_strongreject_prompts()) == 313
    assert len(load_harmbench_standard()) == 200


@pytest.mark.parametrize("load", EVAL_SET_LOADERS, ids=lambda f: f.__name__)
def test_an_eval_set_loads_in_the_shape_the_harness_samples_from(load):
    """``harness.dataset._sample`` reads these four columns off every row it is given,
    so a prompt set missing one becomes samples with a KeyError or a silent blank."""
    df = load()
    assert set(EVAL_SET_COLUMNS) <= set(df.columns)
    assert df["prompt_id"].is_unique


def test_split_disjoint_and_complete():
    """The ids ARE the draw: each split's prompt_id is the parent row it was taken
    from, so the split is stated by the files rather than re-derived from the seed."""
    # Read the extraction file directly: its loader hands back prompts only, because
    # that is all direction extraction reads. This test is about the files.
    ext = pd.read_csv(EXTRACTION_HARMFUL_CSV)
    val = load_validation_prompts()
    parent = load_harmbench_standard()

    assert set(ext["prompt_id"]).isdisjoint(val["prompt_id"])
    assert set(ext["prompt_id"]) | set(val["prompt_id"]) == set(range(200))
    for split in (ext, val):
        taken = parent.iloc[split["prompt_id"]]
        assert list(taken["forbidden_prompt"]) == list(split["forbidden_prompt"])
        assert list(taken["behavior_id"]) == list(split["behavior_id"])


def test_verbatim_overlap_zero():
    sr = load_strongreject_prompts()["forbidden_prompt"].tolist()
    std = load_harmbench_standard()["forbidden_prompt"].tolist()
    for norm in ("none", "strip", "lower", "strip_lower"):
        assert verbatim_overlap(std, sr, normalize=norm) == 0
    assert verbatim_overlap(load_extraction_harmful(), sr) == 0
    assert verbatim_overlap(load_validation_prompts()["forbidden_prompt"], sr) == 0
    assert MANIFEST["verbatim_exact_match_count"] == 0


def test_artifact_hashes_match_manifest():
    paths = {
        HARMBENCH_STANDARD_CSV.name: HARMBENCH_STANDARD_CSV,
        EXTRACTION_HARMFUL_CSV.name: EXTRACTION_HARMFUL_CSV,
        VALIDATION_HARMFUL_CSV.name: VALIDATION_HARMFUL_CSV,
        EXTRACTION_HARMLESS_CSV.name: EXTRACTION_HARMLESS_CSV,
        PILOT_PROMPTS_CSV.name: PILOT_PROMPTS_CSV,
    }
    for name, path in paths.items():
        assert _sha256(path) == MANIFEST["artifacts"][name]["sha256"], name


def test_split_reproduces_from_seed():
    """The committed split is exactly RandomState(SEED) over the canonical order."""
    ext_idx, val_idx = split_indices(200, 128, SEED)
    assert list(pd.read_csv(EXTRACTION_HARMFUL_CSV)["prompt_id"]) == ext_idx
    assert list(load_validation_prompts()["prompt_id"]) == val_idx


def test_parent_is_in_canonical_order():
    """prompt_id is the row index in behavior_id order, which is what makes a split's
    ids meaningful — reorder the parent and every committed id points elsewhere."""
    parent = load_harmbench_standard()
    assert list(parent["prompt_id"]) == list(range(200))
    assert list(parent["behavior_id"]) == sorted(parent["behavior_id"])


def test_pilot_reproduces_from_seed():
    """The pilot is exactly RandomState(SEED) over the 313, keyed by prompt_id."""
    pilot_ids = sample_indices(313, 30, SEED)
    assert pilot_ids == MANIFEST["pilot_prompt_ids"]
    committed = load_pilot_prompts()
    assert sorted(committed["prompt_id"]) == pilot_ids
    # every pilot row matches the StrongREJECT row at that prompt_id
    sr = load_strongreject_prompts().set_index("prompt_id")
    for _, row in committed.iterrows():
        assert row["forbidden_prompt"] == sr.loc[row["prompt_id"], "forbidden_prompt"]


def test_rebuild_refuses_to_refreeze_a_changed_artifact():
    """A drifted derivation must abort the build, not re-freeze itself."""
    spec = importlib.util.spec_from_file_location(
        "build_datasets", REPO / "scripts" / "build_datasets.py"
    )
    build_datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_datasets)

    frozen = MANIFEST["artifacts"]
    build_datasets._assert_frozen_hashes_unchanged(frozen)  # unchanged: silent

    drifted = {name: dict(meta) for name, meta in frozen.items()}
    drifted[EXTRACTION_HARMLESS_CSV.name]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="REBUILD CHANGED A FROZEN ARTIFACT"):
        build_datasets._assert_frozen_hashes_unchanged(drifted)


def test_prompt_id_is_row_index():
    sr = load_strongreject_prompts()
    assert list(sr["prompt_id"]) == list(range(313))
