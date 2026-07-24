"""Frozen-dataset invariants — fully offline (no network, no paid API).

Proves the committed artifacts match their manifest hashes and reproduce exactly
from ``SEED`` via the same ``study.datasets`` helpers the build used. Guards the
preregistration: the split and pilot must be a deterministic function of the seed
and the committed sources, not a one-off draw that can drift.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from study import SEED
from study.datasets import (
    EXTRACTION_HARMFUL_CSV,
    EXTRACTION_HARMLESS_CSV,
    FREEZE_MANIFEST_JSON,
    HARMBENCH_STANDARD_CSV,
    PILOT_PROMPTS_CSV,
    VALIDATION_HARMFUL_CSV,
    load_extraction_harmful,
    load_extraction_harmless,
    load_pilot_prompts,
    load_strongreject,
    load_validation,
    sample_indices,
    split_indices,
    verbatim_overlap,
)

MANIFEST = json.loads(FREEZE_MANIFEST_JSON.read_text())


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_counts():
    assert len(load_extraction_harmful()) == 128
    assert len(load_validation()) == 72
    assert len(load_extraction_harmless()) == 128
    assert len(load_pilot_prompts()) == 30
    assert len(load_strongreject()) == 313
    assert len(pd.read_csv(HARMBENCH_STANDARD_CSV)) == 200


def test_split_disjoint_and_complete():
    ext = pd.read_csv(EXTRACTION_HARMFUL_CSV)["behavior_id"]
    val = pd.read_csv(VALIDATION_HARMFUL_CSV)["behavior_id"]
    std = pd.read_csv(HARMBENCH_STANDARD_CSV)["behavior_id"]
    assert set(ext).isdisjoint(set(val))
    assert set(ext) | set(val) == set(std)
    assert len(std) == 200


def test_verbatim_overlap_zero():
    sr = load_strongreject()["forbidden_prompt"].tolist()
    std = pd.read_csv(HARMBENCH_STANDARD_CSV)["Behavior"].tolist()
    for norm in ("none", "strip", "lower", "strip_lower"):
        assert verbatim_overlap(std, sr, normalize=norm) == 0
    assert verbatim_overlap(load_extraction_harmful(), sr) == 0
    assert verbatim_overlap(load_validation(), sr) == 0
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
    std = pd.read_csv(HARMBENCH_STANDARD_CSV).sort_values(
        "behavior_id", kind="stable"
    ).reset_index(drop=True)
    ext_idx, val_idx = split_indices(200, 128, SEED)
    expected_ext = set(std.iloc[ext_idx]["behavior_id"])
    expected_val = set(std.iloc[val_idx]["behavior_id"])
    assert set(pd.read_csv(EXTRACTION_HARMFUL_CSV)["behavior_id"]) == expected_ext
    assert set(pd.read_csv(VALIDATION_HARMFUL_CSV)["behavior_id"]) == expected_val


def test_pilot_reproduces_from_seed():
    """The pilot is exactly RandomState(SEED) over the 313, keyed by prompt_id."""
    pilot_ids = sample_indices(313, 30, SEED)
    assert pilot_ids == MANIFEST["pilot_prompt_ids"]
    committed = load_pilot_prompts()
    assert sorted(committed["prompt_id"]) == pilot_ids
    # every pilot row matches the StrongREJECT row at that prompt_id
    sr = load_strongreject().set_index("prompt_id")
    for _, row in committed.iterrows():
        assert row["forbidden_prompt"] == sr.loc[row["prompt_id"], "forbidden_prompt"]


def test_prompt_id_is_row_index():
    sr = load_strongreject()
    assert list(sr["prompt_id"]) == list(range(313))
