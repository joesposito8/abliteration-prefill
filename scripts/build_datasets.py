#!/usr/bin/env python3
"""Build and freeze the study datasets from their upstream sources.

Deterministic given the pinned sources: re-running produces byte-identical
artifacts, and a drifted upstream SHA-256 aborts the build rather than re-freezing.

Outputs (all under ``data/``):
  harmbench_standard_behaviors.csv   200 HarmBench standard behaviors (source of truth)
  extraction_harmful.csv             128 harmful behaviors (direction extraction)
  validation_harmful.csv              72 harmful behaviors (primary-layer selection)
  extraction_harmless.csv            128 Alpaca instructions (harmless contrast)
  pilot_prompts.csv                   30 StrongREJECT prompts (seeded pilot slice)
  freeze_manifest.json               seeds, hashes, counts, verbatim-overlap report

The three HarmBench files are one table in three parts: the 200-row parent and the two
splits cut from it share their columns, and a split's ``prompt_id`` is the parent row it
was drawn at. Everything a prompt set needs to become an eval set travels with it.

Run:  python scripts/build_datasets.py
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from study import SEED  # noqa: E402
from study.manifest import sha256_bytes, sha256_file, write_csv, write_manifest  # noqa: E402
from study.datasets import (  # noqa: E402
    _NORMALIZERS,
    DATA_DIR,
    EXTRACTION_HARMFUL_CSV,
    EXTRACTION_HARMLESS_CSV,
    FREEZE_MANIFEST_JSON,
    HARMBENCH_STANDARD_CSV,
    PILOT_PROMPTS_CSV,
    STRONGREJECT_CSV,
    VALIDATION_HARMFUL_CSV,
    load_strongreject_prompts,
    sample_indices,
    split_indices,
    verbatim_overlap,
)

# Pinned upstream sources. SHA-256s are asserted on fetch; bump deliberately only
# if the upstream is intentionally re-vendored.
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
HARMBENCH_SHA256 = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)
ALPACA_SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
STRONGREJECT_SHA256 = (
    "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381"
)

N_EXTRACTION = 128
N_VALIDATION = 72
N_HARMLESS = 128
N_PILOT = 30

HARMBENCH_SOURCE = "harmbench_standard"
COLS_HARMFUL = ["prompt_id", "behavior_id", "category", "source", "forbidden_prompt"]


def _fetch(url: str, expected_sha256: str) -> bytes:
    print(f"fetch {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    got = sha256_bytes(data)
    if got != expected_sha256:
        raise SystemExit(
            f"source hash drift for {url}\n  expected {expected_sha256}\n  got      {got}\n"
            "Upstream changed. Re-vendor deliberately (update the pinned hash) before rebuilding."
        )
    return data


def _assert_frozen_hashes_unchanged(artifacts: dict[str, dict]) -> None:
    """Abort if a rebuild changed an already-frozen artifact.

    The build rewrites the CSVs and the manifest together, so a drifted draw would
    re-freeze itself with every downstream check still passing. Comparing against the
    committed manifest is what makes a rebuild a reproduction test. Catches a changed
    derivation, not a changed source — ``_fetch`` already guards the pinned upstreams.
    """
    if not FREEZE_MANIFEST_JSON.exists():
        return  # first build; there is nothing frozen yet
    frozen = json.loads(FREEZE_MANIFEST_JSON.read_text())["artifacts"]
    drifted = [
        f"  {name}\n    frozen {frozen[name]['sha256']}\n    built  {meta['sha256']}"
        for name, meta in sorted(artifacts.items())
        if name in frozen and frozen[name]["sha256"] != meta["sha256"]
    ]
    if drifted:
        raise SystemExit(
            "REBUILD CHANGED A FROZEN ARTIFACT — refusing to re-freeze:\n"
            + "\n".join(drifted)
            + "\n\nThe committed CSVs on disk have been overwritten; restore them with"
            "\n  git checkout -- data/"
            "\nIf this change is intentional, remove the artifact's entry from"
            f"\n  {FREEZE_MANIFEST_JSON}\nand rebuild, so re-freezing is a deliberate act."
        )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict] = {}

    # --- HarmBench standard behaviors -> the 200-row source of truth ---
    hb_bytes = _fetch(HARMBENCH_URL, HARMBENCH_SHA256)
    hb = pd.read_csv(io.BytesIO(hb_bytes))
    std = hb[hb["FunctionalCategory"] == "standard"].copy()
    std = std.sort_values("BehaviorID", kind="stable").reset_index(drop=True)
    assert len(std) == 200, f"expected 200 standard behaviors, got {len(std)}"
    assert std["BehaviorID"].is_unique, "BehaviorID not unique among standard behaviors"
    behaviors = pd.DataFrame(
        {
            "prompt_id": range(len(std)),
            "behavior_id": std["BehaviorID"].values,
            "category": std["SemanticCategory"].values,
            "source": HARMBENCH_SOURCE,
            "forbidden_prompt": std["Behavior"].values,
        }
    )
    artifacts[HARMBENCH_STANDARD_CSV.name] = {
        "sha256": write_csv(HARMBENCH_STANDARD_CSV, behaviors[COLS_HARMFUL]),
        "rows": len(behaviors),
    }

    # --- 128 extraction / 72 validation split (disjoint complement within 200) ---
    ext_idx, val_idx = split_indices(200, N_EXTRACTION, SEED)
    assert set(ext_idx).isdisjoint(val_idx)
    assert sorted(ext_idx + val_idx) == list(range(200))

    ext_df = behaviors.iloc[ext_idx]
    val_df = behaviors.iloc[val_idx]
    artifacts[EXTRACTION_HARMFUL_CSV.name] = {
        "sha256": write_csv(EXTRACTION_HARMFUL_CSV, ext_df[COLS_HARMFUL]),
        "rows": len(ext_df),
    }
    artifacts[VALIDATION_HARMFUL_CSV.name] = {
        "sha256": write_csv(VALIDATION_HARMFUL_CSV, val_df[COLS_HARMFUL]),
        "rows": len(val_df),
    }

    # --- 128 Alpaca harmless (empty-input, exact-text deduped, seeded) ---
    alpaca_bytes = _fetch(ALPACA_URL, ALPACA_SHA256)
    alpaca = json.loads(alpaca_bytes)
    pool = []  # (original_index, instruction), empty-input, first occurrence of each text
    seen: set[str] = set()
    for i, item in enumerate(alpaca):
        if item.get("input", "").strip() != "":
            continue
        text = item["instruction"]
        if text in seen:
            continue
        seen.add(text)
        pool.append((i, text))
    harmless_idx = sample_indices(len(pool), N_HARMLESS, SEED)
    harmless_rows = [pool[j] for j in harmless_idx]
    harmless_df = pd.DataFrame(
        {
            "prompt": [t for _, t in harmless_rows],
            "alpaca_index": [i for i, _ in harmless_rows],
        }
    )
    assert harmless_df["prompt"].is_unique and len(harmless_df) == N_HARMLESS
    artifacts[EXTRACTION_HARMLESS_CSV.name] = {
        "sha256": write_csv(EXTRACTION_HARMLESS_CSV, harmless_df[["prompt", "alpaca_index"]]),
        "rows": len(harmless_df),
    }

    # --- StrongREJECT-313: verify the committed file, pin its hash, cut the pilot ---
    sr_sha = sha256_file(STRONGREJECT_CSV)
    if sr_sha != STRONGREJECT_SHA256:
        raise SystemExit(
            f"committed StrongREJECT hash changed\n  expected {STRONGREJECT_SHA256}\n  got {sr_sha}"
        )
    sr = load_strongreject_prompts()
    assert len(sr) == 313
    pilot_ids = sample_indices(len(sr), N_PILOT, SEED)
    pilot_df = sr.iloc[pilot_ids].copy()
    assert len(pilot_df) == N_PILOT and pilot_df["prompt_id"].is_unique
    cols_pilot = ["prompt_id", "category", "source", "forbidden_prompt"]
    artifacts[PILOT_PROMPTS_CSV.name] = {
        "sha256": write_csv(PILOT_PROMPTS_CSV, pilot_df[cols_pilot]),
        "rows": len(pilot_df),
    }

    # --- verbatim overlap of each harmful pool with eval, under all normalizers.
    # harmbench200 is the headline; the subset rows are derivable but kept so each
    # split is recorded as independently checked against eval.
    sr_prompts = sr["forbidden_prompt"].tolist()

    def _overlap_report(pool: list[str]) -> dict[str, int]:
        return {n: verbatim_overlap(pool, sr_prompts, normalize=n) for n in _NORMALIZERS}

    verbatim = {
        "harmbench200_vs_strongreject313": _overlap_report(
            behaviors["forbidden_prompt"].tolist()
        ),
        "extraction128_vs_strongreject313": _overlap_report(
            ext_df["forbidden_prompt"].tolist()
        ),
        "validation72_vs_strongreject313": _overlap_report(
            val_df["forbidden_prompt"].tolist()
        ),
        "harmless128_vs_strongreject313": _overlap_report(harmless_df["prompt"].tolist()),
    }
    exact_match_count = verbatim["harmbench200_vs_strongreject313"]["none"]
    assert exact_match_count == 0, f"expected 0 verbatim overlap, got {exact_match_count}"

    _assert_frozen_hashes_unchanged(artifacts)  # before anything reaches the manifest

    manifest = {
        "seed": SEED,
        "rng_method": "numpy.random.RandomState (legacy MT19937); permutation over canonical order",
        "canonical_order": {
            "harmbench_standard": "sorted by BehaviorID, stable (prompt_id = 0-based index)",
            "strongreject": "file row order (prompt_id = 0-based index)",
        },
        "split_method": (
            "RandomState(SEED).permutation(200): [:128]=extraction, [128:]=validation; "
            "non-stratified; disjoint complement within the 200 standard behaviors. "
            "Each split's prompt_id is the parent row index it was drawn at, so the two "
            "id sets partition range(200)"
        ),
        "pilot_method": (
            "RandomState(SEED).permutation(313)[:30] (seeded-random 30 of the 313; "
            "a fixed a-priori slice, not a disjoint held-out set and not the first 30; "
            "reused as the first block of the main evaluation run)"
        ),
        "counts": {
            "harmbench_standard": len(behaviors),
            "extraction_harmful": len(ext_df),
            "validation_harmful": len(val_df),
            "extraction_harmless": len(harmless_df),
            "pilot_prompts": len(pilot_df),
            "strongreject": len(sr),
        },
        "verbatim_overlap": verbatim,
        "verbatim_exact_match_count": exact_match_count,
        "sources": {
            "harmbench_behaviors_text_all.csv": {"url": HARMBENCH_URL, "sha256": HARMBENCH_SHA256},
            "alpaca_data.json": {"url": ALPACA_URL, "sha256": ALPACA_SHA256},
            "strongreject_dataset.csv": {"sha256": STRONGREJECT_SHA256},
        },
        "artifacts": artifacts,
        "pilot_prompt_ids": pilot_ids,
    }
    write_manifest(FREEZE_MANIFEST_JSON, manifest)

    print(f"\nfroze {len(artifacts)} artifacts under {DATA_DIR}")
    print(f"verbatim exact-match count (HarmBench-200 vs StrongREJECT-313): {exact_match_count}")
    for name, meta in sorted(artifacts.items()):
        print(f"  {name:34s} rows={meta['rows']:<4d} sha256={meta['sha256'][:16]}…")
    print(f"  {FREEZE_MANIFEST_JSON.name}")


if __name__ == "__main__":
    main()
