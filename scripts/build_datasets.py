#!/usr/bin/env python3
"""Build and freeze the study datasets from their upstream sources.

Deterministic and idempotent: given the pinned source files, re-running produces
byte-identical artifacts. Network is used only to fetch the two upstream sources,
whose SHA-256s are pinned below and asserted on every run (a mutable-``main``
upstream that drifted aborts the build rather than silently re-freezing).

Outputs (all under ``data/``):
  harmbench_standard_behaviors.csv   200 HarmBench standard behaviors (source of truth)
  extraction_harmful.csv             128 harmful behaviors (direction extraction)
  validation_harmful.csv              72 harmful behaviors (primary-layer selection)
  extraction_harmless.csv            128 Alpaca instructions (harmless contrast)
  pilot_prompts.csv                   30 StrongREJECT prompts (seeded pilot slice)
  freeze_manifest.json               seeds, hashes, counts, verbatim-overlap report

Run:  python scripts/build_datasets.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from study import SEED  # noqa: E402
from study.datasets import (  # noqa: E402
    DATA_DIR,
    EXTRACTION_HARMFUL_CSV,
    EXTRACTION_HARMLESS_CSV,
    FREEZE_MANIFEST_JSON,
    HARMBENCH_STANDARD_CSV,
    PILOT_PROMPTS_CSV,
    STRONGREJECT_CSV,
    VALIDATION_HARMFUL_CSV,
    load_strongreject,
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, expected_sha256: str) -> bytes:
    print(f"fetch {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    got = _sha256(data)
    if got != expected_sha256:
        raise SystemExit(
            f"source hash drift for {url}\n  expected {expected_sha256}\n  got      {got}\n"
            "Upstream changed. Re-vendor deliberately (update the pinned hash) before rebuilding."
        )
    return data


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> str:
    """Write ``df[columns]`` deterministically; return the file SHA-256."""
    df[columns].to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    return _sha256(path.read_bytes())


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict] = {}

    # --- HarmBench standard behaviors -> the 200-row source of truth ---
    hb_bytes = _fetch(HARMBENCH_URL, HARMBENCH_SHA256)
    hb = pd.read_csv(io.BytesIO(hb_bytes))
    std = hb[hb["FunctionalCategory"] == "standard"].copy()
    std = std.sort_values("BehaviorID", kind="stable").reset_index(drop=True)
    std = std.rename(columns={"SemanticCategory": "semantic_category"})
    assert len(std) == 200, f"expected 200 standard behaviors, got {len(std)}"
    assert std["BehaviorID"].is_unique, "BehaviorID not unique among standard behaviors"
    std_out = std.rename(columns={"BehaviorID": "behavior_id"})[
        ["behavior_id", "Behavior", "semantic_category"]
    ]
    artifacts[HARMBENCH_STANDARD_CSV.name] = {
        "sha256": _write_csv(
            std_out, HARMBENCH_STANDARD_CSV, ["behavior_id", "Behavior", "semantic_category"]
        ),
        "rows": len(std_out),
    }

    # --- 128 extraction / 72 validation split (disjoint complement within 200) ---
    ext_idx, val_idx = split_indices(200, N_EXTRACTION, SEED)
    assert set(ext_idx).isdisjoint(val_idx)
    assert sorted(ext_idx + val_idx) == list(range(200))

    def _harmful_frame(idx: list[int]) -> pd.DataFrame:
        f = std.iloc[idx].copy()
        return pd.DataFrame(
            {
                "prompt": f["Behavior"].values,
                "behavior_id": f["BehaviorID"].values,
                "semantic_category": f["semantic_category"].values,
            }
        )

    ext_df = _harmful_frame(ext_idx)
    val_df = _harmful_frame(val_idx)
    cols_harmful = ["prompt", "behavior_id", "semantic_category"]
    artifacts[EXTRACTION_HARMFUL_CSV.name] = {
        "sha256": _write_csv(ext_df, EXTRACTION_HARMFUL_CSV, cols_harmful),
        "rows": len(ext_df),
    }
    artifacts[VALIDATION_HARMFUL_CSV.name] = {
        "sha256": _write_csv(val_df, VALIDATION_HARMFUL_CSV, cols_harmful),
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
        "sha256": _write_csv(harmless_df, EXTRACTION_HARMLESS_CSV, ["prompt", "alpaca_index"]),
        "rows": len(harmless_df),
    }

    # --- StrongREJECT-313: verify the committed file, pin its hash, cut the pilot ---
    sr_sha = _sha256(STRONGREJECT_CSV.read_bytes())
    if sr_sha != STRONGREJECT_SHA256:
        raise SystemExit(
            f"committed StrongREJECT hash changed\n  expected {STRONGREJECT_SHA256}\n  got {sr_sha}"
        )
    sr = load_strongreject()
    assert len(sr) == 313
    pilot_ids = sample_indices(len(sr), N_PILOT, SEED)
    pilot_df = sr.iloc[pilot_ids].copy()
    assert len(pilot_df) == N_PILOT and pilot_df["prompt_id"].is_unique
    cols_pilot = ["prompt_id", "category", "source", "forbidden_prompt"]
    artifacts[PILOT_PROMPTS_CSV.name] = {
        "sha256": _write_csv(pilot_df, PILOT_PROMPTS_CSV, cols_pilot),
        "rows": len(pilot_df),
    }

    # --- verbatim overlap (guardrail 11): headline union + per-subset + variants ---
    sr_prompts = sr["forbidden_prompt"].tolist()
    std_prompts = std["Behavior"].tolist()
    ext_prompts = ext_df["prompt"].tolist()
    val_prompts = val_df["prompt"].tolist()
    verbatim = {
        "harmbench200_vs_strongreject313": {
            n: verbatim_overlap(std_prompts, sr_prompts, normalize=n)
            for n in ("none", "strip", "lower", "strip_lower")
        },
        "extraction128_vs_strongreject313": verbatim_overlap(ext_prompts, sr_prompts),
        "validation72_vs_strongreject313": verbatim_overlap(val_prompts, sr_prompts),
        "harmless128_vs_strongreject313": verbatim_overlap(
            harmless_df["prompt"].tolist(), sr_prompts
        ),
    }
    exact_match_count = verbatim["harmbench200_vs_strongreject313"]["none"]
    assert exact_match_count == 0, f"expected 0 verbatim overlap, got {exact_match_count}"

    manifest = {
        "seed": SEED,
        "rng_method": "numpy.random.RandomState (legacy MT19937); permutation over canonical order",
        "canonical_order": {
            "harmbench_standard": "sorted by BehaviorID (stable)",
            "strongreject": "file row order (prompt_id = 0-based index)",
        },
        "split_method": (
            "RandomState(SEED).permutation(200): [:128]=extraction, [128:]=validation; "
            "non-stratified; disjoint complement within the 200 standard behaviors"
        ),
        "pilot_method": (
            "RandomState(SEED).permutation(313)[:30] (seeded-random 30 of the 313; "
            "not disjoint, not the first 30; reused into the main run per guardrail 18)"
        ),
        "counts": {
            "harmbench_standard": len(std_out),
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
    FREEZE_MANIFEST_JSON.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nfroze {len(artifacts)} artifacts under {DATA_DIR}")
    print(f"verbatim exact-match count (HarmBench-200 vs StrongREJECT-313): {exact_match_count}")
    for name, meta in sorted(artifacts.items()):
        print(f"  {name:34s} rows={meta['rows']:<4d} sha256={meta['sha256'][:16]}…")
    print(f"  {FREEZE_MANIFEST_JSON.name}")


if __name__ == "__main__":
    main()
