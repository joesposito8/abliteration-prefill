#!/usr/bin/env python3
"""Freeze what the waves produced into the portfolio's 4,069 prefills.

Replays the frozen rules over every wave's log one last time — the logs are the
evidence, this CSV is the interface every downstream condition reads. Needs no GPU: the
waves already spent it.

Run:  python scripts/freeze_prefills.py results/prefills
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from generation.gemma import HELPER_DECODING  # noqa: E402
from harness.helper_provider import HELPER_BATCH  # noqa: E402
from prefills import (  # noqa: E402
    HELPER_MODEL,
    HELPER_REVISION,
    PORTFOLIO,
    STATIC_BASELINE,
    PrefillResult,
)
from prefills.families import STATIC_SLOT_ID  # noqa: E402
from study.datasets import (  # noqa: E402
    PORTFOLIO_MANIFEST_JSON,
    PREFILL_MANIFEST_JSON,
    PREFILLS_CSV,
    STRONGREJECT_CSV,
    load_strongreject_prompts,
)
from study.manifest import (  # noqa: E402
    assert_frozen_unchanged,
    rollup_sha256,
    sha256_file,
    write_csv,
    write_manifest,
)

from produce_prefills import drawn_all, replay  # noqa: E402

COLUMNS = (
    "prompt_id",
    "slot",
    "prefill",
    "helper_seed",
    "attempts",
    "failed",
    "resampled_dup",
    "reasons",
)

REPRODUCTION = (
    "a prefill is reproduced from this table's sha256, not by replaying helper_seed: "
    "rows sharing a forward pass draw from one RNG stream, so an individual row is only "
    "reproducible by replaying the identical batch. helper_seed's frozen job is to make "
    "an attempt a different draw from the one it retries, which holds exactly — a retry "
    "is only ever generated in a later wave, and so in a different batch."
)


def _generated_row(prompt_id: int, slot: str, result: PrefillResult) -> dict:
    """One helper-generated cell, as the frozen rules settled it."""
    return {
        "prompt_id": prompt_id,
        "slot": slot,
        "prefill": result.prefill or "",
        # 64 unsigned bits, which a pandas integer column turns into a float.
        "helper_seed": "" if result.seed is None else str(result.seed),
        "attempts": result.attempts,
        "failed": result.failed,
        "resampled_dup": result.resampled_dup,
        "reasons": ";".join(result.reasons),
    }


def _static_row(prompt_id: int) -> dict:
    """The 13th slot: one fixed string, the same for every prompt, never generated."""
    return {
        "prompt_id": prompt_id,
        "slot": STATIC_SLOT_ID,
        "prefill": STATIC_BASELINE,
        "helper_seed": "",
        "attempts": None,
        "failed": False,
        "resampled_dup": False,
        "reasons": "",
    }


def build_table(results: Mapping) -> pd.DataFrame:
    """One row per (prompt, slot), in prompt then portfolio order."""
    rows = []
    for prompt_id in load_strongreject_prompts()["prompt_id"].tolist():
        for slot in PORTFOLIO:
            if slot == STATIC_SLOT_ID:
                rows.append(_static_row(prompt_id))
                continue
            family, variant = slot.split(":")
            result = results[(prompt_id, family)][int(variant)]
            rows.append(_generated_row(prompt_id, slot, result))

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table["attempts"] = table["attempts"].astype("Int64")
    return table


def assert_shape(table: pd.DataFrame) -> None:
    """The preregistered shape: every prompt against every slot, exactly once."""
    prompts = len(load_strongreject_prompts())
    expected = prompts * len(PORTFOLIO)
    if len(table) != expected:
        raise SystemExit(
            f"built {len(table)} rows, expected {prompts} prompts x {len(PORTFOLIO)} "
            f"slots = {expected}"
        )

    repeated = int(table.duplicated(subset=["prompt_id", "slot"]).sum())
    if repeated:
        raise SystemExit(f"{repeated} rows repeat a (prompt_id, slot) already written")

    static = table[table["slot"] == STATIC_SLOT_ID]["prefill"]
    if len(static) != prompts or not (static == STATIC_BASELINE).all():
        raise SystemExit("the static baseline is not verbatim on all of its rows")


def build_manifest(table: pd.DataFrame, csv_sha256: str) -> dict:
    """The rules the table cannot carry, and the files it is only meaningful against."""
    spec = {
        "table": {"file": PREFILLS_CSV.name, "sha256": csv_sha256},
        "helper": {
            "model": HELPER_MODEL,
            "revision": HELPER_REVISION,
            "decoding": dict(HELPER_DECODING),
            "batch": HELPER_BATCH,
        },
        "counts": {
            "rows": len(table),
            "prompts": int(table["prompt_id"].nunique()),
            "slots": int(table["slot"].nunique()),
            "generated": int((table["slot"] != STATIC_SLOT_ID).sum()),
            "failed": int(table["failed"].sum()),
            "resampled_dup": int(table["resampled_dup"].sum()),
            "retried": int((table["attempts"] > 1).sum()),
        },
        "reproduction": REPRODUCTION,
        # The portfolio hash covers the templates and the rules these were drawn under,
        # so a later edit to either is detectable against this frozen set.
        "inputs": {
            path.name: sha256_file(path)
            for path in (STRONGREJECT_CSV, PORTFOLIO_MANIFEST_JSON)
        },
    }
    manifest = dict(spec)
    manifest["prefill_sha256"] = rollup_sha256(spec)
    return manifest


def frozen_table_sha256() -> dict[str, str] | None:
    """What the committed manifest pins, or ``None`` before the first freeze."""
    if not PREFILL_MANIFEST_JSON.exists():
        return None
    return {PREFILLS_CSV.name: json.loads(PREFILL_MANIFEST_JSON.read_text())["table"]["sha256"]}


def main(argv: list[str]) -> None:
    root = Path(argv[0]).resolve()

    drawn = drawn_all(root)
    results, missing = replay(drawn)
    if missing:
        raise SystemExit(
            f"{len(missing)} draws the rules ask for are not in {root}, so the run is "
            f"unfinished (first: {missing[0]}). Re-run scripts/produce_prefills.py."
        )

    table = build_table(results)
    assert_shape(table)

    csv_sha256 = write_csv(PREFILLS_CSV, table)
    assert_frozen_unchanged(
        PREFILL_MANIFEST_JSON, frozen_table_sha256(), {PREFILLS_CSV.name: csv_sha256}
    )
    manifest = build_manifest(table, csv_sha256)
    write_manifest(PREFILL_MANIFEST_JSON, manifest)

    counts = manifest["counts"]
    print(f"froze {counts['rows']:,} prefills -> {PREFILLS_CSV}")
    print(f"  {counts['prompts']} prompts x {counts['slots']} slots, "
          f"{counts['generated']:,} generated from {len(drawn):,} draws")
    print(f"  {counts['retried']} retried, {counts['resampled_dup']} resampled, "
          f"{counts['failed']} failed")
    print(f"  prefill_sha256: {manifest['prefill_sha256']}")
    if counts["failed"]:
        print("\nFAILED CELLS PRESENT — build_dataset will refuse to run on this set.")


if __name__ == "__main__":
    main(sys.argv[1:])
