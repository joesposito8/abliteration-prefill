#!/usr/bin/env python3
"""Freeze the main run's work list: one row per generation, and the batch it belongs to.

313 prompts x 507 generations. The count is asserted against the preregistered workload
before anything downstream can consume it, so a malformed work list is caught here rather
than on a rented GPU.

Run:  python scripts/build_run_manifest.py
"""

from __future__ import annotations

import json
import sys
from itertools import batched, product
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from abliteration.selection import condition_id, sweep_layers  # noqa: E402
from generation.qwen import DECODING, MODEL_ID, N_LAYERS, REVISION  # noqa: E402
from harness.batching import BATCH  # noqa: E402
from harness.dataset import CONTROL, EVAL_SETS  # noqa: E402
from prefills import PORTFOLIO  # noqa: E402
from study import SEED  # noqa: E402
from study.datasets import (  # noqa: E402
    ABLITERATION_MANIFEST_JSON,
    PORTFOLIO_MANIFEST_JSON,
    RUN_MANIFEST_CSV,
    RUN_MANIFEST_JSON,
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

ATTEMPTS = EVAL_SETS["strongreject"].attempts

# The preregistered workload, asserted independently of how the rows are built.
PROMPTS = 313
GENERATIONS_PER_PROMPT = 507

COLUMNS = (
    "prompt_id",
    "layer_id",
    "condition",
    "prefill_id",
    "replicate_id",
    "seed",
    "gen_config_hash",
    "batch_id",
    "batch_size",
)

SEED_SCOPE = (
    "one project-wide seed on every row; the per-forward-pass RNG stream is derived from "
    "that seed and the batch's exact prompts, so every batch samples independently and "
    "matching batches of different conditions share a stream"
)

ORDER = (
    "condition (base, layer_00..layer_35) -> arm (unprefilled, then prefilled) -> "
    "prefill_id (portfolio order) -> prompt_id ascending -> replicate_id ascending. The "
    "unprefilled arm matches harness.dataset.build_dataset row for row; the prefilled arm "
    "is slot-major where build_dataset is prompt-major, so that one declared batch carries "
    "one prefill"
)

PARTITION_SCOPE = (
    "a declared plan, not an observation: harness.batching composes a batch from whatever "
    "arrives while the previous one holds the GPU, so the batch a row actually ran in is "
    "the one its log records through batch_seed, batch_size and batch_position"
)


def primary_layer() -> int:
    """The composition condition's layer, read from the abliteration freeze."""
    return json.loads(ABLITERATION_MANIFEST_JSON.read_text())["primary"]["layer"]


def gen_config() -> dict:
    """What fixes how a row's text is produced, as one hashable object.

    Batch width belongs to the row rather than here: it changes the text too, but the last
    batch of every group is narrower, so a constant would be false for those rows.
    """
    return {"model": MODEL_ID, "revision": REVISION, "decoding": dict(DECODING)}


def groups(primary: int) -> list[tuple[int | None, str, int]]:
    """``(layer, prefill_id, replicates)`` in frozen manifest order.

    Layer and prefill are the partition key: ``harness.run`` applies one weight edit per
    condition, and ``generate_batch`` takes a single prefill for the whole call. The
    prefill half is a property of this plan rather than of the harness, which bakes the
    prefill into the prompt and would mix them happily. Only the base and the composition
    primary get a prefilled arm.
    """
    plan = []
    for layer in sweep_layers(N_LAYERS):
        plan.append((layer, CONTROL, ATTEMPTS))
        if layer is None or layer == primary:
            plan += [(layer, slot, 1) for slot in PORTFOLIO]
    return plan


def build_table(prompt_ids: list[int], primary: int) -> pd.DataFrame:
    """One row per generation, in frozen order, each carrying the batch it belongs to.

    ``batched`` cuts each group at the frozen width and hands back the remainder as the
    last batch, so ``batch_size`` is measured rather than declared, and no batch can span
    a group because the cut happens inside one.
    """
    digest = rollup_sha256(gen_config())
    rows: list[tuple] = []
    batch_id = 0

    for layer, prefill_id, replicates in groups(primary):
        condition = condition_id(layer)
        for batch in batched(product(prompt_ids, range(replicates)), BATCH):
            rows += [
                (prompt_id, layer, condition, prefill_id, replicate,
                 SEED, digest, batch_id, len(batch))
                for prompt_id, replicate in batch
            ]
            batch_id += 1

    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table["layer_id"] = table["layer_id"].astype("Int64")
    assert_workload(table)
    return table


def assert_workload(table: pd.DataFrame) -> None:
    """The preregistered gate: the right number of cells, each enumerated exactly once."""
    if len(table) != PROMPTS * GENERATIONS_PER_PROMPT:
        raise SystemExit(
            f"built {len(table)} rows, but the preregistered workload is {PROMPTS} x "
            f"{GENERATIONS_PER_PROMPT} = {PROMPTS * GENERATIONS_PER_PROMPT}: the study's "
            "shape moved, not just the code's"
        )
    key = ["condition", "prefill_id", "prompt_id", "replicate_id"]
    repeated = int(table.duplicated(subset=key).sum())
    if repeated:
        raise SystemExit(f"{repeated} rows repeat a cell already enumerated, keyed by {key}")


def build_manifest(table: pd.DataFrame, csv_sha256: str, primary: int) -> dict:
    """The rules the table cannot carry, and the files it is only meaningful against."""
    config = gen_config()
    spec = {
        "table": {"file": RUN_MANIFEST_CSV.name, "sha256": csv_sha256},
        "gen_config": config,
        "gen_config_hash": rollup_sha256(config),
        "seed": SEED,
        "seed_scope": SEED_SCOPE,
        "primary": {"condition": condition_id(primary), "layer": primary},
        "counts": {
            "prompts": int(table["prompt_id"].nunique()),
            "generations_per_prompt": GENERATIONS_PER_PROMPT,
            "rows": len(table),
            "batches": int(table["batch_id"].nunique()),
        },
        "order": ORDER,
        "partition": {
            "key": ["condition", "prefill_id"],
            "width": BATCH,
            "rule": (
                "consecutive rows of each group in manifest order; the last batch of a "
                "group carries the remainder"
            ),
            "scope": PARTITION_SCOPE,
        },
        "inputs": {
            path.name: sha256_file(path)
            for path in (STRONGREJECT_CSV, ABLITERATION_MANIFEST_JSON, PORTFOLIO_MANIFEST_JSON)
        },
    }
    manifest = dict(spec)
    manifest["run_sha256"] = rollup_sha256(spec)
    return manifest


def assert_frozen_table_unchanged(csv_sha256: str) -> None:
    """Every generation and every grade is keyed to the committed table, so a silent
    re-freeze would leave finished work pointing at cells that no longer exist."""
    frozen = None
    if RUN_MANIFEST_JSON.exists():
        recorded = json.loads(RUN_MANIFEST_JSON.read_text())["table"]["sha256"]
        frozen = {RUN_MANIFEST_CSV.name: recorded}
    assert_frozen_unchanged(
        RUN_MANIFEST_JSON, frozen, {RUN_MANIFEST_CSV.name: csv_sha256}
    )


def main() -> None:
    primary = primary_layer()
    table = build_table(load_strongreject_prompts()["prompt_id"].tolist(), primary)

    csv_sha256 = write_csv(RUN_MANIFEST_CSV, table)
    assert_frozen_table_unchanged(csv_sha256)
    manifest = build_manifest(table, csv_sha256, primary)
    write_manifest(RUN_MANIFEST_JSON, manifest)

    counts = manifest["counts"]
    print(f"froze {counts['rows']:,} generations -> {RUN_MANIFEST_CSV}")
    print(f"  {counts['prompts']} prompts x {counts['generations_per_prompt']} generations")
    print(f"  {counts['batches']:,} batches at width {BATCH}, "
          f"composition on {manifest['primary']['condition']}")
    print(f"  run_sha256: {manifest['run_sha256']}")


if __name__ == "__main__":
    main()
