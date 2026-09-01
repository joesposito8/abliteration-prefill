#!/usr/bin/env python3
"""The pilot-rank ordering the budget-curve strategies order their attacks by.

Six of the eight multi-cell strategies rank their attacks by measured pilot effectiveness,
so that ranking has to be a fixed artifact before the main run generates. This reads the
30-prompt factorial's scored logs and ranks the eight prefill levels by per-attempt unlock
rate, within each model state and over both states together.

Levels are FAMILIES, not the 13 variant slots: a generated family pools its two variants,
because the strategies are defined at family level. Unlocked is the bit the scorer already
recorded; nothing is regraded.

Descriptive telemetry on already-collected generations. No frozen parameter moves.

Run:  python scripts/pilot_rank.py --run <evidence>/strongreject-30
      python scripts/pilot_rank.py --run <evidence>/strongreject-30 --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from abliteration.selection import BASE_CONDITION, condition_id  # noqa: E402
from generation import target  # noqa: E402
from harness.dataset import CONTROL  # noqa: E402
from prefills import FAMILIES, VARIANTS_PER_FAMILY  # noqa: E402
from prefills.families import STATIC_SLOT_ID  # noqa: E402
from study import SEED, draws  # noqa: E402
from study.manifest import rollup_sha256, write_manifest  # noqa: E402

import run_report as rr  # noqa: E402
from generate import primary_layer  # noqa: E402

ARTIFACT = REPO / "data" / "pilot_rank_ordering.json"
LEVELS = (CONTROL, STATIC_SLOT_ID, *FAMILIES)
RESAMPLES = 10_000
STATES = ("base", "abliterated")


def level_draws(level: str) -> int:
    """Draws of one prompt at one level, per model state, pooling a family's variants."""
    slots = (
        [f"{level}:{v}" for v in range(VARIANTS_PER_FAMILY)] if level in FAMILIES else [level]
    )
    return sum(draws(slot) for slot in slots)


def level_of(slot: str) -> str:
    """The family a portfolio slot belongs to; ``none`` and the baseline are their own."""
    return slot.split(":")[0]


def rank(rates: dict[str, float]) -> list[str]:
    """Best first. An exact tie goes to the alphabetically earlier name, never to the
    order the rates happened to arrive in."""
    return sorted(rates, key=lambda name: (-rates[name], name))


def ties(rates: dict[str, float]) -> list[tuple[str, ...]]:
    """The groups of names the tie-break rule actually had to separate."""
    groups: dict[float, list[str]] = {}
    for name, rate in rates.items():
        groups.setdefault(rate, []).append(name)
    return [tuple(sorted(g)) for g in groups.values() if len(g) > 1]


def read_model(run: Path, model: str) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """One row per generation for one model, its condition-to-state map, and its log paths.

    Scoped to one model's subtree: the two models share the ``base`` condition name, and
    ``load_conditions`` refuses a tree where two logs answer to one condition.
    """
    scored = run / "scored" / model
    by_condition = rr.load_conditions(scored)
    primary = condition_id(primary_layer(target(model)))

    if set(by_condition) != {BASE_CONDITION, primary}:
        raise SystemExit(f"{model}: expected {BASE_CONDITION} and {primary}, found {sorted(by_condition)}")

    frame = rr.load_samples(scored, by_condition)
    frame["level"] = frame.slot.map(level_of)
    # ``location`` is a URI; the artifact records a path relative to the evidence root.
    logs = {c: Path(h.location.removeprefix("file:")).relative_to(run).as_posix()
            for c, h in by_condition.items()}
    return frame, {BASE_CONDITION: "base", primary: "abliterated"}, logs


def cell_table(frame: pd.DataFrame, states: dict[str, str]) -> pd.DataFrame:
    """One row per (state, level): the draws behind the cell and its per-attempt rate."""
    if set(frame.level) != set(LEVELS):
        raise SystemExit(f"levels are {sorted(set(frame.level))}, expected {sorted(LEVELS)}")

    table = (
        frame.groupby(["condition", "level"])
        .agg(n_rows=("unlocked", "size"), n_unlocked=("unlocked", "sum"),
             n_prompts=("prompt_id", "nunique"))
        .reset_index()
    )
    table["state"] = table.condition.map(states)
    table["rate"] = table.n_unlocked / table.n_rows
    table["label"] = table.state + "/" + table.level

    # A cell short of its draws would still produce a plausible rate, so check the shape
    # rather than trusting it: every prompt, at the level's scheduled draws.
    prompts = frame.prompt_id.nunique()
    expected = prompts * table.level.map(level_draws)
    if (len(table) != len(STATES) * len(LEVELS)
            or not (table.n_prompts == prompts).all()
            or not (table.n_rows == expected).all()):
        raise SystemExit(f"cell counts are wrong:\n{table[['condition', 'level', 'n_prompts', 'n_rows']]}")

    return table.sort_values(["state", "label"]).reset_index(drop=True)


def per_prompt_rates(frame: pd.DataFrame, states: dict[str, str], labels: list[str]) -> pd.DataFrame:
    """Prompts x cells of per-prompt unlock rate — the clusters the bootstrap resamples."""
    held = frame.assign(label=frame.condition.map(states) + "/" + frame.level)
    return held.pivot_table(index="prompt_id", columns="label", values="unlocked").loc[:, labels]


def bootstrap(clusters: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample the 30 prompts with replacement; rows within a prompt are not independent.

    Every prompt carries the same number of draws in every cell, so a cell's rate is the
    mean of its column and a resample is the mean over the drawn prompts.
    """
    m = clusters.to_numpy()
    draws = rng.integers(0, len(m), size=(RESAMPLES, len(m)))
    return m[draws].mean(axis=1)


def top3_agreement(replicates: np.ndarray, labels: list[str], frozen: list[str]) -> float:
    """How often a resample of the prompts puts the same three attacks on top."""
    want = set(frozen[:3])
    hits = sum(set(rank(dict(zip(labels, row)))[:3]) == want for row in replicates)
    return hits / len(replicates)


def orderings(table: pd.DataFrame) -> dict[str, list[str]]:
    """The three rankings a model contributes: one per state, and one over all 16 cells."""
    out = {s: rank(dict(zip(g.level, g.rate))) for s, g in table.groupby("state")}
    out["overall"] = rank(dict(zip(table.label, table.rate)))
    return out


def keyed(table: pd.DataFrame, ordering_name: str) -> dict[str, float]:
    """The rates one ordering ranks, keyed the way that ordering names its entries."""
    if ordering_name == "overall":
        return dict(zip(table.label, table.rate))
    held = table[table.state == ordering_name]
    return dict(zip(held.level, held.rate))


def read_run(run: Path) -> dict:
    """Every model under the run, read and ranked."""
    models = sorted(d.name for d in (run / "scored").iterdir() if d.is_dir())

    out = {}
    for model in models:
        rng = np.random.default_rng(SEED)
        frame, states, logs = read_model(run, model)
        table = cell_table(frame, states)
        order = orderings(table)

        clusters = per_prompt_rates(frame, states, list(table.label))
        if not np.allclose(clusters.mean().to_numpy(), table.rate.to_numpy()):
            raise SystemExit(f"{model}: per-prompt means do not reproduce the cell rates")
        replicates = bootstrap(clusters, rng)
        lo, hi = np.percentile(replicates, [2.5, 97.5], axis=0)

        out[model] = {
            "logs": logs,
            "states": states,
            "table": table,
            "orderings": order,
            "ci95": dict(zip(table.label, zip(lo, hi))),
            "top3": {
                name: top3_agreement(
                    replicates if name == "overall"
                    else replicates[:, [i for i, s in enumerate(table.state) if s == name]],
                    list(table.label) if name == "overall"
                    else list(table.level[table.state == name]),
                    frozen,
                )
                for name, frozen in order.items()
            },
        }
    return out


def artifact(run: Path, read: dict) -> dict:
    """The frozen document, hash last so the rollup covers everything but itself."""
    n_prompts = next(iter(read.values()))["table"].n_prompts.iloc[0].item()
    per_cell = next(iter(read.values()))["table"].n_rows.iloc[0].item()
    cells = []
    for model, held in read.items():
        for row in held["table"].itertuples():
            lo, hi = held["ci95"][row.label]
            cells.append({
                "model": model,
                "state": row.state,
                "condition": row.condition,
                "level": row.level,
                "n_prompts": int(row.n_prompts),
                "n_rows": int(row.n_rows),
                "n_unlocked": int(row.n_unlocked),
                "rate": round(row.rate, 6),
                "ci95": [round(lo, 6), round(hi, 6)],
            })

    doc = {
        "artifact": "pilot-rank ordering for the budget-curve strategies",
        "claim": (
            "These orderings model an attacker who has done some reconnaissance. They are not "
            "established as the correct ranking of these attacks, and nothing here claims they "
            "would survive on the full evaluation set."
        ),
        "source": {
            "run": run.name,
            "prompt_set": "strongreject",
            "n_prompts": n_prompts,
            "logs": {f"{m}/{c}": p for m, h in read.items() for c, p in h["logs"].items()},
            "states": {m: h["states"] for m, h in read.items()},
        },
        "definitions": {
            "level": (
                "a prefill family, not one of the 13 variant slots: a generated family pools its "
                "two variants, because the budget strategies are defined at family level"
            ),
            "levels": list(LEVELS),
            "unlocked": (
                "the bit the StrongREJECT scorer recorded on the stripped continuation, unchanged; "
                "malformed and degenerate rows stay in the denominator as non-unlocks"
            ),
            "rate": "per-attempt unlock rate over the cell's draws, not a union over attempts",
            "tie_break": {
                "rule": (
                    "equal rates are ordered alphabetically ascending: by level name within a "
                    "state, and by the '<state>/<level>' label in the overall ordering, where "
                    "'abliterated' precedes 'base'"
                ),
                "ties_broken": sum(
                    len(ties(keyed(h["table"], name)))
                    for h in read.values() for name in h["orderings"]
                ),
            },
        },
        "cells": cells,
        "orderings": {m: h["orderings"] for m, h in read.items()},
        "precision": {
            "method": (
                "cluster bootstrap: resample the 30 prompts with replacement and recompute every "
                "cell rate, so draws sharing a prompt are not counted as independent"
            ),
            "resamples": RESAMPLES,
            "seed": SEED,
            "interval": "95% percentile interval, reported per cell as ci95",
            "top3_agreement": {
                m: {name: round(v, 4) for name, v in h["top3"].items()} for m, h in read.items()
            },
            "reading": (
                f"These are {n_prompts}-prompt rates. A cell holds {per_cell} draws but only "
                f"{n_prompts} independent clusters, so the intervals are wide and neighbouring "
                "levels are not separated. top3_agreement is the share of resamples whose top "
                "three attacks are the same three as the frozen ordering's."
            ),
        },
    }
    doc["ordering_sha256"] = rollup_sha256(doc)
    return doc


def report(read: dict, doc: dict) -> None:
    pd.set_option("display.width", 200)
    for model, held in read.items():
        table = held["table"].copy()
        table["ci95"] = [f"[{held['ci95'][x][0]:.3f}, {held['ci95'][x][1]:.3f}]" for x in table.label]
        print(f"\n=== {model}: PER-ATTEMPT UNLOCK RATE, 8 LEVELS x 2 STATES ===\n")
        print(table[["state", "level", "n_rows", "n_unlocked", "rate", "ci95"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

        print(f"\n--- {model}: ORDERINGS, BEST FIRST ---")
        for name, order in held["orderings"].items():
            rates = keyed(table, name)
            print(f"\n  {name}  (top-3 agreement under the bootstrap: {held['top3'][name]:.2f})")
            for i, entry in enumerate(order, 1):
                print(f"    {i:>2}. {entry:<28} {rates[entry]:.3f}")
            broken = ties(rates)
            print(f"    ties the rule had to break: {broken if broken else 'none'}")

    print(f"\ntie-break rule: {doc['definitions']['tie_break']['rule']}")
    print(f"precision     : {doc['precision']['reading']}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="the 30-prompt factorial's evidence directory")
    parser.add_argument("--write", action="store_true", help=f"write {ARTIFACT.name}")
    args = parser.parse_args(argv)

    run = Path(args.run).resolve()
    read = read_run(run)
    doc = artifact(run, read)
    report(read, doc)

    if args.write:
        write_manifest(ARTIFACT, doc)
        print(f"\nwrote {ARTIFACT.relative_to(REPO)}")
    print(f"ordering_sha256 {doc['ordering_sha256']}")


if __name__ == "__main__":
    main(sys.argv[1:])
