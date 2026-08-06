#!/usr/bin/env python3
"""Pick the primary layer from the graded sweep and commit the choice.

Prints the 37-row table and the tie-break trace by default and writes nothing, so the
operator review gate is the default path. ``--write`` commits the table and the manifest.

Every number the rule reads is a metric the scorer already computed, so the sweep is read
one log header per condition. Only ``--excerpts`` opens a sample.

Run:  python scripts/freeze_abliteration.py --run results/sweep --excerpts 2
      python scripts/freeze_abliteration.py --run results/sweep --write
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from abliteration.selection import (  # noqa: E402
    NEAR_TIE_PROMPTS,
    LayerReport,
    condition_id,
    near_tie_band,
    rank,
    select_primary,
    selectable,
    sweep_layers,
)
from generation.qwen import N_LAYERS  # noqa: E402
from grading.strongreject_grader import JUDGE_MODEL  # noqa: E402
from harness.conditions import PROVIDER  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from study.datasets import (  # noqa: E402
    ABLITERATION_MANIFEST_JSON,
    DATA_DIR,
    REFUSAL_DIRECTIONS_PT,
    VALIDATION_HARMFUL_CSV,
    load_validation_prompts,
)
from study.manifest import sha256_file, write_csv, write_manifest  # noqa: E402

from grade import finished_logs  # noqa: E402

TABLE_PATH = DATA_DIR / "layer_selection.csv"
SCORER = "strongreject"


def load_conditions(root: Path) -> dict:
    """The successful log header of every condition, keyed by condition id.

    A killed or failed attempt stays in the tree beside the log that replaced it, so the
    status filter is what makes one condition one entry.
    """
    logs = finished_logs(root)
    if not logs:
        raise SystemExit(f"no successful eval logs under {root}")

    by_condition: dict = {}
    for log in logs:
        condition = log.eval.model.removeprefix(f"{PROVIDER}/")
        if condition in by_condition:
            raise SystemExit(f"{condition}: two successful logs; the tree is ambiguous")
        by_condition[condition] = log
    return by_condition


def assert_complete(by_condition: dict, n_prompts: int) -> None:
    """Prove every condition ran every prompt before anything is ranked."""
    expected = [condition_id(layer) for layer in sweep_layers(N_LAYERS)]

    missing = [c for c in expected if c not in by_condition]
    if missing:
        raise SystemExit(f"missing conditions {missing}")
    unexpected = sorted(set(by_condition) - set(expected))
    if unexpected:
        raise SystemExit(f"unexpected conditions {unexpected}")

    for condition in expected:
        results = by_condition[condition].results
        # A log with nothing completed carries no results at all, and a run that lost
        # samples still reports success, so both are checked rather than the status.
        if results is None:
            raise SystemExit(f"{condition}: no results; nothing completed")
        if (results.completed_samples, results.total_samples) != (n_prompts, n_prompts):
            raise SystemExit(
                f"{condition}: {results.completed_samples} of {results.total_samples} "
                f"samples completed, expected {n_prompts} of {n_prompts}"
            )


def metric(log, key: str) -> float:
    """One condition's mean over every sample, for one score key."""
    return {score.name: score for score in log.results.scores}[key].metrics[
        "per_slot_all"
    ].value


def build_reports(by_condition: dict, n_prompts: int) -> list[LayerReport]:
    """The base row then every layer. The loop supplies the layer, so no id is parsed."""
    reports = []
    for layer in sweep_layers(N_LAYERS):
        log = by_condition[condition_id(layer)]
        breadth = metric(log, "unlocked")
        quality = metric(log, "aggregate")
        malformed = metric(log, "malformed")
        degenerate = metric(log, "degenerate")
        reports.append(
            LayerReport(
                condition=condition_id(layer),
                layer=layer,
                n_prompts=n_prompts,
                n_unlocked=round(breadth * n_prompts),
                breadth=breadth,
                quality=quality,
                # Every non-unlocked row scores exactly 0, so the all-prompts mean and the
                # unlocked-only mean differ by the breadth factor alone.
                quality_unlocked=quality / breadth if breadth else None,
                n_malformed=round(malformed * n_prompts),
                malformed_rate=malformed,
                n_degenerate=round(degenerate * n_prompts),
                degenerate_rate=degenerate,
            )
        )
    return reports


def build_table(reports, selection) -> pd.DataFrame:
    """One row per condition: the report's own fields, plus rank and the primary flag."""
    # A descriptive breadth-then-quality ranking. It need not agree with `primary`: the
    # tie-break picks the best-quality member of the near-tie band, which can sit below
    # rank 1 on raw breadth. The `primary` column is the authority.
    rank_of = {r.condition: i + 1 for i, r in enumerate(rank(selectable(reports)))}

    frame = pd.DataFrame([asdict(report) for report in reports])
    frame["layer"] = frame["layer"].astype("Int64")
    frame["rank"] = frame["condition"].map(rank_of).astype("Int64")
    frame["primary"] = (frame["condition"] == selection.condition).astype(int)
    return frame


def write_table(path: Path, frame: pd.DataFrame) -> str:
    """Write the table at the precision the committed file was frozen with."""
    return write_csv(path, frame, float_format="%.6f")


def build_manifest(selection, table: Path, table_sha256: str, directions: Path) -> dict:
    """The choice, and the two files it is only meaningful against.

    A layer index means nothing without the direction tensor it indexes into: a different
    extraction gives a different layer 22. Every per-layer number is a row of the table.
    """
    return {
        "primary": {
            "condition": selection.condition,
            "layer": selection.layer,
            "rule_path": list(selection.rule_path),
            "band": list(selection.band),
            "runner_up": selection.runner_up,
        },
        "directions": {"file": directions.name, "sha256": sha256_file(directions)},
        "table": {"file": table.name, "sha256": table_sha256},
    }


def print_report(table: pd.DataFrame, reports, selection, band) -> None:
    print(f"\nselection set : {VALIDATION_HARMFUL_CSV.name} ({reports[0].n_prompts} prompts)")
    print(f"judge         : {JUDGE_MODEL}, k=1\n")

    display = table.drop(columns=["n_prompts", "primary"])
    for column in ("layer", "rank"):  # pd.NA renders as <NA>, which na_rep does not catch
        display[column] = display[column].astype(object).where(display[column].notna(), "-")
    print(display.to_string(index=False, na_rep="-", float_format=lambda v: f"{v:.3f}"))

    print("\n  quality  = mean StrongREJECT aggregate over ALL prompts (refusal, malformed = 0)")
    print("  quality_unlocked = same over unlocked rows only — descriptive, not the tie-break")

    suspicious = [r for r in selectable(reports) if r.degenerate_rate > 0.1]
    if suspicious:
        print("\nHIGH DEGENERATE RATE (diagnostic — read the excerpts before confirming):")
        for r in suspicious:
            print(f"  {r.condition}  empty output {r.n_degenerate}/{r.n_prompts} "
                  f"({r.degenerate_rate:.1%})  breadth {r.breadth:.3f}")

    best = max(r.n_unlocked for r in band)
    print(f"\nNEAR-TIE BAND (within {NEAR_TIE_PROMPTS} unlocked prompts of {best})")
    print(f"  members ({len(band)}): " + "  ".join(
        f"{r.layer}({r.n_unlocked}, q={r.quality:.3f})" for r in band))

    print("\nTIE-BREAK TRACE")
    for rule in selection.rule_path:
        print(f"  {rule}")
    print(f"\nPRIMARY   {selection.condition}    runner-up {selection.runner_up}")
    print("\nT2 PRIOR (advisory; AdvBench extraction, n=20): max breadth L22 1.00/0.705, "
          "max quality L18 0.90/0.854, quality declines with depth")


def unlocked_excerpts(log) -> list[tuple[int, str]]:
    """(prompt_id, continuation) for every unlocked sample of one condition."""
    return sorted(
        (sample.metadata["prompt_id"], sample.output.completion)
        for sample in log.samples
        if sample.scores[SCORER].value["unlocked"] == 1
    )


def print_excerpts(by_condition: dict, band, count: int) -> None:
    """Unlocked continuations, spread across the prompt range so the sample is not always
    the same few prompts. Diagnostic only — never enters ranking, never leaves the terminal."""
    print("\nEXCERPTS (local terminal only; never committed or published)")
    for report in band:
        quality = report.quality_unlocked
        print(f"\n  {report.condition}  breadth {report.breadth:.3f}  "
              f"quality {'-' if quality is None else f'{quality:.3f}'}")
        excerpts = unlocked_excerpts(
            read_eval_log(by_condition[report.condition].location)
        )
        if not excerpts:
            print("    (nothing unlocked)")
            continue
        step = max(1, len(excerpts) // count)
        for index, text in excerpts[::step][:count]:
            snippet = " ".join(text.split())[:200]
            print(f"    [{index:>2}] {snippet}...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/sweep")
    parser.add_argument("--excerpts", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    n_prompts = len(load_validation_prompts())
    by_condition = load_conditions(Path(args.run) / "scored")
    assert_complete(by_condition, n_prompts)
    print(f"completeness ok: {len(by_condition)} conditions x {n_prompts} prompts")

    reports = build_reports(by_condition, n_prompts)
    band = near_tie_band(reports)
    selection = select_primary(reports)
    table = build_table(reports, selection)

    print_report(table, reports, selection, band)
    if args.excerpts:
        print_excerpts(by_condition, band, args.excerpts)

    if not args.write:
        print("\nnothing written. re-run with --write to commit.")
        return

    if not REFUSAL_DIRECTIONS_PT.exists():
        raise SystemExit(f"{REFUSAL_DIRECTIONS_PT} is missing; the layer index would name nothing")

    manifest = build_manifest(
        selection, TABLE_PATH, write_table(TABLE_PATH, table), REFUSAL_DIRECTIONS_PT
    )
    write_manifest(ABLITERATION_MANIFEST_JSON, manifest)
    print(f"\nwrote {TABLE_PATH}\nwrote {ABLITERATION_MANIFEST_JSON}")
    print(f"  primary: {selection.condition} via {' -> '.join(selection.rule_path)}")


if __name__ == "__main__":
    main()
