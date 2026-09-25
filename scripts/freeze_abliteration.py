#!/usr/bin/env python3
"""Pick the primary layer from the graded sweep and commit the choice.

Prints the per-layer table and the tie-break trace by default and writes nothing, so the
operator review gate is the default path. ``--write`` commits the table and the manifest.

Every number the rule reads is a mean over the sweep's own sample rows, so each condition
is read whole. That is affordable here and nowhere else: a selection sweep is one draw per
validation prompt, against the 50,080 samples an evaluation condition holds.

The sweep is read under ``<run>/scored/<model-slug>``: a condition id carries no model, so
a root holding two targets would key both their ``layer_22`` logs to one entry.

Run:  python scripts/freeze_abliteration.py --run results/sweep --model qwen3-4b --excerpts 2
      python scripts/freeze_abliteration.py --run results/sweep --model qwen3-4b --write
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
from generation import target  # noqa: E402
from grading.strongreject_grader import JUDGE_MODEL  # noqa: E402
from harness.conditions import condition_of  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from study.datasets import (  # noqa: E402
    VALIDATION_HARMFUL_CSV,
    load_validation_prompts,
    model_dir,
)
from study.manifest import (  # noqa: E402
    render_csv,
    sha256_bytes,
    sha256_file,
    write_bytes,
    write_manifest,
)

from grade import finished_logs  # noqa: E402

SCORER = "strongreject"


def load_conditions(root: Path) -> dict:
    """Every condition's successful log, whole, keyed by condition id.

    A killed or failed attempt stays in the tree beside the log that replaced it, so the
    status filter over the headers is what makes one condition one entry. The log behind
    each surviving header is then read in full, because the rule means over its samples.
    """
    logs = finished_logs(root)
    if not logs:
        raise SystemExit(f"no successful eval logs under {root}")

    by_condition: dict = {}
    for log in logs:
        condition = condition_of(log.eval.model)
        if condition in by_condition:
            raise SystemExit(f"{condition}: two successful logs; the tree is ambiguous")
        by_condition[condition] = read_eval_log(log.location)
    return by_condition


def assert_complete(by_condition: dict, n_prompts: int, n_layers: int) -> None:
    """Prove every condition ran every prompt before anything is ranked."""
    expected = [condition_id(layer) for layer in sweep_layers(n_layers)]

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
    values = [sample.scores[SCORER].value[key] for sample in log.samples]
    return sum(values) / len(values)


def build_reports(by_condition: dict, n_prompts: int, n_layers: int) -> list[LayerReport]:
    """The base row then every layer. The loop supplies the layer, so no id is parsed."""
    reports = []
    for layer in sweep_layers(n_layers):
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


def build_table(reports, selection, model_id: str) -> pd.DataFrame:
    """One row per condition: the model, the report's own fields, rank and the primary flag."""
    # A descriptive breadth-then-quality ranking. It need not agree with `primary`: the
    # tie-break picks the best-quality member of the near-tie band, which can sit below
    # rank 1 on raw breadth. The `primary` column is the authority.
    rank_of = {r.condition: i + 1 for i, r in enumerate(rank(selectable(reports)))}

    frame = pd.DataFrame([asdict(report) for report in reports])
    frame.insert(0, "model", model_id)
    frame["layer"] = frame["layer"].astype("Int64")
    frame["rank"] = frame["condition"].map(rank_of).astype("Int64")
    frame["primary"] = (frame["condition"] == selection.condition).astype(int)
    return frame


def render_table(frame: pd.DataFrame) -> bytes:
    """Serialize the table at the precision the committed file was frozen with."""
    return render_csv(frame, float_format="%.6f")


def build_manifest(selection, table: Path, table_sha256: str, directions: Path, module) -> dict:
    """The choice, and the two files it is only meaningful against.

    A layer index means nothing without the direction tensor it indexes into: a different
    extraction gives a different layer 22. Every per-layer number is a row of the table.
    """
    return {
        "model": {
            "id": module.MODEL_ID,
            "revision": module.REVISION,
            "n_layers": module.N_LAYERS,
        },
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
        excerpts = unlocked_excerpts(by_condition[report.condition])
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--excerpts", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    module = target(args.model)
    model_dir_ = model_dir(module.MODEL_ID)
    table_path = model_dir_ / "layer_selection.csv"
    manifest_json = model_dir_ / "abliteration_manifest.json"
    directions_pt = model_dir_ / "refusal_directions.pt"

    n_prompts = len(load_validation_prompts())
    by_condition = load_conditions(Path(args.run) / "scored" / args.model)
    assert_complete(by_condition, n_prompts, module.N_LAYERS)
    print(f"completeness ok: {len(by_condition)} conditions x {n_prompts} prompts")

    reports = build_reports(by_condition, n_prompts, module.N_LAYERS)
    band = near_tie_band(reports)
    selection = select_primary(reports)
    table = build_table(reports, selection, module.MODEL_ID)

    print_report(table, reports, selection, band)
    if args.excerpts:
        print_excerpts(by_condition, band, args.excerpts)

    if not args.write:
        print("\nnothing written. re-run with --write to commit.")
        return

    if not directions_pt.exists():
        raise SystemExit(f"{directions_pt} is missing; the layer index would name nothing")

    payload = render_table(table)
    table_sha256 = sha256_bytes(payload)
    write_bytes(table_path, payload)
    manifest = build_manifest(selection, table_path, table_sha256, directions_pt, module)
    write_manifest(manifest_json, manifest)
    print(f"\nwrote {table_path}\nwrote {manifest_json}")
    print(f"  primary: {selection.condition} via {' -> '.join(selection.rule_path)}")


if __name__ == "__main__":
    main()
