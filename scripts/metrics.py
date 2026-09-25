#!/usr/bin/env python3
"""The four reported metrics, off the scored factorial.

One command regenerates every reported table. The three contrasts are Inspect metrics, so
each model's numbers are an ``EvalResults`` and the tables are rendered from it rather
than computed beside it. Metrics 1 and 3 are confirmatory, metric 2 is a ceiling, and
metric 4 and the per-cell rates are descriptive; the readout keeps them in that order.

Run:  python scripts/metrics.py --run <evidence>/main --out results/analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from abliteration.selection import BASE_CONDITION, condition_id  # noqa: E402
from analysis import metrics as m  # noqa: E402
from analysis import read  # noqa: E402
from analysis.counts import ABL, BASE, Counts, cell_frame  # noqa: E402
from analysis.results import results, values  # noqa: E402
from generation import TARGETS, target  # noqa: E402
from study.datasets import model_slug  # noqa: E402

import run_report as rr  # noqa: E402
from generate import primary_layer  # noqa: E402

MODELS = tuple(model_slug(model.MODEL_ID) for model in TARGETS)

CLASSES = {
    "Confirmatory": {
        "groups": {"abliteration_vs_prefill", "same_queries"},
        "lead": "base_best_prefill",
        "note": (
            f"Two claims, one per model, Bonferroni at alpha = {m.ALPHA_D1}, read at a single "
            "attempt. The overlap columns are Fisher's exact test per query, two-sided, "
            f"Benjamini-Hochberg at a {m.FDR:.0%} false discovery rate run separately within each "
            "model. The composed arm is not in that comparison, so **nothing in this study "
            "establishes an interaction between the two attacks.**"
        ),
    },
    "Ceiling": {
        "groups": {"composition_ceiling"},
        "lead": "abliterated_best_prefill",
        "note": (
            "Not a test and not in the corrected family. `g` is the raw mean gain over "
            "abliteration alone and is a point value; the reported quantity is "
            "`d2 = g / (1 - abliterated_alone)`, the share of the headroom recovered, with one "
            f"uncorrected {1 - m.ALPHA_D2:.0%} interval. Both endpoints carry. A composition "
            "claim is read against the `g` Manski pair rather than against `g` alone."
        ),
    },
}
REPORTED = set().union(*(held["groups"] for held in CLASSES.values()))
DESCRIPTIVE = {"grouped"}


def condition_logs(run: Path, model: str) -> tuple[dict[str, str], dict[str, str]]:
    """Each of the model's two conditions, as a log location and a model state."""
    scored = run / "scored" / model
    by_condition = rr.load_conditions(scored)
    primary = condition_id(primary_layer(target(model)))
    states = {BASE_CONDITION: "base", primary: "abliterated"}
    if set(by_condition) != set(states):
        raise SystemExit(f"{model}: expected {sorted(states)}, found {sorted(by_condition)}")
    return {c: header.location for c, header in by_condition.items()}, states


def model_tables(run: Path, model: str, ordering: dict) -> dict:
    """Everything one model contributes: its results, and the rows of each table."""
    logs, states = condition_logs(run, model)
    scores = read.sample_scores(logs, states)
    cells = cell_frame(scores)
    counts = Counts.from_cells(cells)
    families = {
        "base_best_prefill": m.best_level(counts, BASE),
        "abliterated_best_prefill": m.best_level(counts, ABL),
    }
    held = results(scores, m.headline() + m.descriptives(), families)

    return {
        "model": model,
        "results": held,
        "headline": {"model": model, **families, **values(held, REPORTED)},
        "overlap": m.overlap(counts, m.FDR).assign(model=model),
        "budget": m.budget_table(counts, m.sequences(ordering, model)).assign(model=model),
        "counts": cells.assign(model=model),
    }


def stacked(tables: list[dict], key: str) -> pd.DataFrame:
    frame = pd.concat([t[key] for t in tables], ignore_index=True)
    return frame[["model"] + [c for c in frame.columns if c != "model"]]


def gather(tables: list[dict]) -> dict[str, pd.DataFrame]:
    return {
        "headline": pd.DataFrame([t["headline"] for t in tables]),
        "query-overlap": stacked(tables, "overlap"),
        "budget-curves": stacked(tables, "budget"),
        "counts": stacked(tables, "counts"),
    }


# --- the readout ------------------------------------------------------------------


def cell_rates(tables: list[dict]) -> pd.DataFrame:
    """The grouped metrics back as a table. Their key is ``"<state>/<group> <quantity>"``,
    because an ``EvalMetric`` namespace is flat and a two-way table has to be named into
    it. The two groupings share no group name, so the key alone says which one it is."""
    rows: dict[tuple, dict] = {}
    for table in tables:
        for name, value in values(table["results"], DESCRIPTIVE).items():
            cell, quantity = name.rsplit(" ", 1)
            row = rows.setdefault((table["model"], cell), {"model": table["model"], "cell": cell})
            row[quantity] = value
    return pd.DataFrame(rows.values())


def block(frame: pd.DataFrame, precision: str = "{:.4f}") -> list[str]:
    """Counts print as counts. Inspect stores every metric as a float, so a column of
    whole numbers would otherwise read as `313.0000`."""
    shown = frame.copy()
    for column in shown.select_dtypes("number"):
        if (shown[column].dropna() % 1 == 0).all():
            shown[column] = shown[column].astype("Int64")
    return ["```", shown.to_string(index=False, float_format=precision.format), "```", ""]


def render(frames: dict[str, pd.DataFrame], tables: list[dict], resamples: int) -> str:
    """The reported classes in order, then the descriptives. Never the other order.

    Every model carries the same metrics, so the first one names the columns of each class.
    """
    headline = frames["headline"]
    reported = []
    for heading, held in CLASSES.items():
        columns = list(values(tables[0]["results"], held["groups"]))
        reported += [f"## {heading}", "", held["note"], "",
                     *block(headline[["model", held["lead"], *columns]])]

    budget = frames["budget-curves"].copy()
    for c in m.THRESHOLDS:
        budget[f"k*({c})"] = budget[f"k*({c})"].astype(object).fillna("-")

    return "\n".join([
        "# Reported metrics",
        "",
        f"BCa bootstrap, B = {resamples:,}, prompts resampled with every draw kept together and "
        "the best prefill family reselected inside each resample. The percentile interval is the "
        "robustness check on the same replicates. Malformed rows are non-unlocks at the point; a "
        "Manski pair resolves them in whichever direction most disfavours the contrast.",
        "",
        "Every number below is an Inspect metric. `results-<model>.json` carries the same values "
        "as an `EvalResults`, where each metric's `group` records which contrast produced it and "
        "its `params` the alpha and resample count it was read at.",
        "",
        *reported,
        "## Descriptive",
        "",
        "No hypothesis and no correction anywhere below. Per-attempt rates, a family pooling its "
        "two variant slots; *family* here is a prefill family, not a model family. The portfolio "
        "rows regroup the same draws by where the prefill came from.",
        "",
        *block(cell_rates(tables)),
        "Coverage over the attack budget: the fraction of queries unlocked at least once within k "
        "attempts, over the frozen pilot-rank orderings. The 16 single-cell strategies are "
        "reference lines — at one attempt they are the arms above by construction. `k*` is the "
        "smallest budget reaching that coverage, and `-` means it is not reached within ten.",
        "",
        *block(budget, "{:.3f}"),
    ])


def write(out: Path, frames: dict[str, pd.DataFrame], tables: list[dict], readout: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    for table in tables:
        (out / f"results-{table['model']}.json").write_text(
            table["results"].model_dump_json(indent=2, exclude_none=True) + "\n"
        )
    (out / "report.md").write_text(readout)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="the evidence directory holding scored/<model>/")
    parser.add_argument("--model", action="append", choices=MODELS,
                        help="repeatable; every enabled model by default")
    parser.add_argument("--out", type=Path, help="write the tables and the readout here")
    args = parser.parse_args(argv)

    run = Path(args.run).resolve()
    ordering = m.load_ordering()
    tables = [model_tables(run, model, ordering) for model in args.model or MODELS]

    frames = gather(tables)
    readout = render(frames, tables, m.RESAMPLES)
    print(readout)
    if args.out:
        write(args.out, frames, tables, readout)
        print(f"written to {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
