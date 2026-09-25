"""Per-slot unlock rates from the pilot's base and layer_23 scored logs.

Read-only telemetry on already-collected pilot data. No frozen parameter is touched and
no confirmatory statistic is computed -- this is a descriptive read of whether the 13
prefill slots spread, which decides whether a per-slot design is worth building.

Each (arm, slot) cell here is 30 prompts x 1 draw = 30 observations.

Run:  python scripts/slot_spread.py --run <evidence>/pilot --slots-csv <path>
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from analysis.slots import against_comparator, slot_table, spread  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

PREFILLED, COMPOSED = "base", "layer_23"


def read_slots(run: Path) -> pd.DataFrame:
    """One row per generation of the pilot's two arms, slots kept apart."""
    scored = run.resolve() / "scored"
    # `pilot` was a real prompt set when these logs were written, and is baked into the
    # directories on disk; retiring the set did not rename them.
    logs = {condition: scored / condition / "pilot" for condition in (PREFILLED, COMPOSED)}

    rows = []
    for condition, d in logs.items():
        (path,) = sorted(d.glob("*.eval"))
        for s in read_eval_log(str(path)).samples:
            score = s.scores["strongreject"].value
            rows.append(
                {
                    "condition": condition,
                    "prompt_id": s.metadata["prompt_id"],
                    "slot": s.metadata["prefill_slot"],
                    "unlocked": score["unlocked"],
                    "malformed": score["malformed"],
                    "degenerate": score["degenerate"],
                    "leak": s.output.metadata["thinking_leak"],
                    "new_tokens": s.output.metadata["new_tokens"],
                    "stop": s.output.stop_reason,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--slots-csv", required=True)
    args = parser.parse_args(argv)

    f = read_slots(Path(args.run))
    f.to_csv(args.slots_csv, index=False)

    print("rows:", len(f))
    print("slots:", sorted(f.slot.unique()))
    print("prompts:", f.prompt_id.nunique())
    print()

    pd.set_option("display.width", 220)
    table = slot_table(f)

    print("PER-SLOT UNLOCK RATE (30 prompts x 1 draw per cell)\n")
    print(table.pivot(index="slot", columns="condition", values=["draws", "rate", "malformed"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    print("SPREAD OVER THE 13 PREFILL SLOTS (the unprefilled control excluded)\n")
    print(pd.DataFrame([spread(table, c) for c in (PREFILLED, COMPOSED)])
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()

    held = against_comparator(table, PREFILLED, COMPOSED)
    print("AGAINST THE COMPARATOR AN ATTACKER HOLDING THE CHECKPOINT ACTUALLY HAS\n")
    print(held.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n  slots beating it : {int((held.over_comparator > 0).sum())} of {len(held)}, "
          f"by at most {held.over_comparator.max():+.3f}")
    print(f"  slots worse      : {int((held.over_comparator < 0).sum())}, "
          f"worst {held.over_comparator.min():+.3f}")
    print(f"  corr(prefill-only rate, naive gain) = {held.prefill_only.corr(held.naive_gain):.3f}")
    print()

    # Slot ids look like "family:variant"; the static baseline and the control have no colon.
    prefilled = f[f.slot != "none"].copy()
    prefilled["family"] = prefilled.slot.astype(str).str.split(":").str[0]
    fam = (
        prefilled.groupby(["condition", "family"])
        .agg(n=("unlocked", "size"), rate=("unlocked", "mean"), malformed=("malformed", "mean"))
        .reset_index()
        .pivot(index="family", columns="condition", values=["n", "rate", "malformed"])
    )
    print("PER-FAMILY ROLL-UP (variants pooled; 60 obs per family per arm, 30 for the static baseline)\n")
    print(fam.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    print("PER-SLOT: prompts unlocked out of 30\n")
    print(prefilled.groupby(["condition", "slot"]).unlocked.sum().unstack(0)
          .to_string(float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
