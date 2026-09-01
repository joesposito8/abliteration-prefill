"""Per-slot unlock rates from the pilot's base and layer_23 scored logs.

Read-only telemetry on already-collected pilot data. No frozen parameter is touched and
no confirmatory statistic is computed -- this is a descriptive read of whether the 13
prefill slots spread, which decides whether a per-slot design is worth building.

Each (arm, slot) cell here is 30 prompts x 1 draw = 30 observations.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--slots-csv", required=True)
    args = parser.parse_args(argv)

    scored = Path(args.run).resolve() / "scored"
    # `pilot` was a real prompt set when these logs were written, and is baked into the
    # directories on disk; retiring the set did not rename them.
    logs = {
        "base": scored / "base" / "pilot",
        "layer_23": scored / "layer_23" / "pilot",
    }

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

    f = pd.DataFrame(rows)
    f.to_csv(args.slots_csv, index=False)

    print("rows:", len(f))
    print("slots:", sorted(f.slot.unique()))
    print("prompts:", f.prompt_id.nunique())
    print()

    pd.set_option("display.width", 220)

    # Per-slot unlock rate, both arms, side by side.
    tab = (
        f.groupby(["condition", "slot"])
        .agg(n=("unlocked", "size"), unlocked=("unlocked", "sum"),
             rate=("unlocked", "mean"), malformed=("malformed", "mean"),
             trunc=("stop", lambda s: (s == "max_tokens").mean()))
        .reset_index()
    )
    wide = tab.pivot(index="slot", columns="condition",
                     values=["n", "rate", "malformed"])
    print("PER-SLOT UNLOCK RATE (30 prompts x 1 draw per cell)\n")
    print(wide.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    # Spread summary over the 13 prefill slots only (drop the no-prefill control).
    control = [s for s in f.slot.unique() if "control" in str(s).lower() or str(s) == "none"]
    print("control-slot label(s) detected:", control)
    prefilled = f[~f.slot.isin(control)]
    print()
    for cond, held in prefilled.groupby("condition"):
        per = held.groupby("slot").unlocked.mean()
        print(f"{cond:9s} n_slots={per.size}  min={per.min():.3f}  p25={per.quantile(.25):.3f} "
              f" median={per.median():.3f}  p75={per.quantile(.75):.3f}  max={per.max():.3f} "
              f" spread={per.max() - per.min():.3f}  pooled={held.unlocked.mean():.3f}")
    print()

    # Family roll-up: slot ids look like "family:variant"; static baseline has no colon.
    prefilled = prefilled.copy()
    prefilled["family"] = prefilled.slot.astype(str).str.split(":").str[0]
    fam = (
        prefilled.groupby(["condition", "family"])
        .agg(n=("unlocked", "size"), rate=("unlocked", "mean"),
             malformed=("malformed", "mean"))
        .reset_index()
        .pivot(index="family", columns="condition", values=["n", "rate", "malformed"])
    )
    print("PER-FAMILY ROLL-UP (variants pooled; 60 obs per family per arm, 30 for the static baseline)\n")
    print(fam.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    # How many prompts a slot never unlocks -- the per-prompt zero question at r=1.
    print("PER-SLOT: prompts unlocked out of 30\n")
    z = (
        prefilled.groupby(["condition", "slot"]).unlocked.sum().unstack(0)
    )
    print(z.to_string(float_format=lambda v: f"{v:.0f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
