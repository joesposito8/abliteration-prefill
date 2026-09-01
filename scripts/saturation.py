"""Order-free coverage@k per arm, plus how much margin each arm's saturation rests on.

Attempt ORDER is not a frozen study parameter, so "the first k attempts" is not a
well-defined quantity. Coverage@k is the expected union over a uniformly random subset
of k of the arm's attempts: for a prompt with u unlocks out of n attempts,
P(at least one unlock in k draws) = 1 - C(n-u, k) / C(n, k).
"""
import argparse
import sys
from math import comb
from pathlib import Path

import pandas as pd
import run_report as pr


def coverage(u, n, k):
    if k >= n:  return 1.0 if u else 0.0
    if u == 0:  return 0.0
    if k > n - u: return 1.0
    return 1 - comb(n - u, k) / comb(n, k)

KS = [1, 2, 3, 4, 6, 8, 10, 13]


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)

    root = Path(args.run).resolve() / "scored"
    frame = pr.load_samples(root, pr.load_conditions(root))
    masks = pr.arm_masks(frame, 23)

    rows, margins = [], []
    for arm, mask in masks.items():
        held = frame[mask]
        per = held.groupby("prompt_id").unlocked.agg(["sum", "count"])
        n = int(per["count"].max())
        row = {"arm": arm, "n": n}
        for k in KS:
            if k <= n:
                row[f"k={k}"] = sum(coverage(int(u), int(c), k) for u, c in per.values) / len(per)
        rows.append(row)
        margins.append({
            "arm": arm,
            "attempts": n,
            "min_unlocks": int(per["sum"].min()),
            "p25": per["sum"].quantile(.25),
            "median": per["sum"].median(),
            "max": int(per["sum"].max()),
            "prompts_on_1": int((per["sum"] == 1).sum()),
            "prompts_on_<=2": int((per["sum"] <= 2).sum()),
            "prompts_on_0": int((per["sum"] == 0).sum()),
        })

    pd.set_option("display.width", 200)
    print("ORDER-FREE COVERAGE@k  (expected union over a random k-subset of the arm's attempts)\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
    print("\n\nHOW MUCH MARGIN THE SATURATION RESTS ON  (unlocks per prompt, out of `attempts`)\n")
    print(pd.DataFrame(margins).to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main(sys.argv[1:])
