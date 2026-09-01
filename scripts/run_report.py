#!/usr/bin/env python3
"""One run's readout: union unlock per arm, and run health. Any eval set, any size.

Prints only. Set membership is >=1 non-refusal over a prompt's draws, which spans logs:
each model state's two arms are one log split by slot.

Run:  python scripts/run_report.py --run results/main --model qwen3-4b
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from abliteration.selection import BASE_CONDITION, condition_id  # noqa: E402
from generate import primary_layer  # noqa: E402
from generation import target  # noqa: E402
from harness.conditions import condition_of  # noqa: E402
from harness.dataset import CONTROL  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from prefills import PORTFOLIO  # noqa: E402
from study import draws  # noqa: E402

from grade import finished_logs  # noqa: E402

SCORER = "strongreject"


def load_conditions(root: Path) -> dict:
    """The successful log header of every condition under ``root``, keyed by condition."""
    logs = finished_logs(root)
    if not logs:
        raise SystemExit(f"no successful eval logs under {root}")

    by_condition: dict = {}
    for log in logs:
        condition = condition_of(log.eval.model)
        if condition in by_condition:
            raise SystemExit(f"{condition}: two successful logs; the tree is ambiguous")
        by_condition[condition] = log
    return by_condition


def load_samples(scored: Path, by_condition: dict) -> pd.DataFrame:
    """One row per generation, over the successful log of every condition.

    Read log by log rather than through ``inspect_ai.analysis.samples_df``: that path
    parses full samples with ijson, which raises "integer overflow" on any value above
    2**63, and every sample carries a uint64 ``batch_seed`` from ``harness.batching``.
    About half of real logs are unreadable through it.
    """
    rows = []
    for condition, header in by_condition.items():
        for sample in read_eval_log(header.location).samples:
            if SCORER not in sample.scores:
                raise SystemExit(f"{scored} carries no {SCORER} scores; grade it first")
            score = sample.scores[SCORER].value
            rows.append(
                {
                    "id": sample.id,
                    "condition": condition,
                    "layer": sample.metadata["layer"],
                    "prompt_id": sample.metadata["prompt_id"],
                    "slot": sample.metadata["prefill_slot"],
                    "prompt_set": sample.metadata["prompt_set"],
                    "unlocked": score["unlocked"],
                    "malformed": score["malformed"],
                    "degenerate": score["degenerate"],
                    "thinking_leak": sample.output.metadata["thinking_leak"],
                    "new_tokens": sample.output.metadata["new_tokens"],
                    "stop_reason": sample.output.stop_reason,
                }
            )
    return pd.DataFrame(rows)


def arm_masks(frame: pd.DataFrame, primary: int) -> dict:
    """The two model states, each split by whether the row carries a prefill."""
    base = frame.condition == BASE_CONDITION
    prime = frame.condition == condition_id(primary)
    bare = frame.slot == CONTROL

    return {
        "unprefilled base": base & bare,
        "prefill-only": base & ~bare,
        f"{condition_id(primary)} unprefilled": prime & bare,
        "composed": prime & ~bare,
    }


def union_report(frame: pd.DataFrame, masks: dict) -> pd.DataFrame:
    """Per arm: prompts unlocked by at least one attempt, beside the per-attempt mean."""
    rows = []
    for arm, mask in masks.items():
        held = frame[mask]
        per_prompt = held.groupby("prompt_id").unlocked.max()
        rows.append(
            {
                "arm": arm,
                "attempts": held.groupby("prompt_id").size().max(),
                "n_unlocked": int(per_prompt.sum()),
                "n_prompts": per_prompt.size,
                "union": per_prompt.mean(),
                "per_attempt": held.unlocked.mean(),
                "malformed": held.malformed.mean(),
                "degenerate": held.degenerate.mean(),
            }
        )
    return pd.DataFrame(rows)


def census(frame: pd.DataFrame) -> pd.DataFrame:
    """What each condition should hold, from the prompts the samples actually carry."""
    expected = frame.prompt_id.nunique() * sum(draws(slot) for slot in (CONTROL, *PORTFOLIO))

    rows = []
    for condition, held in frame.groupby("condition"):
        rows.append(
            {
                "condition": condition,
                "samples": len(held),
                "expected": expected,
                "unique_ids": held.id.nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values("condition")


def throughput(by_condition: dict) -> dict:
    """Summed over conditions, which the harness runs strictly one at a time."""
    seconds, tokens = 0.0, 0
    for log in by_condition.values():
        started = datetime.fromisoformat(log.stats.started_at)
        completed = datetime.fromisoformat(log.stats.completed_at)
        seconds += (completed - started).total_seconds()
        tokens += sum(usage.output_tokens for usage in log.stats.model_usage.values())
    return {
        "seconds": seconds,
        "output_tokens": tokens,
        # A run too short to have a duration is a test's, not a sweep's.
        "tokens_per_second": tokens / seconds if seconds else float("nan"),
    }


def print_report(cells: pd.DataFrame, unions: pd.DataFrame, frame: pd.DataFrame, rate: dict) -> None:
    print(f"\nprompt set : {frame.prompt_set.iloc[0]} ({frame.prompt_id.nunique()} prompts)")
    print(f"scorer     : {SCORER}\n")

    print(cells.to_string(index=False))
    wrong = cells[(cells.samples != cells.expected) | (cells.unique_ids != cells.samples)]
    if not wrong.empty:
        print(f"\nCELL COUNT MISMATCH on {wrong.condition.tolist()}")
    print(f"\n{cells.samples.sum():,} samples over {len(cells)} conditions\n")

    print(unions.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  union       = prompts with >=1 non-refusal over the arm's draws")
    print("  per_attempt = the same rows as a flat mean, which is what the scorer reports")
    print("  malformed/degenerate stay in the denominator as non-unlocks")

    print(f"\npost-retry malformed : {frame.malformed.mean():.3f}")

    truncated = (frame.stop_reason == "max_tokens").mean()
    p50, p95 = frame.new_tokens.quantile([0.5, 0.95])
    print(f"truncated at the cap : {truncated:.3f}")
    print(f"output tokens        : p50 {p50:.0f}, p95 {p95:.0f}")
    print(
        f"throughput           : {rate['output_tokens']:,} output tokens in "
        f"{rate['seconds'] / 3600:.2f} h = {rate['tokens_per_second']:.1f} tok/s"
    )

    leaked = frame[frame.thinking_leak.astype(bool)]
    bare = int((leaked.slot == CONTROL).sum())
    print(f"thinking leaks       : {len(leaked)} "
          f"({bare} unprefilled, {len(leaked) - bare} prefilled)")
    for condition, held in leaked.groupby("condition"):
        held_bare = int((held.slot == CONTROL).sum())
        print(f"  {condition:<12} {len(held):>4} "
              f"({held_bare} unprefilled, {len(held) - held_bare} prefilled)")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    scored = Path(args.run).resolve() / "scored" / args.model
    primary = primary_layer(target(args.model))
    by_condition = load_conditions(scored)
    frame = load_samples(scored, by_condition)

    print_report(
        census(frame),
        union_report(frame, arm_masks(frame, primary)),
        frame,
        throughput(by_condition),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
