#!/usr/bin/env python3
"""How much of the per-query difference between the two attacks is sampling noise.

The third metric compares abliteration alone against the base model's best prefill on one
query at a time. The spread of that difference across queries has two sources: the draws
are finite, and the queries genuinely differ. Only the first shrinks when draws are added,
so the split is what says whether more draws would buy resolution or only cost money.

Diagnostic telemetry on already-collected generations. No frozen parameter is touched and
nothing is written.

Run:  python scripts/variance_decomposition.py --run <evidence>/strongreject-30-r2 --model qwen3-4b
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from abliteration.selection import BASE_CONDITION, condition_id  # noqa: E402
from generation import target  # noqa: E402
from harness.dataset import CONTROL  # noqa: E402
from prefills import PORTFOLIO, family_of  # noqa: E402
from study import SEED  # noqa: E402

import run_report as rr  # noqa: E402
from generate import primary_layer  # noqa: E402

RESAMPLES = 10_000
AS_BUILT = 20  # the draw schedule's family-level unit, what the projection targets


@dataclass(frozen=True)
class Arms:
    """Per-query rates and the sampling variance of each, one row per query."""

    a: np.ndarray
    p: np.ndarray
    within_a: np.ndarray
    within_p: np.ndarray
    n_a: int
    n_p: int
    family: str


def cell(frame: pd.DataFrame, condition: str, slots: list[str]) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Unlocks and draws per query in each slot, as two queries-by-slots arrays.

    ``slots`` is the portfolio's list rather than the run's, so a slot missing from the
    whole condition is caught here instead of quietly halving the arm's draws.

    Draws come from the data rather than from ``study.draws``: the collected pilot ran at
    a flat 10 per slot and the schedule has since moved, so a script pinned to either
    count is wrong against the other.
    """
    held = frame[(frame.condition == condition) & frame.slot.isin(slots)]
    u = held.pivot_table(index="prompt_id", columns="slot", values="unlocked", aggfunc="sum")
    n = held.pivot_table(index="prompt_id", columns="slot", values="unlocked", aggfunc="size")
    if set(u.columns) != set(slots) or u.isna().to_numpy().any():
        raise SystemExit(f"{condition}: expected every query in {slots}, found {sorted(u.columns)}")
    n = n[slots].to_numpy()
    if (n != n[0]).any():
        raise SystemExit(f"{condition}: draws differ between queries:\n{n}")
    return u[slots].to_numpy(), n, u.index


def sampling_variance(u: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Unbiased estimate of the sampling variance of the pooled rate, per query.

    Estimated slot by slot and then weighted, never on the pooled count: a family's two
    variants may differ in rate, so their sum is a mixture rather than one binomial, and
    treating it as one overstates this term by the square of the difference between them.
    """
    total = n.sum(axis=1, keepdims=True)
    per_slot = u * (n - u) / (n**2 * (n - 1))
    return ((n / total) ** 2 * per_slot).sum(axis=1)


def best_family(frame: pd.DataFrame) -> str:
    """The prefill family the metric reads: the highest unlock rate on the base arm, with
    ties going to the earlier name.

    The metric defines this as the most unlocks summed over queries, which is the same
    ordering only while every family carries the same draws. The as-built schedule does
    give each 20, but the collected pilot does not -- there the static baseline holds one
    slot against a family's two -- so a sum would rank it on its draw count.""" 
    held = frame[(frame.condition == BASE_CONDITION) & (frame.slot != CONTROL)]
    rates = held.groupby(held.slot.map(family_of)).unlocked.mean()
    return sorted(rates.index, key=lambda f: (-rates[f], f))[0]


def read_arms(run: Path, model: str) -> Arms:
    scored = run / "scored" / model
    frame = rr.load_samples(scored, rr.load_conditions(scored))
    primary = condition_id(primary_layer(target(model)))

    family = best_family(frame)
    slots = [s for s in PORTFOLIO if family_of(s) == family]
    a_u, a_n, a_ids = cell(frame, primary, [CONTROL])
    p_u, p_n, p_ids = cell(frame, BASE_CONDITION, slots)
    if not a_ids.equals(p_ids):
        raise SystemExit("the two arms cover different queries; they cannot be paired")

    return Arms(
        a=a_u.sum(axis=1) / a_n.sum(axis=1),
        p=p_u.sum(axis=1) / p_n.sum(axis=1),
        within_a=sampling_variance(a_u, a_n),
        within_p=sampling_variance(p_u, p_n),
        n_a=int(a_n[0].sum()),
        n_p=int(p_n[0].sum()),
        family=family,
    )


def decompose(arms: Arms, idx: np.ndarray) -> dict:
    """Method of moments: the observed spread less the sampling term it must contain.

    Reported unclipped. Under queries that genuinely do not differ the estimator is
    centred on zero, so clipping would push it up and hide the sign of a near-zero read.
    """
    a, p = arms.a[idx], arms.p[idx]
    within_a, within_p = arms.within_a[idx].mean(), arms.within_p[idx].mean()
    total = (a - p).var(ddof=1)

    # Var(a - p) = Var(alpha) + Var(pi) - 2 Cov, so the covariance falls out of the three
    # heterogeneities. Query difficulty is shared, so it is positive and suppresses the
    # paired spread -- without it a small paired number cannot be told from alike queries.
    hetero_a = a.var(ddof=1) - within_a
    hetero_p = p.var(ddof=1) - within_p
    hetero = total - within_a - within_p

    return {
        "total": total,
        "within": within_a + within_p,
        "hetero": hetero,
        "within_a": within_a,
        "within_p": within_p,
        "hetero_a": hetero_a,
        "hetero_p": hetero_p,
        "cov": (hetero_a + hetero_p - hetero) / 2,
        "noise_share": (within_a + within_p) / total,
    }


def project(stats: dict, arms: Arms, n: int) -> dict:
    """The same query-to-query spread had both arms carried ``n`` draws.

    Only the sampling term moves. Heterogeneity is a property of the queries, so the
    total is not rescaled -- the per-draw quantity is recovered and divided again.
    """
    within = stats["within_a"] * arms.n_a / n + stats["within_p"] * arms.n_p / n
    total = stats["hetero"] + within
    return {"within": within, "total": total, "noise_share": within / total}


def interval(arms: Arms, rng: np.random.Generator) -> pd.DataFrame:
    """Cluster bootstrap over queries: draws sharing a query are not independent."""
    keys = ("total", "within", "hetero", "hetero_a", "hetero_p", "cov", "noise_share")
    draws = rng.integers(0, len(arms.a), size=(RESAMPLES, len(arms.a)))
    replicates = pd.DataFrame([decompose(arms, idx) for idx in draws])[list(keys)]
    lo, hi = np.percentile(replicates.to_numpy(), [2.5, 97.5], axis=0)
    return pd.DataFrame({"term": keys, "lo": lo, "hi": hi}).set_index("term")


def report(model: str, arms: Arms, stats: dict, ci: pd.DataFrame, projected: dict) -> None:
    print(f"\n=== {model}: WHERE THE PER-QUERY DIFFERENCE'S SPREAD COMES FROM ===\n")
    print(f"  queries        : {len(arms.a)}")
    print(f"  arm A          : abliteration alone, {arms.n_a} draws, mean rate {arms.a.mean():.3f}")
    print(f"  arm P          : base + {arms.family}, {arms.n_p} draws, mean rate {arms.p.mean():.3f}")

    rows = [
        ("Var(a - p), observed", stats["total"], f"[{ci.lo['total']:.5f}, {ci.hi['total']:.5f}]"),
        ("  within-query sampling", stats["within"], f"[{ci.lo['within']:.5f}, {ci.hi['within']:.5f}]"),
        ("  query-to-query", stats["hetero"], f"[{ci.lo['hetero']:.5f}, {ci.hi['hetero']:.5f}]"),
        ("noise share of the total", stats["noise_share"], f"[{ci.lo['noise_share']:.5f}, {ci.hi['noise_share']:.5f}]"),
    ]
    print("\n  " + pd.DataFrame(rows, columns=["term", "value", "95% interval"])
          .to_string(index=False, float_format=lambda v: f"{v:.5f}").replace("\n", "\n  "))

    split = [
        ("Var(alpha), abliteration", stats["hetero_a"], f"[{ci.lo['hetero_a']:.5f}, {ci.hi['hetero_a']:.5f}]"),
        ("Var(pi), best prefill", stats["hetero_p"], f"[{ci.lo['hetero_p']:.5f}, {ci.hi['hetero_p']:.5f}]"),
        ("Cov(alpha, pi)", stats["cov"], f"[{ci.lo['cov']:.5f}, {ci.hi['cov']:.5f}]"),
    ]
    print("\n  QUERY-TO-QUERY SPREAD OF EACH ARM, AND WHAT THEY SHARE\n")
    print("  " + pd.DataFrame(split, columns=["term", "value", "95% interval"])
          .to_string(index=False, float_format=lambda v: f"{v:.5f}").replace("\n", "\n  "))

    print(f"\n  AT THE AS-BUILT {AS_BUILT} DRAWS ON BOTH ARMS\n")
    print(f"    within-query sampling : {stats['within']:.5f} -> {projected['within']:.5f}")
    print(f"    total                 : {stats['total']:.5f} -> {projected['total']:.5f}")
    print(f"    noise share           : {stats['noise_share']:.3f} -> {projected['noise_share']:.3f}")

    print("\n  Heterogeneity is reported unclipped; a negative read is an estimate centred")
    print("  on zero, not a measurement of less than none.")
    print("  Both arms saturate, and a latent rate anywhere above about 0.9 lands on the")
    print("  same one or two counts, so this is a lower bound on the query-to-query part.")
    print(f"  {len(arms.a)} queries is ~{len(arms.a) - 1} degrees of freedom: directional, not a result.")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args(argv)

    arms = read_arms(Path(args.run).resolve(), args.model)
    stats = decompose(arms, np.arange(len(arms.a)))
    report(
        args.model,
        arms,
        stats,
        interval(arms, np.random.default_rng(SEED)),
        project(stats, arms, AS_BUILT),
    )


if __name__ == "__main__":
    main(sys.argv[1:])
