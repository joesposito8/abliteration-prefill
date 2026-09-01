"""The rule that picks a model's batch width, and the measurements it reads.

Fixed within a model, since batched sampling is not width-invariant; measured per
model, since depth and KV footprint differ. Pure, like ``abliteration.selection``:
the sweep scripts measure, this decides, and a test re-runs it against the curve a
committed number was chosen from.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WidthReport:
    """One measured width. ``peak_gb`` is what the allocator reserved rather than what
    the tensors held: an OOM is against the reservation."""

    width: int
    prompts_per_second: float
    peak_gb: float


@dataclass(frozen=True)
class WidthChoice:
    width: int
    fastest: int  # the quickest that fit, which under_fastest is the shortfall against
    under_fastest: float
    rule_path: tuple[str, ...]
    rejected: tuple[int, ...]  # measured, but over the ceiling


def select_width(
    reports: Sequence[WidthReport], *, ceiling_gb: float, tolerance: float
) -> WidthChoice:
    """Fastest that fits, then down while the loss stays small.

    Smaller is preferred at equal speed because a failed batch re-runs that many
    samples, and the in-flight limit is twice the width.
    """
    eligible = [r for r in reports if r.peak_gb <= ceiling_gb]
    if not eligible:
        raise ValueError(f"every width exceeded {ceiling_gb} GB")

    fastest = max(eligible, key=lambda r: r.prompts_per_second)
    floor = fastest.prompts_per_second * (1 - tolerance)

    chosen, rule_path = fastest, ("fastest_within_ceiling",)
    smaller = sorted(
        (r for r in eligible if r.width < fastest.width),
        key=lambda r: r.width,
        reverse=True,
    )
    for candidate in smaller:
        if candidate.prompts_per_second < floor:
            break
        chosen, rule_path = candidate, ("fastest_within_ceiling", "within_tolerance")

    return WidthChoice(
        width=chosen.width,
        fastest=fastest.width,
        under_fastest=1 - chosen.prompts_per_second / fastest.prompts_per_second,
        rule_path=rule_path,
        rejected=tuple(r.width for r in reports if r.peak_gb > ceiling_gb),
    )
