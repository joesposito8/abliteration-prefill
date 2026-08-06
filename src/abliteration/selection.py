"""A-priori primary-layer selection on the disjoint validation set.

Breadth, then quality within one standard error of it, then lowest layer index.

Pure: no torch, no OpenAI, no file access.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

NEAR_TIE_PROMPTS = 3  # band width, ~1 binomial SE on breadth at n=72

BASE_CONDITION = "base"


def condition_id(layer: int | None) -> str:
    return BASE_CONDITION if layer is None else f"layer_{layer:02d}"


@dataclass(frozen=True)
class LayerReport:
    condition: str
    layer: int | None
    n_prompts: int
    n_unlocked: int
    breadth: float
    quality: float  # mean aggregate over all prompts — the tie-break
    n_malformed: int
    malformed_rate: float
    n_degenerate: int
    degenerate_rate: float
    quality_unlocked: float | None  # descriptive only, not the tie-break


@dataclass(frozen=True)
class Selection:
    condition: str
    layer: int
    rule_path: tuple[str, ...]
    band: tuple[str, ...]  # the near-tie band the primary was chosen from
    runner_up: str | None


def rank(reports: Sequence[LayerReport]) -> list[LayerReport]:
    """Breadth, then quality, then layer index — the order the tie-break walks."""
    return sorted(reports, key=lambda r: (-r.n_unlocked, -r.quality, r.layer))


def selectable(reports: Sequence[LayerReport]) -> list[LayerReport]:
    """Every layer competes; the base row is a reference and never selectable."""
    return [r for r in reports if r.layer is not None]


def near_tie_band(reports: Sequence[LayerReport]) -> list[LayerReport]:
    """Layers whose breadth is indistinguishable from the best at this sample size."""
    candidates = selectable(reports)
    if not candidates:
        raise ValueError("no layer to select from")
    best = max(r.n_unlocked for r in candidates)
    return rank([r for r in candidates if r.n_unlocked >= best - NEAR_TIE_PROMPTS])


def select_primary(reports: Sequence[LayerReport]) -> Selection:
    """Breadth, then quality within the band, then lowest layer index."""
    band = near_tie_band(reports)
    rule_path = ["breadth"]

    remaining = band
    if len(remaining) > 1:
        best_quality = max(r.quality for r in remaining)
        narrowed = [r for r in remaining if r.quality == best_quality]
        if len(narrowed) < len(remaining):
            rule_path.append("quality")
        remaining = narrowed
    if len(remaining) > 1:
        rule_path.append("layer_index")

    primary = min(remaining, key=lambda r: r.layer)
    runner_up = next(
        (r.condition for r in rank(selectable(reports)) if r.condition != primary.condition),
        None,
    )
    return Selection(
        condition=primary.condition,
        layer=primary.layer,
        rule_path=tuple(rule_path),
        band=tuple(r.condition for r in band),
        runner_up=runner_up,
    )
