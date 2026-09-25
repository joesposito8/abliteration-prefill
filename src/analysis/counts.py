"""One row per prompt, one cell per (model state, prefill level): unlocks, draws, malformed.

The level axis is alphabetical so that a first-index argmax breaks a tie the way the
frozen ordering does -- to the alphabetically earlier level.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from harness.dataset import CONTROL
from inspect_ai.scorer import SampleScore
from prefills import FAMILIES, PORTFOLIO, family_of
from prefills.families import STATIC_SLOT_ID
from study import draws

STATES = ("base", "abliterated")
LEVELS = tuple(sorted((CONTROL, STATIC_SLOT_ID, *FAMILIES)))

BASE, ABL = 0, 1
NONE = LEVELS.index(CONTROL)
PREFILL = [i for i, level in enumerate(LEVELS) if level != CONTROL]

LEVEL_DRAWS = {
    level: sum(draws(slot) for slot in (CONTROL, *PORTFOLIO) if family_of(slot) == level)
    for level in LEVELS
}


@dataclass(frozen=True)
class Counts:
    """Arrays of shape ``(prompts, 2 states, 8 levels)``; a family pools its two variants."""

    prompt_ids: np.ndarray
    u: np.ndarray
    n: np.ndarray
    malformed: np.ndarray

    @classmethod
    def from_cells(cls, cells: pd.DataFrame) -> "Counts":
        """``cells`` as ``cell_frame`` returns it: one row per (prompt, state, level), in
        the order the arrays hold."""
        shape = (cells.prompt_id.nunique(), len(STATES), len(LEVELS))
        counts = cls(
            prompt_ids=cells.prompt_id.unique(),
            u=cells.unlocked.to_numpy().reshape(shape),
            n=cells.draws.to_numpy().reshape(shape),
            malformed=cells.malformed.to_numpy().reshape(shape),
        )
        # A cell short of its draws would still print a plausible rate.
        expected = np.array([LEVEL_DRAWS[level] for level in LEVELS])
        if (counts.n != expected).any():
            short = np.argwhere(counts.n != expected)[:5].tolist()
            raise ValueError(f"cells not at the draw schedule (prompt, state, level): {short}")
        return counts

    @classmethod
    def from_scores(cls, scores: list[SampleScore]) -> "Counts":
        return cls.from_cells(cell_frame(scores))


def cell_frame(scores: list[SampleScore]) -> pd.DataFrame:
    """One row per (prompt, state, level): unlocks, draws and malformed rows.

    The long form of the same table ``Counts`` holds as arrays, and what gets published --
    the generations are withheld, so this aggregate is what a reader recomputes from.
    ``state`` and ``level`` are read off the metadata ``analysis.read`` derives, so the
    cell a row belongs to is decided once, where the logs are read.
    """
    held = pd.DataFrame(
        {
            "prompt_id": s.sample_metadata["prompt_id"],
            "state": s.sample_metadata["state"],
            "level": s.sample_metadata["level"],
            "unlocked": s.score.value["unlocked"],
            "malformed": s.score.value["malformed"],
        }
        for s in scores
    )
    cells = held.groupby(["prompt_id", "state", "level"]).agg(
        unlocked=("unlocked", "sum"), draws=("unlocked", "size"), malformed=("malformed", "sum")
    )
    index = pd.MultiIndex.from_product(
        [sorted(held.prompt_id.unique()), STATES, LEVELS], names=cells.index.names
    )
    missing = index.difference(cells.index)
    if not missing.empty:
        raise ValueError(f"cells with no rows: {missing[:5].tolist()}")
    return cells.reindex(index).reset_index()


def resolve(counts: Counts, state: int, levels: list[int]) -> Counts:
    """The corner where every malformed row in the named cells was an unlock."""
    u = counts.u.copy()
    u[:, state, levels] += counts.malformed[:, state, levels]
    return replace(counts, u=u)


def all_prompts(counts: Counts) -> np.ndarray:
    """The index array that makes a statistic its point estimate: one row, every prompt."""
    return np.arange(len(counts.prompt_ids))[None]
