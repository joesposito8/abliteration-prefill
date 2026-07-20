"""Contrast and evaluation prompt sets for abliteration.

Sampling is seeded; the sampled subsets are vendored under ``data/`` and loaded at
runtime (build path lives in a separate one-time script). This module stays offline so
the sampling/dedup logic is unit-testable.
"""

from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXTRACTION_HARMFUL_CSV = _DATA_DIR / "abliteration_extraction_harmful.csv"
EXTRACTION_HARMLESS_CSV = _DATA_DIR / "abliteration_extraction_harmless.csv"
EVAL_HARMFUL_CSV = _DATA_DIR / "abliteration_eval_harmful.csv"

N_HARMFUL = 256
N_HARMLESS = 256
N_EVAL = 20
SAMPLE_SEED = 20260720
NEAR_DUP_RATIO = 0.9  # difflib char-ratio at/above which two prompts are near-duplicates


def normalize(text: str) -> str:
    """Lowercase, drop quotes, non-alphanumerics to spaces, collapse whitespace."""
    text = text.strip().strip("\"'").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _near_any(cand_norm: str, exclude_norm: list[str], ratio: float) -> bool:
    for other in exclude_norm:
        if cand_norm == other:
            return True
        if SequenceMatcher(None, cand_norm, other).ratio() >= ratio:
            return True
    return False


def is_near_dup(candidate: str, exclude: list[str], *, ratio: float = NEAR_DUP_RATIO) -> bool:
    """True if ``candidate`` exactly- or near-matches any prompt in ``exclude``.

    Order-sensitive char similarity, not word overlap, so "...make a bomb" and
    "...make a cake" are not conflated despite shared boilerplate.
    """
    return _near_any(normalize(candidate), [normalize(e) for e in exclude], ratio)


def dedup(candidates: list[str], exclude: list[str], *, ratio: float = NEAR_DUP_RATIO) -> list[str]:
    """``candidates`` with near-duplicates of any ``exclude`` removed, order preserved."""
    exclude_norm = [normalize(e) for e in exclude]
    return [c for c in candidates if not _near_any(normalize(c), exclude_norm, ratio)]


def sample_balanced(grouped: dict[str, list[str]], n_total: int, *, seed: int) -> list[str]:
    """Seeded round-robin sample of ``n_total`` across groups. Raises if too few."""
    import random

    rng = random.Random(seed)
    pools = {}
    for key in sorted(grouped):
        items = list(grouped[key])
        rng.shuffle(items)
        pools[key] = items
    if sum(len(v) for v in pools.values()) < n_total:
        raise ValueError(f"need {n_total} prompts, have fewer")

    chosen: list[str] = []
    keys = list(pools)  # already in sorted-key order
    while len(chosen) < n_total:
        for key in keys:
            if pools[key]:
                chosen.append(pools[key].pop())
                if len(chosen) == n_total:
                    break
    return chosen


def sample_first(candidates: list[str], n: int, *, seed: int) -> list[str]:
    """Seeded shuffle, take first ``n``. Raises if fewer than ``n`` available."""
    import random

    if len(candidates) < n:
        raise ValueError(f"need {n} prompts, have {len(candidates)}")
    items = list(candidates)
    random.Random(seed).shuffle(items)
    return items[:n]


def _read_column(path: Path, column: str) -> list[str]:
    with path.open(newline="") as handle:
        return [row[column] for row in csv.DictReader(handle)]


def load_extraction() -> tuple[list[str], list[str]]:
    """Vendored extraction contrast set as ``(harmful, harmless)``."""
    return (
        _read_column(EXTRACTION_HARMFUL_CSV, "prompt"),
        _read_column(EXTRACTION_HARMLESS_CSV, "prompt"),
    )


def load_eval() -> list[str]:
    """Vendored held-out harmful evaluation prompts."""
    return _read_column(EVAL_HARMFUL_CSV, "prompt")
