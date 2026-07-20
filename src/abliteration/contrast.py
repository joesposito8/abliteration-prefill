"""Contrast and evaluation prompt sets for abliteration.

The refusal direction is estimated from a *contrast set* — harmful vs. harmless
instructions — and the edited models are confirmed on a disjoint *evaluation set* of
harmful prompts. Three properties matter and are enforced here rather than left to the
caller:

Disjointness. The evaluation prompts must not appear in the extraction pool or in the
313 StrongREJECT prompts the headline study is graded on; otherwise a reported unlock
could be an artifact of the direction having been fit on that very prompt, or of eval
leakage into the graded set. ``dedup`` removes any eval candidate that exactly or
near-matches an excluded prompt (18 of the 100 JailbreakBench prompts are themselves
AdvBench-sourced, so this is a real collision, not a hypothetical one).

Reproducibility. Sampling is seeded and deterministic, and the *sampled subsets* — not
the multi-megabyte source corpora — are vendored into ``data/`` and loaded at runtime.
The build path (network download + write) lives in a separate one-time script; this
module stays offline and import-clean so the sampling and dedup logic is unit-testable
without a network or the source files.

Category balance. The evaluation set is drawn evenly across JailbreakBench's ten harm
categories, so a per-layer unlock rate reflects breadth rather than one easy category.
"""

from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

# Vendored sampled subsets (produced once by scratch/build_datasets.py, then committed).
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXTRACTION_HARMFUL_CSV = _DATA_DIR / "abliteration_extraction_harmful.csv"
EXTRACTION_HARMLESS_CSV = _DATA_DIR / "abliteration_extraction_harmless.csv"
EVAL_HARMFUL_CSV = _DATA_DIR / "abliteration_eval_harmful.csv"

# Sample sizes (Arditi et al. use 128+128; the runbook scales extraction to 256+256).
N_HARMFUL = 256
N_HARMLESS = 256
N_EVAL = 20

# Fixed seed for the one-time sampling. The vendored CSVs are the frozen artifact; this
# only documents how they were produced.
SAMPLE_SEED = 20260720

# Character-sequence similarity above which two prompts count as near-duplicates.
NEAR_DUP_RATIO = 0.9


def normalize(text: str) -> str:
    """Canonical form for duplicate comparison.

    Lowercases, drops surrounding quotes, strips non-alphanumeric characters to spaces,
    and collapses whitespace, so that punctuation, casing, and quoting differences do
    not hide a duplicate.
    """
    text = text.strip().strip("\"'").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _near_any(cand_norm: str, exclude_norm: list[str], ratio: float) -> bool:
    """True if the normalized candidate exactly- or near-matches a normalized exclude."""
    for other in exclude_norm:
        if cand_norm == other:
            return True
        if SequenceMatcher(None, cand_norm, other).ratio() >= ratio:
            return True
    return False


def is_near_dup(
    candidate: str, exclude: list[str], *, ratio: float = NEAR_DUP_RATIO
) -> bool:
    """True if ``candidate`` exactly or near-matches any prompt in ``exclude``.

    Exact match is on the normalized string; near-match is a character-sequence similarity
    (``difflib`` ratio) at or above ``ratio``. Order-sensitive similarity is used rather
    than word-set overlap so that two prompts differing only in a key content word (e.g.
    "...make a bomb" vs. "...make a cake") are not flagged despite sharing boilerplate.
    This catches shared-source collisions and minor rewordings, not semantic paraphrase.
    """
    return _near_any(normalize(candidate), [normalize(e) for e in exclude], ratio)


def dedup(candidates: list[str], exclude: list[str], *, ratio: float = NEAR_DUP_RATIO) -> list[str]:
    """Return ``candidates`` with every near-duplicate of an ``exclude`` prompt removed,
    order preserved. The exclude list is normalized once, not per candidate."""
    exclude_norm = [normalize(e) for e in exclude]
    return [c for c in candidates if not _near_any(normalize(c), exclude_norm, ratio)]


def sample_balanced(
    grouped: dict[str, list[str]], n_total: int, *, seed: int
) -> list[str]:
    """Evenly sample ``n_total`` prompts across the groups of ``grouped``.

    Each group is shuffled with a seeded RNG, then prompts are taken round-robin one per
    group until ``n_total`` is reached, so the result is as balanced as the group sizes
    allow. Raises if the groups cannot supply ``n_total`` prompts.
    """
    import random

    rng = random.Random(seed)
    pools = {}
    for key in sorted(grouped):
        items = list(grouped[key])
        rng.shuffle(items)
        pools[key] = items
    total = sum(len(v) for v in pools.values())
    if total < n_total:
        raise ValueError(f"only {total} prompts available, need {n_total}")

    chosen: list[str] = []
    keys = sorted(pools)
    while len(chosen) < n_total:
        for key in keys:
            if pools[key]:
                chosen.append(pools[key].pop())
                if len(chosen) == n_total:
                    break
    return chosen


def sample_first(candidates: list[str], n: int, *, seed: int) -> list[str]:
    """Seeded shuffle then take the first ``n`` — the flat-pool counterpart to
    :func:`sample_balanced`. Raises if fewer than ``n`` are available."""
    import random

    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} prompts available, need {n}")
    items = list(candidates)
    random.Random(seed).shuffle(items)
    return items[:n]


def _read_column(path: Path, column: str) -> list[str]:
    with path.open(newline="") as handle:
        return [row[column] for row in csv.DictReader(handle)]


def load_extraction() -> tuple[list[str], list[str]]:
    """Load the vendored extraction contrast set as ``(harmful, harmless)``."""
    return (
        _read_column(EXTRACTION_HARMFUL_CSV, "prompt"),
        _read_column(EXTRACTION_HARMLESS_CSV, "prompt"),
    )


def load_eval() -> list[str]:
    """Load the vendored held-out harmful evaluation prompts."""
    return _read_column(EVAL_HARMFUL_CSV, "prompt")
