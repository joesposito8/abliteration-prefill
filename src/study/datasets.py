"""Frozen dataset access and the canonical seeded-draw helpers.

Shared by the build script and the tests, so the committed CSVs and their
reproduction can't drift. Kept pandas/numpy-only so it imports offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

STRONGREJECT_CSV = DATA_DIR / "strongreject_dataset.csv"
HARMBENCH_STANDARD_CSV = DATA_DIR / "harmbench_standard_behaviors.csv"
EXTRACTION_HARMFUL_CSV = DATA_DIR / "extraction_harmful.csv"
VALIDATION_HARMFUL_CSV = DATA_DIR / "validation_harmful.csv"
EXTRACTION_HARMLESS_CSV = DATA_DIR / "extraction_harmless.csv"
PILOT_PROMPTS_CSV = DATA_DIR / "pilot_prompts.csv"
FREEZE_MANIFEST_JSON = DATA_DIR / "freeze_manifest.json"


# --- canonical seeded draws ------------------------------------------------
# One mechanism backs every split. Legacy RandomState (MT19937) is version-frozen
# by NumPy policy; default_rng is not.


def _permutation(n: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).permutation(n)


def split_indices(n: int, k: int, seed: int) -> tuple[list[int], list[int]]:
    """Split ``range(n)`` into the first ``k`` of the seeded permutation and the
    rest, each sorted ascending (the seed fixes membership; sorting only makes
    the written files stable)."""
    perm = _permutation(n, seed)
    return sorted(perm[:k].tolist()), sorted(perm[k:].tolist())


def sample_indices(n: int, k: int, seed: int) -> list[int]:
    """``k`` distinct indices from ``range(n)`` (sorted), via the same draw."""
    return split_indices(n, k, seed)[0]


# --- verbatim (exact-string) contamination check ---------------------------

_NORMALIZERS = {
    "none": lambda s: s,
    "strip": lambda s: s.strip(),
    "lower": lambda s: s.lower(),
    "strip_lower": lambda s: s.strip().lower(),
}


def verbatim_overlap(a, b, normalize: str = "none") -> int:
    """Number of items in ``a`` (a harmful pool) that exactly match some item in
    ``b`` (the eval set) — i.e. harmful rows that leaked into eval.

    Exact-string only, no semantic dedup. Counts per matching row, not per
    distinct string, so a duplicate on the ``a`` side isn't silently collapsed.
    ``normalize`` optionally normalizes both sides; ``"none"`` is the headline.
    """
    f = _NORMALIZERS[normalize]
    b_set = {f(y) for y in b}
    return sum(1 for x in a if f(x) in b_set)


# --- loaders ---------------------------------------------------------------
#
# Shape follows job: an eval set the model is scored on loads as a frame carrying the
# columns a sample is built from; a contrast corpus for direction extraction loads as
# a list of prompts, which is all it is ever read for.

EVAL_SET_COLUMNS = ("prompt_id", "category", "source", "forbidden_prompt")


def load_harmbench_standard() -> pd.DataFrame:
    """The 200 HarmBench standard behaviors: the parent of both harmful splits.

    ``prompt_id`` is the row index in ``behavior_id`` order, so a split's ids name the
    rows it took.
    """
    return pd.read_csv(HARMBENCH_STANDARD_CSV)


def load_strongreject_prompts() -> pd.DataFrame:
    """The frozen 313-prompt StrongREJECT eval set, ``prompt_id`` = row index.

    The one loader that derives its id rather than reading it: the file is vendored
    byte-identical to the published set, so no column may be added to it.
    """
    df = pd.read_csv(STRONGREJECT_CSV)
    df.insert(0, "prompt_id", range(len(df)))
    return df


def load_pilot_prompts() -> pd.DataFrame:
    """The seeded-random 30 StrongREJECT prompts (the pilot slice), with prompt_id."""
    return pd.read_csv(PILOT_PROMPTS_CSV)


def load_validation_prompts() -> pd.DataFrame:
    """The 72 HarmBench-standard behaviors used for primary-layer selection."""
    return pd.read_csv(VALIDATION_HARMFUL_CSV)


def load_extraction_harmful() -> list[str]:
    """The 128 HarmBench-standard behaviors used for direction extraction."""
    return pd.read_csv(EXTRACTION_HARMFUL_CSV)["forbidden_prompt"].tolist()


def load_extraction_harmless() -> list[str]:
    """The 128 Alpaca harmless instructions (extraction contrast)."""
    return pd.read_csv(EXTRACTION_HARMLESS_CSV)["prompt"].tolist()
