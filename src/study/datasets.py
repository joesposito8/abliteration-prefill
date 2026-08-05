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
# A prompt file does one of two jobs, and its shape follows the job. An EVAL SET is
# something the target model is evaluated on: it becomes Inspect samples, so it loads
# as a frame carrying prompt_id, category, source and forbidden_prompt. A CONTRAST
# CORPUS is fed to direction extraction, which reads nothing but the text, so it loads
# as a list of prompts. ``harness.dataset.EVAL_SETS`` is the registry of the former.

EVAL_SET_COLUMNS = ("prompt_id", "category", "source", "forbidden_prompt")


def load_harmbench_standard() -> pd.DataFrame:
    """The 200 HarmBench standard behaviors — the parent both harmful splits are cut from.

    ``prompt_id`` is the row index in canonical order (sorted by ``behavior_id``), so a
    split's ids say which rows of this file it took.
    """
    return pd.read_csv(HARMBENCH_STANDARD_CSV)


def load_strongreject_prompts() -> pd.DataFrame:
    """The frozen 313-prompt StrongREJECT eval set with a ``prompt_id`` column.

    The one loader that derives its id instead of reading it: this file is vendored
    byte-identical to the published set, so a column cannot be added to it without
    giving that up. ``prompt_id`` is the 0-based row index in file order, and the
    committed file hash pins that order.
    """
    df = pd.read_csv(STRONGREJECT_CSV)
    df.insert(0, "prompt_id", range(len(df)))
    return df


def load_pilot_prompts() -> pd.DataFrame:
    """The seeded-random 30 StrongREJECT prompts (the pilot slice).

    ``prompt_id`` is the parent's, so pilot rows join onto main-run rows.
    """
    return pd.read_csv(PILOT_PROMPTS_CSV)


def load_validation_prompts() -> pd.DataFrame:
    """The 72 HarmBench-standard behaviors used for primary-layer selection.

    ``prompt_id`` is inherited from the 200-row parent, so these ids and the extraction
    split's partition ``range(200)`` — the disjointness the extraction corpus rests on
    is readable from the two files rather than re-derived from the seed.
    """
    return pd.read_csv(VALIDATION_HARMFUL_CSV)


def load_extraction_harmful() -> list[str]:
    """The 128 HarmBench-standard behaviors used for direction extraction."""
    return pd.read_csv(EXTRACTION_HARMFUL_CSV)["forbidden_prompt"].tolist()


def load_extraction_harmless() -> list[str]:
    """The 128 Alpaca harmless instructions (extraction contrast).

    Keeps its own column names: these are not forbidden prompts, and ``alpaca_index``
    indexes a different corpus than ``prompt_id`` does, so it must not be joinable
    against the harmful splits by accident.
    """
    return pd.read_csv(EXTRACTION_HARMLESS_CSV)["prompt"].tolist()
