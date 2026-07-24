"""Frozen dataset access + the canonical seeded-draw helpers.

The helpers here are the single source of truth for every seeded split in the
study: ``scripts/build_datasets.py`` uses them to write the frozen CSVs and
``tests/test_datasets.py`` uses them to prove the committed CSVs reproduce from
``SEED``. Keep this module pandas/numpy only (no torch/transformers) so it stays
importable in the offline test environment.
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
# One mechanism (a legacy-RandomState permutation over a canonical row order)
# backs every split, so nothing can pick differently than it reproduces. Legacy
# RandomState (MT19937) is version-frozen by NumPy policy; default_rng is not.


def _permutation(n: int, seed: int) -> np.ndarray:
    """Legacy-RandomState permutation of ``range(n)`` — reproducible forever."""
    return np.random.RandomState(seed).permutation(n)


def split_indices(n: int, k: int, seed: int) -> tuple[list[int], list[int]]:
    """Partition ``range(n)`` into a ``k``-sized part and its complement.

    Returns ``(first_sorted, rest_sorted)``: the first ``k`` of the seeded
    permutation and the remaining ``n - k``, each sorted ascending so the
    written files are in canonical order. Membership is what the seed fixes;
    sorting only makes the output stable.
    """
    perm = _permutation(n, seed)
    return sorted(perm[:k].tolist()), sorted(perm[k:].tolist())


def sample_indices(n: int, k: int, seed: int) -> list[int]:
    """``k`` distinct indices from ``range(n)`` (sorted), via the same draw."""
    return sorted(_permutation(n, seed)[:k].tolist())


# --- verbatim overlap (guardrail 11) ---------------------------------------

_NORMALIZERS = {
    "none": lambda s: s,
    "strip": lambda s: s.strip(),
    "lower": lambda s: s.lower(),
    "strip_lower": lambda s: s.strip().lower(),
}


def verbatim_overlap(a, b, normalize: str = "none") -> int:
    """Count exact-string matches between iterables ``a`` and ``b``.

    Exact-string only (no semantic/cosine dedup), per guardrail 11. ``normalize``
    selects an optional normalization applied to both sides before comparison;
    ``"none"`` is the raw verbatim count reported as the headline.
    """
    f = _NORMALIZERS[normalize]
    return len({f(x) for x in a} & {f(y) for y in b})


# --- loaders ---------------------------------------------------------------


def load_strongreject() -> pd.DataFrame:
    """The frozen 313-prompt StrongREJECT set with a ``prompt_id`` column.

    ``prompt_id`` is the 0-based row index in file order; the committed file hash
    pins that order. This is the id the run manifest keys on downstream.
    """
    df = pd.read_csv(STRONGREJECT_CSV)
    df.insert(0, "prompt_id", range(len(df)))
    return df


def load_extraction_harmful() -> list[str]:
    """The 128 HarmBench-standard behaviors used for direction extraction."""
    return pd.read_csv(EXTRACTION_HARMFUL_CSV)["prompt"].tolist()


def load_validation() -> list[str]:
    """The 72 HarmBench-standard behaviors used for primary-layer selection."""
    return pd.read_csv(VALIDATION_HARMFUL_CSV)["prompt"].tolist()


def load_extraction_harmless() -> list[str]:
    """The 128 Alpaca harmless instructions (extraction contrast)."""
    return pd.read_csv(EXTRACTION_HARMLESS_CSV)["prompt"].tolist()


def load_pilot_prompts() -> pd.DataFrame:
    """The seeded-random 30 StrongREJECT prompts (guardrail 18), with prompt_id."""
    return pd.read_csv(PILOT_PROMPTS_CSV)
