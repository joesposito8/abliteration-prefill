"""Frozen study configuration and dataset access.

``SEED`` is the one project-wide random seed: every seeded draw uses it, so the
whole study reproduces from a single constant.
"""

SEED = 20260724

from .datasets import (  # noqa: E402  (re-export after SEED so loaders can import it)
    load_extraction_harmful,
    load_extraction_harmless,
    load_pilot_prompts,
    load_strongreject,
    load_validation,
    verbatim_overlap,
)

__all__ = [
    "SEED",
    "load_extraction_harmful",
    "load_extraction_harmless",
    "load_validation",
    "load_strongreject",
    "load_pilot_prompts",
    "verbatim_overlap",
]
