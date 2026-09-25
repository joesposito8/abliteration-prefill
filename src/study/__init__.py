"""Frozen study configuration and dataset access.

``SEED`` is the one project-wide random seed: everything seeded uses it, so the study
reproduces from a single constant. Sharing it separates nothing that should be separate
during generation, because a forward pass derives its stream from the seed *and* the
batch's own prompts. A draw is different: the permutation is a function of the seed and
the population size alone, so two draws of the same size over the same population return
the same rows.
"""

SEED = 20260724

VARIANT_DRAWS = 10
SLOT_DRAWS = {"none": 20, "static_baseline": 20}


def draws(slot: str) -> int:
    return SLOT_DRAWS.get(slot, VARIANT_DRAWS)


from .datasets import (  # noqa: E402  (re-export after SEED so loaders can import it)
    load_extraction_harmful,
    load_extraction_harmless,
    load_harmbench_standard,
    load_pilot_prompts,
    load_strongreject_prompts,
    load_validation_prompts,
    verbatim_overlap,
)

__all__ = [
    "SEED",
    "draws",
    "load_extraction_harmful",
    "load_extraction_harmless",
    "load_harmbench_standard",
    "load_validation_prompts",
    "load_strongreject_prompts",
    "load_pilot_prompts",
    "verbatim_overlap",
]
