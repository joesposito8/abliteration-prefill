"""The 13-prefill attack portfolio and its helper-generation rules.

Six helper-prompt families (each producing two seeded, request-specific prefills from
the abliterated Gemma helper) plus a static baseline. Templates follow the Struppek
prefill taxonomy (arXiv:2602.14689); decoding, retry, length, and duplicate rules live
in ``rules``.
"""

from .families import (
    CONTRACT_PATH,
    FAMILIES,
    PORTFOLIO,
    VARIANTS_PER_FAMILY,
    family_of,
    fill_prompt,
    load_prompt,
    strategy_path,
)
from .rules import (
    HELPER_SAMPLING,
    MAX_ATTEMPTS,
    MAX_DEDUP_RESAMPLES,
    MAX_NEW_TOKENS,
    PLACEHOLDER,
    STATIC_BASELINE,
    PrefillResult,
    helper_seed,
    produce_family,
    validate,
)

HELPER_MODEL = "mlabonne/gemma-3-27b-it-abliterated"
HELPER_REVISION = "eaa815dffdf0"

__all__ = [
    "FAMILIES",
    "PORTFOLIO",
    "VARIANTS_PER_FAMILY",
    "CONTRACT_PATH",
    "family_of",
    "fill_prompt",
    "load_prompt",
    "strategy_path",
    "HELPER_SAMPLING",
    "MAX_ATTEMPTS",
    "MAX_DEDUP_RESAMPLES",
    "MAX_NEW_TOKENS",
    "PLACEHOLDER",
    "STATIC_BASELINE",
    "PrefillResult",
    "helper_seed",
    "produce_family",
    "validate",
    "HELPER_MODEL",
    "HELPER_REVISION",
]
