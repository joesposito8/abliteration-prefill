"""The frozen 13-prefill attack portfolio and its helper-generation rules.

Six helper-prompt families (each producing two seeded, request-specific prefill
variants from the abliterated Gemma helper) plus Kuo's static baseline. Templates are
authored from the Struppek prefill taxonomy (arXiv:2602.14689); the helper decoding,
retry, length, and duplicate rules are frozen in ``rules``. Everything is committed and
SHA-256-hashed (``scripts/freeze_portfolio.py``) before the main run.
"""

from .families import (
    FAMILIES,
    FAMILY_BY_ID,
    PORTFOLIO,
    VARIANTS_PER_FAMILY,
    Family,
    Slot,
    fill_prompt,
    load_prompt,
    prompt_path,
    shared_contract,
)
from .rules import (
    FIXED_FALLBACK,
    HELPER_SAMPLING,
    MAX_ATTEMPTS,
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
    "FAMILY_BY_ID",
    "PORTFOLIO",
    "VARIANTS_PER_FAMILY",
    "Family",
    "Slot",
    "fill_prompt",
    "load_prompt",
    "prompt_path",
    "shared_contract",
    "FIXED_FALLBACK",
    "HELPER_SAMPLING",
    "MAX_ATTEMPTS",
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
