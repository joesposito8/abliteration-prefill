"""The frozen decoding parameters, shared by every target.

These are Qwen3's recommended settings for the non-thinking mode the study runs it in,
preregistered and then applied to every target: two models sampling differently would
not be comparable, so the second runs under a preset tuned for the first. A cross-model
limitation to disclose, not a per-model knob to turn.

Passing them explicitly is what holds it, and ``generate_prompts`` requires them. Left
off, Qwen3-4B falls back to the 0.6 / 0.95 of its own thinking preset; phi-4, whose
``generation_config.json`` names no sampling parameters at all, falls back to
HuggingFace's. The length cap travels with them, since truncation changes the text too.
"""

from __future__ import annotations

from types import MappingProxyType

# Read-only: mutating these mid-run would silently change the text produced for every
# condition that follows.
DECODING = MappingProxyType(
    {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 1024,
    }
)
