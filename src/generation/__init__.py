"""Target-model generation (Qwen3-4B, thinking disabled)."""

from .qwen import (
    MODEL_ID,
    REVISION,
    SAMPLING,
    THINKING_SENTINEL,
    Generation,
    build_prompt,
    contains_thinking,
    generate,
    generate_batch,
    load_model,
)

__all__ = [
    "MODEL_ID",
    "REVISION",
    "SAMPLING",
    "THINKING_SENTINEL",
    "Generation",
    "build_prompt",
    "contains_thinking",
    "generate",
    "generate_batch",
    "load_model",
]
