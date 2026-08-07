"""Model generation: the shared batched forward pass, and the Qwen3-4B target."""

from .batched import Continuation, batch_seed, generate_prompts
from .qwen import (
    MODEL_ID,
    REVISION,
    DECODING,
    THINKING_SENTINEL,
    Generation,
    build_prompt,
    contains_thinking,
    generate,
    generate_batch,
    load_model,
)

__all__ = [
    "Continuation",
    "batch_seed",
    "generate_prompts",
    "MODEL_ID",
    "REVISION",
    "DECODING",
    "THINKING_SENTINEL",
    "Generation",
    "build_prompt",
    "contains_thinking",
    "generate",
    "generate_batch",
    "load_model",
]
