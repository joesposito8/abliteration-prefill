"""Target model: phi-4, optionally prefilled.

Prefill position. ``add_generation_prompt=True`` opens the assistant's turn and stops,
so the prefill is concatenated onto the render as it is for Qwen3-4B. The template
emits no literal BOS, so unlike the helper's there is nothing to strip.
"""

from __future__ import annotations

MODEL_ID = "microsoft/phi-4"
REVISION = "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"
N_LAYERS = 40

BATCH = 32

# What phi-4's template appends to open the assistant's turn.
TURN_SUFFIX = "<|im_start|>assistant<|im_sep|>"

# Declared: phi-4 has no reasoning mode, so nothing it emits can be a leak.
REASONING_MARKERS = ()


def build_prompt(tokenizer, message: str, prefill: str = "") -> str:
    """Render one user turn as a prompt, with an optional assistant-side prefill.

    Raises:
        ValueError: if the rendered prompt does not end by opening the assistant's turn,
            which means the prefill would be placed outside it and answered rather than
            continued.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not text.endswith(TURN_SUFFIX):
        raise ValueError(
            "rendered prompt does not end by opening the assistant turn, so "
            "add_generation_prompt was not honoured and a prefill would land outside "
            f"the model's own turn. Prompt tail: {text[-80:]!r}"
        )
    return text + prefill


def load_model(model_id: str = MODEL_ID, revision: str = REVISION):
    """Load the target model in BF16 on CUDA. Returns ``(model, tokenizer)``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer
