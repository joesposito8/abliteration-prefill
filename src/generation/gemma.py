"""Helper-model generation: the abliterated Gemma-3-27B that writes the prefills.

What to load and how to render one request as a prompt. The forward pass itself is
:mod:`generation.batched`.

Sampling. Gemma ships no sampling defaults of its own, so every parameter has to arrive
in ``HELPER_DECODING`` or the model falls back to HuggingFace's. The values are the
portfolio's frozen rules, not this module's to choose.
"""

from __future__ import annotations

from types import MappingProxyType

from prefills import HELPER_SAMPLING, MAX_NEW_TOKENS

HELPER_DECODING = MappingProxyType(
    {**HELPER_SAMPLING, "max_new_tokens": MAX_NEW_TOKENS}
)

# What Gemma-3's template appends to open the model's own turn.
MODEL_TURN_HEADER = "<start_of_turn>model\n"


def render_prompt(tokenizer, message: str) -> str:
    """Render one request as a prompt the helper writes a prefill into.

    Raises:
        ValueError: if the rendered prompt does not open the model's turn, which means
            the helper would answer the request rather than write a prefill for it.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": message}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not text.endswith(MODEL_TURN_HEADER):
        raise ValueError(
            "rendered prompt does not open a model turn, so add_generation_prompt was "
            f"not honoured. Prompt tail: {text[-80:]!r}"
        )
    # The template writes a literal <bos>, and tokenizing this string prepends another.
    return text.removeprefix(tokenizer.bos_token)


def load_helper(snapshot: str):
    """Load the helper in BF16 on CUDA from a staged snapshot.

    Returns ``(model, tokenizer)``.
    """
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    # Not AutoProcessor: it needs an image backend that is not in the lock, and it
    # renders from the same chat template.
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer
