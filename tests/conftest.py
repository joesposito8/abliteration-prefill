"""Shared doubles for the harness tests.

Fakes cover only what cannot run locally — a tokenizer's chat template and a batched
forward pass — so a failure means the harness is wrong rather than a fake is.
"""

from __future__ import annotations

import pytest
from generation.qwen import DECODING, THINKING_SENTINEL, Generation, row_prefills


class FakeTokenizer:
    """Renders the shape ``build_prompt`` checks for, sentinel included.

    Honours ``enable_thinking`` the way the real template does, so forgetting the
    flag fails here exactly as it would against the real tokenizer.
    """

    def __init__(self, *, enable_thinking_honoured: bool = True) -> None:
        self.enable_thinking_honoured = enable_thinking_honoured

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ) -> str:
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        head = body + "<|im_start|>assistant\n"
        if enable_thinking or not self.enable_thinking_honoured:
            return head
        return head + THINKING_SENTINEL


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_generate_batch(monkeypatch):
    """Replace ``qwen.generate_batch`` with a recorder returning real Generations.

    Returns the list of calls, so a test can assert what was asked of the GPU.
    """
    calls: list[dict] = []

    def fake(model, tok, messages, *, seed, prefill="", decoding=DECODING):
        # The real broadcast-and-length-check, so the double cannot be more
        # permissive than the function it replaces.
        prefills = row_prefills(prefill, len(messages))
        calls.append(
            {
                "messages": list(messages),
                "seed": seed,
                "prefill": prefill,
                "decoding": dict(decoding),
            }
        )
        generations = []
        for message, row_prefill in zip(messages, prefills):
            continuation = f" continuation for {message}"
            generations.append(
                Generation(
                    message=message,
                    prefill=row_prefill,
                    output=row_prefill + continuation,
                    continuation=continuation,
                    raw_continuation=continuation + "<|im_end|>",
                    seed=seed,
                    prompt_tokens=11,
                    new_tokens=7,
                    max_new_tokens=decoding["max_new_tokens"],
                )
            )
        return generations, 0.5

    monkeypatch.setattr("generation.qwen.generate_batch", fake)
    return calls


@pytest.fixture
def frozen_config_kwargs() -> dict:
    """Derived from ``qwen.DECODING`` rather than restated, so it cannot drift."""
    return {
        "temperature": DECODING["temperature"],
        "top_p": DECODING["top_p"],
        "top_k": DECODING["top_k"],
        "max_tokens": DECODING["max_new_tokens"],
        "extra_body": {"min_p": DECODING["min_p"], "do_sample": DECODING["do_sample"]},
    }
