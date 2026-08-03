"""Shared doubles for the harness tests.

Only what cannot run locally is faked: a tokenizer's chat template and a batched
forward pass.
"""

from __future__ import annotations

import pytest
from generation.qwen import DECODING, THINKING_SENTINEL, Continuation

CONTINUATION = " a continuation"


class FakeTokenizer:
    """Renders the shape ``build_prompt`` checks for, honouring ``enable_thinking``."""

    enable_thinking_honoured = True

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
def fake_generate_prompts(monkeypatch):
    """Replace the forward pass with a recorder returning real Continuations.

    Returns the list of calls, so a test can assert what was asked of the GPU.
    """
    calls: list[dict] = []

    def fake(model, tok, prompts, *, seed, decoding=DECODING):
        calls.append({"prompts": list(prompts), "seed": seed})
        return [
            Continuation(
                continuation=CONTINUATION,
                raw_continuation=CONTINUATION + "<|im_end|>",
                prompt_tokens=11,
                new_tokens=7,
            )
            for _ in prompts
        ], 0.5

    monkeypatch.setattr("harness.batching.generate_prompts", fake)
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
