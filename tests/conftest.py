"""Shared doubles for the harness tests.

Only what cannot run locally is faked: a tokenizer's chat template, a batched forward
pass, and torch itself.
"""

from __future__ import annotations

import sys

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


class FakeTorch:
    """Enough torch for ``generate_prompts`` to reach ``manual_seed`` and no further."""

    class Stop(Exception):
        """Everything past the seeding call needs real tensors."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def manual_seed(self, value: int) -> None:
        self.seeds.append(value)
        raise self.Stop


@pytest.fixture
def fake_torch(monkeypatch) -> FakeTorch:
    """Stands in front of the ``import torch`` inside the forward pass."""
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def prefills() -> dict[tuple[int, str], str]:
    """Stands in for the helper model's output, which needs a GPU to produce."""
    return _FakePrefills()


class _FakePrefills(dict):
    """Every ``(prompt_id, slot)`` resolves, so a test need not enumerate the set."""

    def __missing__(self, key) -> str:
        prompt_id, slot = key
        return f"[{slot} for {prompt_id}]"
