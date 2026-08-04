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


class FakeGeneratePrompts:
    """A forward pass that records what it was asked and returns real Continuations."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.continuation = CONTINUATION

    def __call__(self, model, tok, prompts, *, seed):
        self.calls.append({"prompts": list(prompts), "seed": seed})
        return [
            Continuation(
                continuation=self.continuation,
                raw_continuation=self.continuation + "<|im_end|>",
                prompt_tokens=11,
                new_tokens=7,
            )
            for _ in prompts
        ], 0.5


@pytest.fixture
def fake_generate_prompts(monkeypatch) -> FakeGeneratePrompts:
    """Set ``.continuation`` to change what every row comes back with."""
    fake = FakeGeneratePrompts()
    monkeypatch.setattr("harness.batching.generate_prompts", fake)
    return fake


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


class FakeJudge:
    """A grader model that answers from a rule over the text it was handed.

    Records the text and the resolved config of every call, so a test can assert both
    what the judge saw and what it would have been asked to do.
    """

    def __init__(self, reply) -> None:
        from inspect_ai.model import get_model

        self.reply = reply
        self.judged: list[str] = []
        self.configs: list = []
        self.model = get_model("mockllm/model", custom_outputs=self)

    def __call__(self, input, tools, tool_choice, config):
        from inspect_ai.model import ModelOutput

        self.judged.append(input[-1].text)
        self.configs.append(config)
        return ModelOutput.from_content(
            model="mockllm", content=self.reply(self.judged[-1])
        )


@pytest.fixture
def fake_judge():
    """Stands in for the rubric grader, which is an API call away."""
    return FakeJudge
