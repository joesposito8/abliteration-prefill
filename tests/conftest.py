"""Shared doubles for the harness tests.

Everything here runs offline: no torch, no GPU, no API key. The fakes stand in for
exactly the two things that cannot run locally — a tokenizer's chat template and a
batched forward pass — and nothing else, so a failure means the harness is wrong
rather than a fake is.
"""

from __future__ import annotations

import pytest
from generation.qwen import SAMPLING, THINKING_SENTINEL, Generation


class FakeTokenizer:
    """Renders the shape ``build_prompt`` checks for, sentinel included.

    A real Qwen3 template emits the empty-thinking sentinel only when
    ``enable_thinking=False`` is passed explicitly, so this honours the flag the
    same way: forget it and the sentinel is absent and ``build_prompt`` raises,
    exactly as it would against the real tokenizer.
    """

    def __init__(self, *, enable_thinking_honoured: bool = True) -> None:
        self.enable_thinking_honoured = enable_thinking_honoured

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ) -> str:
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        head = body + "<|im_start|>assistant\n"
        if enable_thinking or not self.enable_thinking_honoured:
            return head
        return head + THINKING_SENTINEL


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_generate_batch(monkeypatch):
    """Replace ``qwen.generate_batch`` with a recorder that returns real Generations.

    Returns the list of calls, so a test can assert what the provider actually asked
    the GPU for. The continuation is derived from the prompt so rows are
    distinguishable, and ``output`` is built the way the real function builds it —
    ``prefill + continuation`` — because the prefill round-trip is what the scorer
    later depends on.
    """
    calls: list[dict] = []

    def fake(model, tok, messages, *, seed, prefill="", max_new_tokens=512):
        prefills = [prefill] * len(messages) if isinstance(prefill, str) else list(prefill)
        calls.append(
            {
                "messages": list(messages),
                "seed": seed,
                "prefill": prefill,
                "max_new_tokens": max_new_tokens,
            }
        )
        generations = []
        for message, row_prefill in zip(messages, prefills, strict=True):
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
                    max_new_tokens=max_new_tokens,
                )
            )
        return generations, 0.5

    monkeypatch.setattr("generation.qwen.generate_batch", fake)
    return calls


@pytest.fixture
def frozen_config_kwargs() -> dict:
    """The generate-config values a task must set for the provider to accept them.

    Derived from ``qwen.SAMPLING`` rather than restated, so this cannot drift from
    the constant it is meant to mirror.
    """
    return {
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "top_k": SAMPLING["top_k"],
        "extra_body": {"min_p": SAMPLING["min_p"], "do_sample": SAMPLING["do_sample"]},
    }
