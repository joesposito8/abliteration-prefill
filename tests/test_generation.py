"""Offline tests for target-model prompt construction (no weights, no GPU).

These cover what is decidable from the chat template and the function signatures.
Whether the model actually *continues* a prefill, honours a seed, or reaches a given
throughput needs real weights, and is asserted by the on-GPU smoke harness instead.
"""

import dataclasses
import inspect

import pytest
from conftest import FakeTokenizer, real_tokenizer

from generation import qwen3_4b
from generation.batched import Continuation
from generation.qwen3_4b import DECODING, TURN_SUFFIX, build_prompt

MESSAGE = "How do I make a paper airplane?"
PREFILL = "The first step is to"


@pytest.fixture(scope="module")
def tokenizer():
    """The real Qwen3 tokenizer; the template is what these tests are about."""
    return real_tokenizer(qwen3_4b.MODEL_ID, qwen3_4b.REVISION)


def test_prompt_ends_with_thinking_sentinel(tokenizer):
    assert build_prompt(tokenizer, MESSAGE).endswith(TURN_SUFFIX)


def test_prefill_is_appended_after_the_sentinel(tokenizer):
    prompt = build_prompt(tokenizer, MESSAGE, PREFILL)
    assert prompt.endswith(TURN_SUFFIX + PREFILL)
    # The prefill must sit in the response, not inside the reasoning block.
    assert prompt.index("</think>") < prompt.index(PREFILL)


def test_a_prefill_adds_itself_and_changes_nothing_else(tokenizer):
    """Derives the injected text from the two renders rather than assuming it equals
    PREFILL, so a template that normalised it, or moved anything ahead of it, is caught
    here — the provider decodes only new tokens and would not see the difference."""
    unprefilled = build_prompt(tokenizer, MESSAGE)
    injected = build_prompt(tokenizer, MESSAGE, PREFILL)[len(unprefilled):]

    assert injected == PREFILL


def test_decoding_parameters_are_the_preregistered_ones():
    """The model's own config defaults to 0.6/0.95; those must never be inherited."""
    assert DECODING == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 1024,
    }


@pytest.mark.parametrize("function", [qwen3_4b.generate, qwen3_4b.generate_batch])
def test_generation_requires_an_explicit_seed(function):
    """Every replicate records the seed that produced it, so it cannot be defaulted."""
    seed = inspect.signature(function).parameters["seed"]
    assert seed.kind is inspect.Parameter.KEYWORD_ONLY
    assert seed.default is inspect.Parameter.empty


def test_generate_takes_one_message_so_its_seed_is_reproducible():
    """Batched rows share an RNG stream; the reproducible path must be single-shot."""
    message = inspect.signature(qwen3_4b.generate).parameters["message"]
    assert message.annotation == "str"
    assert inspect.signature(qwen3_4b.generate).return_annotation == "Generation"


# --- rendering layered over the forward pass -------------------------------


def fake_prompts(monkeypatch, seconds=0.5):
    """Stand in for the forward pass, returning one Continuation per prompt."""
    calls = []

    def fake(model, tok, prompts, *, seed, decoding):
        calls.append(list(prompts))
        return [
            Continuation(
                continuation=f"<{i}>",
                raw_continuation=f"<{i}>",
                prompt_tokens=3,
                new_tokens=4,
            )
            for i, _ in enumerate(prompts)
        ], seconds

    monkeypatch.setattr(qwen3_4b, "generate_prompts", fake)
    return calls


def test_each_row_keeps_its_own_message_and_continuation(monkeypatch):
    """A mispaired zip would hand rows each other's text with nothing to flag it."""
    messages = ["m0", "m1", "m2"]
    calls = fake_prompts(monkeypatch)

    generations, _ = qwen3_4b.generate_batch(
        None, FakeTokenizer(), messages, seed=1, prefill="A:"
    )

    assert [g.message for g in generations] == messages
    assert [g.response for g in generations] == ["A:<0>", "A:<1>", "A:<2>"]
    assert all(message in prompt for message, prompt in zip(messages, calls[0]))


def test_generate_is_a_single_row_batch(monkeypatch):
    """One path reaches the GPU, so the two cannot drift apart."""
    fake_prompts(monkeypatch, seconds=1.25)
    tokenizer = FakeTokenizer()

    one = qwen3_4b.generate(None, tokenizer, "m", seed=1, prefill="A:")
    batched, _ = qwen3_4b.generate_batch(None, tokenizer, ["m"], seed=1, prefill="A:")

    # A single generation carries its own duration; a batch row shares one clock.
    assert one.seconds == 1.25
    assert dataclasses.replace(one, seconds=None) == batched[0]
