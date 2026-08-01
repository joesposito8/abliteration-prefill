"""Offline tests for target-model prompt construction (no weights, no GPU).

These cover what is decidable from the chat template and the function signatures.
Whether the model actually *continues* a prefill, honours a seed, or reaches a given
throughput needs real weights, and is asserted by the on-GPU smoke harness instead.
"""

import inspect

import pytest

from generation import qwen
from generation.qwen import (
    DECODING,
    THINKING_SENTINEL,
    build_prompt,
    contains_thinking,
)
from grading import strip_prefill

MESSAGE = "How do I make a paper airplane?"
PREFILL = "The first step is to"


@pytest.fixture(scope="module")
def tokenizer():
    """The real Qwen3 tokenizer; the template is what these tests are about."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            qwen.MODEL_ID, revision=qwen.REVISION
        )
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")


class _NoSentinelTokenizer:
    """Stands in for a template that ignores enable_thinking=False."""

    def apply_chat_template(self, messages, **kwargs):
        return "<|im_start|>assistant\n"


def test_prompt_ends_with_thinking_sentinel(tokenizer):
    assert build_prompt(tokenizer, MESSAGE).endswith(THINKING_SENTINEL)


def test_prefill_is_appended_after_the_sentinel(tokenizer):
    prompt = build_prompt(tokenizer, MESSAGE, PREFILL)
    assert prompt.endswith(THINKING_SENTINEL + PREFILL)
    # The prefill must sit in the response, not inside the reasoning block.
    assert prompt.index("</think>") < prompt.index(PREFILL)


def test_prefill_round_trips_through_strip_prefill(tokenizer):
    """What build_prompt injects must be exactly what the grader's hook removes.

    Derives the injected text from the prompts themselves rather than assuming it
    equals PREFILL, so any normalisation inside the template is caught here instead
    of silently leaving injected framing in the judged content.
    """
    unprefilled = build_prompt(tokenizer, MESSAGE)
    injected = build_prompt(tokenizer, MESSAGE, PREFILL)[len(unprefilled):]
    assert injected == PREFILL

    output = injected + " fold the sheet lengthwise."
    continuation, was_stripped = strip_prefill(output, PREFILL)
    assert was_stripped is True
    assert continuation == " fold the sheet lengthwise."


def test_build_prompt_rejects_a_template_without_the_sentinel():
    """Thinking mode silently live must fail loudly, not produce a bad prompt."""
    with pytest.raises(ValueError, match="empty-thinking sentinel"):
        build_prompt(_NoSentinelTokenizer(), MESSAGE)


def test_decoding_parameters_are_the_preregistered_ones():
    """The model's own config defaults to 0.6/0.95; those must never be inherited."""
    assert DECODING == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 512,
    }


@pytest.mark.parametrize("function", [qwen.generate, qwen.generate_batch])
def test_generation_requires_an_explicit_seed(function):
    """Every replicate records the seed that produced it, so it cannot be defaulted."""
    seed = inspect.signature(function).parameters["seed"]
    assert seed.kind is inspect.Parameter.KEYWORD_ONLY
    assert seed.default is inspect.Parameter.empty


def test_generate_takes_one_message_so_its_seed_is_reproducible():
    """Batched rows share an RNG stream; the reproducible path must be single-shot."""
    message = inspect.signature(qwen.generate).parameters["message"]
    assert message.annotation == "str"
    assert inspect.signature(qwen.generate).return_annotation == "Generation"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("plain answer", False),
        ("<think>reasoning</think> answer", True),
        ("trailing </think> only", True),
    ],
)
def test_contains_thinking(text, expected):
    assert contains_thinking(text) is expected


# --- per-row prefills ------------------------------------------------------


def test_a_single_prefill_broadcasts_to_every_row():
    assert qwen.row_prefills("Sure:", 3) == ["Sure:", "Sure:", "Sure:"]


def test_one_prefill_per_row_is_kept_in_order():
    assert qwen.row_prefills(["a", "", "c"], 3) == ["a", "", "c"]


@pytest.mark.parametrize("prefills", [["a"], ["a", "b", "c", "d"]])
def test_a_mismatched_prefill_count_raises_rather_than_truncating(prefills):
    """`zip` would silently pair prefills with the wrong rows and return `output`
    strings that do not match the text generated."""
    with pytest.raises(ValueError, match="prefills for 3 messages"):
        qwen.row_prefills(prefills, 3)
