"""Every enabled model, held to the same template contract.

A model module renders one user turn, refuses a template that does not honour its own
invariant, and declares the markers a leaked reasoning block would show. The suite is
parameterised over ``MODELS`` itself rather than a list of rows, so a model cannot be
enabled and left uncovered — being in the tuple is what puts it here.

``contains_reasoning`` is ``False`` by construction for an empty declaration, so a
model claiming no reasoning mode would be indistinguishable from one nobody checked.
The declaration is therefore held against the pinned tokenizer: the delimiters it knows
must be exactly the ones declared.

Prefill placement is a target's contract rather than every model's — the helper writes
prefills and continues none — and is gated in ``test_prefill_exactness.py``.
"""

from __future__ import annotations

import inspect

import pytest
from conftest import (
    MESSAGE,
    OBSERVED_LEAK,
    TruncatedTemplate,
    real_tokenizer,
    template_double,
)
from generation import MODELS, TARGETS
from harness.provider import contains_reasoning

# What a reasoning-block delimiter is called, rather than a list of the ones in
# circulation.
REASONING_WORDS = ("think", "thought", "reason")

# What every model declares, and what a target declares on top of it.
MODEL_INTERFACE = ("MODEL_ID", "REVISION", "TURN_SUFFIX", "REASONING_MARKERS")
TARGET_INTERFACE = ("N_LAYERS", "BATCH", "load_model")

model = pytest.mark.parametrize("module", MODELS, ids=lambda m: m.__name__)
target = pytest.mark.parametrize("module", TARGETS, ids=lambda m: m.__name__)


def reasoning_tokens(tokenizer) -> set[str]:
    """The delimiters this tokenizer knows, whatever the model says about them."""
    return {
        token.content
        for token in tokenizer.added_tokens_decoder.values()
        if any(word in token.content.lower() for word in REASONING_WORDS)
    }


@model
def test_a_model_declares_the_contract(module):
    for name in MODEL_INTERFACE:
        assert hasattr(module, name), f"{module.__name__} declares no {name}"
    assert isinstance(module.REASONING_MARKERS, tuple)
    assert all(isinstance(marker, str) for marker in module.REASONING_MARKERS)
    assert module.build_prompt(template_double(module), MESSAGE)


@target
def test_a_target_declares_what_running_it_needs(module):
    """A target is loaded, edited layer by layer and batched, so it declares more than
    a model that only renders. Missing one of these surfaces on a rented GPU instead."""
    for name in TARGET_INTERFACE:
        assert hasattr(module, name), f"{module.__name__} declares no {name}"
    assert "prefill" in inspect.signature(module.build_prompt).parameters


@model
def test_a_template_that_breaks_the_invariant_is_refused(module):
    """The two renders differ by the turn suffix and nothing else, so the refusal can
    only be that model's own check on it."""
    double = template_double(module)
    assert module.build_prompt(double, MESSAGE)

    with pytest.raises(ValueError) as refused:
        module.build_prompt(TruncatedTemplate(double, module.TURN_SUFFIX), MESSAGE)
    assert "Prompt tail: " in str(refused.value)


@model
def test_the_declaration_is_what_the_pinned_tokenizer_can_emit(module):
    """For a model declaring none, what makes the empty tuple an assertion that there
    is nothing to leak rather than an omission."""
    tokenizer = real_tokenizer(module.MODEL_ID, module.REVISION)

    assert reasoning_tokens(tokenizer) == set(module.REASONING_MARKERS)


@model
def test_the_declared_markers_are_what_the_leak_check_finds(module):
    """A declared no-reasoning model flags nothing, which the check above substantiates."""
    markers = module.REASONING_MARKERS

    assert contains_reasoning("a plain answer", markers) is False
    assert contains_reasoning(OBSERVED_LEAK, markers) is bool(markers)
    for marker in markers:
        assert contains_reasoning(f"some answer {marker} and more", markers) is True


@model
def test_the_real_template_honours_the_invariant(module):
    """The doubles above render the shape we expect; this is the pinned tokenizer's."""
    tokenizer = real_tokenizer(module.MODEL_ID, module.REVISION)

    assert module.build_prompt(tokenizer, MESSAGE).endswith(module.TURN_SUFFIX)


@model
def test_the_rendered_prompt_carries_at_most_one_leading_bos(module):
    """Counted, not asked: ``add_bos_token`` read ``False`` on the helper while its
    template wrote one and tokenizing prepended another."""
    tokenizer = real_tokenizer(module.MODEL_ID, module.REVISION)
    ids = tokenizer(module.build_prompt(tokenizer, MESSAGE))["input_ids"]

    leading = 0
    while leading < len(ids) and ids[leading] == tokenizer.bos_token_id:
        leading += 1
    assert leading <= 1
