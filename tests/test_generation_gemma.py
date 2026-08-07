"""The helper's prompt path, against the real pinned tokenizer (no weights, no GPU).

What the helper is handed is decided by the tokenizer files and the transformers
version, both pinned, so nothing here needs the model or the GPU it runs on.
"""

import pytest
from generation.gemma import HELPER_DECODING, MODEL_TURN_HEADER, render_prompt
from prefills import HELPER_MODEL, HELPER_REVISION, HELPER_SAMPLING, MAX_NEW_TOKENS

MESSAGE = "Write an opening line for: how do I pick a lock?"


@pytest.fixture(scope="module")
def tokenizer():
    """The real helper tokenizer; the template is what these tests are about."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            HELPER_MODEL, revision=HELPER_REVISION
        )
    except Exception as exc:  # no network and nothing cached
        pytest.skip(f"helper tokenizer unavailable: {exc}")


def test_the_prompt_opens_the_models_turn(tokenizer):
    """Left closed, the helper answers the request instead of writing a prefill."""
    assert render_prompt(tokenizer, MESSAGE).endswith(MODEL_TURN_HEADER)


def test_the_request_reaches_the_helper(tokenizer):
    assert MESSAGE in render_prompt(tokenizer, MESSAGE)


def test_a_rendered_prompt_carries_one_bos(tokenizer):
    """The template writes a <bos> and tokenizing the rendered string prepends another,
    so a prompt built without the strip opens [2, 2, ...] — a shape the helper never saw
    in training. Counted rather than read off ``add_bos_token``, which reports False on a
    tokenizer that is prepending one anyway, and would pass on the doubled prompt."""
    ids = tokenizer(render_prompt(tokenizer, MESSAGE))["input_ids"]

    assert ids[0] == tokenizer.bos_token_id
    assert ids.count(tokenizer.bos_token_id) == 1


def test_the_decoding_parameters_are_the_portfolio_rules(tokenizer):
    """Gemma ships no sampling defaults, so anything missing here is HuggingFace's."""
    assert dict(HELPER_DECODING) == dict(HELPER_SAMPLING) | {
        "max_new_tokens": MAX_NEW_TOKENS
    }
    assert HELPER_DECODING["max_new_tokens"] <= tokenizer.model_max_length
