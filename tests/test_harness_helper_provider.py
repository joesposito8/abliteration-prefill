"""The helper provider's contract with Inspect, exercised without torch or a GPU."""

from __future__ import annotations

import anyio
import pytest
from conftest import CONTINUATION
from generation.gemma import HELPER_DECODING, MODEL_TURN_HEADER
from harness.batching import BATCH
from harness.helper_provider import (
    HELPER_BATCH,
    HELPER_FROZEN_CONFIG,
    GemmaHelperAPI,
    require_one_request,
)
from harness.provider import QwenLocalAPI
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

SEED = 20260724


def build(tokenizer=None, module=None) -> GemmaHelperAPI:
    return GemmaHelperAPI(
        model_name="gemma-helper/prefills", module=module, tokenizer=tokenizer
    )


def config(**overrides) -> GenerateConfig:
    return HELPER_FROZEN_CONFIG.model_copy(update={"seed": SEED, **overrides})


async def generate(api: GemmaHelperAPI, messages, cfg):
    return await api.generate(messages, tools=[], tool_choice="none", config=cfg)


# --- construction ----------------------------------------------------------


def test_constructs_weightless_without_complaint():
    """``score()`` rebuilds the model from the log, where model_args are null."""
    api = build()
    assert api.module is None and api.tokenizer is None


def test_generate_refuses_when_weightless():
    api = build()
    with pytest.raises(RuntimeError, match="holds no live module"):
        anyio.run(generate, api, [ChatMessageUser(content="q")], config())


def test_a_misspelled_model_arg_is_refused(helper_tokenizer):
    with pytest.raises(TypeError, match="toknizer"):
        get_model("gemma-helper/typo-check", toknizer=helper_tokenizer)


def test_a_template_that_leaves_the_turn_closed_fails_at_construction():
    """Otherwise the helper answers requests instead of writing prefills for them, and
    the whole first batch reads as ordinary output."""

    class Closed:
        bos_token = "<bos>"

        def apply_chat_template(self, messages, **kwargs) -> str:
            return "<bos><start_of_turn>user\nq<end_of_turn>\n"

    with pytest.raises(ValueError, match="does not open a model turn"):
        build(tokenizer=Closed())


def test_the_entry_point_registers_the_provider_and_defers_torch():
    """Nothing here imports ``harness._registry``; Inspect loads it via the entry point."""
    model = get_model(
        "gemma-helper/prefills",
        config=GenerateConfig(max_connections=HELPER_FROZEN_CONFIG.max_connections),
    )
    assert isinstance(model.api, GemmaHelperAPI)


@pytest.mark.parametrize(
    "api, width",
    [
        pytest.param(GemmaHelperAPI, HELPER_BATCH, id="helper"),
        pytest.param(QwenLocalAPI, BATCH, id="target"),
    ],
)
def test_a_provider_admits_at_least_its_own_batch_width(api, width):
    """A limiter is sized by whichever call creates it first, so anything below the
    width deadlocks a batch that can never fill."""
    assert api(model_name="provider/name").max_connections() >= width


# --- the frozen decoding parameters ----------------------------------------


def test_every_frozen_parameter_reaches_the_config():
    """A HELPER_DECODING value that never reaches the config is neither policed nor
    logged. Equality both ways, so a parameter added to the rules fails here rather than
    running on the GPU unrecorded."""
    assert dict(HELPER_DECODING) == {
        "temperature": HELPER_FROZEN_CONFIG.temperature,
        "top_p": HELPER_FROZEN_CONFIG.top_p,
        "top_k": HELPER_FROZEN_CONFIG.top_k,
        "max_new_tokens": HELPER_FROZEN_CONFIG.max_tokens,
        **HELPER_FROZEN_CONFIG.extra_body,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"temperature": 0.7}, id="the-targets-temperature"),
        pytest.param({"top_p": 0.8}, id="the-targets-top_p"),
        pytest.param({"max_tokens": 1024}, id="the-targets-length-cap"),
        pytest.param({"extra_body": {"do_sample": True}}, id="extra-body-missing-min_p"),
        pytest.param({"extra_body": None}, id="extra-body-absent"),
        pytest.param({"max_connections": 8}, id="width-below-the-batch"),
        pytest.param({"cache": True}, id="cache"),
    ],
)
def test_rejects_any_drift_from_the_frozen_set(
    helper_tokenizer, fake_helper_prompts, overrides
):
    api = build(tokenizer=helper_tokenizer, module=object())
    with pytest.raises(ValueError, match="frozen config"):
        anyio.run(generate, api, [ChatMessageUser(content="q")], config(**overrides))


def test_a_seed_must_be_present(helper_tokenizer, fake_helper_prompts):
    api = build(tokenizer=helper_tokenizer, module=object())
    with pytest.raises(ValueError, match="no seed"):
        anyio.run(generate, api, [ChatMessageUser(content="q")], config(seed=None))


# --- what the helper is asked, and what it answers -------------------------


def test_only_a_single_request_is_rendered():
    assert require_one_request([ChatMessageUser(content="write me an opener")]) == (
        "write me an opener"
    )


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [ChatMessageUser(content="a"), ChatMessageAssistant(content="Sure,")],
            id="a-prefill-of-its-own",
        ),
        pytest.param(
            [ChatMessageSystem(content="be helpful"), ChatMessageUser(content="q")],
            id="system-message",
        ),
        pytest.param([], id="nothing-at-all"),
    ],
)
def test_anything_but_one_request_is_refused(messages):
    """Only the request is rendered, so anything else would vanish without a word."""
    with pytest.raises(ValueError, match="single user message"):
        require_one_request(messages)


def test_the_prompt_opens_the_models_turn(helper_tokenizer, fake_helper_prompts):
    api = build(tokenizer=helper_tokenizer, module=object())
    _, call = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert call.request["prompt"].endswith(MODEL_TURN_HEADER)
    assert not call.request["prompt"].startswith(helper_tokenizer.bos_token)
    assert call.request["seed"] == SEED


def test_the_completion_is_the_prefill(helper_tokenizer, fake_helper_prompts):
    api = build(tokenizer=helper_tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert output.completion == CONTINUATION


def test_the_frozen_decoding_reaches_the_forward_pass(
    helper_tokenizer, fake_helper_prompts
):
    """The two models run at different sampling parameters through one shared pass."""
    api = build(tokenizer=helper_tokenizer, module=object())
    anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert fake_helper_prompts.calls[0]["decoding"] is HELPER_DECODING


def test_metadata_carries_the_batch_a_draw_came_from(
    helper_tokenizer, fake_helper_prompts
):
    """A helper_seed alone does not reproduce a row; the batch it landed in does."""
    api = build(tokenizer=helper_tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert output.metadata["batch_size"] == 1
    assert output.metadata["batch_position"] == 0
    assert isinstance(output.metadata["batch_seed"], int)
    # Control tokens survive here and nowhere else, and the portfolio's validator
    # looks for them.
    assert output.metadata["raw_continuation"] != output.completion
    assert output.usage.output_tokens == 7


def test_truncation_at_the_prefill_cap_shows_up_as_a_stop_reason(
    helper_tokenizer, fake_helper_prompts, monkeypatch
):
    from generation.batched import Continuation

    def capped(model, tok, prompts, *, seed, decoding):
        return [
            Continuation(
                continuation="cut off",
                raw_continuation="cut off",
                prompt_tokens=3,
                new_tokens=HELPER_DECODING["max_new_tokens"],
            )
        ], 0.1

    monkeypatch.setattr("harness.helper_provider.generate_prompts", capped)
    api = build(tokenizer=helper_tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert output.stop_reason == "max_tokens"


def test_a_failed_forward_pass_still_reports_the_prompt(helper_tokenizer, monkeypatch):
    def boom(model, tok, prompts, *, seed, decoding):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("harness.helper_provider.generate_prompts", boom)
    api = build(tokenizer=helper_tokenizer, module=object())

    output, call = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert isinstance(output, RuntimeError)
    assert "q" in call.request["prompt"]
