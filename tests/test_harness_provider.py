"""The provider's contract with Inspect, exercised without torch or a GPU."""

from __future__ import annotations

import anyio
import pytest
from generation.qwen import DECODING, Generation
from harness.provider import IN_FLIGHT, QwenLocalAPI, split_prefill
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

SEED = 20260724


def build(tokenizer=None, module=None) -> QwenLocalAPI:
    return QwenLocalAPI(
        model_name="qwen-local/layer_22", module=module, tokenizer=tokenizer
    )


def config(frozen_config_kwargs, **overrides) -> GenerateConfig:
    kwargs = {"seed": SEED, **frozen_config_kwargs, **overrides}
    return GenerateConfig(**kwargs)


async def generate(api: QwenLocalAPI, messages, cfg):
    return await api.generate(messages, tools=[], tool_choice="none", config=cfg)


def run_eval(tmp_path, tokenizer, frozen_config_kwargs, **eval_kwargs):
    """One sample through the real task/provider stack, returning the written log."""
    from inspect_ai import Task, eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate as generate_solver

    task = Task(
        dataset=MemoryDataset([Sample(id="000/none", input="q", target="")]),
        solver=generate_solver(),
        config=GenerateConfig(seed=SEED, **frozen_config_kwargs),
    )
    return eval(
        task,
        model="qwen-local/base",
        model_args={"module": object(), "tokenizer": tokenizer},
        log_dir=str(tmp_path),
        score=False,
        **eval_kwargs,
    )[0]


# --- construction ----------------------------------------------------------


def test_constructs_weightless_without_complaint():
    """`score()` rebuilds the model from the log, where model_args are null."""
    api = build()
    assert api.module is None and api.tokenizer is None


def test_a_misspelled_model_arg_is_refused(tokenizer):
    """Otherwise it is dropped and resurfaces as a missing-module error much later."""
    with pytest.raises(TypeError, match="toknizer"):
        get_model("qwen-local/typo-check", toknizer=tokenizer)


def test_generate_refuses_when_weightless(frozen_config_kwargs):
    api = build()
    with pytest.raises(RuntimeError, match="holds no live module"):
        anyio.run(
            generate, api, [ChatMessageUser(content="hi")], config(frozen_config_kwargs)
        )


def test_a_broken_chat_template_fails_at_construction(tokenizer):
    tokenizer.enable_thinking_honoured = False
    with pytest.raises(ValueError, match="empty-thinking sentinel"):
        build(tokenizer=tokenizer)


def test_the_entry_point_registers_the_provider_and_defers_torch():
    """Nothing here imports `harness._registry`; Inspect loads it via the entry point."""
    model = get_model("qwen-local/base", config=GenerateConfig(max_connections=IN_FLIGHT))
    assert isinstance(model.api, QwenLocalAPI)
    assert model.api.max_connections() == IN_FLIGHT


# --- the frozen decoding parameters ----------------------------------------


def test_accepts_a_config_that_matches_the_frozen_set(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs)
    )
    assert output.completion


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"temperature": 0.9}, id="named-field-override"),
        pytest.param({"top_p": 0.5}, id="top_p-override"),
        pytest.param(
            {"extra_body": {"min_p": 0.4, "do_sample": True}}, id="extra-body-override"
        ),
        pytest.param({"extra_body": {"do_sample": True}}, id="extra-body-missing-min_p"),
        pytest.param({"extra_body": None}, id="extra-body-absent"),
        pytest.param({"max_tokens": 256}, id="length-cap-override"),
    ],
)
def test_rejects_any_drift_from_the_frozen_set(
    tokenizer, fake_generate_batch, frozen_config_kwargs, overrides
):
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="frozen decoding parameters"):
        anyio.run(
            generate,
            api,
            [ChatMessageUser(content="q")],
            config(frozen_config_kwargs, **overrides),
        )


def test_the_comparison_is_against_qwen_decoding_itself(
    tokenizer, fake_generate_batch, frozen_config_kwargs, monkeypatch
):
    """Were they separate constants, the check could pass while the GPU ran otherwise."""
    monkeypatch.setattr("generation.qwen.DECODING", {**DECODING, "temperature": 0.123})
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="frozen decoding parameters"):
        anyio.run(
            generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs)
        )


def test_a_seed_must_be_present(tokenizer, fake_generate_batch, frozen_config_kwargs):
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="no seed"):
        anyio.run(
            generate,
            api,
            [ChatMessageUser(content="q")],
            config(frozen_config_kwargs, seed=None),
        )


def test_any_seed_value_is_accepted(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """The provider must stay ignorant of how seeds are derived."""
    api = build(tokenizer=tokenizer, module=object())
    anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q")],
        config(frozen_config_kwargs, seed=7),
    )
    assert fake_generate_batch[0]["seed"] == 7


# --- prefill ---------------------------------------------------------------


def test_a_trailing_assistant_message_is_the_prefill():
    message, prefill = split_prefill(
        [ChatMessageUser(content="q"), ChatMessageAssistant(content="Sure, here's how:")]
    )
    assert (message, prefill) == ("q", "Sure, here's how:")


def test_no_trailing_assistant_message_means_no_prefill():
    assert split_prefill([ChatMessageUser(content="q")]) == ("q", "")


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [ChatMessageUser(content="a"), ChatMessageUser(content="b")], id="multi-turn"
        ),
        pytest.param(
            [ChatMessageSystem(content="be helpful"), ChatMessageUser(content="q")],
            id="system-message",
        ),
    ],
)
def test_anything_but_one_user_turn_is_refused(messages):
    """Only the user turn is rendered, so the rest would vanish without a word."""
    with pytest.raises(ValueError, match="single user message"):
        split_prefill(messages)


def test_completion_is_prefill_plus_continuation(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """The scorer strips the prefill back off, so it must be a clean prefix."""
    prefill = "Sure, here's how:"
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q"), ChatMessageAssistant(content=prefill)],
        config(frozen_config_kwargs),
    )
    assert output.completion == prefill + " continuation for q"
    assert output.metadata["prefill"] == prefill


# --- what the log carries --------------------------------------------------


def test_metadata_carries_raw_continuation_and_the_pad_cut_token_count(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """Neither survives elsewhere: completion is cleaned, and usage counts padding."""
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs)
    )
    assert output.metadata["raw_continuation"].endswith("<|im_end|>")
    assert output.metadata["new_tokens"] == 7
    assert output.metadata["prompt_tokens"] == 11
    assert output.metadata["thinking_leak"] is False
    assert output.usage.output_tokens == 7


def test_a_thinking_leak_is_flagged(tokenizer, frozen_config_kwargs, monkeypatch):
    """`<think>` survives skip_special_tokens, so a leak is detectable but not visible."""

    def leaky(model, tok, messages, *, seed, prefill="", decoding=DECODING):
        text = "<think>hmm</think> ok"
        return [
            Generation(
                message=messages[0],
                prefill=prefill,
                output=text,
                continuation=text,
                raw_continuation=text,
                seed=seed,
                prompt_tokens=3,
                new_tokens=5,
                max_new_tokens=decoding["max_new_tokens"],
            )
        ], 0.1

    monkeypatch.setattr("generation.qwen.generate_batch", leaky)
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs)
    )
    assert output.metadata["thinking_leak"] is True


def test_truncation_shows_up_as_a_stop_reason(
    tokenizer, frozen_config_kwargs, monkeypatch
):
    def truncated(model, tok, messages, *, seed, prefill="", decoding=DECODING):
        return [
            Generation(
                message=messages[0],
                prefill=prefill,
                output="cut off",
                continuation="cut off",
                raw_continuation="cut off",
                seed=seed,
                prompt_tokens=3,
                new_tokens=decoding["max_new_tokens"],
                max_new_tokens=decoding["max_new_tokens"],
            )
        ], 0.1

    monkeypatch.setattr("generation.qwen.generate_batch", truncated)
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs)
    )
    assert output.stop_reason == "max_tokens"


def test_the_log_records_a_null_module(
    tmp_path, tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """A serializer added for the module would break this silently."""
    log = run_eval(tmp_path, tokenizer, frozen_config_kwargs)

    assert log.status == "success"
    assert log.eval.model_args["module"] is None
    assert log.eval.model_args["tokenizer"] is None


def test_decoding_provenance_lands_in_the_plan_config(
    tmp_path, tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """`plan.config` is the merged config that ran; `model_generate_config` is not.

    The latter holds model-level settings only, and is empty when every value comes
    from the task.
    """
    log = run_eval(tmp_path, tokenizer, frozen_config_kwargs)

    assert log.plan.config.seed == SEED
    assert log.plan.config.temperature == DECODING["temperature"]
    assert log.plan.config.top_p == DECODING["top_p"]
    assert log.plan.config.top_k == DECODING["top_k"]
    assert log.plan.config.max_tokens == DECODING["max_new_tokens"]
    assert log.plan.config.extra_body == {
        "min_p": DECODING["min_p"],
        "do_sample": DECODING["do_sample"],
    }
    assert log.eval.model_generate_config.seed is None


def test_an_eval_kwarg_beats_the_task_config_and_the_provider_catches_it(
    tmp_path, tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """Why the sampling check sits at the point of use rather than in the task."""
    log = run_eval(tmp_path, tokenizer, frozen_config_kwargs, temperature=0.9)

    assert log.status == "error"
    assert "frozen decoding parameters" in str(log.error.message)
    assert log.plan.config.temperature == 0.9


def test_model_call_records_the_rendered_prompt(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """The sentinel is what makes a prefill a continuation, so record it."""
    from generation.qwen import THINKING_SENTINEL

    api = build(tokenizer=tokenizer, module=object())
    _, call = anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q"), ChatMessageAssistant(content="Sure:")],
        config(frozen_config_kwargs),
    )
    assert call.request["prompt"].endswith(THINKING_SENTINEL + "Sure:")
    assert call.request["seed"] == SEED
    assert call.request["temperature"] == DECODING["temperature"]
