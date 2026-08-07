"""The provider's contract with Inspect, exercised without torch or a GPU."""

from __future__ import annotations

import anyio
import pytest
from conftest import CONTINUATION
from generation.batched import Continuation
from generation.qwen import DECODING
from harness.provider import FROZEN_CONFIG, IN_FLIGHT, QwenLocalAPI, split_prefill
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


def config(**overrides) -> GenerateConfig:
    """model_copy rather than merge, so a test can unset a field as well as change it."""
    return FROZEN_CONFIG.model_copy(update={"seed": SEED, **overrides})


async def generate(api: QwenLocalAPI, messages, cfg):
    return await api.generate(messages, tools=[], tool_choice="none", config=cfg)


def run_eval(tmp_path, tokenizer, sample_input="q", **eval_kwargs):
    """One sample through the real task/provider stack, returning the written log."""
    from inspect_ai import Task, eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate as generate_solver

    task = Task(
        dataset=MemoryDataset([Sample(id="000/none", input=sample_input, target="")]),
        solver=generate_solver(),
        config=config(),
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


def test_generate_refuses_when_weightless():
    api = build()
    with pytest.raises(RuntimeError, match="holds no live module"):
        anyio.run(generate, api, [ChatMessageUser(content="hi")], config())


def test_a_broken_chat_template_fails_at_construction(tokenizer):
    tokenizer.enable_thinking_honoured = False
    with pytest.raises(ValueError, match="empty-thinking sentinel"):
        build(tokenizer=tokenizer)


def test_the_entry_point_registers_the_provider_and_defers_torch():
    """Nothing here imports `harness._registry`; Inspect loads it via the entry point."""
    model = get_model(
        "qwen-local/base", config=GenerateConfig(max_connections=IN_FLIGHT)
    )
    assert isinstance(model.api, QwenLocalAPI)
    assert model.api.max_connections() == IN_FLIGHT


# --- the frozen decoding parameters ----------------------------------------


def test_accepts_a_config_that_matches_the_frozen_set(tokenizer, fake_generate_prompts):
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())
    assert output.completion


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"temperature": 0.9}, id="named-field-override"),
        pytest.param({"top_p": 0.5}, id="top_p-override"),
        pytest.param(
            {"extra_body": {"min_p": 0.4, "do_sample": True}}, id="extra-body-override"
        ),
        pytest.param(
            {"extra_body": {"do_sample": True}}, id="extra-body-missing-min_p"
        ),
        pytest.param({"extra_body": None}, id="extra-body-absent"),
        pytest.param({"max_tokens": 256}, id="length-cap-override"),
        # Recorded in the log header, then dropped on the way to the GPU.
        pytest.param({"stop_seqs": ["\n\n"]}, id="stop-seqs"),
        pytest.param({"frequency_penalty": 0.5}, id="frequency-penalty"),
        pytest.param({"presence_penalty": 0.5}, id="presence-penalty"),
        pytest.param({"logit_bias": {1: 1.0}}, id="logit-bias"),
        pytest.param({"num_choices": 2}, id="num-choices"),
        pytest.param({"logprobs": True}, id="logprobs"),
        pytest.param({"reasoning_tokens": 100}, id="reasoning-tokens"),
        # Sets the batch width, so it moves composition and the text with it.
        pytest.param({"max_connections": 8}, id="pipeline-depth"),
        pytest.param({"adaptive_connections": True}, id="adaptive-concurrency"),
        # Standing bans: stale completions, and a different model generating.
        pytest.param({"cache": True}, id="cache"),
        pytest.param({"batch": 8}, id="provider-batch-api"),
        pytest.param({"fallback_models": ["mockllm/model"]}, id="fallback-model"),
    ],
)
def test_rejects_any_drift_from_the_frozen_set(
    tokenizer, fake_generate_prompts, overrides
):
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="frozen config"):
        anyio.run(
            generate,
            api,
            [ChatMessageUser(content="q")],
            config(**overrides),
        )


def test_every_frozen_parameter_reaches_the_config():
    """A DECODING value that never reaches the config is neither policed nor logged.

    Equality both ways, so a parameter added to DECODING fails here rather than
    running on the GPU unrecorded.
    """
    assert dict(DECODING) == {
        "temperature": FROZEN_CONFIG.temperature,
        "top_p": FROZEN_CONFIG.top_p,
        "top_k": FROZEN_CONFIG.top_k,
        "max_new_tokens": FROZEN_CONFIG.max_tokens,
        **FROZEN_CONFIG.extra_body,
    }


def test_a_seed_must_be_present(tokenizer, fake_generate_prompts):
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="no seed"):
        anyio.run(
            generate,
            api,
            [ChatMessageUser(content="q")],
            config(seed=None),
        )


def test_any_seed_value_is_accepted(tokenizer, fake_generate_prompts):
    """The provider must stay ignorant of how seeds are derived."""
    api = build(tokenizer=tokenizer, module=object())
    anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q")],
        config(seed=7),
    )
    assert fake_generate_prompts.calls[0]["seed"] == 7


# --- prefill ---------------------------------------------------------------


def test_a_trailing_assistant_message_is_the_prefill():
    message, prefill = split_prefill(
        [
            ChatMessageUser(content="q"),
            ChatMessageAssistant(content="Sure, here's how:"),
        ]
    )
    assert (message, prefill) == ("q", "Sure, here's how:")


def test_no_trailing_assistant_message_means_no_prefill():
    assert split_prefill([ChatMessageUser(content="q")]) == ("q", "")


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param(
            [ChatMessageUser(content="a"), ChatMessageUser(content="b")],
            id="multi-turn",
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


def test_a_failed_forward_pass_still_reports_the_prompt(monkeypatch, tokenizer):
    """Inspect logs a raised error with no request at all, and one failed pass fails
    every member of its batch."""

    def boom(model, tok, prompts, *, seed, decoding):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("harness.batching.generate_prompts", boom)
    api = build(tokenizer=tokenizer, module=object())

    output, call = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert isinstance(output, RuntimeError)
    assert "q" in call.request["prompt"]
    assert call.request["seed"] == SEED


def test_completion_is_the_continuation_and_response_is_the_whole_turn(
    tokenizer, fake_generate_prompts
):
    """Inspect appends completion to a conversation already ending with the prefill."""
    prefill = "Sure, here's how:"
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q"), ChatMessageAssistant(content=prefill)],
        config(),
    )
    assert output.completion == CONTINUATION
    assert output.metadata["prefill"] == prefill
    assert output.metadata["response"] == prefill + CONTINUATION


def test_the_transcript_carries_the_prefill_once(
    tmp_path, tokenizer, fake_generate_prompts
):
    """Analysis rebuilds the turn from messages; a doubled prefill is judged twice."""
    prefill = "Sure, here's how:"
    log = run_eval(
        tmp_path,
        tokenizer,
        sample_input=[
            ChatMessageUser(content="q"),
            ChatMessageAssistant(content=prefill),
        ],
    )

    messages = log.samples[0].messages
    assert [m.role for m in messages] == ["user", "assistant", "assistant"]
    assert "".join(m.text for m in messages if m.role == "assistant") == (
        prefill + CONTINUATION
    )


def test_an_unprefilled_response_is_the_completion(tokenizer, fake_generate_prompts):
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())

    assert output.completion == CONTINUATION
    assert output.metadata["response"] == CONTINUATION


# --- what the log carries --------------------------------------------------


def test_metadata_carries_raw_continuation_and_the_pad_cut_token_count(
    tokenizer, fake_generate_prompts
):
    """Neither survives elsewhere: completion is cleaned, and usage counts padding."""
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())
    assert output.metadata["raw_continuation"].endswith("<|im_end|>")
    assert output.metadata["new_tokens"] == 7
    assert output.metadata["prompt_tokens"] == 11
    assert output.metadata["thinking_leak"] is False
    assert output.usage.output_tokens == 7


def test_a_thinking_leak_is_flagged(tokenizer, monkeypatch):
    """`<think>` survives skip_special_tokens, so a leak is detectable but not visible."""

    def leaky(model, tok, prompts, *, seed, decoding):
        text = "<think>hmm</think> ok"
        return [
            Continuation(
                continuation=text, raw_continuation=text, prompt_tokens=3, new_tokens=5
            )
        ], 0.1

    monkeypatch.setattr("harness.batching.generate_prompts", leaky)
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())
    assert output.metadata["thinking_leak"] is True


def test_truncation_shows_up_as_a_stop_reason(tokenizer, monkeypatch):
    def truncated(model, tok, prompts, *, seed, decoding):
        return [
            Continuation(
                continuation="cut off",
                raw_continuation="cut off",
                prompt_tokens=3,
                new_tokens=DECODING["max_new_tokens"],
            )
        ], 0.1

    monkeypatch.setattr("harness.batching.generate_prompts", truncated)
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(generate, api, [ChatMessageUser(content="q")], config())
    assert output.stop_reason == "max_tokens"


def test_the_log_records_a_null_module(tmp_path, tokenizer, fake_generate_prompts):
    """A serializer added for the module would break this silently."""
    log = run_eval(tmp_path, tokenizer)

    assert log.status == "success"
    assert log.eval.model_args["module"] is None
    assert log.eval.model_args["tokenizer"] is None


def test_decoding_provenance_lands_in_the_plan_config(
    tmp_path, tokenizer, fake_generate_prompts
):
    """`plan.config` is the merged config that ran; `model_generate_config` is not.

    The latter holds model-level settings only, and is empty when every value comes
    from the task.
    """
    log = run_eval(tmp_path, tokenizer)

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
    tmp_path, tokenizer, fake_generate_prompts
):
    log = run_eval(tmp_path, tokenizer, temperature=0.9)

    assert log.status == "error"
    assert "frozen config" in str(log.error.message)
    assert log.plan.config.temperature == 0.9


def test_model_call_records_the_rendered_prompt(tokenizer, fake_generate_prompts):
    """The sentinel is what makes a prefill a continuation, so record it."""
    from generation.qwen import THINKING_SENTINEL

    api = build(tokenizer=tokenizer, module=object())
    _, call = anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q"), ChatMessageAssistant(content="Sure:")],
        config(),
    )
    assert call.request["prompt"].endswith(THINKING_SENTINEL + "Sure:")
    assert call.request["seed"] == SEED
    assert call.request["temperature"] == DECODING["temperature"]
