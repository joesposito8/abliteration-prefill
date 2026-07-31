"""The provider's contract with Inspect, exercised without torch or a GPU.

Three properties here are the ones that fail silently in production rather than
loudly, so each gets a test of its own: a weightless rebuild must construct and only
then refuse; the sampling parameters must be checked at the point of use, because an
eval() keyword argument beats the task config; and the prefill must come back out of
``completion`` exactly as it went in, since every downstream score is computed on the
continuation the scorer strips off it.
"""

from __future__ import annotations

import anyio
import pytest
from generation.qwen import SAMPLING
from harness import IN_FLIGHT
from harness.provider import QwenLocalAPI, split_prefill
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

SEED = 20260724


def build(tokenizer=None, module=None, max_new_tokens=512) -> QwenLocalAPI:
    return QwenLocalAPI(
        model_name="qwen-local/layer_22",
        module=module,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )


def config(frozen_config_kwargs, **overrides) -> GenerateConfig:
    kwargs = {"seed": SEED, "max_tokens": 512, **frozen_config_kwargs, **overrides}
    return GenerateConfig(**kwargs)


async def generate(api: QwenLocalAPI, messages, cfg):
    return await api.generate(messages, tools=[], tool_choice="none", config=cfg)


# --- construction ----------------------------------------------------------


def test_constructs_weightless_without_complaint():
    """A scoring pass rebuilds from the log, where model_args are null."""
    api = build()
    assert api.module is None and api.tokenizer is None


def test_generate_refuses_when_weightless(frozen_config_kwargs):
    api = build()
    with pytest.raises(RuntimeError, match="holds no live module"):
        anyio.run(generate, api, [ChatMessageUser(content="hi")], config(frozen_config_kwargs))


def test_a_broken_chat_template_fails_at_construction(tokenizer):
    """Cheap now beats discovering it after GPU-hours."""
    tokenizer.enable_thinking_honoured = False
    with pytest.raises(ValueError, match="empty-thinking sentinel"):
        build(tokenizer=tokenizer)


def test_registered_provider_resolves_and_defers_torch():
    """The registry hands back a factory, so importing the package is cheap."""
    model = get_model(
        "qwen-local/base",
        max_new_tokens=512,
        config=GenerateConfig(max_connections=IN_FLIGHT),
    )
    assert isinstance(model.api, QwenLocalAPI)
    assert model.api.max_connections() == IN_FLIGHT


# --- the frozen sampling parameters ----------------------------------------


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
        pytest.param({"extra_body": {"min_p": 0.4, "do_sample": True}}, id="extra-body-override"),
        pytest.param({"extra_body": {"do_sample": True}}, id="extra-body-missing-min_p"),
        pytest.param({"extra_body": None}, id="extra-body-absent"),
    ],
)
def test_rejects_any_drift_from_the_frozen_set(
    tokenizer, fake_generate_batch, frozen_config_kwargs, overrides
):
    """This is what an `eval(temperature=...)` kwarg looks like by the time it arrives."""
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="frozen sampling parameters"):
        anyio.run(
            generate,
            api,
            [ChatMessageUser(content="q")],
            config(frozen_config_kwargs, **overrides),
        )


def test_the_comparison_is_against_qwen_sampling_itself(
    tokenizer, fake_generate_batch, frozen_config_kwargs, monkeypatch
):
    """Closing the chain: the assertion reads the same constant generate_batch uses.

    If the two could diverge, the config could be validated against one set of
    parameters while the forward pass ran under another.
    """
    monkeypatch.setattr("generation.qwen.SAMPLING", {**SAMPLING, "temperature": 0.123})
    api = build(tokenizer=tokenizer, module=object())
    with pytest.raises(ValueError, match="frozen sampling parameters"):
        anyio.run(
            generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs)
        )


def test_max_tokens_must_agree_with_the_provider(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    api = build(tokenizer=tokenizer, module=object(), max_new_tokens=512)
    with pytest.raises(ValueError, match="max_tokens disagreement"):
        anyio.run(
            generate,
            api,
            [ChatMessageUser(content="q")],
            config(frozen_config_kwargs, max_tokens=256),
        )


def test_missing_max_new_tokens_is_a_construction_argument_error(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    api = build(tokenizer=tokenizer, module=object(), max_new_tokens=None)
    with pytest.raises(ValueError, match="without max_new_tokens"):
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


def test_any_seed_value_is_accepted(tokenizer, fake_generate_batch, frozen_config_kwargs):
    """The provider must not know how the study derives seeds."""
    api = build(tokenizer=tokenizer, module=object())
    anyio.run(
        generate, api, [ChatMessageUser(content="q")], config(frozen_config_kwargs, seed=7)
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


def test_multi_turn_input_is_refused_rather_than_silently_truncated():
    with pytest.raises(ValueError, match="exactly one user message"):
        split_prefill([ChatMessageUser(content="a"), ChatMessageUser(content="b")])


def test_completion_is_prefill_plus_continuation(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """The scorer strips the prefill back off this, so it must be a clean prefix."""
    prefill = "Sure, here's how:"
    api = build(tokenizer=tokenizer, module=object())
    output, _ = anyio.run(
        generate,
        api,
        [ChatMessageUser(content="q"), ChatMessageAssistant(content=prefill)],
        config(frozen_config_kwargs),
    )
    assert output.completion.startswith(prefill)
    assert output.completion == prefill + " continuation for q"
    assert output.metadata["prefill"] == prefill


# --- what the log carries --------------------------------------------------


def test_metadata_carries_raw_continuation_and_the_pad_cut_token_count(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """Neither survives anywhere else: completion is cleaned, and usage is padded."""
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
    """`<think>` survives skip_special_tokens, so a leak is detectable but not obvious."""
    from generation.qwen import Generation

    def leaky(model, tok, messages, *, seed, prefill="", max_new_tokens=512):
        return [
            Generation(
                message=messages[0],
                prefill=prefill,
                output="<think>hmm</think> ok",
                continuation="<think>hmm</think> ok",
                raw_continuation="<think>hmm</think> ok",
                seed=seed,
                prompt_tokens=3,
                new_tokens=5,
                max_new_tokens=max_new_tokens,
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
    from generation.qwen import Generation

    def truncated(model, tok, messages, *, seed, prefill="", max_new_tokens=512):
        return [
            Generation(
                message=messages[0],
                prefill=prefill,
                output="cut off",
                continuation="cut off",
                raw_continuation="cut off",
                seed=seed,
                prompt_tokens=3,
                new_tokens=max_new_tokens,
                max_new_tokens=max_new_tokens,
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
    """Weight hygiene, end to end through a real eval.

    A live module reaches the constructor but cannot be serialised, so Inspect's log
    copy writes null. That is the guarantee no abliterated tensor can reach a log
    file, and it is worth asserting rather than assuming — a serializer added here
    would break it silently.
    """
    from inspect_ai import Task, eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate as generate_solver

    task = Task(
        dataset=MemoryDataset([Sample(id="000/none", input="q", target="")]),
        solver=generate_solver(),
        config=GenerateConfig(**{"seed": SEED, "max_tokens": 512, **frozen_config_kwargs}),
    )
    log = eval(
        task,
        model="qwen-local/base",
        model_args={"module": object(), "tokenizer": tokenizer, "max_new_tokens": 512},
        log_dir=str(tmp_path),
        score=False,
    )[0]

    assert log.status == "success"
    assert log.eval.model_args["module"] is None
    assert log.eval.model_args["tokenizer"] is None
    # max_new_tokens is a plain int, so unlike the weights it survives and can be
    # read back — which is what lets a scoring pass rebuild a consistent provider.
    assert log.eval.model_args["max_new_tokens"] == 512

    # The decoding provenance lives in EvalPlan.config, which is the MERGED config
    # that actually ran. EvalSpec.model_generate_config holds only the model-level
    # config — what eval(seed=...) or get_model(config=...) set — and is empty when
    # everything comes from Task.config, as it does here.
    assert log.plan.config.seed == SEED
    assert log.plan.config.temperature == SAMPLING["temperature"]
    assert log.plan.config.top_p == SAMPLING["top_p"]
    assert log.plan.config.top_k == SAMPLING["top_k"]
    assert log.plan.config.max_tokens == 512
    assert log.plan.config.extra_body == {
        "min_p": SAMPLING["min_p"],
        "do_sample": SAMPLING["do_sample"],
    }
    assert log.eval.model_generate_config.seed is None


def test_an_eval_kwarg_beats_the_task_config_and_the_provider_catches_it(
    tmp_path, tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """Why the sampling check lives at the point of use rather than in the task.

    An eval() keyword argument wins the merge, so a task that declares the frozen
    parameters is not by itself a guarantee that they are what ran.
    """
    from inspect_ai import Task, eval
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.solver import generate as generate_solver

    task = Task(
        dataset=MemoryDataset([Sample(id="000/none", input="q", target="")]),
        solver=generate_solver(),
        config=GenerateConfig(**{"seed": SEED, "max_tokens": 512, **frozen_config_kwargs}),
    )
    log = eval(
        task,
        model="qwen-local/base",
        model_args={"module": object(), "tokenizer": tokenizer, "max_new_tokens": 512},
        log_dir=str(tmp_path),
        score=False,
        temperature=0.9,  # would silently change the sampled distribution
    )[0]

    assert log.status == "error"
    assert "frozen sampling parameters" in str(log.error.message)
    # It really did win the merge — the task's 0.7 is not what arrived.
    assert log.plan.config.temperature == 0.9


def test_model_call_records_the_rendered_prompt(
    tokenizer, fake_generate_batch, frozen_config_kwargs
):
    """The sentinel is what makes a prefill a continuation, so record it, not the raw turn."""
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
    assert call.request["temperature"] == SAMPLING["temperature"]
