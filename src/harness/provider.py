"""An Inspect provider for the in-memory target model.

The built-in ``hf`` provider cannot drive this study for two reasons. It loads a
model from a *name*, and ours is a mutable in-memory object with one state per
abliteration condition and no name at all; and it renders prompts with
``add_generation_prompt=True``, which closes the assistant turn instead of
continuing it, so a prefill would be read as context to respond to rather than
text to complete.

Weights arrive as ``model_args`` on the driver's ``eval()`` call. They reach this
constructor as **live objects**, but Inspect serialises its copy of ``model_args``
for the log with a fallback that writes ``null`` for anything it cannot represent.
So ``EvalSpec.model_args`` records ``"module": null`` — deliberately. That is the
mechanism guaranteeing no abliterated weight tensor can ever reach a log file, and
it must stay: **do not add a serializer for it.**

The consequence is that anything rebuilt *from* a log — a scoring pass, a retry —
constructs weightless. That is a legitimate state, so ``__init__`` accepts it
without complaint and only :meth:`generate` raises.

One hazard this class relies on the caller to avoid. Inspect memoises providers on
a key built from the model name plus the *serialised* ``model_args``; because every
unserialisable object serialises to ``None``, two different live modules under one
model name produce an identical key and the second ``get_model()`` silently returns
the first provider, still holding the first module. This study is safe only because
the condition is encoded in the model *name*. Anything that breaks that invariant
generates one condition's text and labels it as another's, with nothing to flag it.
"""

from __future__ import annotations

from typing import Any

import anyio
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from . import IN_FLIGHT

# The five parameters `qwen.SAMPLING` freezes, in the shape a GenerateConfig
# carries them: three have named fields, two have to ride in extra_body because
# GenerateConfig has no field for them and rejects unknown ones outright.
_NAMED = ("temperature", "top_p", "top_k")
_EXTRA = ("do_sample", "min_p")


class QwenLocalAPI(ModelAPI):
    """Generates from a module handed in as a model argument.

    The model name carries the condition, but this class never parses it and never
    imports the study's selection code. Which weights it holds is the driver's
    business, so the provider cannot mislabel a condition and can be tested offline
    against an unedited model.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        *,
        module: Any = None,
        tokenizer: Any = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        # super() captures initial_api_key before the key-override path runs.
        super().__init__(
            model_name=model_name, base_url=base_url, api_key=api_key, config=config
        )
        self.module = module
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        # Render once now, when a tokenizer is present, so a chat-template
        # regression costs milliseconds instead of surfacing after GPU-hours.
        if tokenizer is not None:
            from generation.qwen import build_prompt

            build_prompt(tokenizer, "startup check")

    def max_connections(self) -> int:
        """Default only for a bare ``get_model()``; ``Task.config`` sets the real value.

        A concurrency limiter's size is fixed by whichever call creates it and is
        never re-read afterwards, so returning the batch width here would let a
        stray bare ``get_model()`` pin the limiter below what the batcher needs to
        fill a batch — a deadlock that would look like a hang.
        """
        return IN_FLIGHT

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        message, prefill = split_prefill(input)
        sampling = require_frozen_sampling(config)
        max_new_tokens = self.require_max_new_tokens(config)
        seed = require_seed(config)

        if self.module is None or self.tokenizer is None:
            raise RuntimeError(
                f"{type(self).__name__} holds no live module, so it can generate "
                "nothing. Model arguments reach a provider as real objects but are "
                "recorded in the log as null, so anything rebuilt from a log — a "
                "scoring or retry pass — arrives weightless. Pass module= and "
                "tokenizer= as model_args on the eval() that generates."
            )

        from generation.qwen import build_prompt, contains_thinking, generate_batch

        # Rendered here as well as inside generate_batch, which costs microseconds
        # and buys two things: the exact prompt in the ModelCall record, and a
        # template regression that raises per sample before any GPU work.
        prompt = build_prompt(self.tokenizer, message, prefill)

        generations, seconds = await anyio.to_thread.run_sync(
            lambda: generate_batch(
                self.module,
                self.tokenizer,
                [message],
                seed=seed,
                prefill=prefill,
                max_new_tokens=max_new_tokens,
            )
        )
        generation = generations[0]

        output = ModelOutput.from_content(
            model=self.model_name,
            # prefill + continuation, i.e. the whole assistant turn. Inspect appends
            # a generated message after a trailing prefill rather than merging it, so
            # returning the continuation alone would force every scorer to reassemble
            # the string a second time.
            content=generation.output,
            stop_reason=(
                "max_tokens" if generation.new_tokens >= max_new_tokens else "stop"
            ),
        )
        output.usage = ModelUsage(
            input_tokens=generation.prompt_tokens,
            output_tokens=generation.new_tokens,
            total_tokens=generation.prompt_tokens + generation.new_tokens,
        )
        output.metadata = {
            # Control tokens intact — the only input to the leak check, and absent
            # from ModelOutput.completion, which is the cleaned string.
            "raw_continuation": generation.raw_continuation,
            # Cut at the pad token, so this is the row's own count and not the
            # padded batch width. The truncation rate is computed from it.
            "new_tokens": generation.new_tokens,
            "prompt_tokens": generation.prompt_tokens,
            "thinking_leak": contains_thinking(generation.raw_continuation),
            "prefill": prefill,
            "seed": seed,
        }
        return output, ModelCall.create(
            request={
                "prompt": prompt,
                "prefill": prefill,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                **sampling,
            },
            response={
                "continuation": generation.continuation,
                "new_tokens": generation.new_tokens,
                "batch_seconds": seconds,
            },
        )

    def require_max_new_tokens(self, config: GenerateConfig) -> int:
        """The generation length, agreed between the config and this provider.

        ``max_new_tokens`` is not part of ``qwen.SAMPLING`` — it is a plain argument
        to ``generate_batch`` — so it is pinned by being passed here as a model
        argument and checked against what the config carries. Unlike the weights it
        serialises cleanly, so it survives into the log and back out of it.
        """
        if self.max_new_tokens is None:
            raise ValueError(
                f"{type(self).__name__} was constructed without max_new_tokens. Pass "
                "it as a model_arg so the generation length is pinned by the same "
                "value the config declares."
            )
        if config.max_tokens != self.max_new_tokens:
            raise ValueError(
                f"max_tokens disagreement: the config asks for {config.max_tokens} "
                f"but this provider was built for {self.max_new_tokens}. These must "
                "match, or the log header would record a length the model never used."
            )
        return self.max_new_tokens


def split_prefill(input: list[ChatMessage]) -> tuple[str, str]:
    """Split a sample's messages into its user turn and an optional prefill.

    A prefill is a trailing assistant message — the start of the model's own turn,
    to be continued rather than responded to. Inspect passes it through verbatim
    (deep-copied, with only ``source`` set), so it arrives here as written.
    """
    if not input:
        raise ValueError("no messages to generate from")

    prefill = ""
    messages = list(input)
    if isinstance(messages[-1], ChatMessageAssistant):
        prefill = messages.pop().text

    user = [m for m in messages if m.role == "user"]
    if len(user) != 1:
        raise ValueError(
            f"expected exactly one user message, got {len(user)}. This task builds "
            "single-turn samples; a multi-turn input would silently drop context."
        )
    return user[0].text, prefill


def require_frozen_sampling(config: GenerateConfig) -> dict[str, Any]:
    """Check the config's sampling parameters against the frozen set, and return them.

    Assert-match rather than reject-and-substitute. ``Task.config`` carries the whole
    tuple so that every value is real provenance in ``eval.model_generate_config``;
    this then confirms nothing rewrote it on the way through. The check belongs here,
    at the point of use, because an ``eval()`` keyword argument beats ``Task.config``
    in the merge and would otherwise change the sampled distribution silently.

    Compared against ``qwen.SAMPLING`` specifically because that is the same object
    ``generate_batch`` splats into ``model.generate``. Config, assertion and forward
    pass therefore all read one constant, so agreement here means agreement there.
    """
    from generation.qwen import SAMPLING

    extra = config.extra_body or {}
    declared = {name: getattr(config, name) for name in _NAMED}
    declared.update({name: extra.get(name) for name in _EXTRA})

    frozen = dict(SAMPLING)
    if declared != frozen:
        differing = {
            name: (declared.get(name), frozen.get(name))
            for name in frozen.keys() | declared.keys()
            if declared.get(name) != frozen.get(name)
        }
        raise ValueError(
            "generate config does not match the frozen sampling parameters "
            f"(name: got, expected): {differing}. min_p and do_sample have no "
            "GenerateConfig field and must be passed in extra_body."
        )
    return declared


def require_seed(config: GenerateConfig) -> int:
    """The seed for this call, which must be present but may be any value.

    Requiring presence rather than a particular value keeps the provider ignorant of
    how the study derives seeds. The log header records whatever actually ran, so a
    driver that passes the wrong one produces a detectable disagreement rather than a
    prevented one.
    """
    if config.seed is None:
        raise ValueError(
            "no seed in the generate config. Batched sampling consumes one RNG "
            "stream, so an unseeded run cannot be characterised afterwards at all."
        )
    return config.seed
