"""An Inspect provider for an in-memory Qwen3 model.

The built-in ``hf`` provider cannot be used: it loads a model from a name, and this
one is a mutable object passed in by the caller; and it renders with
``add_generation_prompt=True``, which closes the assistant turn, so a prefill would
be answered rather than continued.

The module and tokenizer arrive as ``model_args``. They reach this constructor as
live objects but serialise to ``null`` in the log, which is what keeps edited
weights out of log files — **do not add a serializer for them.** The cost is that
anything rebuilt from a log arrives weightless, so construction must tolerate it and
only :meth:`generate` refuses.

Inspect memoises providers on the model name plus the *serialised* ``model_args``,
and everything unserialisable serialises to ``None``. Two different modules under
one model name therefore collide, and the second caller silently gets the first
module. Encoding the condition in the model name is what avoids this.
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

IN_FLIGHT = 64

# min_p and do_sample have no GenerateConfig field, and unknown fields are rejected,
# so they travel in extra_body while the other three are named.
_NAMED = ("temperature", "top_p", "top_k")
_EXTRA = ("do_sample", "min_p")


class QwenLocalAPI(ModelAPI):
    """Generates from a module handed in as a model argument.

    The model name carries the condition, but this class never parses it: which
    weights it holds is the caller's business, so it cannot mislabel a condition and
    can be tested against an unedited model.
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
    ) -> None:
        # No **kwargs, so a misspelled model_arg is a TypeError here rather than a
        # missing-module error much later.
        super().__init__(
            model_name=model_name, base_url=base_url, api_key=api_key, config=config
        )
        self.module = module
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        if tokenizer is not None:
            # Fail on a broken chat template now rather than after hours of generation.
            from generation.qwen import build_prompt

            build_prompt(tokenizer, "startup check")

    def max_connections(self) -> int:
        """Fallback for a bare ``get_model()``; ``Task.config`` sets the real value.

        A limiter's size is fixed by whichever call creates it first, so a value
        below the batch width could deadlock a batch that can never fill.
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
                f"{type(self).__name__} holds no live module. Model arguments are "
                "recorded in the log as null, so anything rebuilt from a log arrives "
                "weightless; pass module= and tokenizer= to the eval that generates."
            )

        from generation.qwen import build_prompt, contains_thinking, generate_batch

        prompt = build_prompt(self.tokenizer, message, prefill)

        # generate_batch blocks on the GPU, which would stall every other sample
        # sharing this event loop.
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
            # The whole assistant turn. Inspect appends the generated message after a
            # prefill rather than merging, so returning the continuation alone would
            # make every scorer reassemble the string again.
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
            # Control tokens intact, unlike completion — the leak check needs them.
            "raw_continuation": generation.raw_continuation,
            # Cut at the pad token, so this row's own count and not the batch width.
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
        """The generation length, which the config and this provider must agree on.

        Not part of ``qwen.SAMPLING``, so it is pinned by being passed in and checked.
        """
        if self.max_new_tokens is None:
            raise ValueError(
                f"{type(self).__name__} was constructed without max_new_tokens; pass "
                "it as a model_arg so the generation length is pinned."
            )
        if config.max_tokens != self.max_new_tokens:
            raise ValueError(
                f"max_tokens disagreement: config asks for {config.max_tokens}, this "
                f"provider was built for {self.max_new_tokens}."
            )
        return self.max_new_tokens


def split_prefill(input: list[ChatMessage]) -> tuple[str, str]:
    """Split a sample's messages into its user turn and an optional prefill.

    A prefill is a trailing assistant message: the start of the model's own turn,
    to be continued rather than responded to.
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
            f"expected exactly one user message, got {len(user)}; a multi-turn input "
            "would silently drop context."
        )
    return user[0].text, prefill


def require_frozen_sampling(config: GenerateConfig) -> dict[str, Any]:
    """Check the config's sampling parameters against the frozen set, and return them.

    Checked here rather than trusted from the task, because an ``eval()`` keyword
    argument wins the config merge and would otherwise change the sampled
    distribution silently. Compared against ``qwen.SAMPLING`` because that is the
    object ``generate_batch`` passes to ``model.generate``, so agreement here means
    agreement at the forward pass.
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

    Presence rather than a particular value, so the provider stays ignorant of how
    seeds are derived and a disagreement is detectable in the log instead.
    """
    if config.seed is None:
        raise ValueError(
            "no seed in the generate config; batched sampling consumes one RNG "
            "stream, so an unseeded run cannot be characterised afterwards."
        )
    return config.seed
