"""An Inspect provider for an in-memory Qwen3 model.

The built-in ``hf`` provider cannot be used: it loads a model from a name, and this
one is a mutable object passed in by the caller; and it renders with
``add_generation_prompt=True``, which closes the assistant turn, so a prefill would
be answered rather than continued.

The module and tokenizer arrive as ``model_args``, reaching this constructor as live
objects but serialising to ``null`` in the log, since JSON cannot hold them.

Construction must therefore succeed without them: ``score()`` rebuilds the model
from the log before running any scorer, so demanding weights here would break every
grading pass. Only :meth:`generate` refuses.

Inspect memoises providers on the model name plus the *serialised* ``model_args``,
and everything unserialisable serialises to ``None``. Two different modules under
one model name therefore collide, and the second caller silently gets the first
module. Encoding the condition in the model name is what avoids this.
"""

from __future__ import annotations

from typing import Any

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

from .batching import IN_FLIGHT, BatchGenerator


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
    ) -> None:
        # No **kwargs, so a misspelled model_arg is a TypeError here rather than a
        # missing-module error much later.
        super().__init__(
            model_name=model_name, base_url=base_url, api_key=api_key, config=config
        )
        self.module = module
        self.tokenizer = tokenizer
        self._batcher = BatchGenerator(module, tokenizer)

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
        decoding = require_frozen_decoding(config)
        seed = require_seed(config)

        if self.module is None or self.tokenizer is None:
            raise RuntimeError(
                f"{type(self).__name__} holds no live module. Model arguments are "
                "recorded in the log as null, so anything rebuilt from a log arrives "
                "weightless; pass module= and tokenizer= to the eval that generates."
            )

        from generation.qwen import build_prompt, contains_thinking

        prompt = build_prompt(self.tokenizer, message, prefill)
        row = await self._batcher.submit(message, prefill, seed)
        generation = row.generation

        output = ModelOutput.from_content(
            model=self.model_name,
            # The whole assistant turn. Inspect appends the generated message after a
            # prefill rather than merging, so returning the continuation alone would
            # make every scorer reassemble the string again.
            content=generation.output,
            stop_reason=(
                "max_tokens"
                if generation.new_tokens >= generation.max_new_tokens
                else "stop"
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
            # Composition is arrival-dependent, so it is recorded, not replayable.
            "batch_size": row.batch_size,
            "batch_index": row.batch_index,
        }
        return output, ModelCall.create(
            request={
                "prompt": prompt,
                "prefill": prefill,
                "seed": seed,
                **decoding,
            },
            response={
                "continuation": generation.continuation,
                "new_tokens": generation.new_tokens,
                "batch_seconds": row.seconds,
            },
        )


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

    if len(messages) != 1 or messages[0].role != "user":
        raise ValueError(
            f"expected a single user message, got roles {[m.role for m in messages]}; "
            "only the user turn is rendered, so anything else would be dropped."
        )
    return messages[0].text, prefill


def require_frozen_decoding(config: GenerateConfig) -> dict[str, Any]:
    """Check the config's decoding parameters against the frozen set, and return them.

    Checked here rather than trusted from the task, because an ``eval()`` keyword
    argument wins the config merge and would otherwise change the text silently.
    Compared against ``qwen.DECODING`` because that is the object ``generate_batch``
    passes to ``model.generate``, so agreement here means agreement at the forward
    pass.
    """
    from generation.qwen import DECODING

    extra = config.extra_body or {}
    declared = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        # GenerateConfig names the length cap differently from transformers.
        "max_new_tokens": config.max_tokens,
        # No GenerateConfig field exists for these two, and unknown fields are
        # rejected, so a task has to pass them in extra_body.
        "do_sample": extra.get("do_sample"),
        "min_p": extra.get("min_p"),
    }

    frozen = dict(DECODING)
    if declared != frozen:
        differing = {
            name: (declared.get(name), frozen.get(name))
            for name in frozen.keys() | declared.keys()
            if declared.get(name) != frozen.get(name)
        }
        raise ValueError(
            "generate config does not match the frozen decoding parameters "
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
