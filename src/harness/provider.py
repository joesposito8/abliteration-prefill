"""An Inspect provider for an in-memory Qwen3 model.

The built-in ``hf`` provider loads a model from a name, and this one is a mutable
object passed in by the caller; it also renders with ``add_generation_prompt=True``,
which closes the assistant turn so a prefill is answered rather than continued.

The module and tokenizer arrive as ``model_args``, live here but serialised to
``null`` in the log. Construction must therefore succeed without them, since
``score()`` rebuilds the model from the log before running any scorer; only
:meth:`generate` refuses.

Providers are memoised on the model name plus the *serialised* ``model_args``, and
everything unserialisable serialises to ``None`` — so two modules under one name
collide, and the second caller gets the first module.
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

from generation.qwen import build_prompt, contains_thinking

from .batching import IN_FLIGHT, BatchGenerator


class QwenLocalAPI(ModelAPI):
    """Generates from a module handed in as a model argument.

    The model name carries the condition, but this class never parses it: which
    weights it holds is the caller's business.
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
            # Fail on a broken chat template before any generation.
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

        prompt = build_prompt(self.tokenizer, message, prefill)
        row = await self._batcher.submit(prompt, seed)

        output = ModelOutput.from_content(
            model=self.model_name,
            # Inspect appends the generated message after a prefill rather than
            # merging, so the whole assistant turn is assembled here instead.
            content=prefill + row.continuation,
            stop_reason=(
                "max_tokens"
                if row.new_tokens >= decoding["max_new_tokens"]
                else "stop"
            ),
        )
        output.usage = ModelUsage(
            input_tokens=row.prompt_tokens,
            output_tokens=row.new_tokens,
            total_tokens=row.prompt_tokens + row.new_tokens,
        )
        output.metadata = {
            # Control tokens intact, unlike completion — the leak check needs them.
            "raw_continuation": row.raw_continuation,
            # Cut at the pad token, so this row's own count and not the batch width.
            "new_tokens": row.new_tokens,
            "prompt_tokens": row.prompt_tokens,
            "thinking_leak": contains_thinking(row.raw_continuation),
            "prefill": prefill,
            "seed": seed,
            "batch_seed": row.batch_seed,
            "batch_position": row.batch_position,
            "batch_size": row.batch_size,
        }
        return output, ModelCall.create(
            request={
                "prompt": prompt,
                "prefill": prefill,
                "seed": seed,
                **decoding,
            },
            response={
                "continuation": row.continuation,
                "new_tokens": row.new_tokens,
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

    Checked at the point of use because an ``eval()`` keyword argument wins the config
    merge, and against ``qwen.DECODING`` because that is the object ``generate_prompts``
    passes to ``model.generate``.
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
    """The seed for this call, which must be present but may be any value."""
    if config.seed is None:
        raise ValueError(
            "no seed in the generate config; batched sampling consumes one RNG "
            "stream, so an unseeded run cannot be characterised afterwards."
        )
    return config.seed
