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

from collections.abc import Mapping
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

from generation.batched import generate_prompts
from generation.qwen import DECODING, build_prompt, contains_thinking

from .batching import IN_FLIGHT, BatchGenerator, Row

FROZEN_CONFIG = GenerateConfig(
    temperature=DECODING["temperature"],
    top_p=DECODING["top_p"],
    top_k=DECODING["top_k"],
    max_tokens=DECODING["max_new_tokens"],
    extra_body={"min_p": DECODING["min_p"], "do_sample": DECODING["do_sample"]},
    max_connections=IN_FLIGHT,
)

MAY_VARY = ("seed", "max_retries", "timeout", "attempt_timeout")


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
        self._batcher = BatchGenerator(
            lambda prompts, *, seed: generate_prompts(
                module, tokenizer, prompts, seed=seed, decoding=DECODING
            )
        )

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
        require_frozen_config(config, FROZEN_CONFIG)
        seed = require_seed(config)
        require_module(self)

        prompt = build_prompt(self.tokenizer, message, prefill)
        request = {"prompt": prompt, "prefill": prefill, "seed": seed, **DECODING}

        try:
            row = await self._batcher.submit(prompt, seed)
        except Exception as exc:
            return exc, ModelCall.create(request=request, response={})

        return batched_output(
            self.model_name,
            row,
            DECODING,
            request,
            extra={
                "thinking_leak": contains_thinking(row.raw_continuation),
                "prefill": prefill,
                "response": prefill + row.continuation,
                "seed": seed,
            },
        )


def require_module(api: ModelAPI) -> None:
    """Refuse to generate from a provider that was rebuilt from a log."""
    if api.module is None or api.tokenizer is None:
        raise RuntimeError(
            f"{type(api).__name__} holds no live module. Model arguments are "
            "recorded in the log as null, so anything rebuilt from a log arrives "
            "weightless; pass module= and tokenizer= to the eval that generates."
        )


def batched_output(
    model_name: str,
    row: Row,
    decoding: Mapping[str, object],
    request: dict,
    extra: dict,
) -> tuple[ModelOutput, ModelCall]:
    """One finished batch row as Inspect's output and call record.

    ``extra`` is merged after the keys every row carries, so what a provider adds of
    its own stays at that provider's call site rather than here.
    """
    output = ModelOutput.from_content(
        model=model_name,
        content=row.continuation,
        stop_reason=(
            "max_tokens" if row.new_tokens >= decoding["max_new_tokens"] else "stop"
        ),
    )
    output.usage = ModelUsage(
        input_tokens=row.prompt_tokens,
        output_tokens=row.new_tokens,
        total_tokens=row.prompt_tokens + row.new_tokens,
    )
    output.metadata = {
        # Control tokens intact, unlike completion — the checks over generated text
        # need them.
        "raw_continuation": row.raw_continuation,
        # Cut at the pad token, so this row's own count and not the batch width.
        "new_tokens": row.new_tokens,
        "prompt_tokens": row.prompt_tokens,
        # What a batched row is reproducible from; its seed alone is not enough.
        "batch_seed": row.batch_seed,
        "batch_position": row.batch_position,
        "batch_size": row.batch_size,
        **extra,
    }
    return output, ModelCall.create(
        request=request,
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


def require_frozen_config(config: GenerateConfig, frozen: GenerateConfig) -> None:
    """Refuse a config that would generate differently from ``frozen``.

    Checked at the point of use because an ``eval()`` keyword argument wins the config
    merge. Anything the provider cannot honour is ``None`` on its frozen config, so
    setting it fails here rather than being recorded in the log header and then dropped
    on the way to the GPU.
    """
    declared = config.model_copy(update=dict.fromkeys(MAY_VARY))
    if declared != frozen:
        differing = {
            name: (getattr(declared, name), getattr(frozen, name))
            for name in GenerateConfig.model_fields
            if getattr(declared, name) != getattr(frozen, name)
        }
        raise ValueError(
            "generate config does not match the frozen config "
            f"(name: got, expected): {differing}"
        )


def require_seed(config: GenerateConfig) -> int:
    """The seed for this call, which must be present but may be any value."""
    if config.seed is None:
        raise ValueError(
            "no seed in the generate config; batched sampling consumes one RNG "
            "stream, so an unseeded run cannot be characterised afterwards."
        )
    return config.seed
