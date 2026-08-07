"""An Inspect provider for an in-memory abliterated Gemma helper.

A sibling of :class:`harness.provider.QwenLocalAPI`, not a generalisation of it. The
helper is handed a request and continues nothing, has no reasoning block to leak, and
renders through its own template — so parameterising the target's provider to serve it
would add four switches that only ever take one value each. What they do share is the
coalescer and the two config checks.

As there, the module and tokenizer arrive as ``model_args`` and serialise to ``null`` in
the log, so construction must succeed without them and only :meth:`generate` refuses.
"""

from __future__ import annotations

from typing import Any

from generation.batched import generate_prompts
from generation.gemma import HELPER_DECODING, render_prompt
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .batching import BatchGenerator, in_flight
from .provider import require_frozen_config, require_seed

HELPER_PROVIDER = "gemma-helper"

# Unlike the target's, this width is not a frozen study parameter: each prefill is
# produced once, committed and hashed, and read identically by every downstream
# condition, so no comparison depends on the width it was produced at. Measured on the
# card rather than assumed.
HELPER_BATCH = 16

HELPER_FROZEN_CONFIG = GenerateConfig(
    temperature=HELPER_DECODING["temperature"],
    top_p=HELPER_DECODING["top_p"],
    top_k=HELPER_DECODING["top_k"],
    max_tokens=HELPER_DECODING["max_new_tokens"],
    extra_body={
        "min_p": HELPER_DECODING["min_p"],
        "do_sample": HELPER_DECODING["do_sample"],
    },
    max_connections=in_flight(HELPER_BATCH),
)


class GemmaHelperAPI(ModelAPI):
    """Generates from a helper module handed in as a model argument."""

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
                module, tokenizer, prompts, seed=seed, decoding=HELPER_DECODING
            ),
            size=HELPER_BATCH,
        )

        if tokenizer is not None:
            # Fail on the wrong tokenizer before any generation.
            render_prompt(tokenizer, "startup check")

    def max_connections(self) -> int:
        """Fallback for a bare ``get_model()``; ``Task.config`` sets the real value.

        A limiter's size is fixed by whichever call creates it first, so a value below
        the batch width could deadlock a batch that can never fill.
        """
        return in_flight(HELPER_BATCH)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput | Exception, ModelCall]:
        message = require_one_request(input)
        require_frozen_config(config, HELPER_FROZEN_CONFIG)
        seed = require_seed(config)

        if self.module is None or self.tokenizer is None:
            raise RuntimeError(
                f"{type(self).__name__} holds no live module. Model arguments are "
                "recorded in the log as null, so anything rebuilt from a log arrives "
                "weightless; pass module= and tokenizer= to the eval that generates."
            )

        prompt = render_prompt(self.tokenizer, message)
        request = {"prompt": prompt, "seed": seed, **HELPER_DECODING}

        try:
            row = await self._batcher.submit(prompt, seed)
        except Exception as exc:
            return exc, ModelCall.create(request=request, response={})

        output = ModelOutput.from_content(
            model=self.model_name,
            content=row.continuation,
            stop_reason=(
                "max_tokens"
                if row.new_tokens >= HELPER_DECODING["max_new_tokens"]
                else "stop"
            ),
        )
        output.usage = ModelUsage(
            input_tokens=row.prompt_tokens,
            output_tokens=row.new_tokens,
            total_tokens=row.prompt_tokens + row.new_tokens,
        )
        output.metadata = {
            # The portfolio's chat-token validator matches Gemma's own control tokens,
            # which completion has already had stripped out of it.
            "raw_continuation": row.raw_continuation,
            "new_tokens": row.new_tokens,
            "prompt_tokens": row.prompt_tokens,
            # What a batched draw is reproducible from, the seed alone not being enough.
            "batch_seed": row.batch_seed,
            "batch_position": row.batch_position,
            "batch_size": row.batch_size,
        }
        return output, ModelCall.create(
            request=request,
            response={
                "continuation": row.continuation,
                "new_tokens": row.new_tokens,
                "batch_seconds": row.seconds,
            },
        )


def require_one_request(input: list[ChatMessage]) -> str:
    """The helper is given one request and continues nothing of its own."""
    if len(input) != 1 or input[0].role != "user":
        raise ValueError(
            f"expected a single user message, got roles {[m.role for m in input]}; "
            "only the user turn is rendered, so anything else would be dropped."
        )
    return input[0].text
