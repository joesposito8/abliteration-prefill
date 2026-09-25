"""An Inspect provider for an in-memory abliterated Gemma helper.

A sibling of :class:`harness.provider.LocalTargetAPI`, not a generalisation of it. The
helper is handed a request and continues nothing, has no reasoning block to leak, and
renders through its own template. What they share is the coalescer and the plumbing in
``provider``: the config and seed checks, and the shaping of a finished batch row into
Inspect's output. What each keeps is its own — how a request becomes a prompt, and what
its logs carry beyond the row.

As there, the module and tokenizer arrive as ``model_args`` and serialise to ``null`` in
the log, so construction must succeed without them and only :meth:`generate` refuses.
"""

from __future__ import annotations

from typing import Any

from generation.batched import generate_prompts
from generation.gemma3_27b import HELPER_DECODING, build_prompt
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .batching import BatchGenerator, in_flight
from .provider import (
    batched_output,
    require_frozen_config,
    require_module,
    require_seed,
)

HELPER_PROVIDER = "gemma-helper"

# Unlike the target's, this width is not a frozen study parameter: each prefill is
# produced once, committed and hashed, and read identically by every downstream
# condition, so no comparison depends on the width it was produced at. Measured on the
# card: data/helper_batch_sweep.json.
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
            build_prompt(tokenizer, "startup check")

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
        require_module(self)

        prompt = build_prompt(self.tokenizer, message)
        request = {"prompt": prompt, "seed": seed, **HELPER_DECODING}

        try:
            row = await self._batcher.submit(prompt, seed)
        except Exception as exc:
            return exc, ModelCall.create(request=request, response={})

        return batched_output(self.model_name, row, HELPER_DECODING, request, extra={})


def require_one_request(input: list[ChatMessage]) -> str:
    """The helper is given one request and continues nothing of its own."""
    if len(input) != 1 or input[0].role != "user":
        raise ValueError(
            f"expected a single user message, got roles {[m.role for m in input]}; "
            "only the user turn is rendered, so anything else would be dropped."
        )
    return input[0].text
