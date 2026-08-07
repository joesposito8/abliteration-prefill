"""Provider registration, reached through the ``inspect_ai`` entry point in pyproject."""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="qwen-local")
def qwen_local() -> type[ModelAPI]:
    from .provider import QwenLocalAPI

    return QwenLocalAPI


@modelapi(name="gemma-helper")
def gemma_helper() -> type[ModelAPI]:
    from .helper_provider import GemmaHelperAPI

    return GemmaHelperAPI
