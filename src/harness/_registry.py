"""Provider registration, reached through the ``inspect_ai`` entry point in pyproject.

Returning a factory rather than the class defers torch until a model is created, so
resolving a model name costs nothing on a machine without a GPU.
"""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="qwen-local")
def qwen_local() -> type[ModelAPI]:
    from .provider import QwenLocalAPI

    return QwenLocalAPI
