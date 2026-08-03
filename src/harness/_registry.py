"""Provider registration, reached through the ``inspect_ai`` entry point in pyproject.

``@modelapi`` registers a factory, not a class. Torch is not a concern either way:
it is imported inside the functions that need it, so nothing here loads it.
"""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="qwen-local")
def qwen_local() -> type[ModelAPI]:
    from .provider import QwenLocalAPI

    return QwenLocalAPI
