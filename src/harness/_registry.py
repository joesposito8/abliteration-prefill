"""Provider registration, reached through the ``inspect_ai`` entry point in pyproject.

Returning a factory rather than the class defers ``generation.qwen`` and torch until
a model is actually created. Registering the class directly would pull torch into
every process that resolves any model name — including the grading pass, which runs
on a machine with no GPU.
"""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="qwen-local")
def qwen_local() -> type[ModelAPI]:
    from .provider import QwenLocalAPI

    return QwenLocalAPI
