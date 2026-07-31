"""Provider registration, kept apart from the provider itself.

The factory defers importing :mod:`harness.provider` until a model is actually
created, which in turn defers ``generation.qwen`` and torch. Registering the class
directly would pull torch into every process that merely imports this package —
including the grading pass, which runs on a machine with no GPU.
"""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="qwen-local")
def qwen_local() -> type[ModelAPI]:
    from .provider import QwenLocalAPI

    return QwenLocalAPI
