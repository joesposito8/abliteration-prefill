"""Provider registration, reached through the ``inspect_ai`` entry point in pyproject.

Two entries, however many models the study enables: every target shares one provider
and is told apart by the model name, and the helper has its own. The names are literals
because the decorator takes one; ``tests/test_harness_provider.py`` resolves every
enabled target through the registry, which is what holds ``local`` here to
``conditions.PROVIDER``.
"""

from __future__ import annotations

from inspect_ai.model import ModelAPI, modelapi


@modelapi(name="local")
def local() -> type[ModelAPI]:
    from .provider import LocalTargetAPI

    return LocalTargetAPI


@modelapi(name="gemma-helper")
def gemma_helper() -> type[ModelAPI]:
    from .helper_provider import GemmaHelperAPI

    return GemmaHelperAPI
