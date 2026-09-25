"""The enabled models.

A model module is one checkpoint: ``MODEL_ID``, ``REVISION``, ``N_LAYERS``, ``BATCH``,
``build_prompt`` and ``REASONING_MARKERS``. Two checkpoints of one architecture are two
models, so nothing here is keyed on the architecture family.

``TARGETS`` receive prefills; ``MODELS`` adds the helper that writes them.
``tests/test_template_contract.py`` holds every model in ``MODELS`` to the template
contract, and ``tests/test_prefill_exactness.py`` every target to byte-exact placement.
"""

from study.datasets import model_slug

from . import gemma3_27b, phi4, qwen3_4b

TARGETS = (qwen3_4b, phi4)
MODELS = (*TARGETS, gemma3_27b)

_BY_SLUG = {model_slug(model.MODEL_ID): model for model in TARGETS}


def target(slug: str):
    """The target module a condition's model slug names."""
    if slug not in _BY_SLUG:
        raise KeyError(f"no enabled target {slug!r}; enabled: {sorted(_BY_SLUG)}")
    return _BY_SLUG[slug]


__all__ = [
    "MODELS",
    "TARGETS",
    "gemma3_27b",
    "phi4",
    "qwen3_4b",
    "target",
]
