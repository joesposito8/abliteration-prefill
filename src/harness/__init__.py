"""Inspect AI harness for the prefill/abliteration comparison.

Importing this package registers the ``qwen-local`` model provider, which is why
the driver imports it before resolving a model. The import stays cheap: the
registry hands back a factory, so neither ``generation.qwen`` nor torch is loaded
until a model is actually created.
"""

from __future__ import annotations

from . import _registry  # noqa: F401  (import for the @modelapi registration)

# Rows per forward pass. A starting point to confirm against real KV-cache
# headroom on the pilot, not a derived number.
BATCH = 32

# Samples allowed inside the provider at once. Deliberately twice BATCH, not equal
# to it: the connection limiter is held across the whole provider call, so at 1x
# every permit belongs to the running batch and the next one cannot begin to
# assemble. 3x buys nothing, since one batch runs at a time regardless.
IN_FLIGHT = 2 * BATCH

__all__ = ["BATCH", "IN_FLIGHT"]
