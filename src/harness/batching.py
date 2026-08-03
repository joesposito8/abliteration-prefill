"""Coalesces concurrent single-sample calls into one batched forward pass.

Every caller queues for the GPU and whoever reaches it first generates whatever
accumulated behind it, so a batch is the arrivals during the previous forward pass.
No timer: while the GPU is busy the next batch fills, and while it is idle there is
nothing to wait for.

Rows need nothing in common, since a prefill is folded into its own prompt before
tokenization and left padding absorbs the length differences. Composition is
therefore arrival-dependent, and the seed characterises a batch rather than a row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
from anyio import get_cancelled_exc_class

if TYPE_CHECKING:
    from generation.qwen import Generation

# A memory bound: the KV cache for this many 512-token sequences must fit alongside
# the weights.
BATCH = 32

# The connection limiter is held across the whole provider call, so at 1x the running
# batch owns every permit and the next one cannot assemble.
IN_FLIGHT = 2 * BATCH


@dataclass
class Row:
    """One member's share of a completed batch."""

    generation: Generation
    seconds: float
    batch_size: int
    batch_index: int


@dataclass
class _Batch:
    seed: int
    rows: list[tuple[str, str]] = field(default_factory=list)
    result: list[Generation] = field(default_factory=list)
    error: BaseException | None = None
    seconds: float = 0.0
    done: bool = False


class BatchGenerator:
    """Gathers ``(message, prefill)`` arrivals and generates them together.

    Every caller runs the same code, so no member is special and there is no
    background task to outlive the run.
    """

    def __init__(self, module, tokenizer, *, size: int = BATCH) -> None:
        self._module = module
        self._tokenizer = tokenizer
        self._size = size
        self._token = None
        self._gpu: anyio.CapacityLimiter | None = None
        self._open: _Batch | None = None

    async def submit(self, message: str, prefill: str, seed: int) -> Row:
        batch, index = self._join(message, prefill, seed)

        async with self._gpu:
            if not batch.done:
                self._close(batch)
                await self._generate(batch)

        if batch.error is not None:
            raise batch.error
        return Row(
            generation=batch.result[index],
            seconds=batch.seconds,
            batch_size=len(batch.rows),
            batch_index=index,
        )

    def _join(self, message: str, prefill: str, seed: int) -> tuple[_Batch, int]:
        """Claim a place in the open batch, or open one.

        Synchronous on purpose: no await between choosing a batch and joining it, so
        it cannot be closed and generated in between.
        """
        self._rebind()

        batch = self._open
        if batch is None:
            batch = self._open = _Batch(seed=seed)
        elif batch.seed != seed:
            raise ValueError(
                f"this batch was opened with seed {batch.seed} but a sample asks for "
                f"{seed}. One seed covers a whole condition, so a disagreement means "
                "two conditions are generating into the same provider."
            )

        index = len(batch.rows)
        batch.rows.append((message, prefill))
        if len(batch.rows) >= self._size:
            self._close(batch)
        return batch, index

    def _close(self, batch: _Batch) -> None:
        """Stop this batch accepting rows; the next arrival opens a new one."""
        if self._open is batch:
            self._open = None

    async def _generate(self, batch: _Batch) -> None:
        try:
            batch.result, batch.seconds = await self._run(batch)
        except get_cancelled_exc_class():
            # Left unfinished rather than failed, so the next member regenerates it.
            raise
        except BaseException as exc:
            batch.error = exc
        batch.done = True

    async def _run(self, batch: _Batch) -> tuple[list[Generation], float]:
        from generation.qwen import generate_batch

        messages = [message for message, _ in batch.rows]
        prefills = [prefill for _, prefill in batch.rows]
        return await anyio.to_thread.run_sync(
            lambda: generate_batch(
                self._module,
                self._tokenizer,
                messages,
                seed=batch.seed,
                prefill=prefills,
            )
        )

    def _rebind(self) -> None:
        """Start each run with no inherited batching state.

        A provider is memoised across ``eval()`` calls, so a batch carried over would
        generate rows whose callers are long gone.
        """
        token = anyio.lowlevel.current_token()
        if token != self._token:
            self._token = token
            self._gpu = anyio.CapacityLimiter(1)
            self._open = None
