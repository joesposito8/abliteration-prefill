"""Coalesces concurrent single-sample calls into one batched forward pass.

Inspect drives many samples at once but hands the provider one at a time, and a
4B model generating a single sequence leaves most of the GPU idle. Arrivals are
therefore collected until the batch is full or a short window passes, then generated
together.

Any arrivals form a batch. A prefill is folded into its own prompt before
tokenization and left padding absorbs the resulting length differences, so rows need
nothing in common and arrival order is free. The consequence is that composition is
wall-clock dependent: the seed characterises a batch, not a row, and a row is
reproducible only by replaying the identical batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
import anyio.lowlevel
from anyio import get_cancelled_exc_class

if TYPE_CHECKING:
    from generation.qwen import Generation

BATCH = 32

# Samples allowed inside the provider at once. Twice the batch width because the
# connection limiter is held across the whole provider call: at 1x the running batch
# owns every permit, so the next one cannot begin to assemble.
IN_FLIGHT = 2 * BATCH

# Fires only when a batch fails to fill — the tail of a condition, or a stall. In
# steady state batches fill on size and this never elapses.
WINDOW = 0.2


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
    full: anyio.Event = field(default_factory=anyio.Event)
    ready: anyio.Event = field(default_factory=anyio.Event)
    result: list[Generation] = field(default_factory=list)
    error: BaseException | None = None
    seconds: float = 0.0


class BatchGenerator:
    """Gathers ``(message, prefill)`` arrivals and generates them together.

    The first caller into an empty batch leads it: it waits for the batch to fill or
    the window to pass, runs the forward pass, and hands every member its row. There
    is no background task, so nothing outlives the run that created it.
    """

    def __init__(
        self, module, tokenizer, *, size: int = BATCH, window: float = WINDOW
    ) -> None:
        self._module = module
        self._tokenizer = tokenizer
        self._size = size
        self._window = window
        self._token = None
        self._gpu: anyio.CapacityLimiter | None = None
        self._open: _Batch | None = None

    async def submit(self, message: str, prefill: str, seed: int) -> Row:
        batch, index = self._join(message, prefill, seed)

        if index == 0:
            await self._lead(batch)
        else:
            await batch.ready.wait()

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

        Deliberately synchronous: with no await between choosing a batch and joining
        it, no other task can close that batch in between and leave this row in one
        that has already been generated.
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
            self._open = None
            batch.full.set()
        return batch, index

    async def _lead(self, batch: _Batch) -> None:
        try:
            with anyio.move_on_after(self._window):
                await batch.full.wait()

            # Intake closes before the GPU is claimed, so the next batch assembles
            # while this one runs rather than after it.
            if self._open is batch:
                self._open = None

            async with self._gpu:
                batch.result, batch.seconds = await self._run(batch)
        except get_cancelled_exc_class():
            # Followers must be woken with an ordinary exception: handing them
            # another task's cancellation is not a valid cancel in their scope.
            batch.error = RuntimeError(
                "the batch leader was cancelled before its forward pass completed"
            )
            raise
        except BaseException as exc:
            # Not re-raised: `submit` raises it for the leader too, so every member
            # of a failed batch fails identically.
            batch.error = exc
        finally:
            # Also on the cancelled path, where the early close above never ran.
            # A batch left open here would collect later arrivals into a batch whose
            # leader is gone, and hand them its failure.
            if self._open is batch:
                self._open = None
            batch.ready.set()

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

        Inspect memoises a provider on its model name, so the same instance is handed
        back across ``eval()`` calls, each on a fresh event loop. anyio's primitives
        themselves survive that, but a batch left open by an abruptly torn-down run
        would collect this run's samples into a batch that will never generate.
        """
        token = anyio.lowlevel.current_token()
        if token != self._token:
            self._token = token
            self._gpu = anyio.CapacityLimiter(1)
            self._open = None
