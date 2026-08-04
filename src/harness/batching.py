"""Coalesces concurrent single-sample calls into one batched forward pass.

A batch is whatever accumulated while the previous one held the GPU. Rows need
nothing in common, since left padding absorbs the length differences, so composition
is arrival-dependent.

Sampling still draws for the whole batch from one stream, so a row's text depends on
the batch it landed in. The stream's seed is derived from that batch's own prompts,
which is what lets a log record composition rather than merely assert it: rows sharing
a forward pass carry the same ``batch_seed``, and it regenerates them.
"""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field

import anyio
import anyio.lowlevel
from generation.qwen import Continuation, batch_seed, generate_prompts

# A memory bound: the KV cache for this many 512-token sequences must fit alongside
# the weights.
BATCH = 32

# The connection limiter is held across the whole provider call, so at 1x the running
# batch owns every permit and the next one cannot assemble.
IN_FLIGHT = 2 * BATCH


@dataclass
class Row(Continuation):
    """One member's share of a completed batch."""

    seconds: float
    batch_seed: int
    batch_position: int
    batch_size: int


@dataclass
class _Batch:
    seed: int
    prompts: list[str] = field(default_factory=list)
    result: list[Continuation] = field(default_factory=list)
    error: BaseException | None = None
    seconds: float = 0.0
    done: bool = False


class BatchGenerator:
    """Gathers arriving prompts and generates them together."""

    def __init__(self, module, tokenizer, *, size: int = BATCH) -> None:
        self._module = module
        self._tokenizer = tokenizer
        self._size = size
        self._token = None
        self._gpu: anyio.CapacityLimiter | None = None
        self._open: _Batch | None = None

    async def submit(self, prompt: str, seed: int) -> Row:
        batch, index = self._join(prompt, seed)

        async with self._gpu:
            if not batch.done:
                self._close(batch)
                await self._generate(batch)

        if batch.error is not None:
            raise copy(batch.error)
        return Row(
            **asdict(batch.result[index]),
            seconds=batch.seconds,
            batch_seed=batch_seed(batch.seed, batch.prompts),
            batch_position=index,
            batch_size=len(batch.prompts),
        )

    def _join(self, prompt: str, seed: int) -> tuple[_Batch, int]:
        """Claim a place in the open batch, or open one.

        Synchronous, so the batch cannot close between being chosen and joined.
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

        index = len(batch.prompts)
        batch.prompts.append(prompt)
        if len(batch.prompts) >= self._size:
            self._close(batch)
        return batch, index

    def _close(self, batch: _Batch) -> None:
        """Stop this batch accepting prompts; the next arrival opens a new one."""
        if self._open is batch:
            self._open = None

    async def _generate(self, batch: _Batch) -> None:
        # Cancellation and process-ending signals are not Exceptions, so they
        # propagate and leave the batch unfinished for the next member to run.
        try:
            batch.result, batch.seconds = await self._run(batch)
        except Exception as exc:
            batch.error = exc
        batch.done = True

    async def _run(self, batch: _Batch) -> tuple[list[Continuation], float]:
        return await anyio.to_thread.run_sync(
            lambda: generate_prompts(
                self._module, self._tokenizer, batch.prompts, seed=batch.seed
            )
        )

    def _rebind(self) -> None:
        """Start each run clean.

        A provider is memoised across ``eval()`` calls, so a batch carried over would
        generate rows whose callers are gone.
        """
        token = anyio.lowlevel.current_token()
        if token != self._token:
            self._token = token
            self._gpu = anyio.CapacityLimiter(1)
            self._open = None
