"""The coalescer's concurrency contract, exercised with a stand-in forward pass.

These properties fail as a hang or as wrong rows rather than as an exception.
"""

from __future__ import annotations

import threading

import anyio
import pytest
from generation.qwen import DECODING, Continuation
from harness.batching import BATCH, IN_FLIGHT, BatchGenerator, _Batch


def echo(prompts, seconds=0.5):
    """One Continuation per prompt, echoing it so rows stay distinguishable."""
    return [
        Continuation(
            continuation=f"<{p}>",
            raw_continuation=f"<{p}>",
            prompt_tokens=3,
            new_tokens=4,
        )
        for p in prompts
    ], seconds


def recording_fake(calls=None, *, fail=None):
    def fake(model, tok, prompts, *, seed, decoding=DECODING):
        if calls is not None:
            calls.append(list(prompts))
        if fail is not None:
            raise fail
        return echo(prompts)

    return fake


def blocking_fake(calls, started, release):
    """First call blocks, so later arrivals accumulate behind it."""

    def fake(model, tok, prompts, *, seed, decoding=DECODING):
        calls.append(list(prompts))
        if len(calls) == 1:
            started.set()
            release.wait(5)
        return echo(prompts, 0.1)

    return fake


def patch(monkeypatch, fake):
    monkeypatch.setattr("harness.batching.generate_prompts", fake)


def batcher(size=BATCH):
    return BatchGenerator(object(), object(), size=size)


async def submit_all(gen, prompts, seed=1):
    """Submit every prompt concurrently, returning rows in submission order."""
    rows: list = [None] * len(prompts)

    async def one(index, prompt):
        rows[index] = await gen.submit(prompt, seed)

    async with anyio.create_task_group() as tg:
        for index, prompt in enumerate(prompts):
            tg.start_soon(one, index, prompt)
    return rows


# --- how batches form ------------------------------------------------------


def test_a_full_batch_generates_together(monkeypatch):
    calls: list = []
    patch(monkeypatch, recording_fake(calls))

    rows = anyio.run(submit_all, batcher(size=4), ["p0", "p1", "p2", "p3"])

    assert calls == [["p0", "p1", "p2", "p3"]]
    assert [r.batch_index for r in rows] == [0, 1, 2, 3]
    assert {r.batch_size for r in rows} == {4}


def test_a_lone_caller_generates_immediately(monkeypatch):
    calls: list = []
    patch(monkeypatch, recording_fake(calls))

    rows = anyio.run(submit_all, batcher(size=32), ["only"])

    assert calls == [["only"]]
    assert rows[0].batch_size == 1


def test_every_member_gets_its_own_row(monkeypatch):
    patch(monkeypatch, recording_fake())

    rows = anyio.run(submit_all, batcher(size=3), ["a", "b", "c"])

    assert [r.continuation for r in rows] == ["<a>", "<b>", "<c>"]


def test_arrivals_during_a_forward_pass_become_the_next_batch(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls: list = []
    patch(monkeypatch, blocking_fake(calls, started, release))
    gen = batcher(size=32)
    rows: dict = {}

    async def one(prompt):
        rows[prompt] = await gen.submit(prompt, 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(one, "first")
            await anyio.to_thread.run_sync(started.wait)
            for p in "cdefgh":
                tg.start_soon(one, p)
            await anyio.sleep(0.05)
            in_flight = list(calls)
            release.set()
        return in_flight

    in_flight = anyio.run(main)

    assert in_flight == [["first"]]
    assert calls == [["first"], ["c", "d", "e", "f", "g", "h"]]
    assert rows["h"].batch_size == 6


def test_a_batch_never_exceeds_the_size_cap(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls: list = []
    patch(monkeypatch, blocking_fake(calls, started, release))
    gen = batcher(size=3)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(gen.submit, "first", 1)
            await anyio.to_thread.run_sync(started.wait)
            for i in range(7):
                tg.start_soon(gen.submit, f"p{i}", 1)
            await anyio.sleep(0.05)
            release.set()

    anyio.run(main)

    assert calls[0] == ["first"]
    assert all(len(call) <= 3 for call in calls)
    assert sum(len(call) for call in calls) == 8


# --- state across runs -----------------------------------------------------


def test_a_run_starts_with_no_inherited_state(monkeypatch):
    """The same provider is memoised across evals."""
    patch(monkeypatch, recording_fake())
    gen = batcher(size=2)

    first = anyio.run(submit_all, gen, ["a", "b"])
    second = anyio.run(submit_all, gen, ["c", "d"])

    assert [r.continuation for r in first] == ["<a>", "<b>"]
    assert [r.continuation for r in second] == ["<c>", "<d>"]


def test_no_batching_state_exists_before_a_run():
    gen = batcher()
    assert gen._gpu is None and gen._open is None and gen._token is None


# --- failure and cancellation ----------------------------------------------


def test_a_failed_batch_reaches_every_member_as_an_ordinary_exception(monkeypatch):
    boom = RuntimeError("CUDA out of memory")
    patch(monkeypatch, recording_fake(fail=boom))
    gen = batcher(size=3)
    errors: list = []

    async def collect(prompt):
        try:
            await gen.submit(prompt, 1)
        except BaseException as exc:  # noqa: BLE001 - the type is the assertion
            errors.append(exc)

    async def main():
        async with anyio.create_task_group() as tg:
            for prompt in ("a", "b", "c"):
                tg.start_soon(collect, prompt)

    anyio.run(main)

    assert len(errors) == 3
    assert all(exc is boom for exc in errors)
    # Not a cancellation, which would be BaseException-only.
    assert all(isinstance(exc, Exception) for exc in errors)


def test_a_cancelled_caller_does_not_poison_later_samples(monkeypatch):
    """Otherwise later arrivals join a batch nobody will generate."""
    started, release = threading.Event(), threading.Event()
    calls: list = []
    patch(monkeypatch, blocking_fake(calls, started, release))
    gen = batcher(size=8)
    rows: list = []

    async def quitter():
        with anyio.move_on_after(0.02):
            await gen.submit("gone", 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(gen.submit, "warmup", 1)  # holds the GPU
            await anyio.to_thread.run_sync(started.wait)
            tg.start_soon(quitter)
            await anyio.sleep(0.05)
            release.set()
        rows.append(await gen.submit("innocent", 1))

    anyio.run(main)

    assert rows[0].continuation == "<innocent>"


def test_cancelling_one_member_leaves_the_rest_intact(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls: list = []
    patch(monkeypatch, blocking_fake(calls, started, release))
    gen = batcher(size=8)
    survived: list = []

    async def member(prompt):
        survived.append(await gen.submit(prompt, 1))

    async def quitter():
        with anyio.move_on_after(0.02):
            await gen.submit("gone", 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(member, "warmup")
            await anyio.to_thread.run_sync(started.wait)
            tg.start_soon(quitter)
            await anyio.sleep(0.005)
            tg.start_soon(member, "a")
            tg.start_soon(member, "b")
            await anyio.sleep(0.05)
            release.set()

    anyio.run(main)

    assert sorted(r.continuation for r in survived) == ["<a>", "<b>", "<warmup>"]


def test_a_cancelled_run_leaves_the_batch_for_another_member():
    """Recording it would hand other members someone else's cancel object.

    Asserted directly: `abandon_on_cancel=False` means the forward pass usually
    completes before the cancellation lands, so this rarely happens in practice.
    """

    async def main():
        gen = batcher(size=8)
        gen._rebind()
        batch = _Batch(seed=1)
        batch.prompts.append("a")

        async def cancelled(_batch):
            raise anyio.get_cancelled_exc_class()()

        gen._run = cancelled
        with pytest.raises(anyio.get_cancelled_exc_class()):
            await gen._generate(batch)
        return batch.done, batch.error

    done, error = anyio.run(main)
    assert not done
    assert error is None


def test_a_process_ending_signal_propagates_untouched():
    """A Ctrl-C must reach the caller, not become every member's error."""

    async def main():
        gen = batcher(size=8)
        gen._rebind()
        batch = _Batch(seed=1)
        batch.prompts.append("a")

        async def interrupted(_batch):
            raise KeyboardInterrupt

        gen._run = interrupted
        with pytest.raises(KeyboardInterrupt):
            await gen._generate(batch)
        return batch.done, batch.error

    done, error = anyio.run(main)
    assert not done
    assert error is None


def test_a_seed_disagreement_is_refused(monkeypatch):
    """One seed covers a whole condition, so two in one batch means two conditions."""
    started, release = threading.Event(), threading.Event()
    calls: list = []
    patch(monkeypatch, blocking_fake(calls, started, release))
    gen = batcher(size=8)
    raised: list = []

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(gen.submit, "warmup", 1)
            await anyio.to_thread.run_sync(started.wait)
            tg.start_soon(gen.submit, "a", 1)  # opens the next batch
            await anyio.sleep(0.005)
            try:
                await gen.submit("b", 2)
            except ValueError as exc:
                raised.append(exc)
            release.set()

    anyio.run(main)

    assert len(raised) == 1 and "asks for 2" in str(raised[0])


# --- the constants ---------------------------------------------------------


def test_in_flight_is_twice_the_batch_width():
    assert IN_FLIGHT == 2 * BATCH
