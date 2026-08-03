"""The coalescer's concurrency contract, exercised with a stand-in forward pass.

These properties fail as a hang or as quietly wrong rows rather than as an exception,
so each is provoked directly.
"""

from __future__ import annotations

import threading

import anyio
import pytest
from generation.qwen import DECODING, Generation, row_prefills
from harness.batching import BATCH, IN_FLIGHT, BatchGenerator, _Batch


def fake_generate_batch(calls=None, *, seconds=0.5, fail=None):
    """A stand-in forward pass that records its arguments."""

    def fake(model, tok, messages, *, seed, prefill="", decoding=DECODING):
        prefills = row_prefills(prefill, len(messages))
        if calls is not None:
            calls.append({"messages": list(messages), "prefills": prefills, "seed": seed})
        if fail is not None:
            raise fail
        return [
            Generation(
                message=message,
                prefill=row_prefill,
                output=row_prefill + f"<{message}>",
                continuation=f"<{message}>",
                raw_continuation=f"<{message}>",
                seed=seed,
                prompt_tokens=3,
                new_tokens=4,
                max_new_tokens=decoding["max_new_tokens"],
            )
            for message, row_prefill in zip(messages, prefills)
        ], seconds

    return fake


def patch(monkeypatch, fake):
    monkeypatch.setattr("generation.qwen.generate_batch", fake)


def batcher(size=BATCH):
    return BatchGenerator(object(), object(), size=size)


async def submit_all(gen, pairs, seed=1):
    """Submit every pair concurrently and return the rows in submission order."""
    rows: list = [None] * len(pairs)

    async def one(i, message, prefill):
        rows[i] = await gen.submit(message, prefill, seed)

    async with anyio.create_task_group() as tg:
        for i, (message, prefill) in enumerate(pairs):
            tg.start_soon(one, i, message, prefill)
    return rows


def blocking_fake(calls, started, release):
    """First call blocks, so later arrivals accumulate behind it."""

    def fake(model, tok, messages, *, seed, prefill="", decoding=DECODING):
        prefills = row_prefills(prefill, len(messages))
        calls.append(list(messages))
        if len(calls) == 1:
            started.set()
            release.wait(5)
        return [
            Generation(
                message=m, prefill=p, output=p + f"<{m}>", continuation=f"<{m}>",
                raw_continuation=f"<{m}>", seed=seed, prompt_tokens=1, new_tokens=1,
                max_new_tokens=decoding["max_new_tokens"],
            )
            for m, p in zip(messages, prefills)
        ], 0.1

    return fake


# --- how batches form ----------------------------------------------------


def test_a_full_batch_fires_on_size(monkeypatch):
    calls = []
    patch(monkeypatch, fake_generate_batch(calls))
    gen = batcher(size=4)

    rows = anyio.run(submit_all, gen, [(f"m{i}", "") for i in range(4)])

    assert len(calls) == 1
    assert calls[0]["messages"] == ["m0", "m1", "m2", "m3"]
    assert [r.batch_index for r in rows] == [0, 1, 2, 3]
    assert {r.batch_size for r in rows} == {4}


def test_a_short_batch_fires_on_the_window(monkeypatch):
    calls = []
    patch(monkeypatch, fake_generate_batch(calls))
    gen = batcher(size=32)

    rows = anyio.run(submit_all, gen, [("only", "")])

    assert len(calls) == 1
    assert rows[0].batch_size == 1


def test_every_member_gets_its_own_row(monkeypatch):
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=3)

    rows = anyio.run(submit_all, gen, [("a", ""), ("b", ""), ("c", "")])

    assert [r.generation.message for r in rows] == ["a", "b", "c"]
    assert [r.generation.continuation for r in rows] == ["<a>", "<b>", "<c>"]


def test_a_batch_of_mixed_prefills_keeps_them_per_row(monkeypatch):
    """Rows share a forward pass but not a prefill."""
    calls = []
    patch(monkeypatch, fake_generate_batch(calls))
    gen = batcher(size=3)

    rows = anyio.run(
        submit_all, gen, [("a", "Sure:"), ("b", ""), ("c", "Step 1.")]
    )

    assert calls[0]["prefills"] == ["Sure:", "", "Step 1."]
    assert [r.generation.output for r in rows] == ["Sure:<a>", "<b>", "Step 1.<c>"]


# --- the loop-bound state --------------------------------------------------


def test_a_cancelled_caller_does_not_poison_later_samples(monkeypatch):
    """Otherwise later arrivals join a batch nobody will generate."""
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=8)

    async def main():
        with anyio.move_on_after(0.01):
            await gen.submit("doomed", "", 1)
        dangled = gen._open is not None
        with anyio.fail_after(1):
            row = await gen.submit("innocent", "", 1)
        return dangled, row

    dangled, row = anyio.run(main)
    assert not dangled
    assert row.generation.message == "innocent"


def test_a_run_starts_with_no_inherited_state(monkeypatch):
    """The same provider is memoised across evals."""
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=2)

    first = anyio.run(submit_all, gen, [("a", ""), ("b", "")])
    second = anyio.run(submit_all, gen, [("c", ""), ("d", "")])

    assert [r.generation.message for r in first] == ["a", "b"]
    assert [r.generation.message for r in second] == ["c", "d"]


def test_no_batching_state_exists_before_a_run():
    """Nothing is built at construction, so nothing binds to the wrong run."""
    gen = batcher()
    assert gen._gpu is None and gen._open is None and gen._token is None


# --- failure and cancellation ----------------------------------------------


def test_a_failed_batch_reaches_every_member_as_an_ordinary_exception(monkeypatch):
    boom = RuntimeError("CUDA out of memory")
    patch(monkeypatch, fake_generate_batch(fail=boom))
    gen = batcher(size=3)
    errors: list = []

    async def collect(message):
        try:
            await gen.submit(message, "", 1)
        except BaseException as exc:  # noqa: BLE001 - the type is the assertion
            errors.append(exc)

    async def main():
        async with anyio.create_task_group() as tg:
            for m in ("a", "b", "c"):
                tg.start_soon(collect, m)

    anyio.run(main)

    assert len(errors) == 3
    assert all(exc is boom for exc in errors)
    # Not a cancellation, which would be BaseException-only.
    assert all(isinstance(exc, Exception) for exc in errors)


def test_cancelling_the_first_arrival_does_not_strand_the_others(monkeypatch):
    """No member is special, so losing one costs only its own row."""
    started, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "generation.qwen.generate_batch", blocking_fake(calls, started, release)
    )
    gen = batcher(size=8)
    survived: list = []

    async def member(message):
        survived.append(await gen.submit(message, "", 1))

    async def quitter():
        with anyio.move_on_after(0.02):
            await gen.submit("gone", "", 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(member, "warmup")  # holds the GPU
            await anyio.to_thread.run_sync(started.wait)
            tg.start_soon(quitter)
            await anyio.sleep(0.005)
            tg.start_soon(member, "a")
            tg.start_soon(member, "b")
            await anyio.sleep(0.05)
            release.set()

    anyio.run(main)

    assert sorted(r.generation.message for r in survived) == ["a", "b", "warmup"]


def test_a_cancelled_run_leaves_the_batch_for_another_member():
    """Recording it would hand other members someone else's cancel object.

    Asserted directly because `abandon_on_cancel=False` means the forward pass
    usually completes before the cancellation lands, so a cancelled member normally
    finishes the batch for everyone else.
    """

    async def main():
        gen = batcher(size=8)
        gen._rebind()
        batch = _Batch(seed=1)
        batch.rows.append(("a", ""))

        async def cancelled(_batch):
            raise anyio.get_cancelled_exc_class()()

        gen._run = cancelled
        with pytest.raises(anyio.get_cancelled_exc_class()):
            await gen._generate(batch)
        return batch.done, batch.error

    done, error = anyio.run(main)
    assert not done
    assert error is None


def test_a_seed_disagreement_is_refused(monkeypatch):
    """One seed covers a whole condition, so two in one batch means two conditions."""
    started, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "generation.qwen.generate_batch", blocking_fake(calls, started, release)
    )
    gen = batcher(size=8)
    raised: list = []

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(gen.submit, "warmup", "", 1)
            await anyio.to_thread.run_sync(started.wait)
            tg.start_soon(gen.submit, "a", "", 1)  # opens the next batch
            await anyio.sleep(0.005)
            try:
                await gen.submit("b", "", 2)
            except ValueError as exc:
                raised.append(exc)
            release.set()

    anyio.run(main)

    assert len(raised) == 1 and "asks for 2" in str(raised[0])


# --- batches form from contention ------------------------------------------


def test_arrivals_during_a_forward_pass_become_the_next_batch(monkeypatch):
    """The wait for the GPU is what a timer would otherwise be.

    Generating alone when nothing is queued is correct, not a compromise.
    """
    started, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "generation.qwen.generate_batch", blocking_fake(calls, started, release)
    )
    gen = batcher(size=32)
    rows: dict = {}

    async def one(message):
        rows[message] = await gen.submit(message, "", 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(one, "first")
            await anyio.to_thread.run_sync(started.wait)
            for m in "cdefgh":
                tg.start_soon(one, m)
            await anyio.sleep(0.05)
            release.set()

    anyio.run(main)

    assert calls == [["first"], ["c", "d", "e", "f", "g", "h"]]
    assert len(rows) == 7
    assert rows["h"].batch_size == 6


def test_a_batch_never_exceeds_the_size_cap(monkeypatch):
    """The cap is a memory bound, so it must hold however many arrive."""
    started, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "generation.qwen.generate_batch", blocking_fake(calls, started, release)
    )
    gen = batcher(size=3)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(gen.submit, "first", "", 1)
            await anyio.to_thread.run_sync(started.wait)
            for i in range(7):
                tg.start_soon(gen.submit, f"m{i}", "", 1)
            await anyio.sleep(0.05)
            release.set()

    anyio.run(main)

    assert calls[0] == ["first"]
    assert all(len(c) <= 3 for c in calls)
    assert sum(len(c) for c in calls) == 8


# --- the constants ---------------------------------------------------------


def test_in_flight_is_twice_the_batch_width():
    """At 1x the running batch owns every connection permit and the next cannot form."""
    assert IN_FLIGHT == 2 * BATCH
