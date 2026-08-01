"""The coalescer's concurrency contract, exercised with a stand-in forward pass.

The properties here are the ones that fail as a hang or as quietly wrong rows rather
than as an exception, so each is provoked directly: state must not outlive the event
loop that built it, a cancelled member must not take the batch with it, and a failed
batch must reach every member as an ordinary exception.
"""

from __future__ import annotations

import threading

import anyio
import pytest
from generation.qwen import DECODING, Generation, row_prefills
from harness.batching import BATCH, IN_FLIGHT, BatchGenerator


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


def batcher(size=BATCH, window=0.05):
    return BatchGenerator(object(), object(), size=size, window=window)


async def submit_all(gen, pairs, seed=1):
    """Submit every pair concurrently and return the rows in submission order."""
    rows: list = [None] * len(pairs)

    async def one(i, message, prefill):
        rows[i] = await gen.submit(message, prefill, seed)

    async with anyio.create_task_group() as tg:
        for i, (message, prefill) in enumerate(pairs):
            tg.start_soon(one, i, message, prefill)
    return rows


# --- what fires a batch ----------------------------------------------------


def test_a_full_batch_fires_on_size(monkeypatch):
    calls = []
    patch(monkeypatch, fake_generate_batch(calls))
    gen = batcher(size=4, window=30)  # a window this long would hang if size did not fire

    rows = anyio.run(submit_all, gen, [(f"m{i}", "") for i in range(4)])

    assert len(calls) == 1
    assert calls[0]["messages"] == ["m0", "m1", "m2", "m3"]
    assert [r.batch_index for r in rows] == [0, 1, 2, 3]
    assert {r.batch_size for r in rows} == {4}


def test_a_short_batch_fires_on_the_window(monkeypatch):
    calls = []
    patch(monkeypatch, fake_generate_batch(calls))
    gen = batcher(size=32, window=0.01)

    rows = anyio.run(submit_all, gen, [("only", "")])

    assert len(calls) == 1
    assert rows[0].batch_size == 1


def test_every_member_gets_its_own_row(monkeypatch):
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=3, window=30)

    rows = anyio.run(submit_all, gen, [("a", ""), ("b", ""), ("c", "")])

    assert [r.generation.message for r in rows] == ["a", "b", "c"]
    assert [r.generation.continuation for r in rows] == ["<a>", "<b>", "<c>"]


def test_a_batch_of_mixed_prefills_keeps_them_per_row(monkeypatch):
    """Rows share a forward pass but not a prefill, which is the point of batching here."""
    calls = []
    patch(monkeypatch, fake_generate_batch(calls))
    gen = batcher(size=3, window=30)

    rows = anyio.run(
        submit_all, gen, [("a", "Sure:"), ("b", ""), ("c", "Step 1.")]
    )

    assert calls[0]["prefills"] == ["Sure:", "", "Step 1."]
    assert [r.generation.output for r in rows] == ["Sure:<a>", "<b>", "Step 1.<c>"]


# --- the loop-bound state --------------------------------------------------


def test_a_cancelled_leader_does_not_poison_later_samples(monkeypatch):
    """A leader cancelled mid-window must not leave its batch open.

    If it does, every later arrival joins a batch nobody will generate and inherits
    its failure — one timeout would take down the rest of the condition.
    """
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=8, window=0.05)

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
    """The same provider is memoised across evals, so each run must begin clean."""
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=2, window=0.01)

    first = anyio.run(submit_all, gen, [("a", ""), ("b", "")])
    second = anyio.run(submit_all, gen, [("c", ""), ("d", "")])

    assert [r.generation.message for r in first] == ["a", "b"]
    assert [r.generation.message for r in second] == ["c", "d"]


def test_no_batching_state_exists_before_a_run():
    """Nothing is built at construction, so a weightless provider costs nothing."""
    gen = batcher()
    assert gen._gpu is None and gen._open is None and gen._token is None


# --- failure and cancellation ----------------------------------------------


def test_a_failed_batch_reaches_every_member_as_an_ordinary_exception(monkeypatch):
    boom = RuntimeError("CUDA out of memory")
    patch(monkeypatch, fake_generate_batch(fail=boom))
    gen = batcher(size=3, window=30)

    errors: list = []

    async def collect(message):
        try:
            await gen.submit(message, "", 1)
        except BaseException as exc:  # noqa: BLE001 - the type is what is asserted
            errors.append(exc)

    async def main():
        async with anyio.create_task_group() as tg:
            for m in ("a", "b", "c"):
                tg.start_soon(collect, m)

    anyio.run(main)

    assert len(errors) == 3
    assert all(exc is boom for exc in errors)
    # A cancellation derives from BaseException, not Exception, so this is the
    # loop-free way to say "an ordinary error reached them, not someone's cancel".
    assert all(isinstance(exc, Exception) for exc in errors)


def test_a_cancelled_leader_wakes_followers_with_an_ordinary_exception(monkeypatch):
    """Handing followers the leader's own cancellation is not a valid cancel in their
    scope, and corrupts the bookkeeping of whatever scope does receive it."""
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=8, window=30)
    caught: list = []

    async def leader():
        with anyio.move_on_after(0.02):
            await gen.submit("a", "", 1)

    async def follower():
        await anyio.sleep(0.005)  # join after the leader has opened the batch
        try:
            await gen.submit("b", "", 1)
        except BaseException as exc:  # noqa: BLE001 - the type is the assertion
            caught.append(exc)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(leader)
            tg.start_soon(follower)

    anyio.run(main)

    assert len(caught) == 1
    assert isinstance(caught[0], Exception)  # not BaseException-only, i.e. not a cancel
    assert "cancelled" in str(caught[0])


def test_one_member_cancelled_mid_batch_leaves_the_rest_intact(monkeypatch):
    """`attempt_timeout` wraps each provider call in its own cancel scope, so this
    is a case the framework really produces."""
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=3, window=0.05)
    survived: list = []

    async def member(message):
        survived.append(await gen.submit(message, "", 1))

    async def quitter():
        with anyio.move_on_after(0.001):
            await gen.submit("gone", "", 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(quitter)
            tg.start_soon(member, "a")
            tg.start_soon(member, "b")

    anyio.run(main)

    assert sorted(r.generation.message for r in survived) == ["a", "b"]


def test_a_seed_disagreement_is_refused(monkeypatch):
    """One seed covers a whole condition, so two in one batch means two conditions."""
    patch(monkeypatch, fake_generate_batch())
    gen = batcher(size=4, window=30)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(gen.submit, "a", "", 1)
            await anyio.sleep(0.01)
            with pytest.raises(ValueError, match="asks for 2"):
                await gen.submit("b", "", 2)
            tg.cancel_scope.cancel()

    anyio.run(main)


# --- pipeline depth --------------------------------------------------------


def test_a_second_batch_assembles_while_the_first_holds_the_gpu(monkeypatch):
    """Intake closes before the GPU is claimed.

    If it closed after, arrivals during a forward pass would be appended to a batch
    that is already generating, and their rows would never be produced at all.
    """
    started, release = threading.Event(), threading.Event()
    calls: list[list[str]] = []

    def fake(model, tok, messages, *, seed, prefill="", decoding=DECODING):
        calls.append(list(messages))
        if len(calls) == 1:
            started.set()
            release.wait(5)  # hold the "GPU" while the next batch forms
        return [
            Generation(
                message=m, prefill="", output=f"<{m}>", continuation=f"<{m}>",
                raw_continuation=f"<{m}>", seed=seed, prompt_tokens=1, new_tokens=1,
                max_new_tokens=decoding["max_new_tokens"],
            )
            for m in messages
        ], 0.1

    monkeypatch.setattr("generation.qwen.generate_batch", fake)
    # Size 8 with one arrival: the batch fires on the window, so the leader's own
    # close is the only thing that can close intake before the forward pass.
    gen = batcher(size=8, window=0.01)
    rows: dict = {}

    async def one(message):
        rows[message] = await gen.submit(message, "", 1)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(one, "a")
            await anyio.to_thread.run_sync(started.wait)

            tg.start_soon(one, "c")
            tg.start_soon(one, "d")
            await anyio.sleep(0.05)
            in_flight = list(calls)
            release.set()
        return in_flight

    in_flight = anyio.run(main)

    # While the first batch held the GPU the second had assembled but not run.
    assert in_flight == [["a"]]
    assert calls == [["a"], ["c", "d"]]
    assert rows["c"].generation.continuation == "<c>"
    assert [rows[m].batch_index for m in ("c", "d")] == [0, 1]


# --- the constants ---------------------------------------------------------


def test_in_flight_is_twice_the_batch_width():
    """At 1x the running batch owns every connection permit and the next cannot form."""
    assert IN_FLIGHT == 2 * BATCH
