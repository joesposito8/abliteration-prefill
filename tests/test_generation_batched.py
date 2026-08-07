"""Offline tests for the shared batched forward pass (no weights, no GPU).

What a batch's RNG stream is derived from, and that the derivation is what the forward
pass actually seeds with. Both models generate through this, so a change here moves the
text of every condition in the study at once.
"""

from types import SimpleNamespace

import pytest
from generation.batched import batch_seed, generate_prompts


class _Encoded:
    """Only what the forward pass touches before it seeds."""

    input_ids = SimpleNamespace(shape=(2, 5))

    def to(self, device):
        return self


def test_a_batch_seed_is_fixed_by_the_seed_and_the_prompts():
    """Rows of one batch share it, so it must not depend on anything else."""
    assert batch_seed(1, ["a", "b"]) == batch_seed(1, ["a", "b"])


@pytest.mark.parametrize(
    "seed, prompts",
    [
        pytest.param(2, ["a", "b"], id="different-seed"),
        pytest.param(1, ["a", "c"], id="different-prompt"),
        pytest.param(1, ["b", "a"], id="different-order"),
        pytest.param(1, ["a", "b", "c"], id="different-width"),
        pytest.param(1, ["a"], id="fewer-prompts"),
    ],
)
def test_any_change_to_a_batch_gives_it_another_seed(seed, prompts):
    """Otherwise a batch reassembled wrongly would still derive the recorded seed."""
    assert batch_seed(seed, prompts) != batch_seed(1, ["a", "b"])


def test_batch_seeds_differ_across_the_batches_of_one_condition():
    """One seed reused would correlate the sampling noise of every batch in a run."""
    condition_seed = 20260803
    batches = [[f"p{i * 4 + j}" for j in range(4)] for i in range(50)]

    seeds = {batch_seed(condition_seed, batch) for batch in batches}

    assert len(seeds) == len(batches)


def test_a_batch_seed_fits_what_torch_accepts():
    """torch.manual_seed takes an unsigned 64-bit value; this is not testable on GPU."""
    seeds = [batch_seed(s, [f"p{s}"]) for s in range(500)]

    assert all(0 <= seed <= 0xFFFFFFFFFFFFFFFF for seed in seeds)


def test_prompts_cannot_be_reshuffled_into_the_same_seed():
    """A separator keeps the joined payload from being ambiguous."""
    assert batch_seed(1, ["ab", "c"]) != batch_seed(1, ["a", "bc"])


def test_the_forward_pass_seeds_from_the_derived_value(fake_torch):
    """Every other test replaces generate_prompts, so only this one reaches it.

    ``decoding`` is empty because the run stops at the seeding call, before anything
    would read it.
    """
    with pytest.raises(fake_torch.Stop):
        generate_prompts(
            SimpleNamespace(device="cpu"),
            lambda prompts, **kwargs: _Encoded(),
            ["a", "b"],
            seed=1,
            decoding={},
        )

    assert fake_torch.seeds == [batch_seed(1, ["a", "b"])]


def test_the_forward_pass_will_not_default_its_decoding():
    """The two models are preregistered on different sampling parameters, so a default
    here would generate one model's text under the other's distribution."""
    import inspect

    decoding = inspect.signature(generate_prompts).parameters["decoding"]
    assert decoding.kind is inspect.Parameter.KEYWORD_ONLY
    assert decoding.default is inspect.Parameter.empty
