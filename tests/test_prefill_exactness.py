"""The prompt the harness renders is the prompt ``build_prompt`` renders, byte for byte.

The prefill is placed by concatenation, and the split between attack text and generated
text is structural rather than searched: the scorer judges refusal on the continuation
alone and quality on ``prefill + continuation``. So one byte altered on the way from a
sample to ``build_prompt`` — whitespace normalised, a fence dropped, a quote re-encoded —
puts framing no portfolio slot contains into the judged text, with the log still recording
the portfolio's version of the prefill. The shapes below are what the six helper families
produce; ``STATIC_BASELINE`` is the one fixed prefill of the 13.
"""

from __future__ import annotations

import pytest
from conftest import CONTINUATION, CORPUS, MESSAGE, template_double
from generation import TARGETS
from harness.conditions import PROVIDER
from harness.task import refusal_unlock
from inspect_ai import eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from study.datasets import model_slug

SEED = 20260803

# Over the targets themselves: a model only reaches this file once it is one, and being
# in the tuple is what gates it here rather than a list that can fall behind.
gated = pytest.mark.parametrize("module", TARGETS, ids=lambda m: m.__name__)


@gated
@pytest.mark.parametrize("prefill", CORPUS)
def test_the_rendered_prompt_is_byte_identical_to_build_prompt(
    tmp_path, fake_generate_prompts, prefill, module
):
    """One sample through the real task and provider, against a direct render."""
    model = model_slug(module.MODEL_ID)
    tokenizer = template_double(module)
    messages = [ChatMessageUser(content=MESSAGE)]
    if prefill:
        messages.append(ChatMessageAssistant(content=prefill))

    log = eval(
        refusal_unlock(
            # The name is the prompt set the task hashes into its metadata; the sample
            # here is synthetic, so any real set will do.
            MemoryDataset(
                [Sample(id="000/gate/00", input=messages, target="")], name="strongreject"
            ),
            seed=SEED,
            model=model,
        ),
        model=f"{PROVIDER}/{model}/gate",
        model_args={"module": object(), "tokenizer": tokenizer},
        log_dir=str(tmp_path),
        score=False,
    )[0]

    assert log.status == "success"
    [rendered] = fake_generate_prompts.calls[0]["prompts"]

    assert rendered == module.build_prompt(tokenizer, MESSAGE, prefill)

    # By index rather than search: the unprefilled render is an exact prefix, so the
    # prefill is the whole difference between the two.
    head = len(rendered) - len(prefill)
    assert rendered[head:] == prefill
    assert rendered[:head] == module.build_prompt(tokenizer, MESSAGE)

    # What the scorer strips on: the recorded turn is the injected text plus what the
    # model added, and nothing else.
    metadata = log.samples[0].output.metadata
    assert metadata["prefill"] == prefill
    assert metadata["response"] == prefill + CONTINUATION
