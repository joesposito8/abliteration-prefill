"""One batched forward pass over already-rendered prompts.

The model, its tokenizer and its decoding parameters all arrive as arguments, so nothing
here is specific to a model.

Seeding. Batched sampling draws for every row from one RNG stream, so a row's text depends
on which prompts shared its forward pass. That stream is seeded by ``batch_seed`` from the
caller's seed and the batch's exact prompts, so each batch runs under its own stream and
any batch can be regenerated from what a log already records.

Note for weight-editing callers: batched work here uses left padding, which is what makes
index -1 the last real token. Anything collecting last-token hidden states must do the
same, or it will average padding positions into the result.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class Continuation:
    """One prompt's generated text and the token counts that characterise it."""

    continuation: str  # special tokens stripped
    raw_continuation: str  # control tokens left in, for leak checks
    prompt_tokens: int  # the row's own length, not the padded batch width
    new_tokens: int  # cut at the pad token


def batch_seed(seed: int, prompts: list[str]) -> int:
    """The RNG seed for one forward pass, from ``seed`` and the batch's exact prompts."""
    payload = "\0".join([str(seed), *prompts]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _decode(tokenizer, token_ids, pad_token_id):
    """Decode one row of generated tokens (a plain list) to ``(clean, raw, count)``.

    Rows that stop early are filled with the pad token, so the tail is cut before
    decoding; otherwise ``raw`` would carry hundreds of literal pad strings and any
    control-token check over it would be useless. A row whose own final token is the
    pad token is undercounted by one, which is within tolerance for these numbers.
    Expects ``token_ids`` already on the host, so the scan adds no device sync.
    """
    end = token_ids.index(pad_token_id) if pad_token_id in token_ids else len(token_ids)
    generated = token_ids[:end]
    return (
        tokenizer.decode(generated, skip_special_tokens=True),
        tokenizer.decode(generated, skip_special_tokens=False),
        end,
    )


def _pad_token_id(model, tokenizer):
    """The id ``generate`` actually pads with, which need not be the tokenizer's."""
    configured = model.generation_config.pad_token_id
    return configured if configured is not None else tokenizer.pad_token_id


def generate_prompts(
    model,
    tokenizer,
    prompts: list[str],
    *,
    seed: int,
    decoding: Mapping[str, object],
) -> tuple[list[Continuation], float]:
    """Run already-rendered prompts through the model in one batched call.

    ``decoding`` has no default: the two models are preregistered on different sampling
    parameters, and a shared forward pass that fell back to either model's would generate
    the other's text under the wrong distribution without saying so.
    """
    import torch

    # Left padding keeps the prompt the same length for every row, so generated
    # tokens start at a single known offset.
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    padded_width = encoded.input_ids.shape[1]

    torch.manual_seed(batch_seed(seed, prompts))
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**encoded, **decoding)
    batch_seconds = time.perf_counter() - start

    # Pull the generated tokens and prompt lengths to the host once, rather than
    # syncing per row inside the loop.
    pad_token_id = _pad_token_id(model, tokenizer)
    rows = outputs[:, padded_width:].tolist()
    prompt_lengths = encoded.attention_mask.sum(dim=1).tolist()
    continuations = []
    for index, row in enumerate(rows):
        text, raw, new_tokens = _decode(tokenizer, row, pad_token_id)
        continuations.append(
            Continuation(
                continuation=text,
                raw_continuation=raw,
                prompt_tokens=prompt_lengths[index],
                new_tokens=new_tokens,
            )
        )
    return continuations, batch_seconds
