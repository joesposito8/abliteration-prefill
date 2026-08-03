"""Target-model generation: Qwen3-4B with thinking disabled, optionally prefilled.

Four properties here fail silently rather than loudly, so each is enforced in code
rather than left to the caller:

Thinking mode. Qwen3's chat template only emits the empty ``<think>\\n\\n</think>``
sentinel when ``enable_thinking=False`` is passed explicitly; omit the argument and the
template emits nothing, the model opens its own reasoning block, and every downstream
score is computed over reasoning text. ``build_prompt`` asserts the sentinel is present.

Prefill position. A prefill is the start of the assistant's own turn, so it is
concatenated after the sentinel and returned as part of the output. Anywhere else and
the model treats it as context to respond to rather than text to continue.

Sampling. The model ships ``generation_config.json`` defaults of temperature 0.6 /
top-p 0.95 (Qwen's thinking-mode recommendation). The study is preregistered on
0.7 / 0.8 / top-k 20 / min-p 0, so the parameters are passed explicitly on every call;
inheriting the model's defaults would silently change the sampled distribution. The
length cap travels with them in ``DECODING``, since truncation changes the text too.

Seeding. ``generate`` produces exactly one generation from one seed, because a seed is
only a reproducibility handle if replaying it reproduces that specific text. Batched
sampling consumes a single RNG stream across the whole batch, so a per-row seed would
be a lie; ``generate_batch`` returns rows that are reproducible only by replaying
the identical batch, and says so.

Note for weight-editing callers: batched work here uses left padding, which is what
makes index -1 the last real token. Anything collecting last-token hidden states must
do the same, or it will average padding positions into the result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from types import MappingProxyType

MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

# Every decoding parameter, as one object. Read-only: mutating these mid-run would
# silently change the text produced for every condition that follows.
DECODING = MappingProxyType(
    {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "max_new_tokens": 512,
    }
)

# What Qwen3's template appends for an assistant turn with thinking disabled.
THINKING_SENTINEL = "<think>\n\n</think>\n\n"


@dataclass
class Generation:
    """One generation and the numbers needed to characterise it."""

    message: str
    prefill: str
    output: str  # prefill + continuation; what grade_stripped expects
    continuation: str  # model-generated text, special tokens stripped
    raw_continuation: str  # same tokens, control tokens left in for leak checks
    seed: int
    prompt_tokens: int
    new_tokens: int
    seconds: float | None = None  # None for batch rows, which share one wall clock

    @property
    def tokens_per_second(self) -> float | None:
        """Single generations only; batch rows have no individual duration."""
        return self.new_tokens / self.seconds if self.seconds else None


@dataclass
class Continuation:
    """One prompt's generated text and the token counts that characterise it."""

    continuation: str  # special tokens stripped
    raw_continuation: str  # control tokens left in, for leak checks
    prompt_tokens: int  # the row's own length, not the padded batch width
    new_tokens: int  # cut at the pad token


def contains_thinking(text: str) -> bool:
    """True if a reasoning-block marker leaked into generated text.

    ``<think>``/``</think>`` are ordinary tokens in this tokenizer, so they survive
    ``skip_special_tokens=True`` and this works on decoded output.
    """
    return "<think>" in text or "</think>" in text


def build_prompt(tokenizer, message: str, prefill: str = "") -> str:
    """Render one user turn as a prompt, with an optional assistant-side prefill.

    A prefill should not end in whitespace: the trailing space becomes its own token,
    which is not how the same text would tokenize during natural generation, and the
    continuation degrades for that condition only.

    Raises:
        ValueError: if the rendered prompt does not end with the thinking sentinel,
            which means thinking mode is live and a prefill would be placed inside
            the reasoning block instead of the response.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not text.endswith(THINKING_SENTINEL):
        raise ValueError(
            "rendered prompt does not end with the empty-thinking sentinel; "
            "the chat template did not honour enable_thinking=False, so thinking "
            f"mode is live. Prompt tail: {text[-80:]!r}"
        )
    return text + prefill


def load_model(model_id: str = MODEL_ID, revision: str = REVISION):
    """Load the target model in BF16 on CUDA. Returns ``(model, tokenizer)``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, tokenizer


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


def generate(
    model,
    tokenizer,
    message: str,
    *,
    seed: int,
    prefill: str = "",
    decoding=DECODING,
) -> Generation:
    """Generate one continuation, reproducible from ``seed`` alone.

    ``seed`` is keyword-only with no default: every replicate records the seed that
    produced it, so it cannot be omitted.
    """
    generations, seconds = generate_batch(
        model, tokenizer, [message], seed=seed, prefill=prefill, decoding=decoding
    )
    # Batch rows leave ``seconds`` unset because they share one clock; with a single
    # row the batch's duration is that row's own.
    return replace(generations[0], seconds=seconds)


def generate_batch(
    model,
    tokenizer,
    messages: list[str],
    *,
    seed: int,
    prefill: str = "",
    decoding=DECODING,
) -> tuple[list[Generation], float]:
    """Generate one continuation per message in a single batched call.

    Returns ``(generations, batch_seconds)``. The seed applies to the batch as a
    whole: rows share one RNG stream, so an individual row is only reproducible by
    replaying the identical batch. Use :func:`generate` when a row must be
    reproducible from its seed alone.
    """
    prompts = [build_prompt(tokenizer, message, prefill) for message in messages]
    continuations, batch_seconds = generate_prompts(
        model, tokenizer, prompts, seed=seed, decoding=decoding
    )
    return [
        Generation(
            message=message,
            prefill=prefill,
            output=prefill + row.continuation,
            continuation=row.continuation,
            raw_continuation=row.raw_continuation,
            seed=seed,
            prompt_tokens=row.prompt_tokens,
            new_tokens=row.new_tokens,
        )
        for message, row in zip(messages, continuations, strict=True)
    ], batch_seconds


def generate_prompts(
    model,
    tokenizer,
    prompts: list[str],
    *,
    seed: int,
    decoding=DECODING,
) -> tuple[list[Continuation], float]:
    """Run already-rendered prompts through the model in one batched call."""
    import torch

    # Left padding keeps the prompt the same length for every row, so generated
    # tokens start at a single known offset.
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, padding_side="left"
    ).to(model.device)
    padded_width = encoded.input_ids.shape[1]

    torch.manual_seed(seed)
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
