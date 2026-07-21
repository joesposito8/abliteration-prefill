# Prefilling and Abliteration

Research code investigating whether inference-time **prefilling** and weight-space
**abliteration** unlock the same refusal failures in an aligned open-weight LLM, and
how much extra coverage comes from combining them. Responses are scored with
StrongREJECT.

**Status:** early. Grading and target generation are built; sections marked _TODO_ are
filled in as each piece lands.

## Setup

```bash
uv sync --extra dev           # openai, pandas, pytest, transformers (no torch)
export OPENAI_API_KEY=...     # for grading API calls
```

Generation additionally needs a GPU and `pip install -e ".[gpu]"`; see below.

## Grading (implemented)

StrongREJECT rubric grader (gpt-4o-mini) plus a prefill-stripping hook.

```python
from grading import grade, grade_stripped

r = grade("<forbidden prompt>", "<model output>")                                # unprefilled
r = grade_stripped("<forbidden prompt>", "<prefill+continuation>", "<prefill>")  # strips prefill first
```

Tests: `uv run pytest -m "not live"` (offline) · `uv run pytest -m live` (paid).
Vendored prompts/data and their sources: `data/SOURCES.md`.

## Target generation (implemented)

Qwen3-4B with thinking disabled, optionally prefilled. Requires a GPU.

```python
from generation import load_model, generate, generate_batch

model, tokenizer = load_model()                            # BF16, pinned revision
gen = generate(model, tokenizer, "...", seed=1)            # unprefilled
gen = generate(model, tokenizer, "...", seed=1, prefill="Sure, here's how: ")
gens, seconds = generate_batch(model, tokenizer, [...], seed=1)   # throughput only
```

`generate` produces one `Generation` from one seed, so replaying the seed reproduces
that text. `generate_batch` is for throughput measurement: batched rows share a single
RNG stream, so no individual row is reproducible from its seed alone.

A `Generation` carries `output` (prefill + continuation, the form `grade_stripped`
expects), `continuation` (model tokens, special tokens stripped), and
`raw_continuation` (control tokens intact, for leak checks), plus seed and token
counts. Sampling is fixed at temperature 0.7 / top-p 0.8 / top-k 20 / min-p 0 — the
model's own config defaults to the thinking-mode values 0.6 / 0.95, which would
otherwise be inherited silently.

### Environment

Any ~16GB+ CUDA GPU runs the 4B model in BF16:

```bash
pip install -e ".[gpu]"
export HF_HOME=/workspace/hf          # keep the model cache on persistent storage
python -c "from generation import load_model, generate; \
m,t = load_model(); print(generate(m, t, 'Say hi.', seed=1).continuation)"
```

Reference environment this was validated on: NVIDIA A100 80GB, torch 2.7.1+cu128,
transformers 5.14.1, BF16, ~8 GB peak, ~29 tok/s single-stream. The base model is
36 layers with `self_attn.o_proj` / `mlp.down_proj` on each — the modules the
abliteration step edits.

## Abliteration

Difference-of-means refusal-direction weight editing (Arditi et al. 2024,
arXiv:2406.11717), ported from `andyrdt/refusal_direction`. Requires a GPU.

```python
from generation import load_model, generate
from abliteration import collect_mean_last_token_states, refusal_directions, abliterated

model, tok = load_model()
directions = refusal_directions(                        # [n_layers, d_model], unit
    collect_mean_last_token_states(model, tok, harmful_prompts),
    collect_mean_last_token_states(model, tok, harmless_prompts),
)
with abliterated(model, directions[22]):                # base weights restored on exit
    gen = generate(model, tok, prompt, seed=1)
```

`abliterated` orthogonalizes `embed_tokens` and every layer's `o_proj`/`down_proj`
against the direction, then restores the base weights — no edited checkpoint is
persisted. Callers supply their own harmful/harmless contrast and evaluation prompts.

## Evaluation

_TODO — run the prefilling-vs-abliteration comparison and analyze coverage._

## Layout

```
src/grading/      StrongREJECT grader + prefill hook
src/generation/   target-model generation (GPU)
data/             prompt set, graded examples, sources
tests/            offline + live tests
```
