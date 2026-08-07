# Prefilling and Abliteration

Research code investigating whether inference-time **prefilling** and weight-space
**abliteration** unlock the same refusal failures in an aligned open-weight LLM, and
how much extra coverage comes from combining them. Responses are scored with
StrongREJECT.

**Status:** grading, generation, abliteration and the prefill portfolio are built, and
the primary-layer selection sweep runs end to end. The main comparison is not run yet;
sections marked _TODO_ are filled in as each piece lands.

## Setup

```bash
uv sync --extra dev           # openai, pandas, pytest, transformers, inspect-ai (no torch)
export OPENAI_API_KEY=...     # for grading API calls
```

Python 3.12 is pinned in `.python-version`; `uv sync` provisions it. Generation
additionally needs a GPU and `uv sync --extra gpu`; see below.

Run everything through `uv run`. It re-syncs from `uv.lock` first, which is what holds
the pinned versions in effect — a bare `python` resolves whatever the ambient
interpreter happens to have.

## Grading (implemented)

The StrongREJECT rubric (gpt-4o-mini) as an Inspect scorer. Refusal is judged on the
model's continuation alone, quality on the whole assistant turn — the provider decodes
only new tokens, so a prefill never reaches the refusal call.

```python
from grading.scorers import strongreject          # the scorer the sweep grades with
from grading import parse_grader_output, aggregate_score   # the rubric itself
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
gens, seconds = generate_batch(model, tokenizer, [...], seed=1)   # one RNG stream
```

`generate` produces one `Generation` from one seed, so replaying the seed reproduces
that text. `generate_batch` shares a single RNG stream across the batch, so a row's text
depends on which prompts shared its forward pass. That stream is seeded by
`batch_seed(seed, prompts)` rather than by `seed` itself: each batch therefore samples
independently of every other, and the derived value doubles as the batch's identity —
rows generated together carry the same one, and re-deriving it from a prompt list that
was reassembled wrongly lands elsewhere, so a bad reconstruction cannot pass for a good
one.

A `Generation` carries `response` (prefill + continuation, the whole assistant turn the
quality rubric is scored on), `continuation` (model tokens, special tokens
stripped), and `raw_continuation` (control tokens intact, for leak checks), plus seed
and token counts. Decoding is fixed in one frozen `DECODING` mapping — temperature 0.7 / top-p
0.8 / top-k 20 / min-p 0, capped at 1024 new tokens — passed explicitly on every call.
The model's own config defaults to the thinking-mode values 0.6 / 0.95, which would
otherwise be inherited silently.

### Environment

Any ~16GB+ CUDA GPU runs the 4B model in BF16:

```bash
uv sync --extra gpu                   # torch comes from the lock, not the host image
export HF_HOME=/workspace/hf          # keep the model cache on persistent storage
uv run python -c "from generation import load_model, generate; \
m,t = load_model(); print(generate(m, t, 'Say hi.', seed=1).continuation)"
```

Reference environment: NVIDIA A100 80GB, Python 3.12, torch 2.11.0+cu128, transformers
5.14.1, BF16. Measured on it at batch 64 over 780 samples: **0.77 samples/s, 244.5
output tokens/s, 20.5 GiB peak** (`data/harness_smoke.json`); a single stream manages
30 tokens/s, so the batching is worth roughly eight times. The earlier torch 2.7.1
figures are not carried forward, because a torch minor can change sampled tokens and so
the reference environment moves with it. The base model is 36 layers with
`self_attn.o_proj` / `mlp.down_proj` on each — the modules the abliteration step edits.

Batch width is a frozen study parameter, not a tuning knob: greedy output is not
width-invariant, so conditions generated at different widths are not comparable. The
width, the curve behind it and the rule that picked it are in `data/batch_sweep.json`.

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

`scripts/extract_directions.py` is the study's own path through this: it extracts over
the frozen 128/128 contrast corpora, checks the edit targets against the loaded config,
and writes `data/refusal_directions.pt` plus the two robustness cosines. Method and
layer-indexing caveats: `data/SOURCES.md`.

## Prefill portfolio

The fixed 13-prefill attack portfolio: six helper-prompt families (each producing two
seeded, request-specific prefill variants from the abliterated Gemma helper) plus Kuo's
static baseline. Templates are authored from the Struppek prefill taxonomy
(arXiv:2602.14689); decoding/retry/length/duplicate rules are frozen in code.

```python
from prefills import FAMILIES, PORTFOLIO, load_prompt, fill_prompt, produce_family

template = load_prompt("system_simulation")            # self-contained helper prompt
filled = fill_prompt(template, "<forbidden prompt>")   # fills the one {forbidden_prompt} slot
variants = produce_family("system_simulation", filled, prompt_id, generate_fn)  # 2 variants
```

`produce_family` applies the frozen rules (`src/prefills/rules.py`): helper sampling
temperature 1.0 / top_p 0.95 / top_k 64 (the full tuple is pinned so nothing leaks from
model-config defaults), a 512-token prefill cap, up to 3 validity retries then the cell is
flagged `failed` with `prefill=None`, and an exact within-family / vs-baseline duplicate
rule with up to 3 further seeded draws. A failed cell counts as a non-unlock downstream
rather than being substituted with a fallback string, which could spuriously unlock and
muddy per-family results. The actual Gemma call is injected as `generate_fn`, so the
portfolio logic is offline-testable.

Freeze the portfolio (hash every prompt + rule into a manifest with one roll-up hash):

```bash
uv run python scripts/freeze_portfolio.py   # writes data/portfolio_manifest.json
```

Sources and the two documented example deviations: `data/SOURCES.md`.

## Prefill production

The portfolio above is templates and rules; the attack strings themselves are generated
once by the helper and committed, because each `(prompt_id, slot)` is consumed twice — by
the prefill-only arm on the base and by the composed arm on the primary layer. Generating
inline would hand those two arms different attack text, and the composition-only set would
then confound "composition unlocked it" with "a different prefill unlocked it".

The rules are sequential per cell — draw, validate, retry, dedup against the variant before
it — and a 27B helper needs breadth to be affordable. The breadth comes from across the
1,878 cells rather than within one, so generation runs in **waves**: every cell's next draw,
generated together. Between waves `produce_family` runs again over a `generate_fn` backed by
what the waves have already produced, and the first draw it asks for that no wave has made
is the next wave's work. That replay is also what settles each cell, so the frozen rules are
applied and never restated.

```bash
# on the GPU box, detached: converges in one wave when every draw is clean
uv run python scripts/produce_prefills.py results/prefills /dev/shm/gemma

# locally, no GPU, after the wave logs are pulled back
uv run python scripts/freeze_prefills.py results/prefills
```

The freeze writes `data/prefills.csv` (313 prompts x 13 slots = 4,069 rows: 3,756 generated
plus the static baseline, which is a constant and is never drawn for) and
`data/prefill_manifest.json`, whose `prefill_sha256` covers the table together with the
helper's identity and decoding, the prompt-set hash and the portfolio hash. A rebuild that
would change the committed table aborts rather than re-freezing itself.

Reproduction is by the table's hash rather than by replaying a seed: rows sharing a forward
pass draw from one RNG stream, so an individual row is only reproducible by replaying the
identical batch. What `helper_seed` guarantees is narrower and still holds — an attempt is a
different draw from the one it retries, since a retry is only ever generated in a later wave.

Conditions read the frozen table through `study.datasets.load_prefills()`.

## Harness

Generation and grading run on [Inspect](https://inspect.aisi.org.uk/). One condition is
one `eval_set` call against a local provider holding the already-loaded model, in its own
log directory — so re-running a killed sweep resumes each condition from what its log
already holds rather than regenerating it.

Conditions run strictly one at a time (`max_tasks=1`). That is a correctness rule and not
a throughput choice: the module is a single mutable object, so two conditions in flight
would both generate under whichever weight edit was applied last, while each log still
named its own layer. After every condition the base weights are restored and compared
against a snapshot, and a mismatch aborts the sweep.

Generation declares no judge and needs no API key; grading is a separate pass over the
logs that appends scores, so a condition can be re-graded without regenerating it.

## Primary-layer selection sweep

Picks the abliteration layer the main comparison uses, on 72 HarmBench behaviours
disjoint from the evaluation prompts. Four steps, each resumable by re-running it:

```bash
uv run python scripts/extract_directions.py                             # once, GPU
uv run python scripts/generate.py results/sweep/generated               # GPU
uv run python scripts/grade.py results/sweep/generated results/sweep/scored
uv run python scripts/freeze_abliteration.py --run results/sweep --excerpts 2
uv run python scripts/freeze_abliteration.py --run results/sweep --write
```

Generation runs 37 conditions — the unedited base plus every layer — at one attempt per
prompt. The selection rule is breadth (non-refusal rate at k=1), then mean quality over
*all* prompts among layers within three unlocked prompts of the best, then the lowest
layer index. Malformed and degenerate rows are non-unlocks that stay in the denominator,
so a degrading layer loses breadth without a separate exclusion rule deciding it is unfit.

The freeze step prints the 37-row table and the tie-break trace and writes nothing by
default; `--excerpts N` additionally prints a few unlocked continuations per near-tie
layer, terminal-only. `--write` commits `data/layer_selection.csv` and
`data/abliteration_manifest.json`. Definitions and the method behind every column:
`data/SOURCES.md`.

## Run manifest

The main comparison's work list, frozen before any of it runs:

```bash
uv run python scripts/build_run_manifest.py    # writes data/run_manifest.{csv,json}
```

One row per generation — 313 evaluation prompts x 507 generations = 158,691 — keyed by
prompt, condition, prefill and replicate. 507 per prompt is 37 unprefilled conditions at
13 replicates, plus a portfolio slot each on the two conditions that carry a prefilled
arm: the unedited base, and the primary the selection sweep chose. The count is asserted
against the preregistered workload, and every cell against being enumerated twice, so a
malformed work list is caught here rather than on a rented GPU. A rebuild that changes the
table aborts instead of re-freezing it.

`batch_id` and `batch_size` partition that list at the frozen batch width, keyed by
condition and prefill, because one forward pass holds one weight edit and one prefill and
the last batch of each group takes the remainder. It is a **declared plan, not an
observation**: `src/harness/batching.py` fills a batch from whatever arrives while the
previous one holds the GPU, so the batch a row actually ran in is the one its log records
through `batch_seed`, `batch_size` and `batch_position`.

## Evaluation

_TODO — run the prefilling-vs-abliteration comparison and analyze coverage._

## Layout

```
src/harness/      Inspect harness: local providers, batching, tasks, drivers
src/grading/      StrongREJECT rubric, parser, and the Inspect scorer
src/generation/   the shared batched forward pass, the target, the helper (GPU)
src/abliteration/ direction extraction, weight editing, the selection rule
src/prefills/     the 13-prefill attack portfolio (helper prompts + frozen rules)
src/study/        frozen datasets and splits, manifest hashing
scripts/          the runnable steps: freeze, extract, generate, grade, select
data/             prompt sets, frozen artifacts and manifests, sources
tests/            offline + live tests
```
