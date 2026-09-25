# Prefilling and Abliteration

Research code for a comparison of two attacks on aligned open-weight models: inference-time
**prefilling**, and weight-space **abliteration**. It runs the generation sweep, grades the
output against the StrongREJECT rubric, and computes the reported metrics.

Information on the study itself (research question, threat model, experimental design, the
four reported metrics, grading, the analysis plan and how each outcome is to be read) is in
the following proposal:

**https://gist.github.com/joesposito8/6a2670b3c9d99df3eacd1e081d85e44e**

This README covers the repository: what was run to produce the committed artifacts, how to
set it up, and how to run it again.

## Results

The baseline for this experiment, the base model with no prefill, unlocks 0.0435 of attempts on
Qwen3-4B and 0.0075 on Phi-4.

At a single attempt on both models, abliteration alone unlocked more queries than the base
model under its best prefill family (`role_chaining`). At an alpha of 0.025, Bonferroni
adjusted to span claims over the two models, the per-query mean difference of abliteration
versus the best prefill family has a point estimate of 0.1107 and a confidence interval of
[0.0810, 0.1455] for Qwen3-4B, and 0.0724 and [0.0362, 0.1097] respectively on Phi-4. The fact
that the lower end of the confidence interval sits above zero on both models is what provides
evidence for the claim.

Additionally, composing the best performing prefill on top of the abliterated model is more
effective than abliteration on its own. Abliteration alone unlocks 0.8502 of attempts on
Qwen3-4B and 0.8379 on Phi-4. The best composed attack, abliteration + `role_chaining`, adds a
gain in the mean unlock rate of 0.0492 and 0.0700 respectively for Qwen3-4B and Phi-4, and the
gain as a percentage share of the non-unlocked samples, is 0.3284 with a 95% interval of
[0.1947, 0.4337] on Qwen3-4B and 0.4315 with [0.3417, 0.5135] on Phi-4.

Out of the ~200,000 generations run for this experiment, 3,749 of them were 'malformed',
meaning the judge did not return a legitimate grading. For the sake of conservatism, these
malformed results were treated as "non-unlocks", and headline results are based on that
information. A Manski pair is a concrete way to gauge the effect of our decision on how to
count malformed judge results: it reports the two extremes instead, resolving every malformed
grade first in the direction that most disfavours the contrast and then in the direction that
most favours it. Neither of the two previous results are changed by how the malformed results
are counted: the per-query mean difference is bounded by [0.0965, 0.1347] on Qwen3-4B and
[0.0642, 0.0891] on Phi-4, and the composed arm's gain over abliteration alone by [0.0252,
0.0871] and [0.0532, 0.0907]. It is also worth mentioning that in a manual audit of 150
malformed generations, 144 were deemed to be true unlocks, which suggests that results are
likely closer to the higher bound of the Manski pair result, and the mechanism of malformed
generations is likely due to guardrails on gpt-4o-mini when judging generations that detail
compliance with harmful queries.

The two attacks also do not reach the same queries. Comparing the base model's best prefill
against abliteration alone query by query by using a two-sided Fisher's exact test on each of
the 313 queries, and Benjamini-Hochberg at a 5% false discovery rate within each model, flags
59 queries on Qwen3-4B and 61 on Phi-4. Both directions are populated on both models: the
prefill arm unlocks at the higher rate on 9 of them, compared to abliteration alone on 50
queries for Qwen3-4B. For Phi-4 it is 17 queries unlocked at a higher rate by the prefill arm
against 44 for abliteration. Each attack reaches queries the other does not, and neither is a
substitute for the other.

Across an attack budget of 1 to 10 attempts, the abliterated arms start saturated and have
little room left: `abliterated/role_chaining` covers 0.899 of queries at a single attempt on
Qwen3-4B and 0.908 on Phi-4. For many of the other attacks, repetition alone is enough to get
to extreme effectiveness: by ten attempts the probability of at least one success is close to
99% for abliteration with no prefill at all (0.983 on Qwen3-4B and 0.986 on Phi-4) and for the
strongest single prefill family on the unmodified base model, `base/role_chaining` (0.988 on
Qwen3-4B and 0.995 on Phi-4). This emphasizes that the most powerful tool available to our
threat model's attacker is repetition, and a budget of ten attempts largely removes the need to
compose the two attacks or to pick a sophisticated strategy at all. What repetition does not do
is rescue an attack that almost never works: the unprefilled base model is still at 0.088 on
Qwen3-4B and 0.027 on Phi-4 after ten attempts.

The computed tables are in `results/analysis/`: `headline.csv`, `report.md`,
`results-<model>.json`, `query-overlap.csv`, `budget-curves.csv`, and the `counts.csv`
aggregate the rest derive from.

## Run record

The reported run was generated **2026-09-14 to 2026-09-17** on one rented NVIDIA
A100-SXM4-80GB, both models sequentially, conditions strictly one at a time.

| | |
|---|---|
| Generation | 73.06 h; the machine billed 73.15 h at $1.59/h — **$118.34** |
| Samples | 50,080 per condition, 100,160 per model, **200,320 for the package** |
| Census | four conditions, each 50,080 samples under 50,080 unique ids, every log `status: success` |
| Grading | 2026-09-17, StrongREJECT rubric via `gpt-4o-mini-2024-07-18` at temperature 0 — $86.48 |

Every generation stamps its own environment into the eval log header at
`eval.metadata.environment`, so a log says what produced it rather than pointing at a figure
measured elsewhere. The reference environment was Python 3.12, torch 2.11.0+cu128,
transformers 5.14.1, BF16.

Inputs frozen before the run, and where each is recorded:

| Frozen input | Recorded in |
|---|---|
| Prompt sets and splits | `data/freeze_manifest.json` |
| The prefill table | `data/prefill_manifest.json` |
| Primary abliteration layer, per model | `data/<model-slug>/abliteration_manifest.json` |
| Batch width, per model | `data/<model-slug>/batch_sweep.json` |
| Budget-curve attack ordering | `data/pilot_rank_ordering.json` |
| Draw schedule | `src/study/__init__.py` — `none` 20, `static_baseline` 20, each of the 12 generated slots 10 |

Batch width is not a tuning knob. Greedy output is not width-invariant, so conditions
generated at different widths are not comparable; the width is fixed across a model's
conditions and measured per model, since depth and KV footprint differ. What is frozen is
the selection rule, in `src/harness/width.py`.

## Setup

```bash
uv sync --extra dev           # openai, pandas, pytest, transformers, inspect-ai (no torch)
export OPENAI_API_KEY=...     # grading calls the judge
```

Python 3.12 is pinned in `.python-version`; `uv sync` provisions it. Generation and prefill
production additionally need a CUDA GPU:

```bash
uv sync --extra gpu           # torch comes from the lock, not the host image
export HF_HOME=/workspace/hf  # keep the model cache on persistent storage
```

Run everything through `uv run`. It re-syncs from `uv.lock` first, which is what holds the
pinned versions in effect — a bare `python` resolves whatever the ambient interpreter happens
to have.

## Re-running the study

In pipeline order. Steps 2-5 and 7 need the GPU extra; grading needs an API key and no GPU;
the rest are local.

| # | Step | Needs | Command |
|---|---|---|---|
| 1 | Freeze the prompt sets and splits | local | `uv run python scripts/build_datasets.py` |
| 2 | Extract per-layer refusal directions | GPU | `uv run python scripts/extract_directions.py <model-slug>` |
| 3 | Measure the batch width, then set `BATCH` in the model's module | GPU | `uv run python scripts/batch_sweep.py <model-slug>` |
| 4 | Selection sweep: generate every layer plus the base | GPU | `uv run python scripts/generate.py --prompt-set validation results/selection/generated <model-slug>` |
| 5 | Grade the sweep | API | `uv run python scripts/grade.py results/selection/generated results/selection/scored` |
| 6 | Pick and freeze the primary layer | local | `uv run python scripts/freeze_abliteration.py --run results/selection --model <model-slug> --excerpts 2`, then again with `--write` |
| 7 | Measure the helper's batch width | GPU | `uv run python scripts/helper_batch_sweep.py /dev/shm/gemma` |
| 8 | Produce the prefills, one wave at a time | GPU | `uv run python scripts/produce_prefills.py results/prefills /dev/shm/gemma` |
| 9 | Freeze the prefill table | local | `uv run python scripts/freeze_prefills.py results/prefills` |
| 10 | Freeze the budget-curve ordering | local | `uv run python scripts/pilot_rank.py --run <pilot-run> --write` |
| 11 | Main run | GPU | `uv run python scripts/generate.py --prompt-set strongreject results/main/generated <model-slug>` |
| 12 | Grade the main run | API | `uv run python scripts/grade.py results/main/generated results/main/scored` |
| 13 | Compute the reported metrics | local | `uv run python scripts/metrics.py --run results/main --out results/analysis` |
| 14 | Run health readout | local | `uv run python scripts/run_report.py --run results/main --model <model-slug>` |

Notes that are operating instructions rather than method:

- **Re-running resumes.** Each condition gets its own log directory and `eval_set` re-runs
  only the samples missing from it, so a killed sweep restarts with the same command. Its
  own retries are off: a failure here is usually systematic, and an automatic retry is the
  one thing that would decide unattended to spend GPU time.
- **A main run needs a fresh log root.** `eval_set` judges a log complete by sample count and
  matches samples by id, so pointing a full run at a directory holding a smaller earlier run
  silently reuses those generations.
- **Step 10 must precede any reading of main-run results.** The ordering the budget curves
  use is derived from a pilot; freezing it afterwards would let the reported curves be
  ordered by the data they describe.
- **Prefills are generated once and frozen, not generated per run.** Each `(prompt, slot)`
  string is consumed by every cell that uses that prefill level — both model states, every
  draw, both models — so generating inline would confound "this model state unlocked it" with
  "a different prefill unlocked it". It is also why adding a model needs no regeneration: the
  strings are keyed by prompt, not by target.
- **Rebuild the prefill table from the wave logs, never by re-running the helper.** Batch
  composition differs between runs, so a fresh draw is different text under a different hash,
  not the same set recovered. `scripts/freeze_prefills.py` is idempotent over the logs.
- **`data/prefills.csv` is not in this repository.** A fresh clone does not have it, and a
  prefilled run fails in `build_dataset` once the model is already loaded. Put the table in
  place first and check it against `table.sha256` in `data/prefill_manifest.json`.
