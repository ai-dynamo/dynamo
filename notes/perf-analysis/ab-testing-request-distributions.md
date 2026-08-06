<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Why Dynamo cannot currently tell a small performance win from noise

Supplement to [gaps.md](gaps.md) gap 3, "No procedure for A/B-ing request-level
distributions". This document is self-contained: it explains the problem from scratch, works
through a benchmark the repository already publishes, and proposes a concrete fix.

Verified against `origin/main` @ `c574e4d7c1a`.

---

## The setting

An A/B here means comparing two configurations of the same running system — KV routing on
versus off, embedding cache on versus off, aggregated versus disaggregated. Each
configuration is an **arm**. You run the same workload against both arms and want to say "B
is faster than A".

This is not about component benchmarks versus system benchmarks. Both arms are whole
deployments. The difficulty comes from something else.

## What a benchmark run actually gives you

Not a number. A serving benchmark issues thousands of requests and measures each one, so a
single run produces thousands of time-to-first-token values and thousands of inter-token
latency values. That is a **distribution**, not a measurement.

This is the root of the problem, and it is worth being precise about why, because one part of
it is already solved.

## What is already solved: summarizing

AIPerf reduces those thousands of values to summary statistics automatically — `avg`, `p50`,
`p90`, `p99`, `max` for time to first token, inter-token latency, and request latency, plus
throughput. Code reads them back by name, for example in
`components/src/dynamo/profiler/utils/aiperf.py`:

```python
float(aiperf_result["time_to_first_token"]["avg"])
float(aiperf_result["output_token_throughput"]["avg"])
```

Recipes go further and define a service-level summary directly:

```bash
--goodput "time_to_first_token:${TTFT_THRESHOLD_MS} inter_token_latency:${ITL_THRESHOLD_MS}"
```

So "thousands of latencies to a handful of numbers" is a solved, automated step. Nobody needs
to invent it.

## What is not solved

Two separate things, one cheap and one hard.

### Gap A — nothing says which summary decides the claim

Published work picks differently. The embedding-cache page leads with output throughput and
average time to first token. Recipes gate on goodput against threshold pairs. The profiler
optimizes against average time to first token and average inter-token latency.

This matters because **the summaries can disagree**. A change that batches more aggressively
usually improves throughput and the median while lengthening the tail. Both "faster" and
"slower" are then true, and whoever writes the page chooses which one becomes the headline.

This is a convention gap. It is cheap to close: state the deciding statistic before running.

### Gap B — nothing says whether an observed difference is real

This is the hard one, and it survives even after Gap A is settled.

Suppose everyone agrees the claim rests on p99 time to first token. You measure 520 ms on arm
A and 545 ms on arm B. Is B worse?

Nobody in this repository can answer that, for four compounding reasons:

1. **One run per arm.** Published benchmarks record no run count, so there is no spread to
   compare the difference against.
2. **You cannot simply run more and average**, at least not without deciding how many, which
   nothing states.
3. **You cannot pool the individual requests instead.** Requests within a run share queueing
   state and cache warmth, so request N is slow partly because request N-1 was. They are not
   independent samples, and treating them as thousands of samples manufactures confidence
   that is not there.
4. **Tail statistics are noisy by construction.** A p99 drawn from 10,000 requests rests on
   the slowest 100 of them, so its own run-to-run variability is large — exactly where you
   want the most precision.

## Why the existing procedure cannot be borrowed

The repository does contain one rigorous A/B procedure, in the offline KV replay parity
workflow. It works, and it is worth understanding why it does not transfer.

That procedure measures **one number per run**: `replay_execution_ms`, timed narrowly around
the replay call itself, deliberately excluding setup, engine preparation, and report
aggregation. It collects a balanced 60-pair schedule per row — 30 baseline-first and 30
candidate-first, randomized — so 120 invocations, each a fresh process, each replaying a
fixed 5,000-request corpus.

Its power comes from four mechanisms, and sample size is the weakest of them:

1. **A narrow timing boundary**, which excludes setup noise from the measurement rather than
   averaging over it.
2. **Pairing**, so both arms run adjacently and are compared as a ratio, which cancels
   common-mode drift such as frequency scaling and background load.
3. **Balanced randomized order**, which cancels ordering effects.
4. **Sample size**, which tightens the interval around the median ratio.

Note what is and is not deterministic there. Replay's *output* is byte-identical across runs;
its *elapsed time* is not. The procedure measures a noisy quantity of a deterministic
computation, which is the ideal case for pairing: the work is provably identical, so a timing
difference is either the change or ambient noise.

None of the top three mechanisms survives the move to hardware. You cannot draw a narrow
timing boundary around a distributed serving path the way you can around one function call,
and pairing is far weaker when each arm needs its own deployment rather than its own process.
And the output is a distribution, not a scalar, so there is nothing for the ratio to operate
on.

---

## A worked example

`recipes/qwen3-vl-30b/vllm/agg-embedding-cache/` compares the multimodal embedding cache on
one GB200 worker. Both arms use the **same** `deploy.yaml`; the only difference is
`DYN_MULTIMODAL_EMBEDDING_CACHE_GB`, set to 10 or 0. Workload from `perf.yaml`:
`REQUEST_COUNT=1000`, `CONCURRENCY=64`, `WARMUP_REQUEST_COUNT=3`, `MAX_TOKENS=150`, against a
200-image pool so roughly 80% of requests reuse an already-encoded image.

Published in `docs/fern/pages/recipes/feature-benchmarks/qwen3-vl-embedding-cache.mdx`:

| Metric | Cache ON | Cache OFF | Delta |
| --- | ---: | ---: | ---: |
| Output TPS (tok/s) | 3,575.6 | 3,072.3 | +16.4% |
| TTFT avg (ms) | 526.0 | 727.5 | -27.7% |
| TTFT p50 (ms) | 356.8 | 510.8 | -30.1% |

### What the procedure actually is

`run-benchmark.sh` patches the cache size in `deploy.yaml`, waits for the worker, patches
`CACHE_MODE` in `perf.yaml`, applies it, and AIPerf runs **once**. Running
`grep -c 'for\|while'` over that driver returns **0**. Someone then reads two summary values
per arm and writes the delta into the README.

### Two structural blockers

- **The artifact path has no run dimension.** `RUN_DIR="${ARTIFACT_BASE_DIR}/${CACHE_MODE}"`,
  so a second run of the same arm overwrites the first. The harness cannot accumulate
  repetitions even if someone wanted to.
- **Repetition is nearly free, and this was never measured.** 1,000 requests at concurrency
  64 with roughly 2.6 s average latency is about **40 seconds of load**. The expensive part
  is deploying the worker, not running AIPerf. Five runs against an already-deployed arm cost
  about three extra minutes.

Rigour was not traded away for cost here. The cost was never checked.

### Why this example also illustrates Gap A

With a 200-image pool and 80% reuse, cache ON produces a **bimodal** time-to-first-token
distribution: about 800 requests skip the vision encoder and about 200 do not. Cache OFF is
roughly unimodal, since every request encodes.

The two arms therefore differ in **shape**, not merely in location, and the summaries diverge
accordingly. The median falls in the fast mode and shows a large win. The p99 falls in the
slow mode and barely moves, because cache misses still pay full encode cost. The mean blends
two populations and describes neither.

The published -30.1% on median time to first token is real, but a substantial part of what it
reports is *"the cache hit rate is 80%"*. Change the reuse ratio and the headline moves with
no code change at all. That is why naming the deciding statistic, and the traffic it was
measured under, is not pedantry.

### The scenario that breaks today

Suppose the result had read 3,195 versus 3,072 — **+4%**.

Someone has to decide: ship it as a win, or investigate? They hold two numbers, each from one
40-second run, with no basis for choosing. If run-to-run spread is around 1%, +4% is solid.
If it is around 6%, even the sign is unestablished. Nothing in the repository records which,
and because the artifact directory is overwritten, it cannot be recovered afterwards.

---

## What to add

Four changes, in dependency order. None requires new statistical machinery.

### 1. Make runs accumulable

```bash
RUN_DIR="${ARTIFACT_BASE_DIR}/${CACHE_MODE}/run-${RUN_INDEX}"
```

One line. Nothing else on this list is possible without it.

### 2. Repeat, and interleave at the deployment level

Deploy arm A once and run AIPerf five times; deploy arm B once and run five times; then
repeat the whole sequence with the arms in the opposite order. Four deployments and twenty
40-second runs, under an hour. The reversal catches deployment-level variation that a single
block ordering hides.

### 3. Adopt a decision rule that needs no statistics library

Per arm, report the **median** of the per-run values and the observed **min-max range**.

| Observation | Verdict |
| --- | --- |
| Ranges do not overlap | Real difference |
| Ranges overlap | **Inconclusive** — collect more, or report it as inconclusive |

This is crude and makes no distributional assumptions, and it is far better than one run per
arm. Its most important property is that it makes **inconclusive an available answer**, which
today it is not.

Applied to the example: if cache ON is 3,575 over a range of 3,540-3,610 and cache OFF is
3,072 over 3,050-3,090, the ranges do not overlap and the claim stands. If a +4% result gave
3,195 over 3,100-3,290 against 3,072 over 3,010-3,140, they overlap and the claim does not
stand on that evidence.

### 4. Record it where the claim lives

Extend the `results` object in
`docs/fern/pages/recipes/feature-benchmarks/_catalog/schema.json`:

```yaml
results:
  available: true
  runs_per_arm: 5
  dispersion: "output TPS range 3540-3610 (ON), 3050-3090 (OFF)"
  interleaved: true
  deciding_statistic: output_token_throughput.avg
  summary: "+16.4% TPS ..."
```

`deciding_statistic` closes Gap A by forcing the author to name in advance which number the
claim rests on, rather than selecting the most favourable one afterwards.

This depends on gap 2 in [gaps.md](gaps.md): the catalog validator is not currently run by
CI, so schema fields are advisory until that is fixed.

---

## Why the embedding cache is the right pilot

Both arms share one manifest and differ by a single environment variable, the workload is
fixed and short, and the effect is large enough that a correct procedure must confirm it. If
the method cannot reproduce a 16% win, the method is wrong. That makes it a calibration case
as well as a first application.

## What remains genuinely open

The four changes above give a defensible answer for large effects. They do not solve the hard
case: deciding whether a small difference in a **tail** statistic is real, at a sample size
anyone can afford on GPUs. That needs real statistical design — choosing the unit of
analysis, handling non-independence within a run, and setting a threshold appropriate to the
percentile in question.

Until that exists, conclusions about tail latency differences should be labelled provisional.
The shipped Performance Analysis Method page says so rather than pointing at a procedure that
does not fit.
