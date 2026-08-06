<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Performance analysis: gaps found

Findings from the investigation behind the performance analysis front door on branch
`gluo/perf-analysis-front-door`. All six are outside that change's scope. All evidence was
verified against `origin/main` @ `c574e4d7c1a`.

Ordered by severity. Each entry is written to be liftable into a GitHub issue as-is.

---

## 1. Simulation fidelity is asserted, never measured

**Severity:** high — sizing decisions are made on this.

**Evidence.** No calibration harness exists anywhere in the repo: nothing runs one trace
through both DynoSim and a real cluster and reports the delta.

`docs/fern/pages/cli/operations/simulation-with-dynosim/simulation-model.md:60`:

> The default model is an uncalibrated synthetic baseline. Prefill latency follows a
> polynomial over the uncached tokens scheduled in the pass.

The same file's **Fidelity Boundaries** section (line 176) lists outright omissions:

- KV capacity and state transitions are modelled; KV tensor payloads are not.
- TensorRT-LLM disaggregation and SGLang KVBM lower-tier movement are "not modeled".
- "Mocker simulates text-token processing; it does not model multimodal encoder or
  cross-attention compute."

**Why it matters.** Simulation is the recommended first step for sizing, so people choose
GPU counts and parallelism from it. The documentation is candid about the boundaries but the
*error* is never quantified, so model accuracy is an assumption rather than a known quantity.
The multimodal omission means DynoSim is structurally blind to encode-stage work.

**What fixing looks like.** A calibration harness that replays one reference workload through
both DynoSim and a real deployment and publishes the delta for TTFT, ITL, and throughput.
Even a single calibrated data point per supported backend would convert "unknown" into
"known and bounded".

**Related.** `docs/fern/pages/cli/operations/simulation-with-dynosim/overview.md:72` already
says "DynoSim narrows the search space; it does not replace real-hardware validation."

---

## 2. The benchmark catalog validator is not wired into CI

**Severity:** high, and nearly free to fix.

**Evidence.** `docs/fern/pages/recipes/_catalog/validate.py` genuinely validates entries
against `schema.json` (via `jsonschema`, with a required-top-level-keys fallback). No
workflow invokes it — grepping `.github/workflows/` and `.pre-commit-config.yaml` for
`validate.py` returns only an unrelated `policy/validate.py` mentioned in a comment, and no
workflow references `_catalog` or `feature-benchmarks` at all.

The `dynamo-docs` skill states it plainly: "The catalog validator is not yet wired into CI,
so run it by hand for any `_catalog/` change."

**Why it matters.** The benchmark catalog is the only declared provenance contract in the
repository. Its schema requires `claim`, `subtype`, `features`, `model`, `hardware`,
`traffic`, `arms`, `results`, and `maintainer`; `additionalProperties: false` catches
misspelled keys; and the validator separately checks that every `deploy` and `perf` asset
path resolves. Unenforced, all of that is convention rather than guarantee.

**What fixing looks like.** One workflow step:

```bash
python3 docs/fern/pages/recipes/_catalog/validate.py
```

gated on changes under `docs/fern/pages/recipes/**/_catalog/**`.

**Do this before gap 4.** Adding statistical fields to a schema nothing validates just
produces more unchecked text.

---

## 3. No enabled playbook for repeated, comparable benchmark runs

**Severity:** high. This is the cheapest large improvement available in performance work.

**Evidence.** Published benchmarks report one run per arm with no sample size or spread.
`docs/fern/pages/recipes/feature-benchmarks/qwen3-vl-embedding-cache.mdx:30-36` publishes:

| Metric | Cache ON | Cache OFF | Delta |
| --- | ---: | ---: | ---: |
| Output TPS (tok/s) | 3,575.6 | 3,072.3 | +16.4% |
| TTFT avg (ms) | 526.0 | 727.5 | -27.7% |

Grepping that page, its source `recipes/qwen3-vl-30b/README.md`, and the `perf.yaml` that
produced it for repetitions, median, standard deviation, or a confidence interval returns
nothing.

This is not carelessness — **the harness cannot currently do better**. Three blockers:

1. **Artifact paths have no run dimension.** Across the 33 `recipes/**/perf.yaml` files the
   directories are variable-driven (`${RUN_DIR}`, `${ARTIFACT_DIR}`) but carry no run index,
   so a second run of the same arm overwrites the first.
2. **No driver repeats a measurement.** Every loop in the four recipe drivers is argument
   parsing (`while [[ $# -gt 0 ]]`) or config iteration (`for cfg in "${CONFIGS[@]}"`). Not
   one repetition loop exists.
3. **Nothing aggregates across runs.** AIPerf summarizes *within* a run;
   `extract_throughput.py` reads one artifact directory. Given five artifact directories for
   one arm, no tool computes a median and range across them.

Recording is **not** blocked. `results.additionalProperties` is `true` in
`docs/fern/pages/recipes/feature-benchmarks/_catalog/schema.json`, so `runs_per_arm` and
`dispersion` can be added to any entry today. They stay unvalidated until the schema requires
them and gap 2 puts the validator in CI.

**Why it matters.** With one run per arm there is no basis for deciding whether a difference
is real. The published +16.4% is almost certainly genuine — the mechanism is understood and
the effect is large — but a +4% result would be indistinguishable from noise, and because
artifacts are overwritten it could not be checked afterwards.

Repetition was never actually unaffordable. The embedding-cache workload is 1,000 requests at
concurrency 64 with roughly 2.6 s average latency, about **40 seconds of load**. The expensive
part is deploying the worker, not running AIPerf; five runs against an already-deployed arm
cost about three extra minutes. The cost was never measured.

**What fixing looks like.** Three small code changes make a playbook followable, then the
playbook itself:

- **P1** — add a run index to artifact paths:
  `RUN_DIR="${ARTIFACT_BASE_DIR}/${CACHE_MODE}/run-${RUN_INDEX}"`.
- **P2** — a repetition loop in the driver, interleaved at the deployment level: deploy arm A
  and run N times, deploy arm B and run N times, then repeat the sequence in the opposite
  order so deployment-level variation is visible.
- **P3** — a cross-run aggregator that reads N artifact directories and emits median and
  min-max per statistic. New tooling, roughly 50 lines.
- **Playbook** — name the deciding statistic in advance; n >= 5 per arm; interleave; report
  median and range; **overlapping ranges mean inconclusive**, which today is not an available
  verdict.
- **Record** — `runs_per_arm`, `dispersion`, `interleaved`, and `deciding_statistic` in the
  catalog entry.

**Pilot.** `recipes/qwen3-vl-30b/vllm/agg-embedding-cache/`. Both arms share one `deploy.yaml`
differing by a single environment variable, the workload is short and fixed, and the effect is
large enough that a correct procedure must reproduce it — so it doubles as a calibration case
for the procedure itself.

**Explicitly not included.** Deciding whether a *small* difference in a *tail* statistic is
real. That is item 4, and folding it in here would make this item unshippable.

**Depends on** gap 2 for enforcement, but not for starting.

**See also** [ab-testing-request-distributions.md](ab-testing-request-distributions.md), a
standalone explainer of what is already solved (summarizing), what is not (which statistic
decides, and whether a difference is real), worked through the same benchmark.

---

## 4. No method for judging small differences in tail statistics

**Severity:** medium. Genuinely open, and the only item on this list that cannot be closed by
deciding to be more careful.

**Evidence.** The repository's one rigorous A/B procedure,
`.agents/skills/dynamo-kv-replay-parity/SKILL.md` Stage 7, consumes **one scalar per run** —
`replay_execution_ms`, timed narrowly around the replay call itself — and collects a balanced
60-pair schedule per row, so 120 invocations, each a fresh process.

Its power comes from four mechanisms, and sample size is the weakest:

1. **Narrow timing boundary** — excludes setup and initialization noise from the measurement
   rather than averaging over it.
2. **Pairing** — both arms run adjacently and are compared as a ratio, so common-mode drift
   such as frequency scaling and background load divides out.
3. **Balanced randomized order** — cancels ordering effects.
4. **n = 60 pairs** — tightens the interval around the median ratio.

None of the first three survives the move to hardware. You cannot draw a narrow timing
boundary around a distributed serving path the way you can around one function call, and
pairing is far weaker when each arm needs its own deployment rather than its own process. The
output is also a distribution rather than a scalar, so there is nothing for the ratio to
operate on.

**Why it matters.** Item 3's playbook resolves large effects through non-overlapping ranges.
It does not answer whether p99 time to first token moving from 520 ms to 545 ms is real.
Requests within a run are not independent — they share queueing state and cache warmth — so
they cannot be pooled as thousands of samples. And a p99 drawn from 10,000 requests rests on
the slowest 100 of them, so it is noisiest exactly where precision is wanted.

**What fixing looks like.** Statistical design work rather than a documentation task: choose
the unit of analysis, handle within-run non-independence, set a threshold appropriate to the
percentile in question, and fit all of it to a sample size affordable on GPUs.

**Status.** Named as a known gap on the shipped Performance Analysis Method page; conclusions
about tail differences are labelled provisional there.

---

## 5. No performance regression gate in CI

**Severity:** medium.

**Evidence.** `.github/workflows/pre-merge.yml` compiles and lints the feature-gated
`dynamo-bench` entrypoints, and runs a KVBM offload trace test for functional correctness.
None of it is timed. `pyproject.toml` registers a `benchmark` pytest marker and silences
`pytest-benchmark` warnings, but nothing is gated on it. Nothing tracks numbers over time.

**Why it matters.** Every performance regression is found by someone deciding to look.

**What fixing looks like.** Start narrow: time one deterministic offline replay on a fixed
trace per merge and fail on a threshold breach. That path is already deterministic and
GPU-free, so it is the cheapest place to put the first gate.

---

## 6. `wait_for_ready()` exists but 79 of 82 launch scripts do not use it

**Severity:** low, but high nuisance value.

**Evidence.** `examples/common/launch_utils.sh:222` defines `wait_for_ready <url> [timeout]`,
which polls an HTTP endpoint until it returns 200. Of the 82 launch scripts under
`examples/backends/*/launch/`:

- 3 call `wait_for_ready`
- ~10 hand-roll their own `/v1/models` poll loop
- 18 use a bare `sleep N`

`allocate_free_port()` in the same file is similarly under-used.

**Why it matters.** Every performance run against a backend topology requires the operator to
hand-roll readiness. The observed result is magic sleeps such as
`sleep 20  # Ensure encode worker and PD worker are not initialized concurrently`, which are
both slower than necessary and unreliable under load.

**What fixing looks like.** Adopt `wait_for_ready` against `/v1/models` in the launch scripts
that currently sleep. Mechanical, script by script, no new abstraction required.

---

## Suggested order

1. **Gap 2** — one workflow step, and it turns an existing contract into a guarantee.
2. **Gap 3** — three small code changes plus a written procedure, with a named pilot and a
   measured cost of about three extra minutes per arm. Best effort-to-value ratio on the list.
   Recording can start before gap 2 lands; enforcement follows it.
3. **Gap 1** — largest correctness risk, since deployments are sized on unvalidated output.
   Independent of the others.
4. **Gaps 5, 6** — as capacity allows.
5. **Gap 4** — last, not because it matters least but because it is the only item requiring
   real statistical design. Do not let it block gap 3.

## How these group

| Theme | Items | Kind of work |
| --- | --- | --- |
| Measurement rigor for published claims | 3, 4 | 3 is plumbing plus policy; 4 is statistical design |
| Enforcement of existing contracts | 2, 5 | CI wiring |
| Correctness of the tooling itself | 1 | needs an external ground truth |
| Ergonomics | 6 | mechanical |

Items 3 and 4 were a single entry in an earlier draft. They were split because 3 is
shippable — small, scoped, with a pilot — while 4 is open-ended, and combining them made the
whole thing unshippable.
