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

## 3. No procedure for A/B-ing request-level distributions

**Severity:** medium.

**Evidence.** The repository has exactly one worked statistical procedure:
`.agents/skills/dynamo-kv-replay-parity/SKILL.md` Stage 7 — paired runs, arm order randomized
within each pair, a bootstrap confidence interval on the median candidate/baseline ratio, an
equivalence threshold, and an escalation rule when the interval straddles it.

That procedure consumes **one elapsed scalar per run** — `replay_execution_ms`, timed narrowly
around the replay call itself, deliberately excluding setup, engine preparation, and report
aggregation — and collects a balanced 60-pair schedule per row (30 baseline-first, 30
candidate-first, randomized), so 120 invocations per row, each a fresh process. That is
affordable only because an offline replay is fast, deterministic, and single-process.
`.agents/skills/dynamo-frontend-benchmark/SKILL.md` offers "interleave arms, take the median
of 3+", which is a heuristic rather than a method.

Note where its power comes from, because this is the part that does not transfer. Four
mechanisms, and sample size is the weakest:

1. **Narrow timing boundary** — excludes setup and initialization noise from the measurement
   entirely, rather than averaging over it.
2. **Pairing** — both arms run adjacently and are compared as a ratio, so common-mode drift
   (frequency scaling, thermal, background load) divides out.
3. **Balanced randomized order** — cancels ordering effects.
4. **n = 60 pairs** — shrinks the interval around the median ratio.

Note also what is and is not deterministic: replay's *output* is byte-identical across runs,
but its *elapsed time* is not. Stage 7 measures a noisy quantity of a deterministic
computation, which is precisely the condition that makes pairing so effective — the work is
provably identical, so a timing difference is either the change or ambient noise.

**Why it matters.** Most real questions are of the form "did p99 time to first token get
worse under load", where the unit of analysis is the request, not the run. Neither existing
procedure covers that, and the replay procedure's constants do not transfer: you cannot run
60 paired GPU sweeps, and a threshold calibrated to simulator runtime means nothing for a
latency percentile.

**What fixing looks like.** A written procedure for comparing latency distributions between
two arms, covering how many runs, how to pool or compare per-request samples, and what
constitutes a real difference at a given percentile.

**Status.** Named as a known gap on the shipped Performance Analysis Method page rather than
papered over with a citation that does not fit.

**Read next.** [ab-testing-request-distributions.md](ab-testing-request-distributions.md) is a
standalone explainer for this gap: what is already solved (summarizing), what is not (which
statistic decides, and whether a difference is real), worked through the embedding-cache
benchmark, with a four-step fix that needs no new statistical machinery.

---

## 4. Published performance numbers carry no sample size or dispersion

**Severity:** medium.

**Evidence.** `docs/fern/pages/recipes/feature-benchmarks/qwen3-vl-embedding-cache.mdx:30-36`
publishes:

| Metric | Cache ON | Cache OFF | Delta |
| --- | ---: | ---: | ---: |
| Output TPS (tok/s) | 3,575.6 | 3,072.3 | +16.4% |
| TTFT avg (ms) | 526.0 | 727.5 | -27.7% |

Grepping that page, its source `recipes/qwen3-vl-30b/README.md`, and the `perf.yaml` that
produced it for repetitions, median, standard deviation, or a confidence interval returns
nothing. As recorded, this is one run per arm, quoted to 0.1 ms.

The catalog schema's `results` object requires only `available: boolean` and accepts a
free-text `summary`, so `+16.4% TPS` validates with no sample size attached.

**Why it matters.** The bar is inverted. The repository demands 120 paired invocations and a
bootstrap interval before calling a small *simulator runtime* change real, while publishing
hardware claims off an unstated sample size. Large effects with understood mechanisms are
probably fine — but the reader cannot distinguish "large effect, obviously real" from
"single sample, unknown dispersion" because the record does not say.

**What fixing looks like.** Extend `results` in
`docs/fern/pages/recipes/feature-benchmarks/_catalog/schema.json` with `runs_per_arm`,
a dispersion field, and an `interleaved: boolean`. Blocked on gap 2 — the schema needs an
enforcer first.

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

1. **Gap 2** — one workflow step, turns an existing contract into a guarantee.
2. **Gap 1** — largest correctness risk; people size deployments on unvalidated output.
3. **Gap 4** — depends on gap 2.
4. **Gaps 3, 5, 6** — as capacity allows.
