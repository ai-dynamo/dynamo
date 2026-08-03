---
name: dynamo-benchmark
description: Routes a Dynamo performance question to the existing tool that answers it, then enforces the protocol that makes the number trustworthy. Use when asked to benchmark, profile, size, or A/B-test Dynamo and the right harness is not already obvious - a change that may have regressed, a GPU-count or parallelism decision, an unexplained latency or throughput problem, or a request for a headline number. Classifies by question type, then by substrate, component under test, and time budget, and hands off to offline replay, the mocker, the profiler, the frontend harness, or AIPerf against a deployment, and to the dynamo-frontend-benchmark, dynamo-recipe-runner, dynamo-router-starter, dynamo-interconnect-check, and dynamo-troubleshoot skills. Also covers preconditions that invalidate a measurement, warmup and interleaving, paired A/B design, and which signal to capture during a run.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - performance
    - benchmarking
    - profiling
    - routing
    - methodology
---

# Dynamo performance analysis

Pick the right tool for a performance question, then measure so the answer holds up.

This skill **routes and gates**. It runs nothing itself: every harness it points to already exists and is tested where it lives.

> **Maintenance rule: no number and no command-with-flags belongs in this file.** Name the file that owns it. Counts, thresholds, and flags stated twice diverge.

## Step 1 — Classify

What would count as an answer? This is interpretation of intent, and it picks the **family** of tool. Resolve it by asking, not guessing: "benchmark the router" is a different job depending on why.

| Class | The question | A finished answer |
| --- | --- | --- |
| REGRESSION | did my change slow things down? | a paired comparison with a stated interval |
| SIZING | how many GPUs, which parallelism? | a configuration plus the curve behind it |
| ATTRIBUTION | where is the time going? | a named function, stage, or component |
| CHARACTERIZATION | what is the throughput of X? | a number with its workload and concurrency |
| DIAGNOSIS | why is this slower than expected? | a root cause, or a violated precondition |

**DIAGNOSIS skips to Step 4.** Never route it to a benchmark first. Benchmarking a broken deployment produces a number describing the breakage — the most common failure mode in this repo's performance work.

## Step 2 — Constrain

Facts about the environment, looked up rather than interpreted. These pick **which member** of the family.

- **Substrate** — no GPU / one development host / Kubernetes. The hard gate; eliminates most options immediately.
- **Component under test** — frontend and router / engine and backend / KV transfer / whole system. This decides whether mock workers are legitimate: they are, and only, when the answer must not depend on the engine.
- **Time budget** — seconds / minutes / hours. Names the fidelity you will get, so nobody quotes a simulated estimate as a measurement.

Keep the class out of this list. Classification is not a constraint.

## Step 3 — Route

| Question and situation | Go to |
| --- | --- |
| REGRESSION — routing, scheduling, admission, cache accounting | Offline replay (`python -m dynamo.replay`, or the Rust entrypoints in `lib/bench`; invocation pattern in `.github/workflows/pre-merge.yml`). Deterministic and GPU-free, so byte-for-byte comparison beats timing. The result must satisfy **skill `dynamo-kv-replay-parity`** — parity before timing, and its Stage 7 *design* for the A/B |
| REGRESSION — frontend, tokenizer, dispatch | **skill `dynamo-frontend-benchmark`**, or `benchmarks/frontend/scripts/sweep_runner.py` for a grid |
| REGRESSION — engine, transport, or anything crossing the wire | Neither of the above sees it. Measure against a real backend |
| SIZING | `python -m dynamo.profiler`; DynoSim sweeps for a wider space. Read the fidelity warning below |
| ATTRIBUTION — CPU time | `benchmarks/frontend/scripts/{flamegraph,bpf,nsight}/`; build with the `profiling` Cargo profile. `dynamo-frontend-benchmark` documents the sharp edges |
| ATTRIBUTION — per-request stages | Request tracing, converted to a timeline |
| ATTRIBUTION — which component on a cluster | Grafana dashboards, then the metrics catalog |
| CHARACTERIZATION — supported model on Kubernetes | **skill `dynamo-recipe-runner`**, then that recipe's `perf.yaml` |
| CHARACTERIZATION — arbitrary endpoint | AIPerf directly |
| CHARACTERIZATION — no GPUs available | Frontend plus `python -m dynamo.mocker`. Measures Dynamo overhead, not serving performance |
| DIAGNOSIS — unhealthy deployment | **skill `dynamo-troubleshoot`** |
| DIAGNOSIS — disaggregated, KV transfer suspect | **skill `dynamo-interconnect-check`**, before quoting any disaggregated number |
| DIAGNOSIS — router mode suspect | **skill `dynamo-router-starter`** for the smoke check, then return here |

Five of those six skills ship scripts and are destinations. `dynamo-kv-replay-parity` ships none: it is a specification, not a runbook, so it is cited as the standard a result must meet, never as the thing to run.

## Step 4 — Gate on preconditions

See [references/measurement-protocol.md](references/measurement-protocol.md). Do not skip on the grounds that the deployment "was fine yesterday".

## Step 5 — Run under protocol

Same reference. Six invariants, no counts — the counts belong to the harness.

## Step 6 — Capture

Choose the signal by the question, not by what is easy to turn on. The reference has the table.

## Step 7 — Report

A result is quotable only if it states workload, concurrency, substrate, revision, how many runs, the spread across them, and which preconditions were verified. Published cross-configuration claims additionally follow the benchmark catalog contract.

## Worked routes

**"I changed the KV router, did it regress?"** REGRESSION, component = router. If the change affects offline replay semantics, replay it: no GPU, deterministic, strongest signal. If it is live-path only, use the frontend harness with mock workers so the engine is not a variable. Escalate to GPUs only if both come back clean and you still suspect an interaction.

**"How many GPUs for this model?"** SIZING. Time budget alone picks the mode: a fast simulation-backed pass for a first cut, a sweep for the design space, a real-GPU pass for a number you will publish. Simulation ranks; it does not measure.

**"Why is TTFT bad on my cluster deployment?"** DIAGNOSIS, not CHARACTERIZATION. Gate first — troubleshoot, then interconnect if disaggregated, then dashboards, then per-request traces. Measure last, if at all.

**"Is configuration A faster than B?"** CHARACTERIZATION plus REGRESSION, and the trap is that "faster" is undefined until workload and concurrency are fixed. Fix those first, deploy both arms, interleave.

## What this skill does not do

- It ships no scripts. Every executable it names is tested where it lives; a wrapper here would be a second implementation to keep in sync.
- It owns no tool's flags, counts, or thresholds.
- It does not decide whether a result is publishable. That is the reporting contract.
