---
name: dynamo-benchmark
description: >-
  Benchmark a deployed NVIDIA Dynamo workload with NVIDIA AIPerf or one
  of the in-tree benchmark suites. Configure ISL/OSL distributions,
  concurrency or request-rate ramps, and duration; run against the
  Frontend service or directly against worker pods; compare results to
  published recipe numbers. Use when measuring TTFT, ITL, throughput,
  KV-cache hit rate, or any other production-grade metric on a Dynamo
  deployment.
version: 1.2.0
author: NVIDIA
tags:
  - dynamo
  - benchmark
  - aiperf
  - performance
  - sla
tools:
  - Shell
  - Read
  - Write
---
<!--
Progressive Disclosure:
- Level 1 (YAML front matter): Trigger matching on "benchmark Dynamo", "AIPerf", "TTFT measurement", "KV cache hit rate", "throughput test"
- Level 2 (this file): 4-phase benchmarking workflow
- Level 3: references/ for AIPerf invocation, benchmark suites, recipe benchmark companions
           scripts/  for the pre-benchmark target check

Scope: post-deployment performance measurement against a running Dynamo workload
Out of scope: deployment (see dynamo-deploy), planning (see dynamo-plan), optimization (see dynamo-optimize),
              troubleshooting (see dynamo-troubleshoot).
-->

# Dynamo Benchmark (Performance Measurement)

Drive a deployed Dynamo workload with NVIDIA AIPerf (or one of the
in-tree suites), capture TTFT/ITL/throughput/KV-hit-rate, and compare
against published recipe numbers. The deployment must already be
serving traffic — this skill does not deploy.

## Workflow

Strict 4-phase workflow. Always follow in order. Never skip a phase.

```
Phase 1: Pre-Check → target deployment is serving, AIPerf installed, /v1/models populated
Phase 2: Configure → benchmark spec: ISL/OSL, concurrency or request-rate, duration
Phase 3: Run       → AIPerf or in-tree suite against the Frontend
Phase 4: Analyze   → parse results, compare to recipe baseline, publish
```

---

## Command Safety

Benchmarking is mostly client-side load generation. Two concerns: load
generation can saturate or evict production traffic, and result
publishing can leak proprietary numbers.

### MUTATING — require confirmation with explanation

| Command Pattern | Impact |
|---|---|
| `aiperf benchmark ...` | Drives load against `<endpoint>`. Saturates the worker(s); other traffic in the same DGD experiences contention. |
| `aiperf benchmark ... --concurrency <large>` | High-concurrency runs evict KV cache from production tenants if the deployment is shared. |

### SAFE — no confirmation needed

```
aiperf --version
aiperf benchmark --help
kubectl get pods -n <ns> -l nvidia.com/dgd-name=<name>
curl http://<frontend>/v1/models
curl http://<frontend>/metrics
ls ai-dynamo/dynamo/recipes/<model>/<framework>/<config>/benchmark/
```

### DESTRUCTIVE — always require explicit confirmation

| Command Pattern | Risk |
|---|---|
| `aws s3 cp results.json s3://<public-bucket>/...` | Publishes results to a public location; verify the numbers are clean before pushing. |
| Posting numbers to a public-facing dashboard or blog post draft | Requires sign-off; the skill refuses unless explicitly approved per the user's environment. |

---

## Phase 1: Pre-Check

**Goal.** Confirm the target deployment is healthy and AIPerf is installed.

**Inputs**:

| Input | How to get it |
|---|---|
| Target DGD name | `kubectl get dgd -A` |
| Frontend service / endpoint | `kubectl get svc -n <ns> -l app.kubernetes.io/component=frontend` |
| AIPerf install | `pip install ai-perf` or via the pinned recipe `requirements.txt` |
| Recipe baseline (optional) | `recipes/<model>/<framework>/<config>/benchmark/` for the comparison target |

**Commands** (SAFE):

```bash
# Confirm AIPerf.
aiperf --version

# Confirm /v1/models is populated.
kubectl port-forward -n <ns> svc/<frontend-svc> 8000:8000 &
sleep 2
curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("data"), "no models registered"; print(d["data"][0]["id"])'

# Confirm pods are healthy.
kubectl get pods -n <ns> -l nvidia.com/dgd-name=<dgd-name>
```

Run the bundled pre-check:

```bash
bash scripts/precheck-target.sh -d <dgd-name> -n <ns>
```

**Verification gate.** Deployment serving, `/v1/models` non-empty, AIPerf installed.

---

## Phase 2: Configure the Benchmark

**Goal.** Specify the workload shape, concurrency or rate, duration, and
which metrics to capture.

The configuration depends on the benchmark mode. See
[references/aiperf-invocation.md](references/aiperf-invocation.md) for
the AIPerf CLI surface and
[references/benchmark-suites.md](references/benchmark-suites.md) for the
in-tree suites under `benchmarks/`.

**Required parameters**:

| Parameter | Source |
|---|---|
| ISL (input sequence length) | Mean of production workload, or recipe `benchmark/run.sh` default |
| OSL (output sequence length) | Same |
| Concurrency **or** request-rate | One required. Concurrency = open-loop fixed parallel requests. Rate = closed-loop target req/s. |
| Duration | At least 5 minutes for stable measurements; 30+ min for KV-cache warm-up |
| Endpoint | `http://localhost:8000/v1/chat/completions` (port-forwarded) or the in-cluster service |
| Metrics | Default: TTFT, ITL, throughput. Add: cache_hit_rate, queue_depth, GPU utilization |

**Decision:** This benchmark will drive ~`<X>` concurrent requests
against the deployment for ~`<duration>`. If this deployment is shared,
other tenants will experience contention. Confirm.

Wait for explicit yes if the deployment is shared.

**Verification gate.** Workload spec, concurrency/rate, duration, and
endpoint all set.

---

## Phase 3: Run

**Goal.** Execute the benchmark and capture the raw results.

### 3.A AIPerf (standard path)

```bash
aiperf benchmark \
  --model <model-id-from-/v1/models> \
  --endpoint-type chat \
  --url http://localhost:8000/v1/chat/completions \
  --synthetic-input-tokens-mean <isl> \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean <osl> \
  --output-tokens-stddev 0 \
  --concurrency <n> \
  --measurement-interval <duration-sec> \
  --output-format json \
  --artifact-dir ./aiperf-artifacts
```

Verbatim invocation patterns per benchmark mode are in
[references/aiperf-invocation.md](references/aiperf-invocation.md).

### 3.B Recipe Benchmark (when a recipe exists)

```bash
RECIPE=/Users/dagil/dynamo/recipes/<model>/<framework>/<config>
bash $RECIPE/benchmark/run.sh
```

The recipe's `benchmark/run.sh` invokes AIPerf with the parameters used
to produce the recipe's **published** numbers. Use this when:

- Reproducing published numbers.
- Verifying a deployment matches the recipe baseline.
- Establishing a regression baseline.

### 3.C In-tree Suite (specialized workloads)

For workloads AIPerf doesn't directly support (BurstGPT traces,
sinusoidal load, agent traces), use the in-tree suites:

```bash
cd /Users/dagil/dynamo/benchmarks/<suite>
bash run.sh --target http://localhost:8000/v1/...
```

Available suites (per lifecycle map and the survey's §9.5):
`agent_trace`, `burstgpt_loadgen`, `frontend`, `incluster`, `llm`,
`multimodal`, `nat_trace`, `omni`, `prefix_data_generator`, `router`,
`sin_load_generator`.

**Decision (MUTATING):** Starting the run. The deployment will be
under load for `<duration>`. Other tenants on the same DGD experience
contention. Proceed?

Wait for explicit yes.

**Verification gate.** Run completes; results in
`./aiperf-artifacts/` (or the suite's output dir).

---

## Phase 4: Analyze and Publish

**Goal.** Parse results, compare to baseline, decide whether to publish.

### 4.1 Parse Results

AIPerf produces a JSON in `./aiperf-artifacts/results.json` with at
least these top-level fields:

```json
{
  "model": "...",
  "concurrency": 32,
  "duration_sec": 300,
  "time_to_first_token_ms": {"p50": 42.1, "p95": 58.3, "p99": 71.4},
  "inter_token_latency_ms": {"p50": 8.7, "p95": 12.4, "p99": 15.1},
  "throughput_req_per_sec": 56.2,
  "throughput_tokens_per_sec": 28100,
  "request_count": 16860
}
```

Suite-specific output formats are documented under each suite's
`benchmarks/<suite>/README.md`.

### 4.2 Compare to Recipe Baseline

If a recipe baseline exists at
`recipes/<model>/<framework>/<config>/benchmark/results.json`:

```bash
python3 - <<PYEND
import json
ours = json.load(open('./aiperf-artifacts/results.json'))
ref  = json.load(open('/path/to/recipe/benchmark/results.json'))
print(f"TTFT  p50: ours={ours['time_to_first_token_ms']['p50']:.1f} vs recipe={ref['time_to_first_token_ms']['p50']:.1f}")
print(f"ITL   p50: ours={ours['inter_token_latency_ms']['p50']:.1f} vs recipe={ref['inter_token_latency_ms']['p50']:.1f}")
print(f"Throughput req/s: ours={ours['throughput_req_per_sec']:.1f} vs recipe={ref['throughput_req_per_sec']:.1f}")
PYEND
```

Regression threshold: typically 10% degradation on TTFT or throughput
is the bar for opening an NVBug. Smaller variance is normal between
runs.

### 4.3 Publish (optional)

**Decision (DESTRUCTIVE):** Publishing the results to `<destination>`
exposes the numbers to `<audience>`. Confirm the numbers are accurate
and that this destination is correct for the release.

Wait for explicit yes.

Publish paths:
- Internal: append to `~/dynamo-tpm/releases/<release>/benchmark_results/`.
- Public: a blog post draft, the `recipes/<model>/<framework>/<config>/benchmark/` companion (a recipe PR), or a published dashboard.

**Verification gate.** Results parsed, baseline comparison done,
publish destination confirmed (or skipped).

---

## Refusal Conditions

- Target deployment is in a production-protected namespace and the
  benchmark would saturate it (only safe in isolated test environments
  unless explicitly approved).
- The deployment is not in a stable state (`/v1/models` empty, recent
  pod restarts, Planner actively scaling).
- Publishing numbers to a public destination without sign-off from the
  user's environment's release process.

---

## Cross-Skill Referencing

| Need | Sibling |
|---|---|
| The deployment under test | [dynamo-deploy](../dynamo-deploy/SKILL.md) — must be running before this skill starts |
| Re-plan based on benchmark results showing SLA miss | [dynamo-plan](../dynamo-plan/SKILL.md) |
| Quantize and re-benchmark | [dynamo-optimize](../dynamo-optimize/SKILL.md) |
| Benchmark reveals a runtime issue | [dynamo-troubleshoot](../dynamo-troubleshoot/SKILL.md) |

---

## Prerequisites

Phase 1 (the workflow's first phase) verifies the full readiness state. The short list:

- A deployed Dynamo workload reporting Ready (`/v1/models` populated).
- AIPerf (`ai-perf`) installed; version aligned with the Dynamo release line via the recipe `benchmark/run.sh` pins.
- Network reachability to the Frontend service (port-forward or gateway).
- Workload spec (ISL / OSL distribution, concurrency or rate target, duration).

---

## Limitations

What this skill does NOT cover:

- Does NOT deploy the workload. See `dynamo-deploy`.
- Does NOT diagnose performance regressions beyond surfacing them. See `dynamo-troubleshoot`.
- Recipe-attached benchmark numbers are reproducible only against the recipe's exact release pin (per); cross-release comparisons require alignment.

See `## Cross-Skill Referencing` for the appropriate sibling skill in each case.

---

## Troubleshooting

When a step in this workflow fails:

- **Per-skill known patterns** are catalogued in [references/known-issues.md](references/known-issues.md). Walk that list first.
- **Cross-skill day-2 patterns** (worker crashloops, conversion-webhook timeouts, KV-transfer fallback, gateway 500s) are owned by [dynamo-troubleshoot](../dynamo-troubleshoot/SKILL.md). The 4-phase day-2 workflow there (Triage → Inspect → Diagnose → Remediate) applies.
- **Per-release bugs** live in the active QA tracker (per the skill's `version` field). Pull the tracker view for the matching release line.

---

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/precheck-target.sh` | Pre-benchmark target readiness check (DGD Ready, /v1/models, AIPerf installed) | `-d <dgd-name> [-n <ns>] [--port-forward]` |

Each script implements the `pass/fail/warn` () or `check()` () pattern and exits non-zero on any failure. Output is structured for agent consumption.

Invocation via the [agentskills.io](https://agentskills.io/) `run_script` protocol:

```python
run_script("scripts/precheck-target.sh", args=["-d", dgd_name, "-n", namespace])
```

Equivalent direct invocation:

```bash
bash scripts/precheck-target.sh -d <dgd> -n <ns>
```

---

## References and Scripts

- [references/aiperf-invocation.md](references/aiperf-invocation.md) —
  AIPerf CLI invocations per benchmark mode (chat, completions, KV-aware
  routing, disagg).
- [references/benchmark-suites.md](references/benchmark-suites.md) —
  In-tree suites under `benchmarks/` with when-to-use guidance.
- [references/known-issues.md](references/known-issues.md) — Benchmarking-
  specific stable issue patterns.
- [scripts/precheck-target.sh](scripts/precheck-target.sh) — Pre-benchmark
  target readiness check. PASS/FAIL/WARN per.
