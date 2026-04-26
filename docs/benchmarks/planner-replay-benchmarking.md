---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Replay Benchmarking
subtitle: Drive the planner in the loop against a saved trace to evaluate SLA behavior and scaling decisions
---

This guide shows how to benchmark the Dynamo Planner against a recorded trace by running it inside the mock replay harness. Use it to compare `agg` vs `disagg` topologies, tune SLA targets, and study how deployment realities (engine startup time, worker counts) affect planner behavior — all without bringing up a live cluster.

For the general mechanics of trace replay (input format, arrival speedup, router modes, synthetic workloads), see [Mocker Offline Trace Replay](mocker-trace-replay.md). This guide focuses on the `--planner-config` path.

## 1. Setup

### Build

Build the Dynamo Python bindings so `python -m dynamo.replay` is available:

```bash
cd lib/bindings/python
.venv/bin/maturin develop --release
```

The `--release` flag is strongly recommended. Replay simulation is largely single-threaded and CPU-bound on the mocker engine core; a debug build can be 5–10× slower, which compounds across sweep runs.

### Key Planner Config Knobs

Passed as JSON via `--planner-config`. Uses the same schema as the live planner. The fields most relevant to benchmarking:

| Field | Purpose |
|---|---|
| `mode` | `"agg"` or `"disagg"` — picks scaling strategy and required engine args. |
| `optimization_target` | `"sla"` uses TTFT/ITL targets; `"throughput"` uses static queue/KV thresholds. |
| `ttft` / `itl` | SLA targets in ms. Drives load-scaling decisions. |
| `enable_throughput_scaling` | Periodic scaling based on predicted steady-state load. |
| `enable_load_scaling` | Reactive scaling to short-term traffic spikes. |
| `throughput_adjustment_interval` | Seconds between throughput-scaling decisions. |
| `load_adjustment_interval` | Seconds between load-scaling decisions. Short intervals mean faster reaction but more flapping. |
| `pre_deployment_sweeping_mode` | `"rapid"` uses the AIC analytical model; leave unset to fall back to recorded profile data. |
| `prefill_engine_num_gpu` / `decode_engine_num_gpu` | GPUs per engine replica. **Must be set explicitly** — both default to `None`, and the replay adapter silently treats `None` as `0`, which collapses the cumulative-GPU-hours metric in the report to zero. |
| `report_filename` | Output HTML filename under `./planner_reports/`. |

### Key Engine Arg Knobs

Passed as JSON via `--extra-engine-args` (agg) or `--prefill-engine-args` / `--decode-engine-args` (disagg). The replay harness uses the mocker engine, so "engine args" means the analytical perf model inputs:

| Field | Purpose |
|---|---|
| `aic_backend` | Backend the analytical model should emulate, e.g. `"vllm"`, `"trtllm"`, `"sglang"`. |
| `aic_system` | GPU SKU for the perf model, e.g. `"h200_sxm"`, `"h100_sxm"`, `"b200"`. |
| `aic_model_path` | HF model identifier used by the perf model. |
| `aic_tp_size` | Tensor-parallel size of each engine replica. |
| `startup_time` | Simulated seconds between a planner scale-up decision and the new worker becoming active. Unset or `0` means workers activate instantly. |

Other fields follow the standard mocker engine protocol (see [Mocker Offline Trace Replay](mocker-trace-replay.md)).

## 2. Example: Agg vs Disagg On The Mooncake Agentic Trace

Download the trace:

```bash
mkdir -p traces/mooncake_fast25 && cd traces/mooncake_fast25
curl -sLO https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl
```

Run agg (2 workers, TP=1):

```bash
.venv/bin/python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --planner-config '{
    "mode": "agg",
    "optimization_target": "sla",
    "ttft": 1500, "itl": 50,
    "enable_throughput_scaling": true, "enable_load_scaling": true,
    "pre_deployment_sweeping_mode": "rapid",
    "throughput_adjustment_interval": 300, "load_adjustment_interval": 10,
    "prefill_engine_num_gpu": 1, "decode_engine_num_gpu": 1,
    "report_filename": "replay_agg.html"
  }' \
  --extra-engine-args '{"aic_backend": "vllm", "aic_system": "h200_sxm", "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1}' \
  --num-workers 2 --arrival-speedup-ratio 1.0
```

Run disagg (1P1D, TP=1):

```bash
.venv/bin/python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --planner-config '{
    "mode": "disagg",
    "optimization_target": "sla",
    "ttft": 1500, "itl": 50,
    "enable_throughput_scaling": true, "enable_load_scaling": true,
    "pre_deployment_sweeping_mode": "rapid",
    "throughput_adjustment_interval": 300, "load_adjustment_interval": 10,
    "prefill_engine_num_gpu": 1, "decode_engine_num_gpu": 1,
    "report_filename": "replay_disagg.html"
  }' \
  --prefill-engine-args '{"aic_backend": "vllm", "aic_system": "h200_sxm", "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1}' \
  --decode-engine-args  '{"aic_backend": "vllm", "aic_system": "h200_sxm", "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1}' \
  --num-prefill-workers 1 --num-decode-workers 1 --arrival-speedup-ratio 1.0
```

Each run prints the AIPerf summary table to stdout and writes an HTML diagnostics report to `./planner_reports/<report_filename>`. For this trace with a long ISL and short OSL, agg is better than disagg, which gets slightly better ITL at the cost noticeably more GPU-hours.

## 3. Example: Cold-Start-Time Sweep

How sensitive is SLA attainment to engine startup time? Sweep `startup_time` from 0 to 300 seconds in 10-second steps and record TTFT/ITL/GPU-hours per run.

```bash
#!/usr/bin/env bash
set -euo pipefail

TRACE=traces/mooncake_fast25/toolagent_trace.jsonl
OUT=planner_reports/sweep_startup
mkdir -p "$OUT"

run_one() {
  local s=$1
  local name=$(printf "replay_agg_startup_%03d.html" "$s")
  local extra
  if [[ "$s" -eq 0 ]]; then
    extra='{"aic_backend":"vllm","aic_system":"h200_sxm","aic_model_path":"nvidia/Llama-3.1-8B-Instruct-FP8","aic_tp_size":1}'
  else
    extra=$(printf '{"aic_backend":"vllm","aic_system":"h200_sxm","aic_model_path":"nvidia/Llama-3.1-8B-Instruct-FP8","aic_tp_size":1,"startup_time":%d}' "$s")
  fi
  .venv/bin/python -m dynamo.replay "$TRACE" \
    --planner-config "$(printf '{"mode":"agg","optimization_target":"sla","ttft":1500,"itl":50,"enable_throughput_scaling":true,"enable_load_scaling":true,"pre_deployment_sweeping_mode":"rapid","throughput_adjustment_interval":300,"load_adjustment_interval":10,"prefill_engine_num_gpu":1,"decode_engine_num_gpu":1,"report_filename":"%s"}' "$name")" \
    --extra-engine-args "$extra" \
    --num-workers 2 --arrival-speedup-ratio 1.0 \
    --report-json "$OUT/startup_$(printf '%03d' "$s").json" \
    >"$OUT/startup_$(printf '%03d' "$s").log" 2>&1
}

export -f run_one
# Run 12 sweeps in parallel; adjust -P for your machine.
seq 0 10 300 | xargs -n1 -P12 -I{} bash -c 'run_one "$@"' _ {}
```

Each run emits the AIPerf metrics table (parse TTFT / ITL avg / p90) and its HTML report (grep `GPU hours: <float>`). Plotting those against `startup_time` gives:

![Planner replay — startup time sweep](../assets/img/planner-replay-startup-sweep.png)

Observations from this sweep (agg, TTFT SLA 1,500 ms, ITL SLA 50 ms, H200-SXM, Llama-3.1-8B-FP8, TP=1):

- **SLA cliff near 100–120 s.** Below that, the planner scales up fast enough to hold TTFT; above it, p99 TTFT diverges and the system stays perpetually backlogged.
- **Scaling-event count drops monotonically** (42 → 8) as startup grows — long-startup runs require load planner to wait for stabilization before the next scaling decision.
- **ITL is less sensitive than TTFT** until the queue saturates. Below the cliff, ITL rises modestly (25 → 30 ms avg); above it, p90 ITL jumps to ~200 ms as decode requests starve.
