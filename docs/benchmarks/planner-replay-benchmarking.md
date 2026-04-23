---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Replay Benchmarking
subtitle: Drive the planner in the loop against a saved trace to evaluate SLA behavior and scaling decisions
---

This guide shows how to benchmark the Dynamo Planner against a recorded trace by running it inside the mock replay harness. Use it to iterate on planner knobs (scaling intervals, SLA targets, predictors), compare `agg` vs `disagg` topologies, and study how deployment realities (engine startup time, worker counts) affect SLA attainment — all without bringing up a live cluster.

For the general mechanics of trace replay (input format, arrival speedup, router modes, synthetic workloads), see [Mocker Offline Trace Replay](mocker-trace-replay.md). This guide focuses on the `--planner-config` path.

## When To Use This

Use planner replay benchmarking to:

- Validate new planner features or config changes against a reproducible workload.
- Answer "what SLA can we hold under X trace, with Y GPUs, and Z startup latency?" before committing GPU time to a live run.
- Sweep across planner configurations in parallel and summarize TTFT / ITL / GPU-hours tradeoffs.
- Debug scaling oscillation or conservative-vs-aggressive planner behavior by inspecting the generated diagnostics HTML.

Note: the replay harness uses the mocker engine (analytical perf model via AI Configurator). Absolute latencies are model-informed estimates, not live measurements. Trends across configurations are what this tool is designed to compare.

## Prerequisites

Build the Dynamo Python bindings so `python -m dynamo.replay` is available:

```bash
cd lib/bindings/python
.venv/bin/maturin develop --release
```

The `--release` flag is strongly recommended. Replay simulation is largely single-threaded and CPU-bound on the mocker engine core; a debug build can be 5–10× slower, which compounds across sweep runs.

Get a trace in Mooncake JSONL format (`{timestamp, input_length, output_length, hash_ids}` per line). The public FAST25 release has four traces you can use directly:

```bash
mkdir -p traces/mooncake_fast25
cd traces/mooncake_fast25
for f in conversation_trace synthetic_trace toolagent_trace; do
  curl -sLO "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/${f}.jsonl"
done
curl -sLO "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
```

## Quick Start

### Aggregated (agg) replay

```bash
.venv/bin/python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --planner-config '{
    "mode": "agg",
    "optimization_target": "sla",
    "ttft": 1500,
    "itl": 50,
    "enable_throughput_scaling": true,
    "enable_load_scaling": true,
    "pre_deployment_sweeping_mode": "rapid",
    "throughput_adjustment_interval": 300,
    "load_adjustment_interval": 10,
    "prefill_engine_num_gpu": 1,
    "decode_engine_num_gpu": 1,
    "report_filename": "replay_agg.html"
  }' \
  --extra-engine-args '{"aic_backend": "vllm", "aic_system": "h200_sxm", "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1}' \
  --num-workers 2 \
  --arrival-speedup-ratio 1.0
```

### Disaggregated (disagg) replay

`disagg` mode requires separate engine args for prefill and decode, and `--num-prefill-workers` / `--num-decode-workers` instead of `--num-workers`:

```bash
.venv/bin/python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --planner-config '{
    "mode": "disagg",
    "optimization_target": "sla",
    "ttft": 1500,
    "itl": 50,
    "enable_throughput_scaling": true,
    "enable_load_scaling": true,
    "pre_deployment_sweeping_mode": "rapid",
    "throughput_adjustment_interval": 300,
    "load_adjustment_interval": 10,
    "prefill_engine_num_gpu": 1,
    "decode_engine_num_gpu": 1,
    "report_filename": "replay_disagg.html"
  }' \
  --prefill-engine-args '{"aic_backend": "vllm", "aic_system": "h200_sxm", "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1}' \
  --decode-engine-args  '{"aic_backend": "vllm", "aic_system": "h200_sxm", "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1}' \
  --num-prefill-workers 1 \
  --num-decode-workers 1 \
  --arrival-speedup-ratio 1.0
```

## Key Planner Config Knobs

The `--planner-config` JSON accepts the same schema as the live planner. For SLA benchmarking the most relevant fields:

| Field | Purpose |
|---|---|
| `mode` | `"agg"` or `"disagg"` — picks scaling strategy and required engine args. |
| `optimization_target` | `"sla"` uses TTFT/ITL targets; `"throughput"` uses static queue/KV thresholds. |
| `ttft` / `itl` | SLA targets in ms. Drives load-scaling decisions. |
| `enable_throughput_scaling` | Enables periodic scaling based on predicted steady-state load. |
| `enable_load_scaling` | Enables reactive scaling to short-term traffic spikes. |
| `throughput_adjustment_interval` | Seconds between throughput-scaling decisions. |
| `load_adjustment_interval` | Seconds between load-scaling decisions. Short intervals mean faster reaction but more flapping. |
| `pre_deployment_sweeping_mode` | `"rapid"` uses the AIC model; leave unset to fall back to recorded profile data. |
| `prefill_engine_num_gpu` / `decode_engine_num_gpu` | GPUs per engine replica. Required for the report's cumulative GPU-hours chart to be non-zero; if omitted, those fields default to `None` and GPU-hours computes as zero. |
| `report_filename` | Output HTML filename under `./planner_reports/`. |

## Engine-Arg Knobs That Matter For Planner Benchmarks

Beyond the standard AIC fields (`aic_backend`, `aic_system`, `aic_model_path`, `aic_tp_size`), one field strongly affects planner evaluation:

- **`startup_time`** (seconds) — simulated time between a planner scale-up decision and the new worker becoming active. When unset or `0`, new workers activate instantly. Real-world Kubernetes engine pods typically take 60–120 seconds to reach readiness, and modeling this is essential for evaluating whether the planner can scale fast enough. See the [case study](#case-study-engine-startup-time-sweep) below.

For `disagg` mode, pass `startup_time` on both `--prefill-engine-args` and `--decode-engine-args`.

## Output Artifacts

Each run emits:

- **AIPerf-style metrics table** printed to stdout — TTFT, ITL, Request Latency, Request Throughput, etc. (avg / min / max / p99 / p90 / p75 / std).
- **`./planner_reports/<report_filename>`** — Plotly HTML with per-tick diagnostics: replica counts, scaling events, predicted vs observed load, cumulative GPU hours. Open this in a browser to visually debug planner behavior.
- **`dynamo_replay_report_<timestamp>.json`** (or `--report-json <path>`) — full per-request timing data and aggregated metrics for downstream analysis.

## Viewing HTML Reports From A Remote Dev Machine

Planner reports regenerate on every run. The cleanest workflow for iterative benchmarking:

On the remote dev machine:

```bash
cd planner_reports && python -m http.server 8765
```

On your local machine:

```bash
ssh -N -L 8765:localhost:8765 <remote-host>
```

Then open `http://localhost:8765/replay_agg.html` locally and reload after each run — no file copying needed. Alternatives: VS Code Remote-SSH + Simple Browser, or `sshfs` mounting the directory.

## Running Sweeps In Parallel

Each replay is a self-contained process with modest CPU usage (~1–2 cores per run), so you can parallelize across a workstation easily. A `concurrent.futures.ProcessPoolExecutor` over the parameter grid is enough — each worker invokes `subprocess.run` with a unique `report_filename` and `--report-json` path, then parses the stdout metrics table and greps `GPU hours:` out of the HTML.

Rough scaling: 12 concurrent runs of the 1-hour `toolagent_trace` complete in ~3–4 minutes on a 24-core workstation.

## Case Study: Engine Startup Time Sweep

This case study quantifies how simulated engine startup time affects SLA attainment and planner behavior. Use this pattern as a template for any planner-config sweep.

### Setup

- Trace: `toolagent_trace.jsonl` (23,608 requests, ~59 min, avg 6.67 rps, avg ISL 8,596 / OSL 182).
- Fixed config: agg mode, TTFT SLA 1,500 ms, ITL SLA 50 ms, H200 SXM, Llama-3.1-8B-Instruct-FP8, TP=1, 2 initial workers.
- Swept: `startup_time` in `{0, 10, 20, ..., 300}` seconds (31 runs).

### Sweep Driver

The driver script is straightforward — iterate the parameter, template the `--planner-config` and `--extra-engine-args` JSON, run `python -m dynamo.replay` in a process pool, then parse TTFT/ITL from the stdout metrics table and `GPU hours:` from the generated HTML.

```python
import json, re, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/path/to/dynamo")
TRACE = REPO / "traces/mooncake_fast25/toolagent_trace.jsonl"
OUT = REPO / "planner_reports/sweep_startup"
OUT.mkdir(parents=True, exist_ok=True)

def run_one(startup_s: int) -> dict:
    report_name = f"replay_agg_startup_{startup_s:03d}.html"
    planner_cfg = {
        "mode": "agg", "optimization_target": "sla",
        "ttft": 1500, "itl": 50,
        "enable_throughput_scaling": True, "enable_load_scaling": True,
        "pre_deployment_sweeping_mode": "rapid",
        "throughput_adjustment_interval": 300, "load_adjustment_interval": 10,
        "prefill_engine_num_gpu": 1, "decode_engine_num_gpu": 1,
        "report_filename": report_name,
    }
    engine_args = {
        "aic_backend": "vllm", "aic_system": "h200_sxm",
        "aic_model_path": "nvidia/Llama-3.1-8B-Instruct-FP8", "aic_tp_size": 1,
    }
    if startup_s > 0:
        engine_args["startup_time"] = startup_s

    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "dynamo.replay", str(TRACE),
         "--planner-config", json.dumps(planner_cfg),
         "--extra-engine-args", json.dumps(engine_args),
         "--num-workers", "2", "--arrival-speedup-ratio", "1.0",
         "--report-json", str(OUT / f"startup_{startup_s:03d}.json")],
        cwd=str(REPO), capture_output=True, text=True,
    )
    # parse AIPerf table rows ("Time to First Token", "Inter Token Latency")
    # and grep "GPU hours: <float>" from the HTML report
    ...

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run_one, s): s for s in range(0, 301, 10)}
        for f in as_completed(futs):
            print(f.result())
```

### Results

At a TTFT SLA of 1,500 ms and ITL SLA of 50 ms, the planner held SLA up to roughly a 50–60 second startup time; beyond ~70 s the p99 tail climbed sharply, and past 100 s the system ran perpetually backlogged.

| Startup (s) | avg TTFT (ms) | p90 TTFT (ms) | avg ITL (ms) | p90 ITL (ms) | GPU-hours | Scale-up events |
|---|---|---|---|---|---|---|
| 0 | 850 | 1,861 | 25.1 | 28.2 | 2.05 | 42 |
| 30 | 954 | 1,957 | 27.0 | 30.0 | 2.04 | 33 |
| 60 | 2,116 | 2,149 | 28.5 | 32.9 | 2.17 | 28 |
| 90 | 3,120 | 2,285 | 29.7 | 41.3 | 2.11 | 23 |
| 120 | 34,797 | 181,991 | 43.9 | 182.3 | 2.17 | 16 |
| 180 | ~60,000 | ~300,000 | ~48 | ~190 | ~2.3 | ~12 |
| 300 | ~76,000 | ~375,000 | ~48 | ~190 | ~2.4 | ~8 |

Takeaways from the sweep:

- **SLA cliff around 100–120 s startup.** Below that, the planner can scale up ahead of load bursts fast enough to stay within TTFT; above that, it cannot catch up within the trace horizon and latency diverges.
- **Scaling-event count drops monotonically** (42 → 8) as startup grows. Long-startup runs take fewer, longer-lived decisions; they also lose the ability to react to short-lived traffic bumps.
- **GPU-hours stays relatively flat** (~2.0 → ~2.4) across the full range. The planner responds to costly scaling by holding a modestly larger steady-state worker count rather than paying for frequent flaps.
- **ITL is less sensitive than TTFT** until the queue saturates. Below the cliff, ITL rises modestly (25 → 30 ms); above it, p90 ITL spikes to ~200 ms as decode requests starve.

### What To Do With This

If your operational environment has a known engine startup budget (typical vLLM pod boot is 60–120 s on Kubernetes), this sweep identifies where the planner transitions from "can keep up" to "must over-provision." In that regime, tune for proactive scaling:

- Use predictive (throughput-based) scaling rather than reactive (load-based) alone — `enable_throughput_scaling: true` with a `throughput_adjustment_interval` that matches your startup budget.
- Over-provision initial replicas (`--num-workers`) to absorb the first ramp before the planner's first reaction can land.
- Consider whether decoupling prefill/decode (`disagg`) gives you a cheaper incremental-scaling unit for the latency-critical role.

## Gotchas

- **Cumulative GPU hours always zero.** `prefill_engine_num_gpu` and `decode_engine_num_gpu` default to `None` on the planner config; the replay adapter falls back to `0`, making the GPU-hours formula collapse to zero. Always set both fields explicitly in the `--planner-config` JSON.
- **Scale-up/down oscillation at low startup.** With `startup_time` unset, new workers activate instantly and the planner can freely oscillate every `load_adjustment_interval`. If you're evaluating planner stability, set a realistic `startup_time`.
- **`disagg` mode rejects `--extra-engine-args`.** Use `--prefill-engine-args` and `--decode-engine-args` instead.
- **Traces have different time bases.** `conversation_trace` and `toolagent_trace` end at 3,537 s; `mooncake_trace` (arxiv) is the same underlying data rescaled to exactly 3,600 s with slightly re-tokenized lengths. Pick one and stick with it across a sweep.
