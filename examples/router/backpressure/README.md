<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router Backpressure with CPU Mockers

Demonstrate how NVIDIA Dynamo's KV router adjusts dispatch to worker capacity.
One frontend routes a fixed stream of concurrent requests to three CPU Mocker
workers. During the run, the example stops worker 3, then restarts it at half
its original simulated speed. No GPU or model weights are required.

The example runs two cases with the same traffic and worker-selection settings:

- **Off:** admission thresholds are unset. The router dispatches the offered
  traffic, allowing work to accumulate at the workers.
- **On:** a policy-class queue holds excess requests until a worker has room.
  Healthy workers should maintain their original receive rates, while the
  restarted worker receives requests at approximately half its original rate.

## Prerequisites

- A Linux environment with Dynamo installed from this checkout, including its
  frontend, Mocker, and compiled Python bindings. See
  [Building from Source](../../../docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md).
- `etcd` and `nats-server` executables on `PATH`.
- Python with `aiohttp` and `matplotlib` installed.
- Access to the model's tokenizer and configuration. The default model is
  `Qwen/Qwen3-0.6B`; the first run may download these files. Use `--model`
  to select a compatible cached model or local directory.

Install the example's Python dependencies in the same environment as Dynamo:

```bash
python3 -m pip install aiohttp matplotlib
```

## Run the Comparison

From the repository root, choose an output directory that does not already exist:

```bash
python3 examples/router/backpressure/run.py \
    --mode both \
    --output-dir /tmp/router-backpressure
```

The runner starts its own etcd and NATS services on dynamically selected ports,
then launches a fresh frontend and three independent Mocker processes for each
case. It stops the processes it started when the case finishes. Use `--mode off`
or `--mode on` to run only one case.

Each request contains 512 uncached input tokens and produces one output token.
Each worker serves up to four concurrent sequences. Prefix caching is disabled.
The client sends at a fixed rate throughout worker removal and restart; it does
not wait for one response before sending the next request.

After warmup, each case measures three stable windows:

| Window | Workers | Expected arrival rates with admission on |
| --- | --- | --- |
| Initial | Three workers at the same speed | `R, R, R` |
| Removed | Worker 3 stopped | `R, R, 0` |
| Restarted | Worker 3 at half speed | `R, R, R/2` |

Here `R` is the observed initial per-worker request rate. The default measurement
window is five seconds, with settling time around transitions. Startup and
discovery transitions are excluded from the stable-window comparisons.

## Read the Results

The output directory contains:

- `summary.json`: overall checks for the requested cases.
- `off/summary.json` and `on/summary.json`: per-window rates and checks.
- `off/plot.svg` and `on/plot.svg` (also PNG): time series with worker transitions.
- `off/samples.csv` and `on/samples.csv`: sampled counters and gauges.
- Per-case `outcomes.json`, process logs, configuration copies, and `metadata.json`.

A successful comparison prints `PASS` with the summary path and exits with status
`0`. A failed behavioral check prints `FAIL` and exits with status `1`; inspect
the per-case checks and logs.

Open the SVG plots and compare backend arrivals with client send rate and router
queue depth. With admission on, aggregate dispatch should decrease from
approximately `3R` to `2R`, then recover to `2.5R`. Excess work waits at the
router. With admission off, aggregate arrivals should remain close to the offered
client rate, with growing backend in-flight work. Normal load-based worker
selection still operates, so the off case need not distribute requests equally.

Backend arrival rate comes from `dynamo_component_requests_total`, filtered to
the inference endpoint on each worker's metrics port. This counter increments
when the worker receives a request, before engine scheduling; it does **not**
count completed responses. The runner starts a new counter baseline when worker
3 restarts. Router queue depth uses
`dynamo_frontend_router_queue_pending_requests`.

## Configuration and Limits

[off.yaml](off.yaml) and [on.yaml](on.yaml) configure the same worker-selection
and load-tracking settings. Both select one named default class and explicitly
use first-come, first-served (FCFS) ordering; controlled-delay expiry remains
disabled. The on case adds the class's
`prefill_busy_threshold_frac` and `request_queue_limit_per_worker`. Queue capacity
must accommodate the finite run even while only two workers remain. These are
policy YAML settings; the example does not use deprecated admission or queue
flags.

The client rate must exceed the initial combined worker capacity for all three
workers to remain saturated. Host scheduling and Mocker timing affect the
achieved rates. If checks fail, inspect the process logs and plots, then use
`--rate`, `--speedup-ratio`, `--window`, and `--settle` to adjust the experiment.
Lowering `--speedup-ratio` slows the simulated workers; increasing `--window`
reduces short-window measurement noise.

The checks allow healthy-worker rates to vary by 25% and the restarted worker
to reach 35–65% of its initial rate. The off case must dispatch within 10% of the
measured client rate. Queue growth, bounded backend work, and response checks
provide additional evidence that the router is controlling dispatch.

Only HTTP 503 responses ending between the shutdown signal and the next stable
window can be classified as expected shutdown failures. The request must have
started before the router confirmed worker removal, including requests sent while
discovery was catching up. The observation time is saved as `worker_removed_at`
in `summary.json`. Requests started after confirmed removal, and all other
request failures, fail validation. Outstanding requests are cancelled at the end
and counted separately in `summary.json`; this is a finite dispatch demonstration,
not a test that drains every request or provides transparent request recovery.
