<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router Backpressure with CPU Mockers

Watch NVIDIA Dynamo's router adjust dispatch when a backend worker stops and
returns at half speed. The example starts one frontend and three CPU Mocker
workers, generates traffic, prints per-worker arrival rates, and shuts down.
No GPU, model weights, etcd, or NATS service is required.

## Run

Use Linux with Dynamo installed from this checkout, including its frontend,
Mocker, compiled Python bindings, and `aiohttp`. See
[Building from Source](../../../../docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md).
The first run may download the tokenizer and configuration for `Qwen/Qwen3-0.6B`.

From the repository root, run:

```bash
python3 examples/router/admission-control/backpressure/run.py
```

Backpressure is **on by default**. For the comparison, run:

```bash
python3 examples/router/admission-control/backpressure/run.py --no-backpressure
```

Both runs use the same workers, `dynamo-default-cost-fn` worker selector, and
fixed-rate client. The selector favors workers with less outstanding load;
cache-reuse credits are disabled, and temperature is zero (minimum cost, with
random tie-breaking).

[on.yaml](on.yaml) sets `prefill_busy_threshold_frac: 1.0` to queue requests
while all workers are busy. With `--no-backpressure`, [off.yaml](off.yaml) leaves
that threshold unset. This is the only configuration difference between modes.
Neither mode limits the router queue size.

## What the Runner Does

1. Creates a fresh directory under `/tmp` for discovery, logs, and results,
   chooses available ports, and starts a frontend and three workers. All use
   file discovery and TCP requests.
2. Sends 90 requests/s continuously, each with 512 input tokens and one output
   token. Each worker allows four concurrent sequences, with an 8192-token
   batch budget and prefix caching disabled.
3. Measures all three healthy workers for five seconds after settling.
4. Stops worker 3, waits for the router to discover its removal, then measures
   the remaining workers for five seconds.
5. Restarts worker 3 with its simulated speed halved, waits for registration,
   then measures another five-second window.
6. Cancels outstanding client requests and stops its servers. Logs and results
   remain in the printed `/tmp` directory. **Ctrl-C** also stops the deployment.

To follow the code, start with `Deployment.start_frontend()` and
`Deployment.start_worker()` for the server commands, `drive_scenario()` for the
worker cycle, and `send_fixed_rate()` for the shared client.

## Read the Output

The terminal refreshes one live row every 0.5 seconds during settling and
measurement: client send rate, each worker's request-arrival rate, router queue
depth, and backend in-flight requests. Live rates use roughly the latest second
of counter changes to smooth short bursts. Each phase ends with an `avg` row
covering its full five-second measurement window; the automated test uses these
phase measurements. Redirected output prints each update on a separate line.

Arrival rates come from each worker's `dynamo_component_requests_total` counter
at the inference endpoint, before engine scheduling. They measure requests
delivered by the router, not completed responses.

With backpressure on, expect approximately:

| Phase | Worker 1 | Worker 2 | Worker 3 |
| --- | --- | --- | --- |
| Healthy | 20 requests/s | 20 requests/s | 20 requests/s |
| Worker 3 stopped | 20 requests/s | 20 requests/s | 0 |
| Worker 3 at half speed | 20 requests/s | 20 requests/s | 10 requests/s |

The client sends 90 requests/s in both modes, exceeding worker capacity in
every phase. The three healthy workers can process about 60 requests/s in total.
Removing worker 3 reduces capacity to about 40 requests/s; restarting it at half
speed brings capacity to about 50 requests/s.

With backpressure on, total worker arrivals follow those capacities: about
60, 40, then 50 requests/s. The selector favors workers with less outstanding
load. Excess requests wait in the router when all workers are busy, while
backend in-flight work stays bounded.

With `--no-backpressure`, total worker arrivals stay near the client's
90 requests/s: initially about 30 per worker, then about 45 each with two
workers. After restart, individual shares depend on worker load, including
the restarted worker's initially empty queue. Excess work accumulates at the
backends in every phase. Compare total arrivals and router queue growth to see
admission backpressure; individual worker shares also reflect load balancing.
Actual rates vary with CPU scheduling.

Shutdown can interrupt requests with HTTP 503. The final output reports those
separately from unexpected errors and requests cancelled when the run ends.

## Automated Test

[test_backpressure.py](../../../../tests/router/test_backpressure.py) imports
[run.py](run.py) and calls its shared deployment and measurement utilities for
both modes. It asserts stable healthy-worker rates, a half-rate restarted
worker, router queue growth and reduced dispatch in every phase when enabled,
and unthrottled total dispatch with growing backend backlog when disabled.

```bash
python3 -m pytest tests/router/test_backpressure.py -v
```

The test reserves randomized frontend and worker metrics ports through pytest
and passes them into the runner. Each case has its own discovery directory and
processes, so the test supports parallel execution:

```bash
python3 -m pytest tests/router/test_backpressure.py -n 2 -v
```

The test is marked for parallel CPU pre-merge coverage. Runner and YAML changes
both trigger the CPU tests.
