<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Global Router Full-KV Benchmark

This branch measures what it costs a Global Router to hold an exact KV index of every worker:
the selection core's CRTC index, its KV-event ingest, and the network carrying those events.
It is a benchmark branch, not for main.

## Pieces

| Binary | Role |
|---|---|
| `gr_core_host` | The standalone selection service, built as `python -m dynamo.select_service` builds it (router config from `DYN_*` env vars, default policy, `--threads` CRTC lanes, stock HTTP API). `--ingress zmq-direct` is the stock ingress; `--ingress velo-relay` takes events from velo streams. Writes a 1 Hz stats file: CRTC lane depths, ZMQ ingest counters, relay counters, velo metrics. |
| `select_service_bench` | One pool: N in-process mocker engines plus closed-loop AgentX sessions that call `/select_and_reserve` on the core and run each turn on the chosen engine. `--event-transport zmq`: each engine publishes raw vLLM ZMQ (token ids included). `--event-transport velo`: the process is a pool relay that converts each engine batch to router events, exactly as the core's ZMQ listener would, and sends it as one record on that worker's velo stream. |

Several pools drive one core: each sends its own workers as `allowed_worker_ids` (one global
partition), replays its own session copies (`--copy-base`), and starts from a shared barrier
(`--ready-file`, `--start-file`).

## Build

```bash
cargo build --release -p dynamo-bench --no-default-features --features gr-full-kv \
  --bin gr_core_host --bin select_service_bench
```

## Run (one host)

```bash
gr_core_host --ingress velo-relay --velo-bind 127.0.0.1:23400 --peer-file /tmp/core.peer \
  --stats-out /tmp/core_stats.jsonl &
select_service_bench --service-url http://127.0.0.1:8092 --event-transport velo \
  --peer-file /tmp/core.peer --velo-bind 127.0.0.1:25500 --trace agentx128-sp16k.mooncake.jsonl \
  --shared-prefix-blocks 256 --flow select-and-reserve --workers 64 --block-size 128 \
  --num-gpu-blocks 8192 --speedup 5 --concurrency 512 --no-decode-progress --no-session-id \
  --out /tmp/bench
```

The multi-node Tyche harness (keeper allocation, per-node roles, samplers, perf folding and
roofline analysis) lives outside the repository.
