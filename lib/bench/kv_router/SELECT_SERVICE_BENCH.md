<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Selection Service Load Benchmark

`select_service_bench` drives Dynamo's selection service with closed-loop, multi-turn
traffic from simulated engines, without GPUs. It measures what the router itself costs
and how well it routes: selection throughput and latency, the service's CPU, and the
predicted and actual KV cache hit ratios.

It can target either of these:
- the standalone HTTP selection service (`python -m dynamo.select_service`);
- `dynamo.llm.SelectionService` embedded in one or more Python processes that follow Ray
  Serve LLM's KV-aware ingress (ray-project/ray#67224 at `53b9942e4f`).

## How it works

| Piece | What it does |
|---|---|
| Engines | N Dynamo mocker engines run in-process in the load generator. Each simulates vLLM-style scheduling and KV cache with prefix caching, block allocation and eviction (`--num-gpu-blocks`), and batches capped by `--max-num-batched-tokens`. Timing comes from the mocker's performance model, scaled by `--speedup`. |
| KV events | Each engine publishes vLLM-wire KV events (msgpack `KVEventBatch` on a ZMQ PUB socket, `--zmq-base-port + i`). With `--zmq-replay-base-port`, it also runs a replay ROUTER and registers it as the worker's `replay_endpoint`, as Ray's vLLM KV-events config does. |
| Traffic | Replays a Mooncake-format trace: per turn, `hash_ids` plus input and output lengths. Token ids are built from the hash ids, so equal ids share prefixes. Each of `--concurrency` sessions keeps one turn in flight and sends the next when it completes. Sessions are copied from the trace's sessions with random starting turns; the first `--shared-prefix-blocks` ids (a shared system prompt) stay identical across copies. |
| Selection | Each turn asks the service for a worker, submits the turn to that engine, and reports the lifecycle back: prefill complete, one output block per generated KV block, completion. |
| Phases | Warmup, measure, then profile. The bench touches `<out>/phase.<name>` at each start so a profiler can line up with them. |

### Flows (`--flow`)

| Flow | Calls |
|---|---|
| `ray` | HTTP `POST /select`, then `POST /reservations` with the same `selection_id` |
| `select-and-reserve` | HTTP `POST /select_and_reserve` |
| `ingress` | `select_service_ray_ingress.py` processes over Unix sockets (below) |

### The Ray-style ingress (`--flow ingress`)

Each `select_service_ray_ingress.py` process stands in for one `LLMRouter` ingress replica. It
follows `KVAwareRouter.choose_replicas` and `KVTokenTracker` from ray-project/ray#67224 at
`53b9942e4f`:

- One asyncio loop owns a `dynamo.llm.SelectionService` with soft session affinity (TTL 1,800 s,
  Ray's default) and a host replication writer id (#16969).
- Each selection is `select_and_reserve` with `token_ids` and `allowed_worker_ids`, with worker ids
  derived by blake2b as Ray does. At most `--max-ongoing-requests` (1,000, Ray Serve's
  `max_ongoing_requests`) selections run on the loop at once.
- Lifecycle calls (`prefill_complete`, `add_output_block`, `free_reservation`) are applied by
  `--lifecycle-shards` concurrent tasks, standing in for one in-flight RPC per engine replica.
- With `INGRESS_N` > 1, each replica broadcasts its load reservations and
  `session_binding_updates()` to its peers. Each peer applies them with `create_reservation` and
  `apply_session_binding`, awaiting one update at a time as Ray's `_apply_replica_updates` does.
  The Ray RPC is replaced by pickled batches over Unix sockets, and the sender waits for every
  peer's acknowledgement.
- The load generator sends each request to a random replica and every lifecycle event to every
  replica, as Ray broadcasts them.
- Functions named `_ipc_*` exist only for the benchmark transport, so profiles can separate them.

Two what-if switches change the emulated Ray behavior. Neither is Ray's current behavior.
- `--concurrent-apply` awaits each round of peer updates together.
- `--binding-batch N` drains up to N queued session bindings per stream item. It relies on a
  prototype `max_batch` argument to `session_binding_updates()` that exists only on this branch.

## What this branch contains

The numbers in the ai-dynamo/dynamo#16969 review were measured on this exact stack:

| Commit | Contents |
|---|---|
| `main` at `e7ccd47f08` | Base |
| merge of #17238 at `654d716824` | Request parsing without serde `flatten` or path tracking |
| merge of #17253 at `26fc8604c3` | Shared ZMQ contexts; replay connection per replay |
| merge of #16969 at `e8f3b88252` | Host session-affinity replication, which the multi-ingress flow needs |
| bench commits | This load generator, the ingress emulation, and the what-if switches (the last one also adds the `--binding-batch` prototype to the bindings) |

## Build

```bash
# Python bindings with the selection service, from the repository root
uv venv .venv && source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv --release --features select-service)

# Load generator
cargo build --release -p dynamo-bench --no-default-features --features select-service \
    --bin select_service_bench
```

## Trace

Any Mooncake-format JSONL works if its rows carry `session_id`, `hash_ids`, `input_length` and
`output_length`; rows of a session are replayed in file order. The review used the AgentX trace
(128 sessions, 33,101 requests, 64 tokens per hash id) with one shared 16,384-token system prompt
added in front of every request:

```bash
python3 lib/bench/kv_router/select_service/add_system_prompt.py \
    agentx128.mooncake.jsonl agentx128-sp16k.mooncake.jsonl 256
```

Then pass `--shared-prefix-blocks 256` so the prompt stays shared across session copies.

## Run

`select_service/run.sh OUT_DIR FLOW [bench args...]` starts the service processes, waits until
they are ready, runs the bench, samples every process's CPU, and prints a measure-phase summary.
Its environment variables are listed at the top of the script; `TRACE` is required.

Quick check on a workstation, 8 engines:

```bash
export TRACE=$PWD/agentx128-sp16k.mooncake.jsonl
export PY=$PWD/.venv/bin/python
SMOKE="--workers 8 --concurrency 32 --speedup 4 --shared-prefix-blocks 256 \
    --zmq-replay-base-port 22000 --no-decode-progress \
    --ramp-secs 5 --warmup-secs 20 --measure-secs 20 --profile-secs 10"
INGRESS_N=2 lib/bench/kv_router/select_service/run.sh runs/smoke ingress $SMOKE
```

The #16969 review configuration ran 1,024 engines and 8,192 sessions at 5× speed on one
two-socket node. Each ingress replica was pinned to 8 cores on socket 0 from core 8 up, and the
load generator ran on socket 1:

```bash
REVIEW="--workers 1024 --concurrency 8192 --speedup 5 --shared-prefix-blocks 256 \
    --zmq-replay-base-port 22000 --no-decode-progress --warmup-secs 120 --measure-secs 60"
PIN="INGRESS_CPUS_EACH=8 INGRESS_CPU_BASE=8 BENCH_CPUS=<socket 1 cpus>"
```

| Arm | Command |
|---|---|
| N=1 | `env $PIN INGRESS_N=1 run.sh runs/n1 ingress $REVIEW` |
| N=4, Ray as written | `env $PIN INGRESS_N=4 run.sh runs/n4 ingress $REVIEW` |
| N=4, no session ids | `env $PIN INGRESS_N=4 run.sh runs/n4-nosess ingress $REVIEW --no-session-id` |
| W1: concurrent peer apply | `env $PIN INGRESS_N=4 SVC_ARGS=--concurrent-apply run.sh runs/w1 ingress $REVIEW` |
| W2: W1 + batched bindings | `env $PIN INGRESS_N=4 SVC_ARGS="--concurrent-apply --binding-batch 1024" run.sh runs/w2 ingress $REVIEW` |

The HTTP flows take the same bench arguments, for example
`SVC_CPUS=8-23 run.sh runs/http ray $REVIEW`.

### Sizing

- Every service process opens a ZMQ subscription and, during replays, a replay connection per
  engine. The script raises the open-file limit to the hard limit.
- Engines use ports `--zmq-base-port` to `--zmq-base-port + workers - 1` (default 20000), and the
  replay ROUTERs use the same range from `--zmq-replay-base-port`. Keep the two ranges apart.
- On smaller machines, scale `--workers` and `--concurrency` down together and check
  `bench.cpu.jsonl`: the load generator should not be the limit.

## Output

| File | Contents |
|---|---|
| `bench.log` | One line per window: requests/s, selection p50/p99, error rate, predicted and engine hit ratio, KV blocks stored and removed per second |
| `bench/summary.json` | Per-phase totals and per-endpoint latency percentiles; `gated_error_fraction` covers measure and profile |
| `bench/windows.jsonl` | The window lines as JSON |
| `service.<i>.log` | Each service process's log |
| `service.<i>.stats.jsonl` | Ingress counters every 10 s: `selects`, `load_reservation_out` / `_applied` / `_skipped`, `session_affinity_out` / `_applied`, `broadcasts`, and peer-apply and lifecycle lag in ms (`_sum`, `_n`, `_max`) |
| `service.<i>.cpu.jsonl`, `bench.cpu.jsonl` | Per-second CPU time, threads and RSS, with CPU grouped by thread name |

`select_service_bench` exits 2 when errors exceed `--max-error-fraction` (default 1e-4).

At #16969's `e8f3b88252`, bindings dropped by the replication channel are logged only at trace
level. To count them, add
`DYN_LOG=info,dynamo_kv_router::services::common::replica_sync=trace` to the environment and count
`dropping best-effort session affinity replica update` in each `service.<i>.log`.

## Profiling

`PROFILE=1` records `perf -F 999 -g` on every service process for `PROFILE_SECS` (default 30)
once the profile phase starts. Ingress processes run with `python -X perf`, so Python frames have
names. For Rust stacks, build the bindings with frame pointers
(`RUSTFLAGS=-Cforce-frame-pointers=yes`) and debug info (`CARGO_PROFILE_RELEASE_DEBUG=2`).
