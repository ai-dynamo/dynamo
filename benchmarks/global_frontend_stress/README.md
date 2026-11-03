<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Global Frontend Stress Test

Measures how many requests per second one Global Frontend can proxy, and what each request
costs it in CPU, when the pools behind it cost nothing. One Global Frontend sits in front of
16 mock pool frontends. Two aiperf clients replay Claude Code agent traces (AgentX) at 8,192
concurrent sessions. The mocks stream back the requested number of tokens without any
inference, so the Global Frontend is the only real component in the request path.

## Layout

Four nodes, held exclusively through Slurm:

| Node | Runs |
|---|---|
| A | The Global Frontend, `--router-policy least-in-flight` over the 16 pools, pinned to a CPU list on NUMA domain 0 |
| B | 16 `mock-pool` processes, 8 cores each, half on each NUMA domain |
| C, D | One aiperf 0.13 each, at 4,096 concurrent sessions |

Every node also runs a 1 s telemetry sampler: CPU per NUMA domain, CPU and RSS per tracked
process, NIC bytes, and TCP retransmits.

The scripts assume nodes with two 72-core NUMA domains (they were written on GB200 Grace
nodes: aarch64, 144 cores) with `numactl`, `perf`, `nstat`, and `ss` installed.

## How it measures

aiperf can't keep up with per-request records at 10k req/s. Each record holds about 870 chunk
timestamps, the record processors handle about 770 records/s, and the backlog fills a client
node's memory in about two minutes. So aiperf only generates load: each arm runs a 60 s
profiling phase (15 s settle, 45 s measure) and kills aiperf once the phase finishes sending.

The figures come from the Global Frontend's own `/metrics`, scraped at the measure window's
edges:

- requests per second, from `dynamo_global_frontend_request_duration_seconds_count`;
- TTFB and duration p50 and p99, from its `time_to_first_byte_seconds` and
  `request_duration_seconds` histograms;
- non-200 responses, rejections, and response errors.

The telemetry adds the frontend's cores, CPU ms per request (split into user and system time),
peak RSS, NIC throughput, how busy the client nodes were, and the TCP segments the frontend
node sent per request.

The `direct` arm points aiperf straight at the mocks, to find the clients' own ceiling. The
clients are close to it near 10k req/s, so check their busy share before reading a frontend
arm's rate as the frontend's limit.

## Mock pool

`mock_pool/` is a small hyper server, built outside the Dynamo workspace with its own
lockfile. `POST /v1/chat/completions` answers with `max_tokens` chunks in the format Dynamo's
frontend streams: one SSE event per token, `role` only in the first, `finish_reason:
"length"` on the last, a usage event when asked for, then `data: [DONE]`. It doesn't parse
the request JSON.

By default every token goes out at once. `--itl-ms N` paces them N ms apart, as an engine
would produce them. Tokens that came due while the reader was behind go out back to back.
Tokio timers have millisecond resolution, so 1 ms is the smallest step.

## Running it

1. Export the settings and lay out the campaign directory, which every node must share:

   ```bash
   export CAMPAIGN=/shared/gf-stress NIC=eth0
   mkdir -p "$CAMPAIGN"/{control,src}
   cp -r benchmarks/global_frontend_stress "$CAMPAIGN/bench"
   git worktree add "$CAMPAIGN/src/main" origin/main   # the baseline
   git worktree add "$CAMPAIGN/src/pr" my-branch       # the change under test
   ```

   The builds need `cargo` and `protoc` on `PATH`. Set `AIPERF` if aiperf 0.13.0 isn't on
   `PATH` as `aiperf`. On nodes without internet access, set `SWAGGER_UI_ZIP` to a downloaded
   Swagger UI release zip.

2. Fill aiperf's caches, because the clients run offline. Put the `Qwen/Qwen3-0.6B` tokenizer
   (at the revision in `slurm/env.sh`) and the
   `semianalysisai/cc-traces-weka-with-subagents-060526` dataset in `$AIPERF_HF_HOME`
   (default `$CAMPAIGN/cache/huggingface`).

3. Hold four nodes. The keeper writes the node list (A, B, C, D in order) and its job ID to
   `$CAMPAIGN/control`:

   ```bash
   cd "$CAMPAIGN/control" && sbatch -p PARTITION -A ACCOUNT "$CAMPAIGN/bench/slurm/keeper.sbatch"
   ```

4. Build both Global Frontends and the mock in parallel on the held nodes, each in its own
   target directory under `$CAMPAIGN/builds/`:

   ```bash
   "$CAMPAIGN/bench/slurm/build_all.sh"
   ```

5. Run arms. Each one writes `$CAMPAIGN/results/LABEL/`:

   ```bash
   S=$CAMPAIGN/bench/slurm
   $S/run_arm.sh direct-1 direct                            # clients straight to the mocks
   $S/run_arm.sh main-full frontend main 0-71               # all of NUMA domain 0
   $S/run_arm.sh pr-full frontend pr 0-71
   $S/run_arm.sh main-k40 frontend main 0-39                # 40 cores
   PROFILE=20 $S/run_arm.sh main-prof frontend main 0-71    # 20 s on-CPU perf profile
   MOCK_ARGS="--itl-ms 1" $S/run_arm.sh main-paced frontend main 0-71
   ```

   Arguments after the CPU list go to the Global Frontend. Each arm starts with a preflight
   that fails if an earlier arm's process is still running on any node. A smoke request must
   then get exactly 7 tokens back through the path under test.

6. Summarize. `summarize.py` prints one table row per arm and writes each arm's
   `summary.json`. `netdiff.py RESULTS_DIR` gives TCP and NIC counter rates over the window,
   and `sssum.py` summarizes an `ss -tin` sample.

   ```bash
   python3 "$S/summarize.py" "$CAMPAIGN"/results/*
   ```

7. Release the nodes:

   ```bash
   touch "$CAMPAIGN/control/RELEASE"
   ```

Things to know:

- Compare arms within one allocation, and alternate their order (A B C, then C B A). The same
  build gave different rates on different node sets.
- A profiled pass runs 45–60% slower. Read its profile, not its rate.
- `PERF_EVENT=cycles` adds kernel stacks to `PROFILE`, which records user space only by default.
- `perf script` right-aligns six-digit thread IDs with a leading space, so split its
  headers on whitespace.

## Results, October 30–31, 2026

GB200 nodes, AgentX at 8,192 concurrent sessions. Each table comes from one allocation. The
clients alone reached 18.3k req/s.

### Burst tokens

A Global Frontend that serves with `axum::serve` stalls on that server's shutdown watch
channel: 38% of its time went to lock contention, and every worker thread sat about 25% busy.
[#17717](https://github.com/ai-dynamo/dynamo/pull/17717) replaces it with a hyper accept loop that gives each connection its own shutdown
signal. Before the fix, main f945f1a8d5 ran 1,194 req/s on 10 cores, with TTFB p50 36.6 ms.
After it, on main 86eea2a1d9:

| Frontend | req/s | Frontend cores | CPU ms/req | TTFB p50 / p99 | Duration p50 / p99 | RSS |
|---|---:|---:|---:|---:|---:|---:|
| main + #17717 | 11,357 | 69.5 | 6.12 | 3.3 / 9.8 ms | 1.1 / 7.8 s | 12.7 GiB |

The clients were 85–99% busy, so this rate is close to the clients' limit too.

A prototype data plane on Pingora (`--data-plane pingora`, branch
`jthomson04/global-frontend-pingora`), against the same binary's axum plane with the fix:

| | axum + #17717 | Pingora |
|---|---:|---:|
| All of NUMA domain 0: req/s (both client-bound) | 8,770 | 15,653 |
| All of NUMA domain 0: CPU ms/req | 7.37 | 4.29 |
| All of NUMA domain 0: RSS | 15.4 GiB | 8.7 GiB |
| 40 cores: req/s | 6,546 | 8,749 |
| 40 cores: CPU ms/req | 5.68 | 4.26 |
| TCP segments sent per request | ~936 | ~407 |

User CPU per request is about the same; Pingora halves kernel time by writing up to four
ready chunks per system call.

### Paced tokens, 1 ms apart

With `--itl-ms 1` that batching rarely happens. Both planes on one shared multi-threaded
runtime plateaued at 3.2–3.6k req/s whether given 72 or 40 cores. Pingora's
`--pingora-runtimes N` (N single-threaded runtimes, each with its own listener, connection
pool, and I/O driver) removes that limit:

| Pingora | req/s | CPU ms/req | TTFB p50 / p99 |
|---|---:|---:|---:|
| One shared runtime | 2,854 | 9.62 | 20 / 193 ms |
| 72 runtimes | 10,575 (client-bound) | 6.73 | 14 / 44 ms |
| 40 runtimes on 40 cores | 7,979 | 4.97 | — |
