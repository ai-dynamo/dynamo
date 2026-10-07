<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# E2E indexer-contention tooling (experiment only)

Campaign `e2e-indexer-contention-20261006`. None of this is for upstream.

The goal is to load one serving indexer (`python -m dynamo.router --serve-indexer`, default
4 event threads) the way roughly ten frontends and about 2k+ workers would:

- One real frontend (`--use-remote-indexer`) plus live mockers carry about 1/10 of the load.
- Phantom publishers and a side query driver carry the other 9/10. Both use the indexer's
  production interfaces: direct-ZMQ KV event envelopes and the `kv_indexer_query`
  request-plane endpoint.

## Pieces

| Piece | Where | Role |
|---|---|---|
| Static-source patch | commit `chore(kv-router): EXPERIMENT static direct-ZMQ KV sources`; `artifacts/tooling/static-kv-sources.patch` | Lets the serving indexer subscribe to and accept phantom publishers without discovery or serving membership. |
| Stream exporter | `mooncake_bench --export-phantom-streams DIR` (phase-1 harness, `lib/bench`) | Captures AgentX on the mocker, or loads a corpus cache, and writes one base stream per captured worker. |
| `phantom_plan` | this crate | Prints rates, speedup, and coverage. Splits the phantoms across publisher processes and writes the indexer's static-source file. |
| `phantom_publisher` | this crate | Runs N phantom workers per process, each on its own ZMQ PUB socket. Sends the production wire format with open-loop pacing. |
| `query_driver` | this crate | Sends the real `kv_indexer_query` RPC with the phantoms' own lookups on the same clock, and reports RTT, issue lag, rate, and self-hit fraction. |
| `scripts/local_smoke.sh` | this crate | Loopback plumbing smoke. Never report numbers from it. |

### Static-source patch

Set `DYN_EXPERIMENT_STATIC_KV_SOURCES` to a list separated by commas or whitespace, or to
`@<file>` for a file with one entry per line and `#` comments. Entry forms:

- `<worker_id>@<zmq endpoint>`
- `<first_worker_id>+<count>@tcp://<host>:<first_port>`: worker `first + k` publishes on
  `first_port + k`.

Behavior:

- A static source has publisher ID equal to its worker ID, DP rank 0, and no recovery target
  (live-only).
- Static sources appear only in a private membership view, which is used by the ingress's
  `WorkerQueryClient` and its direct-ZMQ supervisor. Their events therefore take the full
  production live-batch path:
  - publisher binding lookup, slot lock, and event-ID cursor;
  - queue admission into the indexer.
- Static sources never become serving workers. The frontend never sees them, so they are never
  routable.
- A static ID that collides with a discovered worker or publisher is skipped and logged.
- If the event plane is not direct ZMQ, startup fails.
- The env var applies to any process whose KvRouter subscribes to direct-ZMQ events. That
  includes a frontend's embedded router, through the same `start_subscriber` path. Set it only
  on the serving indexer.

CRTC neutrality:

- Every file the patch touches has the same blob at MAIN `e61319d830`, STACK `2b20fc1d35`, the
  D2 jemalloc tree (`82f4bbe95b`), and the campaign base `5233229717`:
  - `lib/llm/src/kv_router/indexer/recovery/{direct_zmq,mod,subscriber,worker_query}.rs`, plus
    the new `static_sources.rs`;
  - the membership code the patch relies on.
- `git apply --cached --check` passes against all three arm trees.

To apply on an arm tree, use either:

```bash
git -C <arm-tree> apply /path/to/static-kv-sources.patch
git -C <arm-tree> cherry-pick <patch commit>
```

### Wire fidelity

The publisher starts from the raw engine events the offline mocker capture produced. At load
time it runs them through `OfflinePublisherPipeline`, a campaign-only shim in
`lib/llm/src/kv_router/publisher/offline.rs`. The shim reuses the production publisher's
`BatchingState`, `EventDedupFilter`, `emit`, and `event_plane_event_batches`, with the mocker's
default batching (no timeout). The result:

- the same store/remove coalescing and dedup;
- contiguous outbound event IDs starting at 1;
- envelopes of at most 128 events and 8192 blocks.

Each envelope is encoded with the runtime's own `Codec` (msgpack `Vec<RouterEvent>` inside an
`EventEnvelope`) and sent through `ZmqPubTransport`, giving the 4-frame multipart on topic
`kv-events`. The indexer's `ValidatedZmqSource` therefore accepts the envelopes unchanged.

Phantom `i` is built from its base `b(i)`:

- It uses worker ID `--worker-id-base + i`. The default base is `0x7F00_0000_0000_0000`, which
  is above the 2^53 discovery-safe publisher-ID space.
- Every block hash is remapped with a per-phantom bijection, `fmix64(h ^ key_i)`, so equality
  and parent/child structure survive while copies share no blocks.
- Bases are assigned to phantoms in contiguous ranges: `b(i) = floor(i * bases / total)`.

## Workflow

### 1. Build (campaign worktree)

```bash
cargo build --release -p dynamo-e2e-indexer-tools                # phantom_publisher, query_driver, phantom_plan
cargo bench --no-run -p dynamo-bench --no-default-features --features mooncake --bench mooncake_bench
```

The tools link `dynamo-llm` without the `block-manager` feature. The arm wheels are built
separately, from each arm tree with the patch applied.

### 2. Export streams (one capture per page size; cluster for real sizes)

Use the phase-1 capture arguments (`artifacts/offline-sizing/scripts/sizing_driver.py`):

```bash
mooncake_bench <pool>.pool.msgpack --workload agentic --agentic-engine sglang \
  --agentic-pool-sha256 3753eceb7d88e1d9dda3f3fc5ad6d138d8ec68ecda8734d4cc19a5f5e586ccba \
  --block-size <1|16> --num-gpu-blocks $((786432 / <page>)) \
  --num-unique-inference-workers <bases> --agentic-plays-per-worker <P> --agentic-lanes-per-worker 4 \
  --agentic-sim-ms <T> --agentic-warmup-sim-ms <W> --agentic-phase-spread 0.05 --seed 42 \
  [--agentic-corpus-cache <cache.bin>] --result-json-output export.json \
  --export-phantom-streams <streams-dir> concurrent-radix-tree-compressed --num-event-workers 4
```

- If `--agentic-corpus-cache` names an existing cache whose key matches, the exporter loads it
  instead of capturing. Otherwise it captures and writes the cache.
- The output is `manifest.json` plus `base-NNNNN.bin`. Each base contains:
  - raw warm-up lists (untimed);
  - timed lists, one per capture timestamp;
  - timed lookups.

Sizing: there is no looping, so a phantom stops when its timed section ends.
`timed_span_virtual_s / speedup` must cover the measurement duration plus the start spread;
`phantom_plan --duration-s` enforces this. Per-worker load is constant across `<bases>`
(weak scaling), so a longer `--agentic-sim-ms` with fewer bases buys coverage. For example,
32 bases × 24,500 s costs about as many worker-seconds as phase 1's 512 × 1,600 s. Size
`--agentic-plays-per-worker` so lanes do not run dry; the export report shows
`lanes_exhausted_before_cap`.

### 3. Plan one load point

```bash
phantom_plan --streams <dir> --total-phantoms 1800 --target-write-blocks-per-sec <R> \
  --duration-s 1200 --start-spread-ms 5000 \
  --publishers hostA:30000:900,hostB:30000:900 --sources-out sources.txt
```

This prints:

- natural and aggregate rates: writes, events, queries, lookup blocks, and queries per phantom;
- the speedup;
- one JSON line per publisher process with its exact `phantom_publisher` arguments.

`sources.txt` holds the indexer's static-source list. Keep at most 1000 phantoms per publisher
process (libzmq allows 1023 sockets per context).

### 4. Launch (same order as the smoke)

1. Start the live mockers and the serving indexer:

   ```bash
   DYN_EXPERIMENT_STATIC_KV_SOURCES=@sources.txt \
     python -m dynamo.router --endpoint <ns>.backend.generate --serve-indexer --router-block-size <page>
   ```

   - Consider `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB` to group phantom endpoints per SUB socket, and
     a high `ulimit -n`.
   - Use the same environment in both arms.
2. Start the frontend: `python -m dynamo.frontend --router-mode kv --use-remote-indexer
   --kv-cache-block-size <page>`.
3. Start every publisher with the shared values `--total-phantoms`, `--worker-id-base`,
   `--salt-seed`, `--speedup` or `--target-write-blocks-per-sec`, `--start-at-unix-ms`,
   `--start-spread-ms`, and `--duration-s`. Also pass `--warmup-blocks-per-sec`; warm-up is
   untimed and must finish before `--start-at-unix-ms` (see `late_warmups`).
4. Start the query drivers with the same shared values, plus:
   - `--component <ns>.backend`;
   - `--model-name <served model name>`;
   - optionally `--first-phantom/--count` to shard phantoms across driver processes.

Outputs:

- Each binary prints one JSON line per `--report-interval-s` and a summary (also written to
  `--summary-out`).
- Publisher: achieved write blocks/s and lag versus schedule. Driver: RTT p50/p99, issue lag,
  achieved queries/s and lookup blocks/s, and the self-hit fraction.
- Indexer CPU comes from the host (pidstat or perf), not from these tools.

### 5. Local smoke (loopback; plumbing only)

```bash
WORK=<scratch> STREAMS=<tiny stream dir> SPEEDUP=4 DURATION_S=30 lib/e2e-indexer-tools/scripts/local_smoke.sh
# page size 1: add BLOCK_SIZE=1 MOCKER_ARGS="--engine-type sglang"
```

The smoke uses file discovery, the direct-ZMQ event plane, and the TCP request plane. It runs a
live mocker, the patched serving indexer, a remote-indexer frontend (one real chat request),
4 phantoms, and the driver.

## Caveats (carry into any report)

- **Membership.** Membership filtering and recovery are skipped for phantom sources, identically
  in both arms.
- **Hash remapping.** Non-root sequence hashes are remapped rather than re-chained from the
  remapped local hashes. Neither arm's event-driven indexer (CRTC and ThreadPoolIndexer)
  recomputes sequence hashes from local hashes; grep `compute_next_seq_hash` at both SHAs.
- **No sharing across phantoms.** As in phase 1, workers share no prefixes: the harness salts
  each base, and the publisher remaps each copy. A real fleet shares system prompts.
- **Copies share timing.** Phantom copies of one base replay the same timing pattern, shifted
  only by `--start-spread-ms`.
- **Envelope boundaries.**
  - Timed lists are capture-timestamp groups. At page size 16 the tiny local capture gave one
    event per list, so that stream may send more, smaller envelopes than a live mocker that
    publishes a whole pass at once. At page size 1, the 41k events formed 24.8k lists.
  - Warm-up lists are runs of one worker in the merged warm-up order.
- **No looping.** The tools do not loop. Size the capture; the plan refuses short coverage.
- **Overload drops events.** A PUB socket drops sends once its high-water mark (100k messages)
  is reached. A resulting event-ID gap resets that phantom's rank at the indexer
  (live-only `ResetDegraded`); this is production behavior. Watch the indexer logs for gap
  resets under overload, and compare the publisher's achieved rate with the plan.
- **Per-query logging.** The serving indexer logs every `kv_indexer_query` at INFO
  (`push_handler` "request received/completed"). At thousands of queries per second that is
  real CPU in both arms. Keep `DYN_LOG` identical across arms, and consider filtering it.
- **Slow activation at scale.** Activation is O(sources) per readiness signal
  (`WorkerQueryClient::reconcile_view` and `ready_sources`), so startup is about O(N²) for N
  sources. Expect slow activation, plus one warning line per live-only source, at 10× (about
  18k sources).
- **Version skew.** The tools are built from the campaign base `5233229717`. Between it and the
  arm SHAs, the event-plane codec and frames, ZMQ PUB, and the TCP request plane are unchanged;
  the runtime diff touches only QUIC typed prologues and NATS. Keep the request and response
  planes on TCP.
- **Warm-up skip.** `--warmup-blocks-per-sec 0` skips the warm-up. Use it only for smoke tests:
  the timed section then references blocks the indexer never saw.
