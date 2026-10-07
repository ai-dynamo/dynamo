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
| Static-source patch | commits `chore(kv-router): EXPERIMENT static direct-ZMQ KV sources` and `chore(kv-router): EXPERIMENT static-source delivery accounting`; squashed in `artifacts/tooling/static-kv-sources.patch` | Lets the serving indexer subscribe to and accept phantom publishers without discovery or serving membership, and counts what it applied from them. |
| Stream exporter | `mooncake_bench --export-phantom-streams DIR` (phase-1 harness, `lib/bench`) | Captures AgentX on the mocker, or loads a corpus cache, and writes one base stream per captured worker. Reports per-stream eviction. |
| `phantom_plan` | this crate | Prints rates, speedup, coverage, eviction, the indexer's socket budget, and the delivery rule. Splits the phantoms across publisher processes and writes the indexer's static-source and environment files. |
| `phantom_publisher` | this crate | Runs N phantom workers per process, each on its own ZMQ XPUB socket. Waits until the indexer subscribed to every socket, then sends a format-identical copy of the production wire with open-loop pacing. |
| `query_driver` | this crate | Sends the real `kv_indexer_query` RPC with the phantoms' own lookups on the same clock, and reports RTT, issue lag, rate, and self-hit fraction. |
| `delivery_check` | this crate | Applies the delivery rule to one arm: publisher summaries against the indexer's accounting. |
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
- Static sources require an explicit `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB`; startup fails without
  it. Use the value `phantom_plan` prints, identical in both arms.
- The env var applies to any process whose KvRouter subscribes to direct-ZMQ events. That
  includes a frontend's embedded router, through the same `start_subscriber` path. Set it only
  on the serving indexer.

Delivery accounting (both arms, same code):

- For each static source, the live path counts what it admits to the indexer queue: events,
  stored and removed blocks, and the first and last event ID. It also counts gap resets
  (`ResetDegraded`), rank resets after the source had indexed events (any cause, gap resets
  included, shutdown excluded), and events dropped because the source was inactive. Counters sit
  on one cache line per source and are published once per envelope.
- A source whose first admitted event is not event 1 is logged once at WARN: its prefix was lost
  before the indexer applied it. A live-only cursor accepts the first event it sees as initial,
  so without this check such a loss would produce no reset and no log.
- Admission is the last point that is identical in both arms. The indexer's own
  `dynamo_kvrouter_kv_cache_events_applied{event_type,status}` counts events (not blocks) for
  all sources, has no static-source split and no gap counts, and is recorded inside the
  arm-specific indexer backends. Keep it only as a cross-check, for example for
  `status="block_not_found"` removes.
- Environment:

  | Variable | Meaning |
  |---|---|
  | `DYN_EXPERIMENT_STATIC_KV_REPORT_S` | Report interval in seconds (default 10). |
  | `DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT` | Rewrite each report as compact JSON to this path (atomic rename), plus one row per static source. |
  | `DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS` | Mark the timed start (set it to `--start-at-unix-ms`). Must lie in the future when the indexer starts. |
  | `DYN_EXPERIMENT_STATIC_KV_TIMED_END_UNIX_MS` | Mark the timed end: the publishers' stop (`start + --duration-s`) plus 1–10 s. Needs the start. |

- Window marks. At each mark the reporter reads the admitted totals (`taken_unix_ms` records
  when, since the reporter runs on a loaded runtime), then times a FIFO barrier through every
  event-thread queue of the indexer (`ThreadPoolIndexer::flush_and_wait`, same blob in both
  arms). `drain_ms` is about zero when the indexer kept up and the backlog's drain time when it
  did not. Admission feeds unbounded queues, so without it an arm that falls behind would
  neither drop nor push back, and warm-up still queued at the start would load the window.
- Each report is also logged at WARN as `EXPERIMENT static KV source accounting report=<json>`.
  It carries `kind` (`interval`, `split` at the start mark, `end` at the end mark, or `final`),
  `t_unix_ms`, `report_interval_s`, `static_sources`, `endpoints_per_sub`, the `start` and `end`
  marks, `warmup` (the start mark's totals), `timed` (start to end mark, or to now before the
  end mark), `after_end`, and `accounting` (totals, `sources_with_events`,
  `sources_first_event_late`, `gap_resets`, `rank_resets`, `dropped_events`, and up to 16
  anomalous sources). The file adds `source_columns` and `sources`: per static source its
  events, write blocks, first and last admitted event ID, and gap resets.
- A `final` report is best effort. It is written only when the subscriber is cancelled, and a
  router stopped by a signal can exit first; the local smoke's SIGTERM produced none. Read the
  file at least three report intervals after the last publisher exited.

CRTC neutrality:

- Every file the patch touches has the same blob at MAIN `e61319d830`, STACK `41803f6f2d` (the
  stack top merged onto the jemalloc L1), and the campaign base `5233229717`. So do the code the
  patch relies on (`discovery/kv_source_{watch,membership}.rs`, `direct_zmq_sub_pool.rs`,
  `recovery/{target,worker_query_state,worker_query_transport}.rs`) and
  `lib/kv-router/src/protocols.rs`:
  - `lib/llm/src/kv_router/indexer/recovery/{direct_zmq,mod,subscriber,worker_query}.rs`, plus
    the new `static_sources.rs`.
- The patch touches no CRTC or other `lib/kv-router` code. `git apply --cached --check` passes
  against all three trees; STACK was fetched from
  `artifacts/d2-jemalloc/stack-mu/stack-jemalloc.bundle` (`d2-jemalloc/L4`).

To apply on an arm tree, use one of:

```bash
git -C <arm-tree> apply /path/to/static-kv-sources.patch       # or: git -C <arm-tree> am ...
git -C <arm-tree> cherry-pick <static-sources commit> <delivery-accounting commit>
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
`EventEnvelope`) and sent as the same 4-frame multipart `ZmqPubTransport` sends on topic
`kv-events`: topic, big-endian publisher ID, big-endian sequence, and the encoded `Frame`. A
unit test compares the frames with the runtime transport's. The wire is format-identical, not
byte-identical, to a live worker's: hashes, worker IDs, and timestamps differ by construction.
The indexer's `ValidatedZmqSource` and SUB sockets accept the envelopes unchanged.

The sending socket is an XPUB, not a PUB. The subscriber side cannot tell the difference. The
publisher gains two things:

- **Subscription gate.** A PUB socket drops everything sent before the subscriber joins. Each
  publisher therefore sends nothing until the indexer has subscribed to every phantom socket it
  hosts. `--subscribe-timeout-s` (default 600) fails the run and names the first missing worker
  IDs. Warm-up starts `--warmup-delay-s` (default 1) after the last subscription. The publisher
  reports per-phantom subscribe latency, and counts phantoms whose subscriber later reconnected
  or left.
- **Counted drops.** With `ZMQ_XPUB_NODROP`, a send at the 100k-message high-water mark returns
  `EAGAIN`. The message is dropped, as a PUB socket would drop it, but it is counted
  (`hwm_dropped`).

Phantom `i` is built from its base `b(i)`:

- It uses worker ID `--worker-id-base + i`. The default base is `0x7F00_0000_0000_0000`, which
  is above the 2^53 discovery-safe publisher-ID space.
- Every block hash is remapped with a per-phantom bijection, `fmix64(h ^ key_i)`, so equality
  and parent/child structure survive while copies share no blocks.
- Bases are assigned to phantoms in contiguous ranges: `b(i) = floor(i * bases / total)`.

## Workflow

### 1. Build (campaign worktree)

```bash
cargo build --release -p dynamo-e2e-indexer-tools   # phantom_publisher, query_driver, phantom_plan, delivery_check
cargo bench --no-run -p dynamo-bench --no-default-features --features mooncake --bench mooncake_bench
```

The tools link `dynamo-llm` without the `block-manager` feature. The arm wheels are built
separately, from each arm tree with the patch applied.

### 2. Export streams (one capture per page size; cluster for real sizes)

Use the phase-1 capture arguments (`artifacts/offline-sizing/scripts/sizing_driver.py`):

```bash
mooncake_bench <pool>.pool.msgpack --workload agentic --agentic-engine sglang \
  --agentic-pool-sha256 3753eceb7d88e1d9dda3f3fc5ad6d138d8ec68ecda8734d4cc19a5f5e586ccba \
  --block-size <1|16> --num-gpu-blocks <capacity> \
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

Coverage: there is no looping, so a phantom stops when its timed section ends.
`timed_span_virtual_s / speedup` must cover the measurement duration plus the start spread;
`phantom_plan --duration-s` enforces this. Per-worker load is constant across `<bases>`
(weak scaling), so a longer `--agentic-sim-ms` with fewer bases buys coverage. For example,
32 bases × 24,500 s costs about as many worker-seconds as phase 1's 512 × 1,600 s. Size
`--agentic-plays-per-worker` so lanes do not run dry; the export report shows
`lanes_exhausted_before_cap`. Keep `--agentic-phase-spread × --agentic-sim-ms` near phase 1's
80 s: the phase offset delays a worker's start, so a long capture with the default 0.05 would
start some workers after their warm-up.

Stationarity: the streams are not stationary. The request rate stays flat (about 0.13 per
worker-second), but writes per request rise as agentic sessions turn over: the page-size-16
production capture wrote about 140–240 blocks per worker-second in the first ~14,000 virtual
seconds after the warm-up and about 430–475 later (phase 1's 400 s window saw about 142). A
window that plays only part of the span therefore does not run at the span average, so
`phantom_plan` solves the speedup on the planned window (section 3).

Eviction (removes): the timed window must run at steady-state eviction, where the cache is full
and every stored block eventually evicts another. Otherwise the stream under-represents
removes, about half the write mix at steady state.

- The export report's `eviction` section lists, per stream:
  - warm-up and timed stored and removed blocks;
  - `warmup_resident_fraction`: blocks resident after the warm-up over the capture's
    `--num-gpu-blocks`;
  - `timed_remove_ratio`: removed over stored blocks.
- The exporter flags streams below 0.8 (`bases_below_min`) and still writes them.
  `phantom_plan` and `phantom_publisher` refuse them unless `--allow-low-eviction`.
- The publisher checks the exact planned window too: `planned.timed_remove_ratio` in its plan
  line.
- To size the capture, make each worker's warm-up store clearly more blocks than
  `--num-gpu-blocks`, so `warmup_resident_fraction` is about 1 when the timed section starts. The
  levers are a longer `--agentic-warmup-sim-ms` or a smaller `--num-gpu-blocks`.
  - The tiny local capture (4 workers, 120 s warm-up) stored 0.27–0.42 M warm-up blocks per
    worker at page size 1 and 16–31 k at page size 16. That is far below the production
    786,432-token capacity (786,432 blocks at page size 1, 49,152 at page size 16), so the
    original tiny streams had no removes at all.
  - At production capacity and page size 1, a linear extrapolation from that capture gives
    about 225–350 s of warm-up sim time per worker. That is an estimate only; check the
    report.
  - A smaller capacity reaches steady state sooner, but it changes hit rates and so the
    overlap and remove mix. Record it as a deviation.
  - The capacity must hold the largest request. Below that, the capture fails with "offline
    replay detected an effect-free zero-duration pass". At page size 16, 2,048 and 8,192
    blocks failed this way and 12,288 worked.

### 3. Plan one load point

```bash
phantom_plan --streams <dir> --total-phantoms 1800 --target-write-blocks-per-sec <R> \
  --duration-s 1200 --start-spread-ms 5000 --live-sources <live mocker workers x DP ranks> \
  --publishers hostA:30000:900,hostB:30000:900 --sources-out sources.txt --indexer-env-out indexer.env
```

This prints:

- natural and aggregate rates: writes, events, queries, lookup blocks, and queries per phantom.
  With `--duration-s` the aggregate rates are the planned window's (`window`, `basis`), counted
  with the publisher's and driver's own stop rule from every base's timeline;
- the speedup and timed coverage. With `--duration-s` and `--target-write-blocks-per-sec`, the
  speedup is solved so that the planned window itself carries the target
  (`speedup_span_average` shows the naive whole-span value). Pass the printed `speedup` to every
  publisher (it is in each process's arguments) and driver; their own
  `--target-write-blocks-per-sec` uses the whole-span average and is exact only for stationary
  streams. Window events are engine events before the publisher pipeline coalesces them; the
  publisher's plan line has the wire counts;
- `eviction`: the aggregate timed remove ratio and any streams below the minimum;
- `indexer.sockets`: the serving indexer's ZMQ socket budget and the mandatory
  `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB` (below);
- `acceptance`: the delivery rule (section 5);
- one JSON line per publisher process with its exact `phantom_publisher` arguments.

`sources.txt` holds the indexer's static-source list. `indexer.env` holds the `export` lines for
`DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB` and `DYN_EXPERIMENT_STATIC_KV_SOURCES`; source the same file in
both arms.

Keep at most 1000 phantoms per publisher process. libzmq allows 1023 sockets per context, and
each phantom uses about 4 file descriptors: the socket mailbox, the listener, the connection, and
spare.

Socket budget (libzmq caps a process's shared context at 1023 sockets):

- The serving indexer groups KV event endpoints `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB` to a SUB
  socket. That fan-in covers phantoms and live workers alike.
- It also opens, per live worker, one ungrouped SUB socket each for `kv_metrics` and
  `active_sequences_events`; each mocker worker has its own runtime and publishers. The local
  smoke's indexer log shows both. `--sockets-per-live-source` defaults to 2 for these, and
  `--socket-reserve` (default 64) covers the rest.
- The plan picks the smallest fan-in that fits: `E = ceil((phantoms + live) / (1023 - 2 * live
  - 64))`. That is 4 at 1x (1,800 phantoms, 200 live) and 33 at 10x with the same live workers.
- The plan fails when the live workers' ungrouped sockets alone reach the cap, at about 480
  live workers with the defaults. No fan-in fixes that.
- It also prints a suggested `ulimit -n` (`min_nofile`).
- `--endpoints-per-sub N` pins the fan-in instead (it must fit, else the plan fails). Pin one
  value of at least 2 for a whole sweep, in both arms: otherwise the smallest fitting value
  changes with the load point (1, the dedicated per-source SUB path, at small points; 2+, the
  grouped `direct_zmq_sub_pool` path, at larger ones), which confounds CPU against load.
- `DYN_ROUTER_ZMQ_ENDPOINTS_PER_SUB` is mandatory: the patched indexer refuses static sources
  without it. Every accounting report echoes it (`endpoints_per_sub`), so you can confirm both
  arms used the same value.

### 4. Launch (same order as the smoke)

1. Pick `START_AT` (unix ms) far enough ahead to cover indexer startup, activation, the warm-up
   (`planned.warmup.write_blocks / --warmup-blocks-per-sec`), and slack. The indexer needs it at
   launch.
2. Start the live mockers and the serving indexer, in both arms with the same `indexer.env`:

   ```bash
   (set -a; source indexer.env
    DYN_EXPERIMENT_STATIC_KV_ACCOUNTING_OUT=<run>/accounting.json \
    DYN_EXPERIMENT_STATIC_KV_TIMED_START_UNIX_MS=$START_AT \
    DYN_EXPERIMENT_STATIC_KV_TIMED_END_UNIX_MS=$((START_AT + DURATION_S * 1000 + 2000)) \
    DYN_EXPERIMENT_STATIC_KV_REPORT_S=10 \
      python -m dynamo.router --endpoint <ns>.backend.generate --serve-indexer --router-block-size <page>)
   ```

   Raise `ulimit -n` to at least the plan's `min_nofile`.
3. Start the frontend: `python -m dynamo.frontend --router-mode kv --use-remote-indexer
   --kv-cache-block-size <page>`.
4. Start every publisher with the plan's arguments plus the shared values: `--speedup` or
   `--target-write-blocks-per-sec`, `--start-at-unix-ms $START_AT`, `--start-spread-ms`,
   `--duration-s`, and `--warmup-blocks-per-sec`.
   - Publishers may start before the indexer; they hold at the subscription gate.
   - The warm-up is untimed and must end before `--start-at-unix-ms`. `late_warmups` counts
     phantoms that missed, which invalidates the run.
5. Start the query drivers with the same shared values, plus:
   - `--component <ns>.backend`;
   - `--model-name <served model name>`;
   - optionally `--first-phantom/--count` to shard phantoms across driver processes.

Outputs:

- Each binary prints one JSON line per `--report-interval-s` and a summary (also written to
  `--summary-out`).
- Publisher plan line: `planned` warm-up and timed totals for this process. Summary: `sent`,
  `hwm_dropped`, the following fields, and `finished_unix_ms` (stamped after the ZMQ context
  terminated, i.e. after the sockets' 10 s linger flush; `sends_done_unix_ms` is before it):
  - `subscribe_latency`: from bind, including any wait for the indexer;
  - `timed_lag`: measured after each send returns;
  - `stop_at_unix_ms`, `first_timed_send_unix_ms`, `last_timed_send_unix_ms`;
  - `late_warmups`;
  - `resubscribed_phantoms` and `unsubscribed_phantoms`;
  - `per_phantom` (file only; columns in `per_phantom_columns`): planned events, write blocks
    and last event ID, and what was sent.
- Driver: RTT p50/p99, issue lag, achieved queries/s and lookup blocks/s, and the self-hit
  fraction.
- Indexer: the accounting reports (above).
- Indexer CPU comes from the host (pidstat or perf), not from these tools.

### 5. Delivery check (rejection rule)

After every publisher of a run has exited, wait three accounting intervals. Then, per arm:

```bash
delivery_check --publisher-summary pubA.json --publisher-summary pubB.json \
  --indexer-accounting <run>/accounting.json --expect-endpoints-per-sub <plan's value> --label <arm>
```

The load point is invalid if either arm is invalid. An arm is invalid when any of these holds:

- **Not exact per phantom.** Any phantom's admitted events, write blocks, or first and last
  event ID differ from what its publisher planned (IDs run from 1 through the warm-up and the
  timed lists before the stop). In a valid run (no gap, no high-water-mark drop) delivered
  equals planned once the indexer drains, at any load, so there is no tolerance: a 2% tail loss
  that an aggregate 98% rule would pass is invalid.
- **Window.** The indexer did not mark both window edges; at the start mark the admitted totals
  differ from the planned warm-up (warm-up spilled into the window, or timed traffic preceded
  it); admitted write blocks between the marks are below `--min-window-fraction` (0.99) of the
  planned timed blocks, or above them; the start mark is not the publishers' start; or the end
  mark is not within `--max-end-grace-ms` (10 s) after the publishers' stop.
- **Not drained.** The barrier took longer than `--max-drain-ms` (1000) at either mark, or did
  not run.
- **Not paced.** Any publisher's timed lag p99 exceeds `--max-lag-p99-ms` (50) or its maximum
  `--max-lag-ms` (1000), or its last timed send came more than `--max-send-overrun-ms` (1000)
  after its stop.
- Any gap reset (`ResetDegraded`), any other rank reset after a phantom indexed events, or any
  phantom whose first admitted event was not event 1.
- A warm-up ended after the timed start, a publisher hit send errors or was interrupted, the
  indexer accounted for a different number of static sources than the publishers host, its
  `endpoints_per_sub` differs from `--expect-endpoints-per-sub`, or its file was written less
  than two report intervals after the last publisher's `finished_unix_ms`.

High-water-mark drops and resubscriptions are reported as warnings; they also break exactness.
The defaults are printed in `phantom_plan`'s `acceptance` section.

### 6. Local smoke (loopback; plumbing only)

```bash
WORK=<scratch> STREAMS=<tiny stream dir with eviction> SPEEDUP=4 DURATION_S=30 lib/e2e-indexer-tools/scripts/local_smoke.sh
# page size 1: add BLOCK_SIZE=1 MOCKER_ARGS="--engine-type sglang"
# streams without eviction: add PLAN_ARGS=--allow-low-eviction (skips the removes check)
```

The smoke uses file discovery, the direct-ZMQ event plane, and the TCP request plane. It runs a
live mocker, 4 phantoms, the patched serving indexer, a remote-indexer frontend (one real chat
request), and the driver. It checks the following:

- (a) The publisher, started before the indexer, holds at the gate for `GATE_HOLD_S` and then
  delivers the whole warm-up.
- (b) Removes reach the indexer and are counted.
- (c) The full delivery rule passes (`delivery_check`), including both window marks and drains.
- (d) The indexer ran the grouped SUB path with the pinned `ENDPOINTS_PER_SUB` (default 2).

Tiny streams with eviction: page size 16, `--num-gpu-blocks 12288`, 4 workers, 120 s warm-up,
240 s sim (`artifacts/tooling/local-smoke/streams-ps16-evict`).

## Caveats (carry into any report)

Headline caveats:

- **No prefix sharing across phantoms.** As in phase 1, workers share no prefixes: the harness
  salts each base, and the publisher remaps each copy. A real fleet shares system prompts and
  tool definitions. The index therefore holds more distinct blocks, and lookups overlap less
  across workers, than in production.
- **Event cadence at page size 1.** The streams come from the SGLang-mode mocker. It emits about
  one Stored event per decode token, about 112 per request. Real SGLang 0.5.21 at page size 1
  emits about 2 Stored events per request (about 370 blocks each: the prefill tail, then the
  outputs at finish) and about 0.5 Removed events (about 1.4k blocks)
  (`artifacts/sglang-cadence/`). Block counts are comparable, but events and envelopes per
  block are inflated about 50x. Per-event indexer costs are overstated at page size 1 until the
  mocker cadence changes (ledger D7).

Other caveats:

- **Membership.** Membership filtering and recovery are skipped for phantom sources, identically
  in both arms.
- **Hash remapping.** Non-root sequence hashes are remapped rather than re-chained from the
  remapped local hashes. Neither arm's event-driven indexer (CRTC and ThreadPoolIndexer)
  recomputes sequence hashes from local hashes; grep `compute_next_seq_hash` at both SHAs.
- **Copies share timing.** Phantom copies of one base replay the same timing pattern, shifted
  only by `--start-spread-ms`.
- **Envelope boundaries.**
  - Timed lists are capture-timestamp groups. At page size 16 the tiny local capture gave one
    event per list, so that stream may send more, smaller envelopes than a live mocker that
    publishes a whole pass at once. At page size 1, the 41k events formed 24.8k lists.
  - Warm-up lists are runs of one worker in the merged warm-up order.
- **No looping.** The tools do not loop. Size the capture; the plan refuses short coverage.
- **Overload drops events.** At the high-water mark (100k messages per socket) a send is
  dropped, as in production. The publisher counts the drop (`hwm_dropped`). The resulting
  event-ID gap resets that phantom's rank at the indexer (live-only `ResetDegraded`), which the
  accounting counts. Either invalidates the load point.
- **Buffering before activation.** The gate proves subscription, not activation. Until a source
  activates, its envelopes wait in the indexer's SUB queue or group channel (100k each). At 10×,
  activation is slow (next item); if the warm-up outruns it, the overflow shows up as drops,
  gap resets, or late first events. Lengthen `--warmup-delay-s` or lower
  `--warmup-blocks-per-sec` if it does.
- **Per-query logging.** The serving indexer logs every `kv_indexer_query` at INFO
  (`push_handler` "request received/completed"). At thousands of queries per second that is
  real CPU in both arms. Keep `DYN_LOG` identical across arms, and consider filtering it. The
  accounting reports log at WARN, so they survive `DYN_LOG=warn`.
- **Accounting overhead.** Counting adds a hash lookup per envelope and a few atomic adds on a
  per-source cache line, identically in both arms.
- **Slow activation at scale.** Activation is O(sources) per readiness signal
  (`WorkerQueryClient::reconcile_view` and `ready_sources`), so startup is about O(N²) for N
  sources. Expect slow activation, plus one warning line per live-only source, at 10× (about
  18k sources).
- **Live-worker sockets.** Each live worker costs the serving indexer two ungrouped ZMQ sockets
  (and the remote-indexer frontend similar), so about 480 live workers is the ceiling under the
  1023-socket cap (section 3).
- **Version skew.** The tools are built from the campaign base `5233229717`. Between it and the
  arm SHAs, the event-plane codec and frames, the ZMQ transport (except one guidance string),
  and the TCP request plane are unchanged; the runtime diff touches only QUIC typed prologues
  and NATS. Keep the request and response planes on TCP.
- **Warm-up skip.** `--warmup-blocks-per-sec 0` skips the warm-up. Use it only for smoke tests.
  The timed section then references blocks the indexer never saw, and every phantom's first
  event is not event 1, so the delivery rule fails.
