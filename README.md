# KV indexer year-over-year benchmark

Achieved versus offered throughput for global KV-cache event indexers, replaying the same Mooncake trace on AMD EPYC 9654P hosts of identical configuration: the Rust indexers on one host, llm-d on a second.

## Figures

- `yoy.png` / `yoy.svg`: the Rust indexers built with mimalloc, plus llm-d.
  - The Oct 2026 CRTC line is the top of the four-PR CRTC stack with mimalloc, [ai-dynamo/dynamo#15606](https://github.com/ai-dynamo/dynamo/pull/15606) through [ai-dynamo/dynamo#15611](https://github.com/ai-dynamo/dynamo/pull/15611).
- `yoy_glibc.png` / `yoy_glibc.svg`: the same replay with every Rust indexer on glibc malloc.
  - The CRTC line there is `main` as of Oct 2026, which ships CRTC on glibc and hits glibc's arena-lock ceiling.

The left panel plots achieved against offered block ops/s on log-log axes; the dashed line is achieved = offered. The right panel plots lookup p99 from scheduled to answered, queueing included, only at offered loads an indexer sustains in every repetition: under overload that latency measures the replay backlog, not the indexer. The dotted line marks a matched offered load of 129M block ops/s, about half of SMG's capacity, where both CRTC and SMG keep up.

- `yoy_bars.png` / `yoy_bars.svg`: the same data as bar charts on linear axes.
  - Left: each indexer's highest sustained throughput, i.e. achieved block ops/s at the heaviest window where every repetition kept up.
  - Right: lookup p99 (scheduled → answered) at 51.8M block ops/s offered, the heaviest load comfortably within all three Rust indexers' capacity.
  - llm-d is drawn at its true 0.66M on the left, which is too short to see at this scale; its value is printed above the axis. On the right it is marked "cannot keep up", because it never sustains 51.8M.
  - Lookup bars more than 10× the next tallest are cropped at the top, with their value printed inside.

## Workload and method

- **Trace:** the public Mooncake trace (`mooncake_trace.jsonl`, SHA-256 `b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711`).
  - 128 inference workers, trace duplication factor 20, trace length factor 4, block size 128 tokens.
  - 472,160 requests and 1,974,035 KV events per replay; 320M block operations per replay.
- **Replay:** open loop. Each replay compresses the whole trace into a window W, so the offered rate is total operations / W.
  - Lookups are issued from 128 lanes. Stores and removes go to the indexer's event workers.
  - A block op is a requested block, a stored block, or a removed block.
- **Achieved rate:** block ops / (last completion − start).
  - A trial is discarded when its generator could not issue on schedule, that is, when the issue span exceeds 1.01 × W.
- **Reps:** three fresh-process repetitions per window. Points are medians and error bars span min–max.
- **Host:** AMD EPYC 9654P (96 cores, one socket), one logical CPU per physical core.
  - 8 cores issue events, 1 core issues lookups, and the indexer runs on the remaining 87.

## Indexers

| Line | Version | Event workers | Notes |
|---|---|---|---|
| Concurrent Radix Tree Compressed | Dynamo, Oct 2026 ([#15611](https://github.com/ai-dynamo/dynamo/pull/15611) @ `e234880162`) | 64 | mimalloc; top of the stack that starts at [#15606](https://github.com/ai-dynamo/dynamo/pull/15606) |
| Concurrent Radix Tree | Dynamo, Feb 2026 (Flash Indexer blog, `222c2e85c8`) | 64 | Tree and thread pool from that commit, ported unchanged into the current harness |
| SMG PositionalIndexer | SMG ([`smg-project/smg`](https://github.com/smg-project/smg)) `kv_index` @ `0f9f219` | 64 | Event-driven, sticky per-worker pool mirroring `KvEventMonitor`; built at opt-level 3 (SMG ships opt-level `z`) |
| llm-d precise prefix index | llm-d-router v0.11.0 `InMemoryIndex` | 4 shards (default) | Go driver replaying the same corpus with sequence-hash keys |

- **Event workers:** each line uses its best count from a scan of 16–80 workers. In `yoy_glibc`, `main`'s CRTC runs at 16, its best count on glibc; the other Rust lines stay at 64.
- **Score check:** on a 1,000-request Mooncake fixture, the Feb 2026 tree and SMG returned the same overlap scores as CRTC for every lookup. So did a sequence-hash-keyed model of llm-d's index (0 mismatches each).
- **Not included:** indexers that are approximate or that mix request history into the index.

## Caveats

- **Feb 2026 allocator.** In the main figure, the Feb 2026 tree is rebuilt with mimalloc, so allocators match across the Rust lines. It shipped on glibc, and its glibc line is in `yoy_glibc`.
- **Latency panel.** Scheduled → answered includes queueing. The Feb 2026 tree's lookups queue behind its writes, so its p99 reaches about 2 s at its last sustained load even though throughput keeps up.

## Files

- `summary_mimalloc.csv`, `summary_glibc.csv`: per-window medians behind the figures.
- `trials.csv`: every trial. Columns: offered and achieved rates, generator validity, `kept_up`, drain, lookup p50/p99/p99.9, update p99, and operation totals.
