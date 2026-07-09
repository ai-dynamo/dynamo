# HA Valkey JSON Configuration Weka-Derived Smoke Benchmark

Status: **passed**

## Topology

- Three Dynamo frontend processes, each started with the same
  `--router-valkey-config` JSON string.
- Four logical `dynamo.mocker` workers with direct Valkey KV-event publication.
- One router Valkey primary/replica pair loaded with `dynkv.so` and using
  synchronous `WAIT 1` mutation acknowledgement.
- One tokenizer-cache Valkey primary/replica pair configured with
  `allkeys-lru`.
- Three shared Sentinel processes. All three selected the router primary under
  one name and the tokenizer primary under a second name.
- TCP request plane and host-local etcd discovery.

The exact frontend configuration is in `router-valkey-config.json`.

## Dataset

The source topology was derived from
`semianalysisai/cc-traces-weka-062126-256k`. The bounded replay contains 476
independent full-context requests. Each source 64-token hash topology was
grouped into stable 512-token hashes for AIPerf 0.10's file-loader default.
Input length was capped at 8,192 source tokens and output length at 16 tokens.

The final derived JSONL SHA-256 is
`d09127c37fb0f37ae547c97039bda764faafaeedec1c29ba76180f018044338b`.

## AIPerf Results

| Metric | Result |
| --- | ---: |
| Completed requests | 476 / 476 |
| Errored or cancelled requests | 0 |
| Request throughput | 686.76 requests/s |
| Request latency p50 / p95 / p99 | 22.70 / 57.63 / 65.78 ms |
| TTFT p50 / p95 / p99 | 22.70 / 57.63 / 65.78 ms |
| ITL p50 / p95 / p99 | 0.000 / 0.000 / 0.115 ms |
| ISL average / p50 | 7,641.33 / 8,192 tokens |
| OSL average / p50 | 16.42 / 16 tokens |

The profiling window was 0.693 seconds. Treat throughput and tail percentiles
as a topology smoke result, not a production capacity result.

## State and Cache Checks

- Router worker ranks registered: 4 / 4.
- Router primary/replica database sizes: 1 / 1.
- Tokenizer primary/replica database sizes: 18 / 18.
- Tokenizer L1 hits/misses: 361 / 115.
- Tokenizer L2 hits/misses: 25 / 90, with 0 lookup errors, 0 write errors, and
  0 dropped writes.
- Mean L2 lookup time: 0.532 ms across 115 lookups.
- All three Sentinels agreed on both configured primary groups.

## Caveats

- The run used CPU-hosted mock workers with a `100000` speedup ratio, not a GPU
  backend.
- The run verified Sentinel discovery and two replicated groups but did not
  kill a live primary. The tokenizer client failover retry has a focused Rust
  test with three mock Sentinel witnesses and old/new primary endpoints.
- Teardown can log etcd watcher `channel closed` messages after measured
  traffic completes; no AIPerf request failed.
- This is not an in-process versus Valkey A/B run.
