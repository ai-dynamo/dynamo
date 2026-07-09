# Weka-Derived In-Process versus HA Valkey A/B

Status: **passed**

The campaign alternated complete, fresh topologies over 3 repetitions
per arm. Both arms used three frontends, four logical mock workers, TCP request
transport, synchronized frontend-local admission, the same release binary,
the same dataset, 128 closed-loop concurrency, unlimited request
rate, 1,024 warmup requests, and 15,232 measured
requests per sample.

The control used an in-process router and a per-frontend tokenizer L1. The
candidate used a replicated router Valkey pair plus a separate replicated
tokenizer-cache pair, with both groups discovered through three shared
Sentinels.

## Median Results

| Metric | In-process median | HA Valkey median | Valkey delta |
| --- | ---: | ---: | ---: |
| Request throughput | 3311.966 requests/s | 3350.708 requests/s | +1.17% |
| Request latency p50 | 6.191 ms | 6.154 ms | -0.60% |
| Request latency p95 | 46.825 ms | 47.024 ms | +0.42% |
| TTFT p50 | 5.575 ms | 5.674 ms | +1.78% |
| TTFT p95 | 46.772 ms | 46.982 ms | +0.45% |
| ITL p99 | 0.541 ms | 0.473 ms | -12.53% |
| ISL average | 7641.328 tokens | 7641.328 tokens | +0.00% |
| OSL average | 16.355 tokens | 16.354 tokens | -0.01% |

Positive throughput delta is better; positive latency delta is worse.

## Per-Repetition Throughput

| Repetition | In-process RPS | HA Valkey RPS | Valkey delta |
| ---: | ---: | ---: | ---: |
| 1 | 3311.966 | 3330.740 | +0.57% |
| 2 | 3328.212 | 3350.708 | +0.68% |
| 3 | 3259.837 | 3367.695 | +3.31% |

The median paired throughput delta was
+0.68%. The small absolute
difference should be treated as parity rather than proof of a material speedup.

## Cache and Integrity Validation

- In-process median tokenizer L1 hits/misses:
  16256 / 0.
- HA Valkey median tokenizer L1 hits/misses and L2 hits/misses:
  15982 / 274
  and 38 / 217.
- Tokenizer L2 lookup/write errors: 0
  / 0.
- Completed records across all samples: 91,392;
  request errors/cancellations: 0 /
  0.
- Direct-Valkey integrity failure markers: 0.

## Caveats

- CPU mock workers used a 100000 speedup ratio; this isolates frontend/router
  overhead and does not model GPU inference capacity.
- The Weka-derived workload caps ISL at 8,192 and OSL at 16 tokens and groups
  source 64-token hashes into stable 512-token hashes for AIPerf 0.10.
- This campaign did not kill a primary during measured traffic.
- Sub-millisecond ITL percentiles are quantized in this fast mocker setup; do
  not interpret their relative percentage changes as production token latency.
