# Agentic growing-history tokenizer-cache A/B

Date: 2026-07-06 (America/Los_Angeles)

## Outcome

This test isolates the tokenizer cache: both arms use the same in-process KV
router. The baseline is the existing in-process tokenizer L1. The candidate is
the new per-frontend L1 plus a shared, single-node Valkey L2.

- Pinned conversations: Valkey does not help after placement because every
  turn returns to the same frontend. Median aggregate throughput was 1,909.4
  RPS versus 1,965.5 RPS in-process (-2.85%).
- Stateless frontend handoff: median effective throughput across all five
  growing-history stages was 1,254.3 RPS versus 1,088.5 RPS (+15.23%).
- The benefit is concentrated where a request reaches a frontend without its
  session prefix. At 4 messages, Valkey improved RPS 59.97% and reduced TTFT
  p50 64.60%. At 8 messages, it improved RPS 47.57% and reduced TTFT p50
  58.23%.

## Median results (3 repetitions)

### Pinned conversation

| Messages | Nominal ISL | In-proc RPS | Valkey RPS | RPS delta | In-proc TTFT p50 | Valkey TTFT p50 | TTFT delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3,517 | 439.8 | 423.6 | -3.68% | 36.69 ms | 44.95 ms | +22.50% |
| 4 | 4,173 | 439.7 | 423.5 | -3.69% | 48.48 ms | 48.83 ms | +0.71% |
| 8 | 5,485 | 457.9 | 446.6 | -2.46% | 46.79 ms | 47.14 ms | +0.74% |
| 16 | 8,109 | 456.9 | 458.8 | +0.43% | 50.87 ms | 50.56 ms | -0.60% |
| 32 | 13,657 | 433.0 | 436.7 | +0.85% | 66.09 ms | 67.49 ms | +2.12% |

Per-depth pinned RPS is the completion rate within that turn's overlapping
window. The aggregate AIPerf request-throughput medians are the appropriate
overall comparison: 1,965.5 RPS in-process and 1,909.4 RPS with Valkey.

### Stateless frontend handoff

| Messages | Nominal ISL | In-proc RPS | Valkey RPS | RPS delta | In-proc TTFT p50 | Valkey TTFT p50 | TTFT delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3,517 | 1,418.9 | 1,382.5 | -2.57% | 47.76 ms | 50.10 ms | +4.89% |
| 4 | 4,173 | 1,279.5 | 2,046.9 | +59.97% | 58.83 ms | 20.83 ms | -64.60% |
| 8 | 5,485 | 1,113.7 | 1,643.5 | +47.57% | 68.90 ms | 28.78 ms | -58.23% |
| 16 | 8,109 | 1,215.6 | 1,227.0 | +0.94% | 56.68 ms | 52.66 ms | -7.09% |
| 32 | 13,657 | 731.1 | 756.4 | +3.45% | 110.21 ms | 114.32 ms | +3.73% |

ITL p50 was 0 ms in nearly every cell because the mock workers used a 100,000x
speedup and generated only 16 output tokens. ITL p95 varied from roughly 0.01
to 2.64 ms and is dominated by the mock streaming/timer path. The tokenizer
cache affects preprocessing and therefore TTFT, not decode-token cadence.

## Cache evidence

- Every handoff Valkey repetition made 1,155 L2 lookups. The median run had
  768 L2 hits and 387 misses.
- At 4 messages, the three runs recorded 383, 384, and 382 L2 hits. Median L2
  lookup time was 0.197 ms.
- At 8 messages, every run recorded 384/384 L2 hits. Median L2 lookup time was
  0.459 ms.
- The cold 2-message stage recorded no L2 hits and had a median 3.261 ms lookup.
- Pinned runs recorded no L2 hits: after the cold first turn, their local L1
  had the useful session prefix.
- Across the three handoff Valkey repetitions there were 3 fail-open lookup
  errors out of 3,465 lookups (0.087%), 3 background write errors, and zero
  queue-full write drops. All 5,760 handoff requests completed. No pinned run
  recorded an L2 error or write error.

## Test configuration

- Model/tokenizer: `Qwen/Qwen3-0.6B`
- Agent histories: 2, 4, 8, 16, and 32 complete messages
- Nominal ISL: 3,517, 4,173, 5,485, 8,109, and 13,657 tokens
- OSL: requested 16; observed averages were 16.31-16.41 tokens
- Load: infinite-rate closed loop, concurrency 128, 384 sessions per depth
- Topology: 3 frontend processes, 4 mock workers, TCP request plane, KV router
- Repetitions: 3, with arm order reversed on the second repetition
- CPU placement: mocker 0-2, Valkey 3, frontends 4-15, AIPerf 16-23
- Valkey: local single server, pool size 16, 50 ms timeout, persistence disabled
- L1: 64 MiB per frontend, prefix extension enabled
- Release binding SHA-256:
  `f529665984098ccd94a39b88be7bcdf536ae19b30081342b765c2471995aa6db`

Each arm/repetition used a fresh mocker, fresh frontends, and (for the Valkey
arm) a fresh Valkey server and scope. The handoff workload retained the
topology between depths but rotated frontend URL order so the same logical
session moved to a different frontend. The pinned workload used AIPerf's normal
same-URL session affinity. The repository retains the aggregate summary,
provenance, workload manifest, report, and plot. High-volume datasets, raw
records, and process logs are deliberately excluded and can be regenerated
with the adjacent benchmark driver.

## Findings and next optimization

1. Shared L2 is valuable for frontend churn, load-balancer handoff, and
   autoscaling. It is not a throughput win for already-sticky conversations;
   there the local L1 is both sufficient and cheaper.
2. The current lookup order stops on any L1 prefix. When URL rotation returns
   a conversation to an older frontend at 16 and 32 messages, that frontend's
   stale L1 prefix prevents checking Valkey for a newer, deeper prefix. This is
   why the large gains at 4 and 8 messages disappear later. A targeted fix is
   to query L2 for candidates deeper than an L1 hit only when the local hit is
   several message boundaries behind the newest safe boundary.
3. The candidate path has a cold-burst stampede: the custom L1+L2 path sent
   384 concurrent cold lookups, while the existing synchronous in-process L1
   quickly shared its common system/tool prefix. Single-flight/coalescing for
   identical prefix lookup and fill work should recover the roughly 2.6-3.7%
   cold/pinned overhead.
4. Increase or adapt the connection pool before production defaults are set.
   Pool 16 at concurrency 128 still produced three fail-open lookup timeouts in
   one repetition, although no user request failed.

This is a CPU-host software-path benchmark with a local Valkey server and mock
workers. It demonstrates cache-path behavior, not cross-node network latency,
Valkey HA/failover, GPU model throughput, or a maximum-capacity RPS claim.
