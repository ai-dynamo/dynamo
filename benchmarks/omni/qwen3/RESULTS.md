# Qwen3-Omni Benchmark Results (DYN-2581)

Per-cell AIPerf results from `run_sweep.sh`. Topologies compared: vllm_serve.

## chat — prompt=short

### TTFT (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 4 | — | — | — | — |

### ITL (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 4 | — | — | — | — |

### E2E (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 4 | 931.2 | 911.6 | 1,001 | 1,001 |

### Out Tok/s

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 4 | 572.8 | — | — | — |

### Req/s

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 4 | 4.193 | — | — | — |


## Summary

_Recommendation paragraph: which (workload, concurrency, prompt-len) regimes favor disagg, where vllm-serve still wins. Fill in once the full sweep completes._

