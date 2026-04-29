# Qwen3-Omni Benchmark Results (DYN-2581)

Per-cell AIPerf results from `run_sweep.sh`. Topologies compared: vllm_serve.

## chat — prompt=long

### TTFT (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 85.8 | 90.0 | 92.4 | 93.7 |
| 4 | 124.2 | 125.6 | 133.2 | 142.7 |
| 8 | 152.5 | 153.6 | 171.9 | 221.3 |
| 16 | 218.9 | 217.8 | 286.0 | 286.8 |
| 32 | 308.1 | 310.5 | 473.3 | 474.4 |

### ITL (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 4.554 | 4.544 | 4.784 | 4.976 |
| 4 | 6.043 | 6.019 | 6.369 | 6.658 |
| 8 | 7.222 | 7.152 | 7.625 | 7.790 |
| 16 | 8.752 | 8.704 | 9.392 | 9.818 |
| 32 | 11.1 | 11.0 | 12.4 | 13.2 |

### E2E (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 717.6 | 721.2 | 724.4 | 736.0 |
| 4 | 960.8 | 959.9 | 975.3 | 981.1 |
| 8 | 1,154 | 1,154 | 1,175 | 1,208 |
| 16 | 1,429 | 1,422 | 1,481 | 1,481 |
| 32 | 1,830 | 1,819 | 1,952 | 1,953 |

### Out Tok/s

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 194.6 | — | — | — |
| 4 | 580.3 | — | — | — |
| 8 | 967.4 | — | — | — |
| 16 | 1,558 | — | — | — |
| 32 | 2,402 | — | — | — |

### Req/s

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 1.390 | — | — | — |
| 4 | 4.156 | — | — | — |
| 8 | 6.920 | — | — | — |
| 16 | 11.2 | — | — | — |
| 32 | 17.4 | — | — | — |


## chat — prompt=short

### TTFT (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 63.7 | 63.5 | 64.5 | 68.2 |
| 4 | 71.9 | 70.7 | 77.8 | 81.9 |
| 8 | 82.5 | 76.6 | 82.4 | 191.3 |
| 16 | 121.7 | 111.5 | 234.0 | 234.9 |
| 32 | 180.2 | 134.3 | 309.7 | 310.7 |

### ITL (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 4.543 | 4.559 | 4.781 | 4.939 |
| 4 | 6.074 | 6.046 | 6.397 | 6.580 |
| 8 | 7.273 | 7.295 | 7.630 | 7.893 |
| 16 | 8.650 | 8.684 | 9.062 | 9.544 |
| 32 | 10.0 | 10.0 | 10.7 | 11.0 |

### E2E (ms)

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 690.8 | 689.9 | 692.2 | 707.5 |
| 4 | 908.4 | 909.3 | 921.3 | 927.4 |
| 8 | 1,082 | 1,074 | 1,092 | 1,186 |
| 16 | 1,322 | 1,315 | 1,435 | 1,436 |
| 32 | 1,573 | 1,509 | 1,686 | 1,687 |

### Out Tok/s

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 201.2 | — | — | — |
| 4 | 610.7 | — | — | — |
| 8 | 1,018 | — | — | — |
| 16 | 1,660 | — | — | — |
| 32 | 2,815 | — | — | — |

### Req/s

| Concurrency | vllm_serve avg | vllm_serve p50 | vllm_serve p90 | vllm_serve p99 |
|---|---|---|---|---|
| 1 | 1.444 | — | — | — |
| 4 | 4.396 | — | — | — |
| 8 | 7.340 | — | — | — |
| 16 | 11.9 | — | — | — |
| 32 | 20.0 | — | — | — |


## Summary

_Recommendation paragraph: which (workload, concurrency, prompt-len) regimes favor disagg, where vllm-serve still wins. Fill in once the full sweep completes._

