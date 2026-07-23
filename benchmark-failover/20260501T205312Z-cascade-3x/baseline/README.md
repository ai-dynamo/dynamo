# baseline

3-worker plain TRT-LLM aggregated serving. No failover surface, no frontend migration. Worker recovery is whatever Kubernetes does when the container exits and `restartPolicy` fires (~5 min cold reload of the engine + EAGLE3 head + graph ladder).

## Configuration

- **Date**: 2026-05-01
- **Cluster**: nv-prd-dgxc nscale, 3 × B200 (tep8x1)
- **Workers**: 3 replicas across DRA nodes `s2877`, `l9nsv`, `tx5tk`
- **Image**: `dynamoci.azurecr.io/ai-dynamo/dynamo:failover-v2-675ae24fe21-trtllm-runtime` (TRT-LLM rc11, gms-refactor — failover surface OFF)
- **Failover**: OFF · **Migration**: OFF
- **Engine**: chunked_prefill on, autotuner on, max_batch=128, fp8 KV, EAGLE3 spec decode, MoE backend=TRTLLM, `free_gpu_memory_fraction=0.75`
- **aiperf**: `--concurrency 24 --benchmark-duration 1800 --concurrency-ramp-duration 60`

## Cascade timeline

| Event | Wall-clock (UTC) | Pod | Node | Container |
|---|---|---|---|---|
| aiperf start | 21:16:07 | — | — | — |
| Kill #1 (T+600 s) | 21:26:09 | `kimi-baseline-3x-0-trtllmworker-6g8t4` | s2877 | `main` |
| Kill #2 (T+660 s) | 21:27:10 | `kimi-baseline-3x-0-trtllmworker-ckfrj` | l9nsv | `main` |
| Kill #3 (T+720 s) | 21:28:11 | `kimi-baseline-3x-0-trtllmworker-tp9dd` | tx5tk | `main` |
| aiperf wraps | 21:46:24 | — | — | — |

Kill recipe: `kubectl exec -- kill -9 $(pgrep -f "orted|mpi4py.futures.server")` against the worker pod's `main` container. MPI children dying cascades to parent python via `MPI_ABORT`, container exits, K8s restarts.

## Headline metrics

| Metric | Value |
|---|---|
| Total requests | 2,439 |
| Successes (HTTP 200) | 1,806 |
| True failures | 633 |
| TTFT (ms) — avg / p50 / p90 | 1,549 / 961 / 2,890 |
| ITL (ms) — avg / p50 / p90 | 12.45 / 10.90 / 18.31 |
| Tok/s/user — avg / p50 / p90 | 96.5 / 91.8 / 142.3 |

## Charts

### Cumulative successes
![cumulative_successes](charts/cumulative_successes.png)

### Request outcome over time
![http_status_over_time](charts/http_status_over_time.png)

### Per-user decode rate
![tok_per_user_over_time](charts/tok_per_user_over_time.png)

### TTFT — 30 s mean
![ttft_avg_per_window](charts/ttft_avg_per_window.png)

### TTFT scatter
![ttft_scatter](charts/ttft_scatter.png)

### ITL — 30 s mean
![itl_avg_per_window](charts/itl_avg_per_window.png)

Raw artifacts (per-request records, kill plans, run logs, DGDs, harness scripts) are kept on the internal benchmark branch.
