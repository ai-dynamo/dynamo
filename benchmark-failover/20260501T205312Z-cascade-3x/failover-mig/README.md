# failover-mig

3-worker GMS shadow-failover (PR #8572) **plus** frontend request migration (`--migration-limit 10`). Failover handles the engine-side handoff via prewarmed engine-1 standby; migration handles client-visible disconnects via transparent retry on a different worker (re-prefilling original prompt + tokens already streamed).

## Configuration

- **Date**: 2026-05-02
- **Cluster**: nv-prd-dgxc nscale, 3 × B200 (tep8x1)
- **Workers**: 3 replicas across DRA nodes `l9nsv`, `d6dn5`, `s2877`
- **Image**: `dynamoci.azurecr.io/ai-dynamo/dynamo:failover-v2-675ae24fe21-trtllm-runtime`
- **Failover**: ON (`--gms-shadow-mode --load-format gms`) · **Migration**: ON (`--migration-limit 10` on Frontend)
- **Engine**: chunked_prefill on, autotuner on, max_batch=128, fp8 KV, EAGLE3 spec decode, MoE backend=TRTLLM, `free_gpu_memory_fraction=0.75`
- **aiperf**: `--concurrency 24 --benchmark-duration 1800 --concurrency-ramp-duration 60`

## Cascade timeline

| Event | Wall-clock (UTC) | Pod | Node | Container |
|---|---|---|---|---|
| aiperf start | 01:24:08 | — | — | — |
| Kill #1 (T+600 s) | 01:34:08 | `kimi-failover-mig-3x-0-trtllmworker-rv6m6` | l9nsv | `engine-0` |
| Kill #2 (T+660 s) | 01:35:10 | `kimi-failover-mig-3x-0-trtllmworker-tkpz9` | d6dn5 | `engine-0` |
| Kill #3 (T+720 s) | 01:36:11 | `kimi-failover-mig-3x-0-trtllmworker-tlxf6` | s2877 | `engine-0` |
| aiperf wraps | 01:54:21 | — | — | — |

Kill recipe: `kubectl exec -- kill -9 $(pgrep -f "orted|mpi4py.futures.server")` against the worker pod's `engine-0` container.

## Headline metrics

| Metric | Value |
|---|---|
| Total requests | 2,048 |
| Successes (HTTP 200) | 1,987 |
| True failures | 61 |
| TTFT (ms) — avg / p50 / p90 | 1,309 / 911 / 2,572 |
| ITL (ms) — avg / p50 / p90 | 13.50 / 12.97 / 19.90 |
| Tok/s/user — avg / p50 / p90 | 84.4 / 77.1 / 126.1 |

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
