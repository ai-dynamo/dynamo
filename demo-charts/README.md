# Failover demo — chart artifacts

Snapshots of the perf and failover demo runs, captured 2026-04-30 on `nv-prd-dgxc nscale` cluster (B200, 1 worker × 8 GPU = tep8x1).

- **Model**: `nvidia/Kimi-K2.5-NVFP4` + `nvidia/Kimi-K2.5-Thinking-Eagle3`
- **Stack**: dynamo branch `mabdulwahhab/trtllm-failover-build-20260429` (TRT-LLM rc11 fork at SHA `9ba5ee40`, gms-refactor)
- **Workload**: `kv-reuse-difficult_200k-fixed/dataset.jsonl` mooncake-trace replay via aiperf 0.7.0
- **Engine config**: graphs `[1,2,4,8,16,32,64,128]+padding`, autotuner on, chunked_prefill on, overlap on, CUTLASS NVFP4 GEMM, max_batch=128, fp8 KV, `free_gpu_memory_fraction=0.75`, Eagle3 spec decode

---

## Failover demo — baseline vs failover, kill at t=0

Two 10-min runs at concurrency 8 against single-worker deployments. Primary worker process killed at t=5:00. Same client load on both, same trace, same SLA.

| | Baseline (failover OFF) | Failover ON |
|---|---|---|
| Total requests | 486 | 271 |
| **Successful requests** | **98** | **131 (+34%)** |
| Errors | 388 | 140 |
| **Time from kill → first successful response** | **284 s** (4m44s) | **46 s** |
| TTFT p50 pre-kill | 456 ms | 519 ms |
| TTFT p50 post-kill | 532 ms | 627 ms |

### Cumulative successful requests

The cleanest view: both lines climb in lockstep pre-kill; baseline flatlines for ~4 minutes; failover dips briefly and resumes within ~50 s.

![cumulative_successes](demo-failover/cumulative_successes.png)

### TTFT scatter

Each dot is one request's TTFT. Both lines look similar pre-kill. Post-kill: failover scatters resume after a brief gap; baseline shows a long blackout.

![ttft_scatter](demo-failover/ttft_scatter.png)

### Per-user decode rate (tok/s/user) over time

Each dot is one request's `output_token_throughput_per_user`; bold lines are the 30 s mean per setup. Pre-kill both setups ramp from ~200 down to ~95 as concurrency saturates. Post-kill: baseline scatter goes silent until ~+250 s; failover line stays continuous, hovering near the 100 tok/s/user SLA threshold.

![tok_per_user_over_time](demo-failover/tok_per_user_over_time.png)

### Goodput per 30-second window

Successful (SLA-meeting) request count per 30 s. Pre-kill both setups deliver ~10–14 successes per 30 s. Post-kill: baseline holds at zero for the entire ~5-min cold restart window, then surfaces a few late completions; failover dips briefly and re-establishes a steady ~5–7 successes/30 s.

![goodput_per_window](demo-failover/goodput_per_window.png)

> **Bin choice (30 s)**: Kimi 50 K-token completions arrive ~once every 5-10 s, so 1-second bins were mostly 0/1 spikes. 30 s captures 3-6 completions per bin in steady state — smooth enough to read while keeping pre-kill and post-kill regions clearly separated.

Raw aggregates in [demo-failover/summary.csv](demo-failover/summary.csv).

---

## Methodology

- aiperf args: `--streaming --extra-inputs ignore_eos:true --concurrency 8 --benchmark-duration 600 --benchmark-grace-period 60 --concurrency-ramp-duration 60 --goodput "time_to_first_token:5000 inter_token_latency:10" --workers-max 200 --record-processors 8`
- Kill recipe: `kubectl exec -- kill -9 $(pgrep -f "orted|mpi4py.futures.server")` — destroys MPI worker tree, cascades to parent python via MPI_ABORT, terminates container, K8s restarts (baseline) / failover lock releases (failover)
- "Successful" = `aiperf.metrics.good_request_count >= 1`, i.e. request met both `TTFT ≤ 5000ms` and `ITL ≤ 10ms`
- "First successful response after kill" = smallest `request_end_ns - kill_ns` across all good requests
