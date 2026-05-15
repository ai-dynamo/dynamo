# H200 Kimi Counterpart Data

This directory contains the H200 counterpart data for the DynoSim blog draft.
It is intentionally separate from the B200 MiniMax and planner data used by the
main post.

## Files

- `router_data.csv` - normalized router sweep data for round robin and KV
  Router.
- `optimize_evaluated.csv` - all H200 replay-optimizer candidate rows.
- `optimize_summary.json` - optimizer setup plus best infeasible result.
- `optimize_setup.json` - serialized setup from the remote run.
- `optimize_progress.log` - remote optimizer progress log.
- `plot.py` - renders the H200 router figure into `../../images/`.

## Render Figures

```bash
.venv/bin/python docs/blogs/digital-twin-simulation/scripts/h200_kimi_counterpart/plot.py
```

Generated figures:

- `../../images/h200_kv_router_exp.png`

## Shared Setup

| Category | Value |
|---|---|
| Workload | Full 23,608-request Mooncake FAST25 `toolagent_trace.jsonl`, `trace_format=mooncake`, `trace_block_size=512` |
| Model/system | `moonshotai/Kimi-K2.5`, H200-SXM, vLLM 0.19.0 through AIC |
| Engine config | `block_size=512`, `num_gpu_blocks=16384`, `max_num_batched_tokens=16384` |
| MoE config | `moe_tp_size=tp_size`, `moe_ep_size=1`, `attention_dp_size=1` |

## Router Sweep

- Workers: 8 aggregated workers
- TP: 4, total GPUs: 32
- Concurrencies: 32, 64, 128, 256, 512
- Router modes:
  - `round_robin`
  - `kv_router`

The KV Router raises average prefix reuse from 0.413 to 0.492 and cuts TTFT by
46-58% versus round robin across the sweep.

## Replay Optimize

- Remote run: SC-01 x86 A30 node, CPU-only replay workload
- Arrival speedup ratio: 0.25
- GPU budget: 16
- Objective: maximize output throughput
- SLA: mean TTFT <= 4000 ms, mean TPOT <= 75 ms, mean E2E <= 20000 ms
- Search rounds: 2
- Parallel evaluations: 64

No candidate was feasible under the strict 4 s TTFT SLA. The best near-miss is
summarized in the same category/result shape as the main blog draft:

| Category | Result |
|---|---|
| Workload | `moonshotai/Kimi-K2.5`, vLLM 0.19.0, H200-SXM, full 23,608-request `toolagent_trace.jsonl`, `arrival_speedup_ratio=0.25` |
| Engine config | `block_size=512`, `num_gpu_blocks=16384`, `max_num_batched_tokens=16384` |
| Budget | 16 GPUs |
| Objective | Maximize output throughput subject to mean TTFT <= 4,000 ms, mean TPOT <= 75 ms, and mean end-to-end latency <= 20,000 ms |
| Best near-miss layout | `prefill_tp=2`, `decode_tp=1`, `prefill_workers=5`, `decode_workers=6` |
| Router | `kv_router`, `prefill_load_scale=0.5` |
| Key metrics | `output_throughput_tok_s=303.13`, `prefix_cache_reused_ratio=0.5383`, `mean_ttft_ms=4058.07`, `mean_tpot_ms=58.57`, `mean_e2e_latency_ms=13319.99` |

The optimizer log column is named `overlap_score_weight`, but in this code path
the backward-compatible config maps it to `prefill_load_scale`.
