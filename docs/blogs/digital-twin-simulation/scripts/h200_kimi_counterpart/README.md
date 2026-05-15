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

- Trace: full 23,608-request Mooncake FAST25 `toolagent_trace.jsonl`
- Trace format: `mooncake`
- Trace block size: 512
- Model: `moonshotai/Kimi-K2.5`
- System: `h200_sxm`
- Backend: vLLM 0.19.0 through AIC
- Engine block size: 512
- `num_gpu_blocks=16384`
- `max_num_batched_tokens=16384`
- MoE plumbing: `moe_tp_size=tp_size`, `moe_ep_size=1`,
  `attention_dp_size=1`

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

No candidate was feasible under the strict 4 s TTFT SLA. The best near-miss:

| Field | Value |
|---|---:|
| `prefill_tp / decode_tp` | `2 / 1` |
| `prefill_workers / decode_workers` | `5 / 6` |
| `prefill_load_scale` | `0.5` |
| Output throughput | `303.13 tok/s` |
| Prefix reuse | `0.5383` |
| Mean TTFT | `4058.07 ms` |
| Mean TPOT | `58.57 ms` |
| Mean E2E | `13319.99 ms` |

The optimizer log column is named `overlap_score_weight`, but in this code path
the backward-compatible config maps it to `prefill_load_scale`.
