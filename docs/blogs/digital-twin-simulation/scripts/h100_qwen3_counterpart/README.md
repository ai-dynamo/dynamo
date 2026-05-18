# H100 Qwen3 Counterpart Data

This directory contains the H100 Qwen3-32B counterpart data for the DynoSim
blog draft. It is intentionally separate from the B200 MiniMax figure in the
main post and from the H200 Kimi counterpart.

## Files

- `data.csv` - normalized long-format data for silicon, offline Mocker replay,
  and AIC.
- `plot.py` - renders the H100 Qwen3-32B Pareto and 4-panel figures into
  `../../images/`.

## Render Figures

```bash
.venv/bin/python docs/blogs/digital-twin-simulation/scripts/h100_qwen3_counterpart/plot.py
```

Generated figures:

- `../../images/h100_hw_mocker_aic_pareto.png`
- `../../images/h100_hw_mocker_aic_4panel.png`

## Shared Setup

| Category | Value |
|---|---|
| Silicon source | `Qwen/Qwen3-32B` H100-SXM aggregate rows from [ai-dynamo/aiconfigurator `silicon_sample.csv`](https://github.com/ai-dynamo/aiconfigurator/blob/6477924179f2503c9191361dc036f38de3bfc7a2/src/aiconfigurator/systems/silicon_sample.csv#L1494-L1589) |
| Model/system | `Qwen/Qwen3-32B`, H100-SXM, vLLM |
| Shape | TP=2, ISL=1024, OSL=1024 |
| Concurrency sweep | 8, 16, 32, 64, 128 |
| Replay timing DB | vLLM 0.14.0 AIC timing database, shared by AIC estimates and offline Mocker replay |
| Replay config | `block_size=64`, `max_num_batched_tokens=2048`, `max_num_seqs=256`, `gpu_memory_utilization=0.9` |
| Replay workload | Synthetic closed-loop replay with `request_count=10*concurrency` |
| Throughput metric | Derived from TPOT as `concurrency * (1000 / TPOT_ms) / TP` |

The replay rows here are offline Mocker replay results, not live Mocker
deployment measurements. The silicon rows are used as the external reference
point, while AIC and Mocker replay use the same vLLM 0.14.0 timing database for
parity in this comparison.

## Plotted Metrics

For the plotted points, the MAPE values versus silicon are:

| Metric | Mocker replay | AIC |
|---|---:|---:|
| TPS/GPU | 16.1% | 17.3% |
| TPS/User | 16.1% | 17.3% |
| TPOT | 13.8% | 14.6% |
| TTFT | 8.5% | 24.7% |
| TTFT+TPOT average | 11.1% | 19.7% |

At concurrency 128, the measured TTFT values are 652.3 ms for silicon,
586.8 ms for offline Mocker replay, and 291.7 ms for AIC.
