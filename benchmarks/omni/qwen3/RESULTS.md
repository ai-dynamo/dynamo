# Qwen3-Omni Benchmark Results (DYN-2581)

_Placeholder — populated by `analyze.py` after the sweep runs on Hopper._

Run order:

1. Stand up all three deployments (see [`recipes/qwen3-omni/vllm/`](../../../recipes/qwen3-omni/vllm/)).
2. `bash benchmarks/omni/qwen3/run_sweep.sh --topology {agg,disagg,vllm_serve} --url ...`
3. `python3 benchmarks/omni/qwen3/analyze.py` overwrites this file.

## Recommendation (TBD)

Will summarize which (workload, concurrency, prompt-len) regimes favor disagg
once the data lands.
