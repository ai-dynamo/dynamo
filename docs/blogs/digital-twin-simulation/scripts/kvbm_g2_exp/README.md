# KVBM G2 Host-Memory KV Offload Experiment

Mocker simulation comparing single-engine performance on B200 MiniMax-M2.5 with
and without the G2 (host-memory) KV cache tier enabled. The chart is referenced
from the KV Block Manager Simulation subsection in §2.2 of the blog.

## Files

- `data.csv` — one row per concurrency. Pair of columns per metric:
  `baseline_*` (G1-only) vs `g2_*` (with `num_g2_blocks=32768`), plus
  precomputed `*_reduction_pct` / `*_delta_pct` columns.
- `plot.py` — reads `data.csv`, writes `kvbm_g2_exp.png` to `../../images/`.

## Running

```bash
pip install matplotlib
python plot.py
```

## Setup

- Trace: Mooncake FAST25 `toolagent_trace.jsonl` (the full ~23,600-request
  trace used elsewhere in the blog)
- Workers: 1, replay mode: offline
- Engine: vLLM 0.14.0 via AIC (`aic_system=b200_sxm`)
- Model: MiniMax-M2.5 FP8, TP=4, attention_dp=1, moe_tp=4, moe_ep=1
- `max_num_batched_tokens=2048`
- Concurrencies: 8, 16, 32, 64
- G2 condition: `num_g2_blocks=32768`

## Reproducing the data

For each concurrency `<C>`, run twice (baseline and G2):

```bash
# Baseline (G1 only)
python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --num-workers 1 --replay-mode offline --replay-concurrency <C> \
  --report-json /tmp/mocker_b200_minimax_tp4_c<C>_baseline.json \
  --extra-engine-args '{"max_num_batched_tokens":2048,"aic_backend":"vllm","aic_system":"b200_sxm","aic_backend_version":"0.14.0","aic_model_path":"MiniMaxAI/MiniMax-M2.5","aic_tp_size":4,"aic_attention_dp_size":1,"aic_moe_tp_size":4,"aic_moe_ep_size":1}'

# With G2 — add "num_g2_blocks":32768 to --extra-engine-args
python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --num-workers 1 --replay-mode offline --replay-concurrency <C> \
  --report-json /tmp/mocker_b200_minimax_tp4_c<C>_g2.json \
  --extra-engine-args '{"max_num_batched_tokens":2048,"aic_backend":"vllm","aic_system":"b200_sxm","aic_backend_version":"0.14.0","aic_model_path":"MiniMaxAI/MiniMax-M2.5","aic_tp_size":4,"aic_attention_dp_size":1,"aic_moe_tp_size":4,"aic_moe_ep_size":1,"num_g2_blocks":32768}'
```

Then aggregate the relevant fields from the two report JSONs per concurrency
into `data.csv` (one row each).
