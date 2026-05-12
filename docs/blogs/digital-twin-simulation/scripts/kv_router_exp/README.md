# KV Router Experiment

Mocker simulation comparing round-robin routing against Dynamo's KV Router on
the full Mooncake FAST25 `toolagent_trace.jsonl` trace. The chart is referenced
from the Router simulation subsection in section 2.2 of the blog.

## Files

- `data.csv` - one row per `(router_mode, concurrency)` with throughput,
  latency, request throughput, prefix reuse, and completed request count.
- `plot.py` - reads `data.csv`, writes `kv_router_exp.png` to `../../images/`.

## Running

```bash
python plot.py
```

## Setup

- Trace: Mooncake FAST25 `toolagent_trace.jsonl`, full 23,608-request trace
- Workers: 8, replay mode: offline
- Router modes: `round_robin`, `kv_router`
- Engine: vLLM 0.14.0 via AIC (`aic_system=b200_sxm`)
- Model: MiniMax-M2.5 FP8, TP=4, attention_dp=1, moe_tp=4, moe_ep=1
- `max_num_batched_tokens=2048`
- Concurrencies: 64, 128, 256, 512
- G2 offload disabled

## Reproducing The Data

For each router mode `<MODE>` and concurrency `<C>`:

```bash
python -m dynamo.replay traces/mooncake_fast25/toolagent_trace.jsonl \
  --num-workers 8 --router-mode <MODE> \
  --replay-mode offline --replay-concurrency <C> \
  --report-json /tmp/mocker_b200_minimax_tp4_8w_<MODE>_c<C>.json \
  --extra-engine-args '{"max_num_batched_tokens":2048,"aic_backend":"vllm","aic_system":"b200_sxm","aic_backend_version":"0.14.0","aic_model_path":"MiniMaxAI/MiniMax-M2.5","aic_tp_size":4,"aic_attention_dp_size":1,"aic_moe_tp_size":4,"aic_moe_ep_size":1}'
```

Then aggregate the report fields into `data.csv`. `tps_gpu` divides cluster
output token throughput by 32 GPUs: 8 workers times TP=4.
