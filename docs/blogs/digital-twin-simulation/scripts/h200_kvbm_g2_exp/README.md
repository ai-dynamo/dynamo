# H200 KVBM G2 Host-Memory KV Offload Experiment

Mocker simulation comparing single-engine performance on H200 Kimi-K2.5 with
and without the G2 (host-memory) KV cache tier enabled. The chart is referenced
from the KV Block Manager Simulation subsection in section 2.2 of the blog.

## Files

- `data.csv` - one row per `(source, concurrency)` pair. `source=baseline`
  means G1-only, and `source=g2` enables G2 with `num_g2_blocks=32768` and
  `kv_bytes_per_token=70272`.
- `plot.py` - reads `data.csv`, writes `h200_kvbm_g2_exp.png` to
  `../../images/`.

## Running

```bash
pip install matplotlib
python plot.py
```

## Setup

- Trace: Mooncake FAST25 `toolagent_trace.jsonl` (the full ~23,600-request
  trace used elsewhere in the blog)
- Workers: 1, replay mode: offline
- Engine: vLLM 0.19.0 via AIC (`aic_system=h200_sxm`)
- Model: Kimi-K2.5, TP=4, attention_dp=1, moe_tp=4, moe_ep=1
- `max_num_batched_tokens=2048`
- Concurrencies: 8, 16, 32, 64
- G2 condition: `num_g2_blocks=32768`, `kv_bytes_per_token=70272`

Kimi-K2.5 uses MLA-style latent KV. The transfer size follows the AIC model
definition `num_hidden_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes`,
which is `61 * (512 + 64) * 2 = 70272` bytes per token here.

If the local Python environment does not include the H200 vLLM 0.19.0 AIC
timing data, point `PYTHONPATH` at an AIC source checkout after fetching the
relevant LFS-backed timing files:

```bash
export PYTHONPATH=/path/to/aiconfigurator/src:components/src:lib/bindings/python/src
```

If the installed AIC package already includes `h200_sxm/vllm/0.19.0`, the normal
project Python environment is enough.

## Reproducing The Data

For each concurrency `<C>`, run twice (baseline and G2):

```bash
PY=python
TRACE=traces/mooncake_fast25/toolagent_trace.jsonl

# Baseline (G1 only)
$PY -m dynamo.replay "$TRACE" \
  --num-workers 1 --replay-mode offline --replay-concurrency <C> \
  --report-json /tmp/h200_kvbm_g2_exp/c<C>_baseline.json \
  --extra-engine-args '{"max_num_batched_tokens":2048,"aic_backend":"vllm","aic_system":"h200_sxm","aic_backend_version":"0.19.0","aic_model_path":"moonshotai/Kimi-K2.5","aic_tp_size":4,"aic_attention_dp_size":1,"aic_moe_tp_size":4,"aic_moe_ep_size":1}'

# With G2
$PY -m dynamo.replay "$TRACE" \
  --num-workers 1 --replay-mode offline --replay-concurrency <C> \
  --report-json /tmp/h200_kvbm_g2_exp/c<C>_g2.json \
  --extra-engine-args '{"max_num_batched_tokens":2048,"aic_backend":"vllm","aic_system":"h200_sxm","aic_backend_version":"0.19.0","aic_model_path":"moonshotai/Kimi-K2.5","aic_tp_size":4,"aic_attention_dp_size":1,"aic_moe_tp_size":4,"aic_moe_ep_size":1,"num_g2_blocks":32768,"kv_bytes_per_token":70272}'
```

`kv_bytes_per_token=70272` is required by the current mocker offload path; with
only `num_g2_blocks`, replay falls back to the baseline cache behavior. In this
environment the G2 offload path also needed to run outside the default sandbox
permissions because the offload engine initialization otherwise returned
`Operation not permitted`. The runs used `DYN_LOG=warn` to keep KVBM transfer
logging from dominating the replay.

Then aggregate these JSON fields into `data.csv`:

- `output_throughput_tok_s` -> `tps_gpu`
- `mean_output_token_throughput_per_user` -> `tps_user`
- `mean_tpot_ms` -> `tpot_ms`
- `mean_ttft_ms` -> `ttft_ms`
