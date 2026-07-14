# Qwen3-VL CustomEncoder image benchmark

This recipe compares three single-H100 runtimes with
`Qwen/Qwen3-VL-2B-Instruct`:

- upstream `vllm serve`;
- native aggregated `dynamo.vllm`;
- aggregated Dynamo with
  `examples.custom_encoder.qwen3_vl_vision_encoder.Qwen3VLVisionEncoder`.

The workload contains three disjoint pools of 1,000 deterministic JPEGs, one
pool for each offered rate (16, 24, and 32 requests/second). Every request has
one globally unique 500×500 RGB image between 50 and 60 KiB, exact server-side
ISL 515, and exact OSL 70. The native and custom JSONLs share the same image
order and semantic prompt; the custom prompt adds two harmless filler tokens to
compensate for its template omitting Qwen3-VL's two vision-boundary tokens.

Generate and audit the workload:

```bash
WORKLOAD_DIR=/dynamo-tmp/logs/$(date +%m-%d)/qwen3-vl-custom-sweep/workload
python examples/custom_encoder/benchmark/generate_workload.py \
    --output-dir "$WORKLOAD_DIR"
python examples/custom_encoder/benchmark/validate_workload.py "$WORKLOAD_DIR"
```

Run the nine benchmark cells and write `benchmark.md` plus `benchmark.csv`:

```bash
RUN_DIR=/dynamo-tmp/logs/$(date +%m-%d)/qwen3-vl-custom-sweep
WORKLOAD_DIR="$RUN_DIR/workload" OUTPUT_DIR="$RUN_DIR" \
    examples/custom_encoder/benchmark/run_sweep.sh
```

The canonical multimodal sweep runner launches each server independently and
invokes an equivalent of:

```bash
aiperf profile -m Qwen/Qwen3-VL-2B-Instruct -u http://localhost:8000 \
    --request-rate 16 --conversation-num 1000 --warmup-request-count 20 \
    --input-file image_native_qps16_1000_isl515.jsonl \
    --custom-dataset-type single_turn \
    --extra-inputs max_tokens:70 --extra-inputs min_tokens:70 \
    --extra-inputs ignore_eos:true --extra-inputs stream:true --streaming \
    --endpoint-type chat --endpoint /v1/chat/completions \
    --warmup-request-rate 1000 --warmup-arrival-pattern constant \
    --random-seed 42 --workers-max 20 --record-processors 32 \
    --use-server-token-count --request-timeout-seconds 300 \
    --artifact-dir <cell-dir> --ui none --no-server-metrics
```

No `AIPERF_HTTP_CONNECTION_LIMIT` override is used. Each cell stores its exact
expanded command in `command.txt`.

## Custom encoder loading

`Qwen3VLVisionEncoder.build()` loads `AutoProcessor` and
`Qwen3VLForConditionalGeneration` in bf16, retains `model.visual` as the ViT,
detaches that module from the full checkpoint, and releases the remaining
weights. It captures CUDA graphs for the configured image-grid and batch-bucket
pairs. This benchmark disables both CPU embedding and preprocessing caches and
uses a 1 ms custom-encoder queue wait, so every unique image executes the ViT.

The native Qwen3-VL path also injects DeepStack features into intermediate
language-model layers. The current CustomEncoder interface carries only primary
image embeddings. This is a performance comparison of the custom vision path,
not a numerical-output parity claim.
