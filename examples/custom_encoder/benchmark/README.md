# Qwen2.5-VL CustomEncoder image benchmark

This recipe compares three single-H100 runtimes with
`Qwen/Qwen2.5-VL-3B-Instruct`:

- upstream `vllm serve`;
- native aggregated `dynamo.vllm`;
- aggregated Dynamo with
  `examples.custom_encoder.qwen2_vl_vision_encoder.Qwen2VLVisionEncoder`.

The workload contains three disjoint pools of 1,000 deterministic JPEGs, one
pool for each offered rate (16, 24, and 32 requests/second). Every request has
one globally unique 500×500 RGB image between 50 and 60 KiB, exact server-side
ISL 515, and exact OSL 70. The native and custom JSONLs share the same image
order and semantic prompt. Each prompt is calibrated independently: the native
path uses the checkpoint processor, while the custom path counts its exact Jinja
tokens and replaces the single image placeholder with the processor-derived
number of merged image tokens.

Generate and audit the workload:

```bash
WORKLOAD_DIR=/dynamo-tmp/logs/$(date +%m-%d)/qwen2-vl-custom-sweep/workload
python examples/custom_encoder/benchmark/generate_workload.py \
    --output-dir "$WORKLOAD_DIR"
python examples/custom_encoder/benchmark/validate_workload.py "$WORKLOAD_DIR"
```

Run the nine benchmark cells and write `benchmark.md` plus `benchmark.csv`:

```bash
RUN_DIR=/dynamo-tmp/logs/$(date +%m-%d)/qwen2-vl-custom-sweep
WORKLOAD_DIR="$RUN_DIR/workload" OUTPUT_DIR="$RUN_DIR" \
    examples/custom_encoder/benchmark/run_sweep.sh
```

The canonical multimodal sweep runner launches each server independently and
invokes an equivalent of:

```bash
aiperf profile -m Qwen/Qwen2.5-VL-3B-Instruct -u http://localhost:8000 \
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

`Qwen2VLVisionEncoder.build()` loads `AutoProcessor` and
`Qwen2_5_VLForConditionalGeneration` in bf16, retains `model.visual` as the ViT,
detaches that module from the full checkpoint, and releases the remaining
weights. It captures CUDA graphs for the configured image-grid and batch-bucket
pairs. This benchmark disables the preprocessing cache and uses a 1 ms
custom-encoder queue wait, so every unique image executes the ViT.

Native HF vision output and the static graph adapter are checked directly for
numerical parity before the sweep.

## Qwen2.5-1.5B performance-only concurrency sweep

The separate concurrency recipe serves `Qwen/Qwen2.5-1.5B-Instruct` as the
decoder while still computing the complete vision tower from
`Qwen/Qwen2.5-VL-3B-Instruct`. Because their hidden widths differ, it truncates
the native 2048-wide vision output to its first 1536 columns. This is deliberately
a performance-only adapter: it is not a trained projection and has no output
quality or model-parity claim.

It runs only `dynamo-custom-encoder` at closed-loop client concurrency 8, 16,
and 32. Each cell gets a disjoint 1,000-image pool, exact ISL 515 and OSL 70,
and uses preprocessing concurrency 64, maximum batch cost 64, graph buckets
`1,2,4,8,16,32,64`, and only the 500×500 graph shape. Since every request has
one cost-1 image, concurrency at most 32 cannot select bucket 64 even though the
server captures it.

```bash
OUTPUT_DIR=/dynamo-tmp/logs/$(date +%m-%d)/qwen2-vl-1.5b-custom-concurrency \
    examples/custom_encoder/benchmark/run_concurrency_sweep.sh
```

The script generates and audits 3,000 unique JPEGs, runs CUDA-graph checks, a
smoke request, and the three measured cells, then requires
`BENCHMARK_AUDIT=PASS cells=3`. It writes the complete Markdown, CSV, JSON,
commands, workload manifest, graph-verification log, and sweep log below the
chosen output directory.

## Custom encoder ablation

After the fixed nine-cell comparison, run the separate 18-cell ablation using
the same workload:

```bash
ABLATION_DIR="$RUN_DIR/ablation"
WORKLOAD_DIR="$RUN_DIR/workload" OUTPUT_DIR="$ABLATION_DIR" \
    examples/custom_encoder/benchmark/run_ablation.sh
```

It isolates immediate-drain eager execution (0 ms queue wait, maximum batch cost
1 or 8), CUDA graphs without batching, and full, coarse, or single-rung graph
ladders (1 ms queue wait) at all three offered rates. Results are written to
`ablation.md`, `ablation.csv`, and `ablation_validation.json` without changing
the fixed nine-cell benchmark audit.
