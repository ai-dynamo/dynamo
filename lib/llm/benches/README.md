# Tokenizer Benchmarks

## tokenizer_simple

Uses Criterion with local test data. Runs automatically via `cargo bench`.

```bash
cargo bench --bench tokenizer_simple -p dynamo-llm
```

## tokenizer_dataset

Downloads a real dataset from HuggingFace Hub (LongBench-v2, ~500 samples) and
measures encode throughput, comparing `HuggingFaceTokenizer` against `FastTokenizer`.

This benchmark is **opt-in**: it exits immediately unless `RUN_BENCH=1` is set.
This prevents it from running during `cargo test --all-targets` in CI, since it
takes several minutes to complete.

### Basic run (default: Qwen/Qwen3-0.6B, LongBench-v2, 503 samples)

```bash
RUN_BENCH=1 cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Override tokenizer

```bash
RUN_BENCH=1 TOKENIZER_PATH=deepseek-ai/DeepSeek-V3 \
  cargo bench --bench tokenizer_dataset -p dynamo-llm
```

Use a local `tokenizer.json` file:

```bash
RUN_BENCH=1 TOKENIZER_PATH=/path/to/tokenizer.json \
  cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Override dataset and sample count

```bash
RUN_BENCH=1 DATASET=RyokoAI/ShareGPT52K MAX_SAMPLES=50 \
  cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Batched mode

```bash
RUN_BENCH=1 BATCH_SIZE=64 cargo bench --bench tokenizer_dataset -p dynamo-llm
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `RUN_BENCH` | unset | Must be set to any value to run the benchmark |
| `TOKENIZER_PATH` | `Qwen/Qwen3-0.6B` | HuggingFace model name or local path to `tokenizer.json` |
| `DATASET` | `zai-org/LongBench-v2` | HuggingFace dataset name (`zai-org/LongBench-v2` or `RyokoAI/ShareGPT52K`) |
| `MAX_SAMPLES` | `503` | Maximum number of samples to process |
| `BATCH_SIZE` | unset | If set, runs batched mode instead of sequential |
