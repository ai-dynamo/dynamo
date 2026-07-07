# LLM Benchmarks

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

## video_decode

Measures the `video-rs` decoder. The default benchmark uses the checked-in
320x240 H.264 fixture and samples 30 of its 100 frames:

```bash
cargo bench -p dynamo-llm --features media-ffmpeg --bench video_decode -- \
  video_decode_h264_320x240_100_to_30
```

The C1, C8, and C32 sweep is opt-in so `cargo test --all-targets` executes only
the small benchmark. Generate or provide a larger input, then run:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i testsrc2=size=1280x720:rate=24:duration=10 \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -g 48 -an \
  /tmp/dynamo-video-720p24-10s.mp4

RUN_VIDEO_DECODE_SWEEP=1 \
VIDEO_DECODE_BENCH_INPUT=/tmp/dynamo-video-720p24-10s.mp4 \
VIDEO_DECODE_BENCH_NUM_FRAMES=30 \
  cargo bench -p dynamo-llm --features media-ffmpeg --bench video_decode -- \
    video_decode_concurrent
```

Each concurrent iteration decodes one video per worker. Override
`VIDEO_DECODE_BENCH_NUM_FRAMES` to change the number of uniformly sampled output
frames. If `VIDEO_DECODE_BENCH_INPUT` is unset, the sweep uses the small
checked-in fixture.

## request_trace_finish_metadata

Measures request finish metadata overhead with tracing disabled/enabled, plus
tool-call metadata recording across increasing numbers of calls.

```bash
cargo bench --bench request_trace_finish_metadata -p dynamo-llm --features request-trace-bench
```
