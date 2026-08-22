# Aggregated / colocated EPD sweep

This experiment compares Aggregated and colocated EPD topologies for
Dynamo vLLM and SGLang.

Test with `vllm==0.26.0` with the
`vllm/vllm-openai:v0.26.0-ubuntu2404` runtime image and `sglang==0.5.16` with
`lmsysorg/sglang:v0.5.16-cu130-runtime`.

## Download images

```bash
cd benchmarks/multimodal/sweep/experiments/epd
python download_dataset.py \
  --output-dir /to/your/path
```

The downloader downloads 50 images by default. Pass `--count N` to download
more.

## Run the benchmark

### Run a single workload

```bash
# sglang or vllm
python run_experiment.py \
  --backend sglang \
  --topology aggregate epd \
  --image-count 5 \
  --osl 128 \
  --model nvidia/Qwen3.5-122B-A10B-NVFP4 \
  --image-dir /to/your/path/images \
  --output-dir /to/your/path/results
```

### Full sweep

```bash
# sglang or vllm
python run_experiment.py \
  --backend sglang \
  --topology aggregate epd \
  --image-count 5,10,30 \
  --image-token-budget 128 256 \
  --osl 128,512,2048 \
  --model nvidia/Qwen3.5-122B-A10B-NVFP4 \
  --image-dir /to/your/path/images \
  --output-dir /to/your/path/results
```
