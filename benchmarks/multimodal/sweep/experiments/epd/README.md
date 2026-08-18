# Aggregated / colocated EPD sweep

This experiment compares one-GPU Aggregated and colocated EPD topologies for
Dynamo vLLM and SGLang.

The vLLM workflow requires Dynamo PR #12004 or a branch containing the same
frontend-decoding interfaces.

This revision pins `vllm==0.26.0` with the
`vllm/vllm-openai:v0.26.0-ubuntu2404` runtime image, and `sglang==0.5.16`
with `lmsysorg/sglang:v0.5.16-cu130-runtime`. Use these exact images for matched
comparisons.

## Download images

```bash
python benchmarks/multimodal/sweep/experiments/epd/download_dataset.py \
  --output-dir /to/your/path
```

The downloader writes 50 images by default and pins the source revision and
checksum. Images are stored in row order under `images/` without resizing,
cropping, padding, or changing aspect ratio.

## Run the sweep

Run AIPerf inside the same Kubernetes pod as the model, not from a workstation:

```bash
DYN_RUNTIME_SOURCE_MODE=worktree \
python benchmarks/multimodal/sweep/experiments/epd/run_experiment.py \
  --backend vllm \
  --topology aggregate epd \
  --image-count 5,10,30 \
  --image-token-budget 128 256 \
  --isl 9000 \
  --osl 128,256,512,1024,2048 \
  --qps 0.5 \
  --model /models/path/to/Qwen3.5-122B-A10B-NVFP4 \
  --served-model-name qwen35-122b-a10b-nvfp4 \
  --image-dir /to/your/path/images \
  --aiperf-bin aiperf \
  --output-dir /workspace/epd-sweep-results/run
```

Each selector accepts one value or a comma- or space-separated list; the runner
executes their Cartesian product. Run each backend in its matching runtime pod,
or pass both only when the image contains both runtimes. Use `--dry-run` to print
the expanded cells without loading a model.
