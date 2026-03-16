# Disaggregated Diffusion Inference

Split a video diffusion pipeline (Text Encoder -> DiT Denoiser -> 3D VAE) into
independent stages on separate GPUs for pipeline parallelism and independent scaling.

Design doc: [docs/design/disaggregated_diffusion.md](../../docs/design/disaggregated_diffusion.md)

## Quick Start (HunyuanVideo, 4 GPUs)

```bash
# Single request — generates a 2.5s video at 544x960
python phase1_workers/run_e2e_sglang.py

# 4 requests, 2 concurrent
NUM_REQUESTS=4 CONCURRENCY=2 python phase1_workers/run_e2e_sglang.py

# Custom prompt and parameters
PROMPT="A rocket launching into space" NUM_FRAMES=33 NUM_STEPS=30 \
  python phase1_workers/run_e2e_sglang.py
```

Default GPU layout: Encoder (GPU 0) | Denoiser TP=2 (GPU 1,2) | VAE (GPU 3).

Override with `GPU_ENC`, `GPU_DEN`, `GPU_VAE` environment variables.

## Architecture

```
  Encoder (GPU 0)         Denoiser (GPU 1,2)         VAE (GPU 3)
  ┌──────────────┐        ┌──────────────────┐       ┌─────────────┐
  │ Llama3-8B    │ embeds │ HunyuanVideo DiT │ lats  │ 3D Causal   │
  │ + CLIP       │───ZMQ──│ 13B (TP=2)       │──ZMQ──│ VAE         │──► video
  │ ~16 GB       │        │ ~13 GB/GPU       │       │ ~2 GB       │
  └──────────────┘        └──────────────────┘       └─────────────┘
```

Each stage runs as a separate SGLang scheduler subprocess with its own CUDA context.

## Supported Models

| Model | Encoders | DiT | Status |
|-------|----------|-----|--------|
| **HunyuanVideo v1** (`hunyuanvideo-community/HunyuanVideo`) | Llama3-8B + CLIP | 13B | Default |
| Wan2.2-TI2V-5B (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | T5 | 5B | Supported |

To use Wan: `MODEL_PATH=Wan-AI/Wan2.2-TI2V-5B-Diffusers GUIDANCE=5.0 python phase1_workers/run_e2e_sglang.py`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `hunyuanvideo-community/HunyuanVideo` | HuggingFace model ID or local path |
| `PROMPT` | `A cat walking on green grass` | Text prompt |
| `GPU_ENC` | `0` | GPU(s) for encoder |
| `GPU_DEN` | `1,2` | GPU(s) for denoiser |
| `GPU_VAE` | `3` | GPU(s) for VAE |
| `TP_SIZE` | auto from `GPU_DEN` | Tensor parallelism degree |
| `NUM_FRAMES` | `61` | Video frames (~2.5s at 24fps) |
| `NUM_STEPS` | `50` | Denoising steps |
| `HEIGHT` | `544` | Frame height |
| `WIDTH` | `960` | Frame width |
| `GUIDANCE` | `1.0` | Guidance scale (>1.0 enables CFG) |
| `NUM_REQUESTS` | `1` | Number of pipeline runs |
| `CONCURRENCY` | `1` | Max concurrent pipelines |

## Files

| File | Purpose |
|------|---------|
| `phase1_workers/run_e2e_sglang.py` | Orchestrator: launches stages, runs E2E pipeline, reports timing |
| `phase1_workers/partial_gpu_worker.py` | `PartialGPUWorker`, `IntermediateOutputStage`, subprocess launcher |
| `phase1_workers/sglang_utils.py` | Partial pipeline loading, tensor injection/extraction, config sync |
| `phase0_validate/` | Offline split validation (FLUX.1-schnell, diffusers) |
