# Disaggregated Diffusion Inference

Decomposes a monolithic video diffusion pipeline into three independently
deployable stages — **Text Encoder**, **DiT Denoiser**, and **3D VAE** —
each running on separate GPUs. Inter-stage tensor data transfers use
**NIXL RDMA** (GPU-direct); only small metadata (~1.5 KB) travels over
the ZMQ control plane.

## 1. Design

### 1.1 Motivation

In monolithic diffusion inference, all model components occupy a single GPU
for the entire request lifetime. The encoder (~16 GB) sits idle during the
denoising loop (96%+ of wall time), and the VAE (~2 GB) sits idle during
encoding + denoising. Disaggregation enables:

- **Independent scaling** per stage (e.g. 1 encoder : N denoisers : 1 VAE)
- **Pipeline parallelism** — while request N is denoising, request N+1 encodes
- **Memory efficiency** — each GPU loads only its stage's weights
- **Heterogeneous hardware** — encoder on cost-efficient GPUs, denoiser on H100/H200

### 1.2 Architecture

```
serve.py (orchestrator, no GPU)
│
├── Encoder worker      GPU 0         Llama3-8B + CLIP  (~16 GB)
│        │
│        │ ── NIXL RDMA (embeddings, ~3.6 MB) ──►
│        ▼
├── Denoiser worker 0   GPU 1  ┐
├── Denoiser worker 1   GPU 2  ┘── HunyuanVideo DiT 13B, TP=2  (~13 GB/GPU)
│        │
│        │ ── NIXL RDMA (latents, ~1.4 MB) ──►
│        ▼
├── VAE worker          GPU 3         3D Causal VAE  (~2 GB)
│
└── FastAPI HTTP server   :8090       routes ZMQ metadata between stages
```

**Data flow per request:**

1. Orchestrator sends `Req(prompt)` → Encoder via ZMQ
2. Encoder runs `TextEncodingStage`, registers embeddings as NIXL-readable,
   returns metadata via ZMQ (~1.5 KB, no tensor data)
3. Orchestrator forwards NIXL metadata → Denoiser via ZMQ
4. Denoiser's `NixlReceiveStage` RDMA-pulls embeddings directly from Encoder GPU,
   runs 50 denoising steps, registers latents as NIXL-readable
5. Orchestrator forwards NIXL metadata → VAE via ZMQ
6. VAE's `NixlReceiveStage` RDMA-pulls latents from Denoiser GPU,
   runs 3D VAE decode, returns video frames

### 1.3 Key Implementation Details

**Partial pipeline loading** — Each stage loads only its required modules via
`build_partial_pipeline()`, which creates a dynamic subclass that suppresses
automatic stage creation and syncs component configs for unloaded modules.

**NIXL transfer** (`nixl_transfer.py`) — `NixlTensorSender` flattens GPU
tensors into a contiguous buffer, registers it as NIXL-readable, and returns
a metadata dict. `NixlTensorReceiver` allocates a GPU buffer and RDMA-pulls
the data. Falls back to ZMQ pickle when NIXL is unavailable.

**Dual-encoder support** — HunyuanVideo uses Llama3-8B + CLIP (incompatible
tensor shapes). `NixlSendStage` indexes each element separately
(`prompt_embeds_0`, `prompt_embeds_1`); `NixlReceiveStage` reconstructs the list.

**SGLang compatibility patches** — HunyuanConfig missing `task_type` default
(wrapped `__init__`); triton `norm_infer` non-contiguous assertion (patched
in subprocess entry point).

## 2. Quick Start

### HTTP Server (recommended)

```bash
# Start server — 4 GPU worker processes + HTTP endpoint
python phase1_workers/serve.py

# Health check
curl http://localhost:8090/health
# → {"status":"ok", "total_gpu_workers":4, "transfer":"NIXL RDMA", ...}

# Quick test (9 frames, 3 steps, ~10s)
curl -X POST http://localhost:8090/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat walking on green grass", "num_frames": 9, "num_steps": 3}' \
  --output test.mp4

# Full quality (61 frames ≈ 2.5s video, 50 steps, ~8 min on H20)
curl -X POST http://localhost:8090/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A rocket launching into space", "num_frames": 61, "num_steps": 50}' \
  --output rocket.mp4
```

### Batch / Benchmark Mode

```bash
python phase1_workers/run_e2e_sglang.py                          # single request
NUM_REQUESTS=4 CONCURRENCY=2 python phase1_workers/run_e2e_sglang.py  # concurrent
```

### Custom GPU Layout

```bash
# 6 GPUs: encoder=0, denoiser TP=4 on GPUs 1-4, VAE=5
GPU_ENC=0 GPU_DEN=1,2,3,4 GPU_VAE=5 TP_SIZE=4 python phase1_workers/serve.py
```

## 3. Supported Models

| Model | Encoders | DiT | Status |
|-------|----------|-----|--------|
| **HunyuanVideo v1** (`hunyuanvideo-community/HunyuanVideo`) | Llama3-8B + CLIP | 13B | Default, validated |
| Wan2.2-TI2V-5B (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`) | T5 | 5B | Supported |

Encoder module detection is automatic via `model_index.json`. Any SGLang-supported
diffusion model with the standard 3-stage structure should work.

## 4. Code Structure

```
phase1_workers/
├── serve.py               # HTTP server (FastAPI), launches stage workers
├── run_e2e_sglang.py      # Batch mode orchestrator with timing report
├── nixl_transfer.py       # NixlTensorSender / NixlTensorReceiver (GPU-direct RDMA)
├── partial_gpu_worker.py  # PartialGPUWorker, NixlSendStage, NixlReceiveStage
└── sglang_utils.py        # build_partial_pipeline(), tensor inject/extract helpers
```

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `hunyuanvideo-community/HunyuanVideo` | HuggingFace model ID or local path |
| `GPU_ENC` | `0` | GPU for encoder |
| `GPU_DEN` | `1,2` | GPUs for denoiser (comma-separated) |
| `GPU_VAE` | `3` | GPU for VAE |
| `TP_SIZE` | auto | Tensor parallelism degree for denoiser |
| `SERVE_PORT` | `8090` | HTTP server port |
| `NUM_FRAMES` | `61` | Video frames (batch mode) |
| `NUM_STEPS` | `50` | Denoising steps (batch mode) |

## 6. Roadmap

- [ ] **Dynamo Router integration** — replace manual ZMQ orchestrator with
  Router-orchestrated multi-stage chaining, typed stage endpoints
- [ ] **Independent stage scaling** — N:M:K ratio with load-balanced routing
- [ ] **Encoder caching** — LRU cache for repeated prompts
- [ ] **Sequence parallelism** — Ulysses/Ring SP for denoiser beyond TP
- [ ] **Continuous batching** — batch requests at different denoising steps
- [ ] **Heterogeneous hardware** — encoder on L4/T4, denoiser on H100/H200

## 7. Dependencies

```bash
pip install sglang nixl ai-dynamo-runtime imageio imageio-ffmpeg fastapi uvicorn
```
