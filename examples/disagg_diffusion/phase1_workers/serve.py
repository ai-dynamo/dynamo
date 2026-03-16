#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""HTTP serving mode for disaggregated diffusion pipeline.

Launches 3 stage worker groups (Encoder, Denoiser, VAE) on separate GPUs,
then serves video generation requests via HTTP.

Process architecture::

    serve.py (main)
    ├── Encoder worker subprocess     (GPU 0)      ← SGLang Scheduler + Llama + CLIP
    ├── Denoiser worker subprocess 0  (GPU 1)  ┐
    ├── Denoiser worker subprocess 1  (GPU 2)  ┘── SGLang Scheduler + DiT (TP=2)
    ├── VAE worker subprocess         (GPU 3)      ← SGLang Scheduler + 3D VAE
    └── FastAPI HTTP server           (no GPU)     ← orchestrates stages via ZMQ

Inter-stage tensor data transfers use NIXL RDMA (GPU-direct).
Only metadata (~1.5 KB) travels over ZMQ.

Usage::

    # Start server (default: 4 GPUs)
    python serve.py

    # Custom GPU layout
    GPU_ENC=0 GPU_DEN=1,2,3,4 GPU_VAE=5 TP_SIZE=4 python serve.py

    # Send requests
    curl http://localhost:8090/health

    curl -X POST http://localhost:8090/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "A cat walking on green grass", "num_frames": 9, "num_steps": 3}' \\
      --output output.mp4

    # Longer video (61 frames ≈ 2.5s at 24fps, ~8 min on H20)
    curl -X POST http://localhost:8090/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "A rocket launching into space", "num_frames": 61, "num_steps": 50}' \\
      --output rocket.mp4

Environment variables:
    MODEL_PATH    HuggingFace model (default: hunyuanvideo-community/HunyuanVideo)
    GPU_ENC       GPU for encoder (default: 0)
    GPU_DEN       GPUs for denoiser, comma-separated (default: 1,2)
    GPU_VAE       GPU for VAE (default: 3)
    TP_SIZE       Tensor parallelism degree (default: auto from GPU_DEN count)
    SERVE_HOST    HTTP bind address (default: 0.0.0.0)
    SERVE_PORT    HTTP port (default: 8090)
"""

from __future__ import annotations

import io
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Optional

WORKERS_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKERS_DIR not in sys.path:
    sys.path.insert(0, WORKERS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("serve")

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "hunyuanvideo-community/HunyuanVideo")
GPU_ENC = os.environ.get("GPU_ENC", "0")
GPU_DEN = os.environ.get("GPU_DEN", "1,2")
GPU_VAE = os.environ.get("GPU_VAE", "3")
TP_SIZE = int(os.environ.get("TP_SIZE", str(len(GPU_DEN.split(",")))))
SERVE_HOST = os.environ.get("SERVE_HOST", "0.0.0.0")
SERVE_PORT = int(os.environ.get("SERVE_PORT", "8090"))

# ── Reuse core components ────────────────────────────────────────────────
from run_e2e_sglang import (
    _patch_hunyuan_config_task_type,
    _detect_encoder_modules,
    _launch_stage,
    StageClient,
    terminate_processes,
)

# Global state
enc_client: Optional[StageClient] = None
den_client: Optional[StageClient] = None
vae_client: Optional[StageClient] = None
_all_procs = []
_request_counter = 0


# ── Pipeline execution ───────────────────────────────────────────────────

async def run_pipeline(
    prompt: str,
    negative_prompt: str = "",
    height: int = 544,
    width: int = 960,
    num_frames: int = 61,
    num_steps: int = 50,
    guidance_scale: float = 1.0,
    seed: int = 42,
) -> bytes:
    """Run full E2E pipeline, return mp4 bytes."""
    global _request_counter
    import torch
    from sglang_utils import build_req, inject_tensors_to_req

    req_id = _request_counter
    _request_counter += 1

    req_kwargs = dict(
        prompt=prompt, negative_prompt=negative_prompt,
        height=height, width=width, num_frames=num_frames,
        num_inference_steps=num_steps, guidance_scale=guidance_scale, seed=seed,
    )

    t_total = time.monotonic()

    # ── Encoder ──
    t0 = time.monotonic()
    enc_output = await enc_client.forward([build_req(**req_kwargs)])
    t_enc = time.monotonic() - t0
    if enc_output.error:
        raise RuntimeError(f"Encoder error: {enc_output.error}")
    enc_result = enc_output.output
    nixl_enc = "_nixl_transfer_meta" in enc_result

    # ── Denoiser ──
    t0 = time.monotonic()
    den_req = build_req(**req_kwargs)
    den_req.do_classifier_free_guidance = (guidance_scale > 1.0)
    if nixl_enc:
        den_req._nixl_transfer_meta = enc_result["_nixl_transfer_meta"]
    else:
        inject_tensors_to_req(den_req, enc_result)
    den_output = await den_client.forward([den_req])
    t_den = time.monotonic() - t0
    if den_output.error:
        raise RuntimeError(f"Denoiser error: {den_output.error}")
    den_result = den_output.output
    nixl_den = "_nixl_transfer_meta" in den_result

    # ── VAE ──
    t0 = time.monotonic()
    vae_req = build_req(prompt="", height=height, width=width,
                        num_frames=num_frames, num_inference_steps=num_steps,
                        guidance_scale=0.0, seed=seed)
    if nixl_den:
        vae_req._nixl_transfer_meta = den_result["_nixl_transfer_meta"]
    else:
        vae_req.latents = den_result["latents"].cpu()
    vae_output = await vae_client.forward([vae_req])
    t_vae = time.monotonic() - t0
    if vae_output.error:
        raise RuntimeError(f"VAE error: {vae_output.error}")

    t_elapsed = time.monotonic() - t_total

    # ── Encode as mp4 ──
    import numpy as np
    import imageio

    frames_tensor = vae_output.output
    if hasattr(frames_tensor, "cpu"):
        frames_tensor = frames_tensor.cpu().float().numpy()
    frames = (frames_tensor[0].transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    imageio.mimwrite(buf, frames, format="mp4", fps=24, codec="libx264")
    mp4_bytes = buf.getvalue()

    transfer = "NIXL" if (nixl_enc and nixl_den) else "ZMQ"
    logger.info(
        "req %d | %d frames %dx%d | enc=%.2fs den=%.2fs vae=%.2fs total=%.2fs | %s | %.1f KB",
        req_id, frames.shape[0], width, height,
        t_enc, t_den, t_vae, t_elapsed, transfer, len(mp4_bytes) / 1024,
    )
    return mp4_bytes


# ── FastAPI app ──────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Disaggregated Diffusion Server")


class GenerateRequest(BaseModel):
    prompt: str = "A cat walking on green grass"
    negative_prompt: str = ""
    height: int = 544
    width: int = 960
    num_frames: int = 61
    num_steps: int = 50
    guidance_scale: float = 1.0
    seed: int = 42


@app.get("/health")
async def health():
    n_enc = len(GPU_ENC.split(","))
    n_den = len(GPU_DEN.split(","))
    n_vae = len(GPU_VAE.split(","))
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "gpus": {
            "encoder": f"GPU {GPU_ENC} ({n_enc} process{'es' if n_enc > 1 else ''})",
            "denoiser": f"GPU {GPU_DEN} ({n_den} processes, TP={TP_SIZE})",
            "vae": f"GPU {GPU_VAE} ({n_vae} process{'es' if n_vae > 1 else ''})",
        },
        "total_gpu_workers": n_enc + n_den + n_vae,
        "transfer": "NIXL RDMA",
        "requests_served": _request_counter,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        mp4_bytes = await run_pipeline(
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            height=req.height, width=req.width, num_frames=req.num_frames,
            num_steps=req.num_steps, guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
        return Response(content=mp4_bytes, media_type="video/mp4")
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Stage lifecycle ──────────────────────────────────────────────────────

def launch_all_stages():
    global enc_client, den_client, vae_client, _all_procs

    from partial_gpu_worker import build_encoder_stages, build_denoiser_stages, build_vae_stages

    _patch_hunyuan_config_task_type()

    t0 = time.monotonic()
    logger.info("=" * 60)
    logger.info("  Launching disaggregated diffusion stages")
    logger.info("  Model:    %s", MODEL_PATH)
    logger.info("  Encoder:  GPU %s  (1 process)", GPU_ENC)
    logger.info("  Denoiser: GPU %s  (%d processes, TP=%d)", GPU_DEN, len(GPU_DEN.split(",")), TP_SIZE)
    logger.info("  VAE:      GPU %s  (1 process)", GPU_VAE)
    logger.info("  Total:    %d GPU worker processes", len(GPU_ENC.split(",")) + len(GPU_DEN.split(",")) + len(GPU_VAE.split(",")))
    logger.info("=" * 60)

    enc_procs, enc_args = _launch_stage(
        "Encoder", GPU_ENC,
        required_modules=_detect_encoder_modules(MODEL_PATH),
        custom_stages_fn=build_encoder_stages,
        tp_size=1, scheduler_port=15600,
    )
    den_procs, den_args = _launch_stage(
        "Denoiser", GPU_DEN,
        required_modules=["transformer", "scheduler"],
        custom_stages_fn=build_denoiser_stages,
        tp_size=TP_SIZE, scheduler_port=15700,
    )
    vae_procs, vae_args = _launch_stage(
        "VAE", GPU_VAE,
        required_modules=["vae", "scheduler"],
        custom_stages_fn=build_vae_stages,
        tp_size=1, scheduler_port=15800,
    )

    _all_procs.extend(enc_procs + den_procs + vae_procs)

    enc_client = StageClient(enc_args.scheduler_endpoint(), "encoder")
    den_client = StageClient(den_args.scheduler_endpoint(), "denoiser")
    vae_client = StageClient(vae_args.scheduler_endpoint(), "vae")

    logger.info("=" * 60)
    logger.info("  All %d workers ready in %.1fs", len(_all_procs), time.monotonic() - t0)
    logger.info("  HTTP server: http://%s:%d", SERVE_HOST, SERVE_PORT)
    logger.info("  Transfer:    NIXL RDMA (GPU-direct)")
    logger.info("")
    logger.info("  Try:")
    logger.info("    curl http://localhost:%d/health", SERVE_PORT)
    logger.info('    curl -X POST http://localhost:%d/generate \\', SERVE_PORT)
    logger.info('      -H "Content-Type: application/json" \\')
    logger.info('      -d \'{"prompt": "A cat on grass", "num_frames": 9, "num_steps": 3}\' \\')
    logger.info("      --output test.mp4")
    logger.info("=" * 60)


def shutdown_stages():
    for client in [enc_client, den_client, vae_client]:
        if client:
            try:
                client.close()
            except Exception:
                pass
    for p in _all_procs:
        p.terminate()
    for p in _all_procs:
        p.join(timeout=10)
    logger.info("All workers terminated.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    launch_all_stages()

    import uvicorn
    try:
        uvicorn.run(app, host=SERVE_HOST, port=SERVE_PORT, log_level="info")
    finally:
        shutdown_stages()
