#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""HTTP serving mode for disaggregated diffusion pipeline.

Launches Encoder, Denoiser (TP=2), and VAE stages, then serves
requests via a FastAPI HTTP server.

Usage:
    python serve.py

    # Then send requests:
    curl -X POST http://localhost:8090/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "A cat walking on green grass", "num_frames": 9, "num_steps": 3}' \
      --output output.mp4

    # Check server health:
    curl http://localhost:8090/health
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import multiprocessing as mp
import os
import sys
import time
from typing import Optional

# Ensure workers dir on path
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
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_e2e")


# ── Reuse run_e2e components ─────────────────────────────────────────────
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
    """Run the full E2E pipeline and return mp4 bytes."""
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

    t_start = time.monotonic()

    # Encoder
    t0 = time.monotonic()
    enc_output = await enc_client.forward([build_req(**req_kwargs)])
    enc_time = time.monotonic() - t0
    if enc_output.error:
        raise RuntimeError(f"Encoder error: {enc_output.error}")
    logger.info("req %d | Encoder %.2fs", req_id, enc_time)

    # Denoiser
    t0 = time.monotonic()
    den_req = build_req(**req_kwargs)
    inject_tensors_to_req(den_req, enc_output.output)
    den_req.do_classifier_free_guidance = (guidance_scale > 1.0)
    den_output = await den_client.forward([den_req])
    den_time = time.monotonic() - t0
    if den_output.error:
        raise RuntimeError(f"Denoiser error: {den_output.error}")
    logger.info("req %d | Denoiser %.2fs", req_id, den_time)

    # VAE
    t0 = time.monotonic()
    vae_req = build_req(prompt="", height=height, width=width,
                        num_frames=num_frames, num_inference_steps=num_steps,
                        guidance_scale=0.0, seed=seed)
    vae_req.latents = den_output.output["latents"].cpu()
    vae_output = await vae_client.forward([vae_req])
    vae_time = time.monotonic() - t0
    if vae_output.error:
        raise RuntimeError(f"VAE error: {vae_output.error}")

    total = time.monotonic() - t_start
    logger.info("req %d | VAE %.2fs | Total %.2fs", req_id, vae_time, total)

    # Convert to mp4 bytes
    import numpy as np
    import imageio

    frames_tensor = vae_output.output
    if hasattr(frames_tensor, "cpu"):
        frames_tensor = frames_tensor.cpu().float().numpy()
    frames = (frames_tensor[0].transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)

    buf = io.BytesIO()
    imageio.mimwrite(buf, frames, format="mp4", fps=24, codec="libx264")
    mp4_bytes = buf.getvalue()

    logger.info(
        "req %d | Done: %d frames, %dx%d, %.1f KB, enc=%.2fs den=%.2fs vae=%.2fs total=%.2fs",
        req_id, frames.shape[0], width, height, len(mp4_bytes) / 1024,
        enc_time, den_time, vae_time, total,
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
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/generate")
async def generate(req: GenerateRequest):
    try:
        mp4_bytes = await run_pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
        return Response(content=mp4_bytes, media_type="video/mp4")
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── Startup / Shutdown ───────────────────────────────────────────────────

def launch_all_stages():
    """Launch all 3 stage schedulers and return clients."""
    global enc_client, den_client, vae_client, _all_procs

    from partial_gpu_worker import build_encoder_stages, build_denoiser_stages, build_vae_stages

    _patch_hunyuan_config_task_type()

    logger.info("Launching stages: Encoder=%s  Denoiser=%s (TP=%d)  VAE=%s",
                GPU_ENC, GPU_DEN, TP_SIZE, GPU_VAE)

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

    logger.info("All stages ready. Serving at http://%s:%d", SERVE_HOST, SERVE_PORT)


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
    logger.info("All stages terminated.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    launch_all_stages()

    import uvicorn
    try:
        uvicorn.run(app, host=SERVE_HOST, port=SERVE_PORT, log_level="info")
    finally:
        shutdown_stages()
