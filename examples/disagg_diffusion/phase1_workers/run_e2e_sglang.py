#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end disaggregated diffusion pipeline with SGLang backend.

Launches Encoder, Denoiser (TP=2), and VAE as separate SGLang scheduler
processes on different GPUs, then runs the full pipeline:
    Encoder -> Denoiser -> VAE

Measures per-stage timing.  Supports concurrent requests for benchmarking.

GPU assignment:
    Encoder :  GPU 0   (1 GPU)
    Denoiser:  GPU 1,2 (TP=2)
    VAE     :  GPU 3   (1 GPU)

Usage:
    # Single request (correctness check)
    python run_e2e_sglang.py

    # Benchmark: 4 requests, 2 concurrent
    NUM_REQUESTS=4 CONCURRENCY=2 python run_e2e_sglang.py

Environment variables:
    MODEL_PATH      Model to use (default: hunyuanvideo-community/HunyuanVideo)
    PROMPT          Text prompt (default: A cat walking on green grass)
    GPU_ENC         GPU(s) for encoder  (default: 0)
    GPU_DEN         GPU(s) for denoiser (default: 1,2)
    GPU_VAE         GPU(s) for VAE      (default: 3)
    TP_SIZE         Tensor parallelism for denoiser (default: auto from GPU_DEN)
    NUM_REQUESTS    Number of pipeline runs (default: 1)
    CONCURRENCY     Max concurrent pipelines (default: 1)
    NUM_FRAMES      Number of video frames (default: 61)
    NUM_STEPS       Denoising steps (default: 50)
    HEIGHT          Frame height (default: 544)
    WIDTH           Frame width  (default: 960)
    GUIDANCE        Guidance scale (default: 1.0; >1.0 enables CFG)
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import List

# --- ensure workers dir is on sys.path so subprocesses find sglang_utils ---
WORKERS_DIR = os.path.dirname(os.path.abspath(__file__))
if WORKERS_DIR not in sys.path:
    sys.path.insert(0, WORKERS_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("e2e")

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "hunyuanvideo-community/HunyuanVideo")
PROMPT = os.environ.get("PROMPT", "A cat walking on green grass")
GPU_ENC = os.environ.get("GPU_ENC", "0")
GPU_DEN = os.environ.get("GPU_DEN", "1,2")
GPU_VAE = os.environ.get("GPU_VAE", "3")
TP_SIZE = int(os.environ.get("TP_SIZE", str(len(GPU_DEN.split(",")))))
NUM_REQUESTS = int(os.environ.get("NUM_REQUESTS", "1"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "1"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "61"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "50"))
HEIGHT = int(os.environ.get("HEIGHT", "544"))
WIDTH = int(os.environ.get("WIDTH", "960"))
GUIDANCE = float(os.environ.get("GUIDANCE", "1.0"))
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/disagg_e2e")


# ── SGLang compatibility patches ─────────────────────────────────────────

def _patch_hunyuan_config_task_type():
    """HunyuanConfig inherits ``task_type`` from PipelineConfig without a
    default value, so ``HunyuanConfig()`` crashes.  Wrap __init__ to supply
    ``task_type=T2V`` when omitted.  Idempotent.
    """
    from sglang.multimodal_gen.configs.pipelines.base import ModelTaskType
    try:
        from sglang.multimodal_gen.configs.pipelines.hunyuan import (
            HunyuanConfig, FastHunyuanConfig,
        )
    except ImportError:
        return

    for cls in (HunyuanConfig, FastHunyuanConfig):
        if getattr(cls, "_task_type_patched", False):
            continue
        orig = cls.__init__

        def _patched(self, *a, task_type=ModelTaskType.T2V, _orig=orig, **kw):
            _orig(self, *a, task_type=task_type, **kw)

        cls.__init__ = _patched
        cls._task_type_patched = True


def _detect_encoder_modules(model_path: str) -> List[str]:
    """Return the required_modules list for the encoder stage."""
    try:
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model_index, verify_model_config_and_directory,
        )
        config = (verify_model_config_and_directory(model_path)
                  if os.path.exists(model_path)
                  else maybe_download_model_index(model_path))
        modules = ["text_encoder", "tokenizer"]
        if "text_encoder_2" in config:
            modules += ["text_encoder_2", "tokenizer_2"]
        modules.append("scheduler")
        return modules
    except Exception:
        pass
    # Fallback: include dual encoders for known models
    if "hunyuan" in model_path.lower():
        return ["text_encoder", "text_encoder_2",
                "tokenizer", "tokenizer_2", "scheduler"]
    return ["text_encoder", "tokenizer", "scheduler"]


# ── Non-singleton ZMQ client ────────────────────────────────────────────

class StageClient:
    """Async ZMQ REQ client that talks to a SGLang Scheduler."""

    def __init__(self, endpoint: str, name: str = ""):
        import zmq.asyncio
        self._name = name
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(endpoint)
        self._lock = asyncio.Lock()
        logger.info("StageClient(%s) connected to %s", name, endpoint)

    async def forward(self, reqs):
        """Send request(s) and receive response."""
        async with self._lock:
            await self._sock.send_pyobj(reqs)
            return await self._sock.recv_pyobj()

    def close(self):
        self._sock.close()
        self._ctx.term()


# ── Stage launchers ─────────────────────────────────────────────────────

def _launch_stage(
    stage_name: str,
    cuda_devices: str,
    required_modules: List[str],
    custom_stages_fn,
    tp_size: int = 1,
    scheduler_port: int = 15600,
):
    """Launch a partial scheduler for one pipeline stage.

    Returns (processes, server_args).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    num_gpus = len(cuda_devices.split(","))

    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs, set_global_server_args,
    )
    from partial_gpu_worker import launch_partial_server

    server_args = ServerArgs.from_kwargs(
        model_path=MODEL_PATH,
        num_gpus=num_gpus,
        tp_size=tp_size,
        scheduler_port=scheduler_port,
    )
    set_global_server_args(server_args)

    logger.info(
        "Launching %s: CUDA_VISIBLE_DEVICES=%s  num_gpus=%d  tp=%d  port=%d",
        stage_name, cuda_devices, num_gpus, tp_size, server_args.scheduler_port,
    )

    processes = launch_partial_server(
        server_args,
        required_modules=required_modules,
        custom_stages_fn=custom_stages_fn,
    )

    logger.info("%s ready (%d processes)", stage_name, len(processes))
    return processes, server_args


def terminate_processes(processes, name=""):
    for p in processes:
        p.terminate()
    for p in processes:
        p.join(timeout=10)
    logger.info("Terminated %s processes", name)


# ── Pipeline execution ──────────────────────────────────────────────────

async def run_single_pipeline(
    req_id: int,
    encoder_client: StageClient,
    denoiser_client: StageClient,
    vae_client: StageClient,
    seed: int,
    save_output: bool = False,
) -> dict:
    """Run one Encoder -> Denoiser -> VAE pipeline, return timing dict."""
    import torch
    from sglang_utils import build_req, inject_tensors_to_req

    timings = {"req_id": req_id}
    negative_prompt = "bad quality" if GUIDANCE > 1.0 else ""
    req_kwargs = dict(
        prompt=PROMPT, negative_prompt=negative_prompt,
        height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE, seed=seed,
    )
    t_pipeline = time.monotonic()

    # ── Encoder ──────────────────────────────────────────────────────
    t0 = time.monotonic()
    enc_output = await encoder_client.forward([build_req(**req_kwargs)])
    timings["encoder_s"] = time.monotonic() - t0
    if enc_output.error:
        raise RuntimeError(f"Encoder error: {enc_output.error}")
    enc_result = enc_output.output
    # enc_result is either {"_nixl_transfer_meta": {...}} (NIXL) or
    # {"prompt_embeds": tensor, ...} (ZMQ fallback)
    nixl_mode = "_nixl_transfer_meta" in enc_result
    logger.info(
        "req %d | Encoder done %.2fs — transfer: %s",
        req_id, timings["encoder_s"],
        "NIXL RDMA" if nixl_mode else f"ZMQ (keys: {list(enc_result.keys())})",
    )

    # ── Denoiser ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    den_req = build_req(**req_kwargs)
    den_req.do_classifier_free_guidance = (GUIDANCE > 1.0)
    if nixl_mode:
        # Pass NIXL metadata — NixlReceiveStage will RDMA-pull the tensors
        den_req._nixl_transfer_meta = enc_result["_nixl_transfer_meta"]
    else:
        # ZMQ fallback — tensors already in enc_result
        inject_tensors_to_req(den_req, enc_result)
    den_output = await denoiser_client.forward([den_req])
    timings["denoiser_s"] = time.monotonic() - t0
    if den_output.error:
        raise RuntimeError(f"Denoiser error: {den_output.error}")
    den_result = den_output.output
    nixl_mode_den = "_nixl_transfer_meta" in den_result
    logger.info(
        "req %d | Denoiser done %.2fs — transfer: %s",
        req_id, timings["denoiser_s"],
        "NIXL RDMA" if nixl_mode_den else "ZMQ",
    )

    # ── VAE ──────────────────────────────────────────────────────────
    t0 = time.monotonic()
    vae_req = build_req(prompt="", height=HEIGHT, width=WIDTH,
                        num_frames=NUM_FRAMES, num_inference_steps=NUM_STEPS,
                        guidance_scale=0.0, seed=seed)
    if nixl_mode_den:
        vae_req._nixl_transfer_meta = den_result["_nixl_transfer_meta"]
    else:
        vae_req.latents = den_result["latents"].cpu()
    vae_output = await vae_client.forward([vae_req])
    timings["vae_s"] = time.monotonic() - t0
    if vae_output.error:
        raise RuntimeError(f"VAE error: {vae_output.error}")

    timings["total_s"] = time.monotonic() - t_pipeline
    logger.info(
        "req %d | VAE done %.2fs — total pipeline: %.2fs",
        req_id, timings["vae_s"], timings["total_s"],
    )

    # Save output as mp4
    if save_output and vae_output.output is not None:
        _save_video(vae_output.output, req_id)

    return timings


def _save_video(frames_tensor, req_id: int):
    """Save decoded video tensor [B,C,T,H,W] as mp4."""
    try:
        import numpy as np
        import imageio

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if hasattr(frames_tensor, "cpu"):
            frames_tensor = frames_tensor.cpu().float().numpy()
        # [B, C, T, H, W] -> [T, H, W, C]
        frames = (frames_tensor[0].transpose(1, 2, 3, 0) * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(OUTPUT_DIR, f"output_{req_id}.mp4")
        imageio.mimwrite(out_path, frames, fps=24, codec="libx264")
        logger.info("req %d | Saved %d frames to %s", req_id, frames.shape[0], out_path)
    except Exception as e:
        logger.warning("req %d | Could not save video: %s", req_id, e)


def print_timing_report(all_timings: list, wall_elapsed: float):
    """Print per-stage timing statistics."""
    stages = ["encoder_s", "denoiser_s", "vae_s", "total_s"]
    n = len(all_timings)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  Timing Report  (%d requests, concurrency=%d)", n, CONCURRENCY)
    logger.info("=" * 72)

    for t in all_timings:
        logger.info(
            "  req %2d | enc=%6.2fs  den=%6.2fs  vae=%6.2fs  total=%6.2fs",
            t["req_id"], t["encoder_s"], t["denoiser_s"], t["vae_s"], t["total_s"],
        )

    if n > 1:
        logger.info("-" * 72)
        for stage in stages:
            vals = [t[stage] for t in all_timings]
            mean = statistics.mean(vals)
            med = statistics.median(vals)
            mn, mx = min(vals), max(vals)
            std = statistics.stdev(vals) if n >= 2 else 0.0
            logger.info(
                "  %-10s mean=%6.2fs  median=%6.2fs  min=%6.2fs  max=%6.2fs  std=%5.2fs",
                stage, mean, med, mn, mx, std,
            )

    logger.info("-" * 72)
    throughput = n / wall_elapsed if wall_elapsed > 0 else 0
    logger.info("  Wall time: %.2fs | Throughput: %.2f req/s", wall_elapsed, throughput)
    logger.info("=" * 72)


# ── Main ────────────────────────────────────────────────────────────────

async def main():
    from partial_gpu_worker import build_encoder_stages, build_denoiser_stages, build_vae_stages

    # Apply patches once before any SGLang config is created
    _patch_hunyuan_config_task_type()

    logger.info("=" * 72)
    logger.info("  Disaggregated Diffusion E2E — SGLang Backend")
    logger.info("  Model:    %s", MODEL_PATH)
    logger.info("  Prompt:   %s", PROMPT)
    logger.info("  Encoder:  GPU %s", GPU_ENC)
    logger.info("  Denoiser: GPU %s (TP=%d)", GPU_DEN, TP_SIZE)
    logger.info("  VAE:      GPU %s", GPU_VAE)
    logger.info("  Requests: %d  Concurrency: %d", NUM_REQUESTS, CONCURRENCY)
    logger.info("  Frames: %d  Steps: %d  Size: %dx%d  Guidance: %.1f",
                NUM_FRAMES, NUM_STEPS, WIDTH, HEIGHT, GUIDANCE)
    logger.info("=" * 72)

    enc_procs = den_procs = vae_procs = None
    enc_client = den_client = vae_client = None

    try:
        # ── Launch all 3 stages ──────────────────────────────────────
        t_launch = time.monotonic()

        enc_procs, enc_args = _launch_stage(
            "Encoder", GPU_ENC,
            required_modules=_detect_encoder_modules(MODEL_PATH),
            custom_stages_fn=build_encoder_stages,
            tp_size=1,
            scheduler_port=15600,
        )

        den_procs, den_args = _launch_stage(
            "Denoiser", GPU_DEN,
            required_modules=["transformer", "scheduler"],
            custom_stages_fn=build_denoiser_stages,
            tp_size=TP_SIZE,
            scheduler_port=15700,
        )

        vae_procs, vae_args = _launch_stage(
            "VAE", GPU_VAE,
            required_modules=["vae", "scheduler"],
            custom_stages_fn=build_vae_stages,
            tp_size=1,
            scheduler_port=15800,
        )

        logger.info("All stages launched in %.1fs", time.monotonic() - t_launch)

        # ── Connect clients ──────────────────────────────────────────
        enc_client = StageClient(enc_args.scheduler_endpoint(), "encoder")
        den_client = StageClient(den_args.scheduler_endpoint(), "denoiser")
        vae_client = StageClient(vae_args.scheduler_endpoint(), "vae")

        # ── Warmup ───────────────────────────────────────────────────
        logger.info("Warmup request …")
        warmup = await run_single_pipeline(
            -1, enc_client, den_client, vae_client, SEED, save_output=False,
        )
        logger.info(
            "Warmup done — enc=%.2fs den=%.2fs vae=%.2fs total=%.2fs",
            warmup["encoder_s"], warmup["denoiser_s"],
            warmup["vae_s"], warmup["total_s"],
        )

        # ── Run pipeline(s) ──────────────────────────────────────────
        if NUM_REQUESTS <= 1:
            t_wall = time.monotonic()
            timings = await run_single_pipeline(
                0, enc_client, den_client, vae_client, SEED, save_output=True,
            )
            wall_elapsed = time.monotonic() - t_wall
            print_timing_report([timings], wall_elapsed)
        else:
            logger.info("Firing %d requests (concurrency=%d) …", NUM_REQUESTS, CONCURRENCY)
            sem = asyncio.Semaphore(CONCURRENCY)

            async def _run_one(i):
                async with sem:
                    return await run_single_pipeline(
                        i, enc_client, den_client, vae_client,
                        SEED + i, save_output=(i == 0),
                    )

            t_wall = time.monotonic()
            tasks = [asyncio.create_task(_run_one(i)) for i in range(NUM_REQUESTS)]
            all_timings = list(await asyncio.gather(*tasks))
            wall_elapsed = time.monotonic() - t_wall
            print_timing_report(all_timings, wall_elapsed)

    finally:
        for client in [enc_client, den_client, vae_client]:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        for procs, name in [
            (enc_procs, "encoder"), (den_procs, "denoiser"), (vae_procs, "vae"),
        ]:
            if procs is not None:
                terminate_processes(procs, name)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())
