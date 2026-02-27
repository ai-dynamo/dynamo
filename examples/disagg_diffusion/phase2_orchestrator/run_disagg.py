#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Phase 2: Disaggregated Diffusion Orchestrator

Connects to the three stage workers (Encoder, Denoiser, VAE) via Dynamo
RPC and chains them into an end-to-end image generation pipeline.

This plays the same role as the Frontend/Global Router in the EPD
architecture, but implemented as a lightweight client for the POC.

Usage:
    # Ensure the three workers are already running (see launch/run_all.sh)
    python run_disagg.py \\
        --prompt "A photo of a cat sitting on a windowsill" \\
        --output /tmp/disagg_output.png
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time

import uvloop

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1_workers"))

from protocol import (  # noqa: E402
    DenoiserRequest,
    EncoderRequest,
    VAEDecodeRequest,
)

from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

PROMPT = os.environ.get("PROMPT", "A photo of a cat sitting on a windowsill")
MODEL = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.1-schnell")
OUTPUT = os.environ.get("OUTPUT", "/tmp/disagg_output.png")
HEIGHT = int(os.environ.get("HEIGHT", "512"))
WIDTH = int(os.environ.get("WIDTH", "512"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "4"))
SEED = int(os.environ.get("SEED", "42"))


async def call_stage(client, request_json: str) -> dict:
    """Call a Dynamo endpoint, collect the streamed response.

    Each stage worker yields exactly one response dict.
    """
    result = None
    stream = await client.generate(request_json)
    async for chunk in stream:
        # Dynamo streams return objects with a .data() method or raw dicts
        data = chunk.data() if hasattr(chunk, "data") else chunk
        if isinstance(data, str):
            data = json.loads(data)
        result = data

    if result is None:
        raise RuntimeError("Empty response from stage")
    return result


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    """Orchestrator: chain Encoder → Denoiser → VAE."""

    # Create clients to the three stage endpoints
    encoder_client = await runtime.endpoint(
        "disagg_diffusion.encoder.generate"
    ).client()
    denoiser_client = await runtime.endpoint(
        "disagg_diffusion.denoiser.generate"
    ).client()
    vae_client = await runtime.endpoint(
        "disagg_diffusion.vae.generate"
    ).client()

    logger.info("Connected to all three stage endpoints")
    timings = {}

    # ── Stage 1: Encoder ─────────────────────────────────────────────
    logger.info("[1/3] Encoding prompt …")
    t0 = time.monotonic()

    encoder_req = EncoderRequest(prompt=PROMPT, model=MODEL)
    encoder_resp = await call_stage(encoder_client, encoder_req.model_dump_json())

    timings["encoder_s"] = time.monotonic() - t0
    logger.info("  Done in %.2fs — shapes: %s", timings["encoder_s"], encoder_resp.get("shapes"))

    # ── Stage 2: Denoiser ────────────────────────────────────────────
    logger.info("[2/3] Denoising %dx%d, %d steps …", WIDTH, HEIGHT, NUM_STEPS)
    t0 = time.monotonic()

    denoiser_req = DenoiserRequest(
        embeddings_b64=encoder_resp["embeddings_b64"],
        model=MODEL,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_STEPS,
        guidance_scale=0.0,
        seed=SEED,
    )
    denoiser_resp = await call_stage(denoiser_client, denoiser_req.model_dump_json())

    timings["denoiser_s"] = time.monotonic() - t0
    logger.info("  Done in %.2fs — latent shape: %s", timings["denoiser_s"], denoiser_resp.get("shape"))

    # ── Stage 3: VAE Decode ──────────────────────────────────────────
    logger.info("[3/3] VAE decoding …")
    t0 = time.monotonic()

    vae_req = VAEDecodeRequest(
        latents_b64=denoiser_resp["latents_b64"],
        model=MODEL,
    )
    vae_resp = await call_stage(vae_client, vae_req.model_dump_json())

    timings["vae_s"] = time.monotonic() - t0
    logger.info("  Done in %.2fs", timings["vae_s"])

    # ── Save output ──────────────────────────────────────────────────
    image_bytes = base64.b64decode(vae_resp["image_b64"])
    with open(OUTPUT, "wb") as f:
        f.write(image_bytes)

    timings["total_s"] = sum(timings.values())

    logger.info("")
    logger.info("=" * 50)
    logger.info("Pipeline complete!")
    logger.info("  Encoder:  %.2fs", timings["encoder_s"])
    logger.info("  Denoiser: %.2fs", timings["denoiser_s"])
    logger.info("  VAE:      %.2fs", timings["vae_s"])
    logger.info("  Total:    %.2fs", timings["total_s"])
    logger.info("  Output:   %s", OUTPUT)
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvloop.install()
    asyncio.run(worker())
