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

"""Disaggregated Diffusion — VAE Worker

Loads only the VAE decoder.  Accepts denoised latents and returns
the final image as base64-encoded PNG.

Usage:
    python vae_worker.py --model black-forest-labs/FLUX.1-schnell
"""

import asyncio
import base64
import io
import logging
import os
import sys

import numpy as np
import torch
import uvloop
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import VAEDecodeRequest, VAEDecodeResponse, b64_to_tensors  # noqa: E402

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.1-schnell")
DEVICE = os.environ.get("DEVICE", "cuda")


class VAEStage:
    """VAE decode stage: latents → image pixels."""

    def __init__(self):
        self.vae = None

    def load_model(self):
        from diffusers import AutoencoderKL

        logger.info("Loading VAE from %s …", MODEL_PATH)
        self.vae = AutoencoderKL.from_pretrained(
            MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.vae.to(DEVICE)

        vram = torch.cuda.memory_allocated() / 1e6
        logger.info("VAE ready — VRAM: %.0f MB", vram)

    @dynamo_endpoint(VAEDecodeRequest, VAEDecodeResponse)
    async def generate(self, request: VAEDecodeRequest):
        logger.info("Decoding latents …")

        data = b64_to_tensors(request.latents_b64, DEVICE)
        latents = data["latents"]
        scaling_factor = data["scaling_factor"].item()
        shift_factor = data.get("shift_factor")
        if shift_factor is not None:
            shift_factor = shift_factor.item()

        # Undo pipeline's latent scaling
        if shift_factor is not None:
            latents = latents / scaling_factor + shift_factor
        else:
            latents = latents / scaling_factor

        loop = asyncio.get_event_loop()

        def _decode():
            with torch.no_grad():
                decoded = self.vae.decode(latents, return_dict=False)[0]
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            return decoded.cpu().permute(0, 2, 3, 1).float().numpy()

        pixels = await loop.run_in_executor(None, _decode)

        img = Image.fromarray((pixels[0] * 255).round().astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        response = VAEDecodeResponse(image_b64=image_b64)
        logger.info("Decoded — image %dx%d", img.width, img.height)
        yield response.model_dump()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("disagg_diffusion.vae.generate")

    stage = VAEStage()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, stage.load_model)

    logger.info("Serving VAE endpoint: disagg_diffusion.vae.generate")
    await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvloop.install()
    asyncio.run(worker())
