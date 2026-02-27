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

"""Disaggregated Diffusion — Denoiser Worker

Loads only the Transformer (DiT) and scheduler.  Accepts pre-computed
embeddings and returns denoised latents (skips VAE decode).

Usage:
    python denoiser_worker.py --model black-forest-labs/FLUX.1-schnell
"""

import asyncio
import logging
import os
import sys

import torch
import uvloop

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import (  # noqa: E402
    DenoiserRequest,
    DenoiserResponse,
    b64_to_tensors,
    tensors_to_b64,
)

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.1-schnell")
DEVICE = os.environ.get("DEVICE", "cuda")


class DenoiserStage:
    """Denoiser stage: embeddings → latents (no text encoder, no VAE)."""

    def __init__(self):
        self.pipe = None
        self.vae_scaling_factor = 1.0
        self.vae_shift_factor = None

    def load_model(self):
        from diffusers import FluxPipeline

        logger.info("Loading transformer from %s …", MODEL_PATH)
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        self.pipe.to(DEVICE)

        # Capture VAE config before discarding
        self.vae_scaling_factor = self.pipe.vae.config.scaling_factor
        self.vae_shift_factor = getattr(self.pipe.vae.config, "shift_factor", None)

        # Free text encoders + VAE
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        self.pipe.tokenizer = None
        self.pipe.tokenizer_2 = None
        self.pipe.vae = None
        torch.cuda.empty_cache()

        vram = torch.cuda.memory_allocated() / 1e6
        logger.info("Denoiser ready — VRAM: %.0f MB (transformer only)", vram)

    @dynamo_endpoint(DenoiserRequest, DenoiserResponse)
    async def generate(self, request: DenoiserRequest):
        logger.info(
            "Denoising %dx%d, %d steps, seed=%d",
            request.width, request.height,
            request.num_inference_steps, request.seed,
        )

        embeddings = b64_to_tensors(request.embeddings_b64, DEVICE)
        generator = torch.Generator(device=DEVICE).manual_seed(request.seed)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.pipe(
                prompt_embeds=embeddings["prompt_embeds"],
                pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
                height=request.height,
                width=request.width,
                output_type="latent",
            ),
        )
        latents = result.images

        latent_payload = {
            "latents": latents,
            "scaling_factor": torch.tensor(self.vae_scaling_factor),
        }
        if self.vae_shift_factor is not None:
            latent_payload["shift_factor"] = torch.tensor(self.vae_shift_factor)

        response = DenoiserResponse(
            latents_b64=tensors_to_b64(latent_payload),
            shape=list(latents.shape),
        )
        logger.info("Denoised — latents %s", list(latents.shape))
        yield response.model_dump()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("disagg_diffusion.denoiser.generate")

    stage = DenoiserStage()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, stage.load_model)

    logger.info("Serving denoiser endpoint: disagg_diffusion.denoiser.generate")
    await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvloop.install()
    asyncio.run(worker())
