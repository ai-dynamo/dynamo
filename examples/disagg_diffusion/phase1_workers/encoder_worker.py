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

"""Disaggregated Diffusion — Encoder Worker

Loads only the text encoders (CLIP + T5) and serves an endpoint that
converts text prompts into serialized embeddings.

Usage:
    python encoder_worker.py --model black-forest-labs/FLUX.1-schnell
"""

import asyncio
import logging
import os
import sys

import torch
import uvloop

# Allow importing protocol.py from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import EncoderRequest, EncoderResponse, tensors_to_b64  # noqa: E402

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.1-schnell")
DEVICE = os.environ.get("DEVICE", "cuda")


class EncoderStage:
    """Text encoder stage: CLIP + T5 → embeddings."""

    def __init__(self):
        self.pipe = None

    def load_model(self):
        from diffusers import FluxPipeline

        logger.info("Loading text encoders from %s …", MODEL_PATH)
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16
        )
        self.pipe.to(DEVICE)

        # Free transformer + VAE — we only need text encoders
        self.pipe.transformer = None
        self.pipe.vae = None
        torch.cuda.empty_cache()

        vram = torch.cuda.memory_allocated() / 1e6
        logger.info("Encoder ready — VRAM: %.0f MB (text encoders only)", vram)

    @dynamo_endpoint(EncoderRequest, EncoderResponse)
    async def generate(self, request: EncoderRequest):
        logger.info("Encoding prompt: %.80s…", request.prompt)

        loop = asyncio.get_event_loop()
        prompt_embeds, pooled_prompt_embeds, text_ids = await loop.run_in_executor(
            None,
            lambda: self.pipe.encode_prompt(prompt=request.prompt, prompt_2=None),
        )

        embeddings = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

        response = EncoderResponse(
            embeddings_b64=tensors_to_b64(embeddings),
            shapes={k: list(v.shape) for k, v in embeddings.items()},
        )
        logger.info("Encoded — prompt_embeds %s", list(prompt_embeds.shape))
        yield response.model_dump()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("disagg_diffusion.encoder.generate")

    stage = EncoderStage()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, stage.load_model)

    logger.info("Serving encoder endpoint: disagg_diffusion.encoder.generate")
    await endpoint.serve_endpoint(stage.generate)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    uvloop.install()
    asyncio.run(worker())
