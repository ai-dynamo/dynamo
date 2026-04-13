#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion XL Worker for Dynamo on Intel XPU

Registers a HuggingFace diffusers StableDiffusionXLPipeline as a Dynamo backend
endpoint compatible with the /v1/images/generations frontend endpoint. The
endpoint generates images from text prompts and returns them as base64-encoded
PNG data in the response.

Generation parameters (size, num_inference_steps, guidance_scale, etc.) are
taken from the request body, so the same worker instance can serve requests
with different settings without restarting.

One request at a time (asyncio.Lock — the diffusers pipeline is not re-entrant).

Usage:
  python worker_xpu.py [--model MODEL] [--dtype DTYPE]
                       [--enable-torch-compile]

Options:
  --model              HuggingFace model path
                       (default: stabilityai/stable-diffusion-xl-base-1.0)
  --dtype              Data type: bf16 or fp16 (default: bf16)
  --enable-torch-compile
                       Compile the UNet with torch.compile for faster inference

Request format (sent to /v1/images/generations):
  prompt:   text description of the desired image
  model:    HuggingFace model path (must match what the worker registered)
  size:     one of "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
  n:        number of images to generate (default: 1)
  nvext:
    num_inference_steps: denoising steps (default: 30)
    guidance_scale:      CFG scale (default: 7.5)
    seed:                RNG seed (optional)
    negative_prompt:     text to avoid (optional)
"""

import argparse
import asyncio
import base64
import io
import logging
import os
import time
import uuid

import torch
import uvloop
from diffusers import StableDiffusionXLPipeline

from dynamo.common.protocols.image_protocol import (
    ImageData,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.llm import ModelInput, ModelType, register_llm  # type: ignore[attr-defined]
from dynamo.runtime import DistributedRuntime, dynamo_endpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_worker_namespace() -> str:
    """Resolve Dynamo namespace for endpoint registration."""
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    suffix = os.environ.get("DYN_NAMESPACE_WORKER_SUFFIX")
    if suffix:
        namespace = f"{namespace}-{suffix}"
    return namespace


def _parse_size(size: str | None) -> tuple[int, int]:
    """Parse a 'WxH' size string into (width, height)."""
    if size is None:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    try:
        width_str, height_str = size.lower().split("x", 1)
        width, height = int(width_str), int(height_str)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid size format '{size}', expected 'WxH'") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size '{size}', width and height must be positive")
    return width, height


def _encode_image_to_b64(image) -> str:
    """Encode a PIL Image to base64 PNG string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ── Backend ──────────────────────────────────────────────────────────────────


class SDXLBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_name: str = args.model
        self.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
        self.enable_torch_compile: bool = args.enable_torch_compile
        self.device = torch.device("xpu")

        # One request at a time — pipeline is not re-entrant
        self._generate_lock = asyncio.Lock()
        self.pipe: StableDiffusionXLPipeline | None = None

    async def initialize_model(self) -> None:
        logger.info(
            "Loading SDXL pipeline model=%s dtype=%s device=xpu",
            self.model_name,
            self.dtype,
        )
        loop = asyncio.get_running_loop()

        def _load():
            if not torch.xpu.is_available():
                raise RuntimeError(
                    "Intel XPU is not available. Check that:\n"
                    "  1. Intel GPU drivers are installed (level-zero, i915)\n"
                    "  2. PyTorch was installed with XPU support "
                    "(pip install torch --index-url https://download.pytorch.org/whl/xpu)\n"
                    "  3. The XPU device is visible (run: python -c 'import torch; print(torch.xpu.device_count())')"
                )

            device_name = torch.xpu.get_device_name(0)
            device_count = torch.xpu.device_count()
            logger.info("XPU device: %s (total devices: %d)", device_name, device_count)

            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            )
            pipe = pipe.to(self.device)

            if self.enable_torch_compile:
                logger.info("Compiling UNet with torch.compile (inductor backend)...")
                pipe.unet = torch.compile(pipe.unet, backend="inductor")

            return pipe

        self.pipe = await loop.run_in_executor(None, _load)
        logger.info("SDXL pipeline ready")

    def _generate_images(
        self,
        request_id: str,
        prompt: str,
        width: int,
        height: int,
        num_images: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
        negative_prompt: str | None,
    ) -> list[str]:
        """Generate images and return as list of base64-encoded PNG strings."""
        assert self.pipe is not None

        kwargs: dict = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        if seed is not None:
            kwargs["generator"] = torch.Generator(device="xpu").manual_seed(seed)
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt

        result = self.pipe(**kwargs)
        images = result.images

        logger.info(
            "[%s] Generated %d image(s), encoding to PNG", request_id, len(images)
        )
        encoded = []
        for img in images:
            encoded.append(_encode_image_to_b64(img))
        return encoded

    # ── Dynamo endpoint ──────────────────────────────────────────────────────

    @dynamo_endpoint(NvCreateImageRequest, NvImagesResponse)
    async def generate_image(self, request: NvCreateImageRequest):
        """
        Non-streaming endpoint.

        Generates images using the SDXL pipeline from the request parameters,
        then yields a single NvImagesResponse with base64-encoded PNG data.
        """
        if self.pipe is None:
            raise RuntimeError("Pipeline is not initialized")

        width, height = _parse_size(request.size)
        num_images = max(request.n or 1, 1)

        nvext = request.nvext
        num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = DEFAULT_GUIDANCE_SCALE
        seed = None
        negative_prompt = None
        if nvext is not None:
            if nvext.num_inference_steps is not None:
                num_inference_steps = nvext.num_inference_steps
            if nvext.guidance_scale is not None:
                guidance_scale = nvext.guidance_scale
            seed = nvext.seed
            negative_prompt = nvext.negative_prompt

        request_id = f"img_{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        logger.info(
            "[%s] generate_image: prompt='%s...' size=%dx%d n=%d steps=%d cfg=%.1f",
            request_id,
            request.prompt[:60],
            width,
            height,
            num_images,
            num_inference_steps,
            guidance_scale,
        )
        logger.info(
            "[%s] Waiting for generate lock (locked=%s)",
            request_id,
            self._generate_lock.locked(),
        )

        async with self._generate_lock:
            t = time.perf_counter()
            try:
                b64_images = await asyncio.to_thread(
                    self._generate_images,
                    request_id=request_id,
                    prompt=request.prompt,
                    width=width,
                    height=height,
                    num_images=num_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    negative_prompt=negative_prompt,
                )
            except Exception as exc:
                logger.exception("[%s] Generation failed", request_id)
                raise RuntimeError(
                    f"Image generation failed for request {request_id}"
                ) from exc

            elapsed = time.perf_counter() - t
            logger.info(
                "[%s] Generation done in %.1fs — %d image(s)",
                request_id,
                elapsed,
                len(b64_images),
            )

            yield NvImagesResponse(
                created=created_ts,
                data=[ImageData(b64_json=b64) for b64 in b64_images],
            ).model_dump()

        logger.info("[%s] Request finished", request_id)


# ── Dynamo wiring ────────────────────────────────────────────────────────────


async def _register_model(endpoint, model_name: str) -> None:
    try:
        await register_llm(
            ModelInput.Text,  # type: ignore[attr-defined]
            ModelType.Images,
            endpoint,
            model_name,
            model_name,
        )
        logger.info("Successfully registered model: %s", model_name)
    except Exception as e:
        logger.error("Failed to register model: %s", e, exc_info=True)
        raise RuntimeError("Model registration failed") from e


async def backend_worker(runtime: DistributedRuntime, args: argparse.Namespace) -> None:
    namespace_name = _get_worker_namespace()
    component_name = "backend"
    endpoint_name = "generate"

    endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.{endpoint_name}")
    logger.info(
        "Serving endpoint %s/%s/%s", namespace_name, component_name, endpoint_name
    )

    backend = SDXLBackend(args)
    await backend.initialize_model()

    await asyncio.gather(
        endpoint.serve_endpoint(backend.generate_image),  # type: ignore[arg-type]
        _register_model(endpoint, backend.model_name),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stable Diffusion XL Worker for Dynamo on Intel XPU"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for inference (default: bf16)",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        dest="enable_torch_compile",
        help="Compile UNet with torch.compile (inductor backend) for faster inference",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", _get_worker_namespace())
    runtime = DistributedRuntime(loop, discovery_backend, "tcp", False)
    await backend_worker(runtime, args)


if __name__ == "__main__":
    _args = _parse_args()
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    uvloop.install()
    asyncio.run(main(_args))
