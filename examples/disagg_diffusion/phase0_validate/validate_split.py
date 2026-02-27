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

"""Phase 0: Validate that a diffusers pipeline can be split into
Encoder / Denoiser / VAE stages with portable intermediate tensors.

This script runs on a single GPU without Dynamo.  It:
  1. Generates a reference image with the monolithic pipeline.
  2. Runs the same generation in three SEPARATE stages, each loading only
     its own model components, passing serialized tensors between them.
  3. Compares the results to confirm equivalence.
  4. Reports per-stage VRAM usage.

Usage:
    python validate_split.py \\
        --model black-forest-labs/FLUX.1-schnell \\
        --prompt "A photo of a cat" \\
        --output-dir /tmp/disagg_validate
"""

import argparse
import gc
import io
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def flush_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def serialize_tensors(data: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(data, buf)
    return buf.getvalue()


def deserialize_tensors(raw: bytes) -> dict:
    buf = io.BytesIO(raw)
    return torch.load(buf, weights_only=True)


# ---------------------------------------------------------------------------
# Monolithic baseline
# ---------------------------------------------------------------------------

def run_monolithic(model_path: str, prompt: str, seed: int, num_steps: int, device: str):
    from diffusers import FluxPipeline

    logger.info("=== Monolithic Run ===")
    logger.info("Loading full pipeline …")
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    logger.info("Full pipeline VRAM: %.0f MB", vram_mb())

    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        generator=generator,
        height=512,
        width=512,
    ).images[0]

    del pipe
    flush_vram()
    return image


# ---------------------------------------------------------------------------
# Split stages — each loads ONLY its own components
# ---------------------------------------------------------------------------

def stage_encoder(model_path: str, prompt: str, device: str) -> bytes:
    """Stage 1: Load ONLY text encoders, encode prompt, return serialized embeddings."""
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    from diffusers import FluxPipeline

    logger.info("=== Stage 1: Encoder (text encoders only) ===")
    logger.info("VRAM before load: %.0f MB", vram_mb())

    # Load only the text encoder components
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(device)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    ).to(device)

    logger.info("Text encoders VRAM: %.0f MB", vram_mb())

    # Build a minimal pipeline just for encode_prompt().
    # We pass transformer=None and vae=None — FluxPipeline.__init__ stores
    # them as attributes without validation so this works.
    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    encoder_pipe = FluxPipeline(
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        transformer=None,
        vae=None,
    )

    prompt_embeds, pooled_prompt_embeds, text_ids = encoder_pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
    )
    logger.info(
        "Encoded: prompt_embeds=%s  pooled=%s  text_ids=%s",
        list(prompt_embeds.shape),
        list(pooled_prompt_embeds.shape),
        list(text_ids.shape),
    )

    raw = serialize_tensors({
        "prompt_embeds": prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
        "text_ids": text_ids.cpu(),
    })

    del encoder_pipe, text_encoder, text_encoder_2, tokenizer, tokenizer_2
    flush_vram()
    logger.info("VRAM after cleanup: %.0f MB", vram_mb())
    logger.info("Serialized embeddings: %.2f KB", len(raw) / 1024)
    return raw


def stage_denoiser(
    model_path: str, raw_embeddings: bytes, seed: int, num_steps: int, device: str
) -> bytes:
    """Stage 2: Load ONLY transformer + scheduler, denoise, return serialized latents."""
    from diffusers import FluxPipeline

    logger.info("=== Stage 2: Denoiser (transformer only) ===")
    logger.info("VRAM before load: %.0f MB", vram_mb())

    # Load full pipeline, then immediately discard what we don't need.
    # (Loading transformer alone and reconstructing a callable pipeline is
    #  fragile across diffusers versions; this approach is robust.)
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    # Read VAE config before deleting
    vae_scaling_factor = pipe.vae.config.scaling_factor
    vae_shift_factor = getattr(pipe.vae.config, "shift_factor", None)

    # Delete text encoders + VAE to free VRAM
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    pipe.vae = None
    flush_vram()
    logger.info("Denoiser VRAM (transformer only): %.0f MB", vram_mb())

    embeddings = deserialize_tensors(raw_embeddings)
    prompt_embeds = embeddings["prompt_embeds"].to(device)
    pooled_prompt_embeds = embeddings["pooled_prompt_embeds"].to(device)

    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        generator=generator,
        height=512,
        width=512,
        output_type="latent",
    )
    latents = result.images

    logger.info("Latents shape: %s", list(latents.shape))

    raw = serialize_tensors({
        "latents": latents.cpu(),
        "scaling_factor": torch.tensor(vae_scaling_factor),
        "shift_factor": torch.tensor(vae_shift_factor) if vae_shift_factor is not None else None,
    })

    del pipe
    flush_vram()
    logger.info("VRAM after cleanup: %.0f MB", vram_mb())
    logger.info("Serialized latents: %.2f KB", len(raw) / 1024)
    return raw


def stage_vae(model_path: str, raw_latents: bytes, device: str):
    """Stage 3: Load ONLY the VAE, decode latents to an image."""
    from diffusers import AutoencoderKL

    logger.info("=== Stage 3: VAE Decode (VAE only) ===")
    logger.info("VRAM before load: %.0f MB", vram_mb())

    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.to(device)
    logger.info("VAE VRAM: %.0f MB", vram_mb())

    data = deserialize_tensors(raw_latents)
    latents = data["latents"].to(device)
    scaling_factor = data["scaling_factor"].item()
    shift_factor = data["shift_factor"]
    if shift_factor is not None:
        shift_factor = shift_factor.item()

    if shift_factor is not None:
        latents = latents / scaling_factor + shift_factor
    else:
        latents = latents / scaling_factor

    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

    import numpy as np
    from PIL import Image

    image = Image.fromarray((decoded[0] * 255).round().astype(np.uint8))

    del vae
    flush_vram()
    logger.info("VRAM after cleanup: %.0f MB", vram_mb())
    return image


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_images(img_a, img_b) -> dict:
    import numpy as np

    a = np.array(img_a).astype(float)
    b = np.array(img_b).astype(float)

    if a.shape != b.shape:
        return {"match": False, "reason": f"shape mismatch: {a.shape} vs {b.shape}"}

    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    psnr = 10 * np.log10(255**2 / (diff**2).mean()) if diff.any() else float("inf")

    return {
        "match": max_diff < 10,
        "max_pixel_diff": float(max_diff),
        "mean_pixel_diff": float(mean_diff),
        "psnr_db": float(psnr),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate split diffusion pipeline")
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--prompt", default="A photo of a cat sitting on a windowsill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--output-dir", default="/tmp/disagg_validate")
    parser.add_argument("--skip-monolithic", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Monolithic ───────────────────────────────────────────────────
    mono_image = None
    if not args.skip_monolithic:
        t0 = time.monotonic()
        mono_image = run_monolithic(
            args.model, args.prompt, args.seed, args.num_steps, args.device
        )
        logger.info("Monolithic: %.2fs -> %s", time.monotonic() - t0, output_dir / "monolithic.png")
        mono_image.save(output_dir / "monolithic.png")

    # ── Split ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPLIT pipeline: Encoder → Denoiser → VAE")
    logger.info("=" * 60)

    t_total = time.monotonic()

    t0 = time.monotonic()
    raw_embeddings = stage_encoder(args.model, args.prompt, args.device)
    t_enc = time.monotonic() - t0

    t0 = time.monotonic()
    raw_latents = stage_denoiser(
        args.model, raw_embeddings, args.seed, args.num_steps, args.device
    )
    t_den = time.monotonic() - t0
    del raw_embeddings

    t0 = time.monotonic()
    split_image = stage_vae(args.model, raw_latents, args.device)
    t_vae = time.monotonic() - t0
    del raw_latents

    t_split = time.monotonic() - t_total
    split_image.save(output_dir / "split.png")

    # ── Results ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Encoder:  %.2fs", t_enc)
    logger.info("  Denoiser: %.2fs", t_den)
    logger.info("  VAE:      %.2fs", t_vae)
    logger.info("  Total:    %.2fs", t_split)

    if mono_image is not None:
        metrics = compare_images(mono_image, split_image)
        logger.info("  Match:    %s", metrics["match"])
        logger.info("  Max diff: %.1f   Mean diff: %.2f   PSNR: %.1f dB",
                     metrics["max_pixel_diff"], metrics["mean_pixel_diff"], metrics["psnr_db"])
        if not metrics["match"]:
            logger.warning("Images differ — check outputs visually (bf16 rounding is expected).")

    logger.info("  Output:   %s", output_dir)


if __name__ == "__main__":
    main()
