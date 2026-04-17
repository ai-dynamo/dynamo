#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate pre-computed vision embeddings in safetensors format using
TRT-LLM's standalone MultimodalEncoder.

Usage:
    python generate_safetensors_embeddings.py \
        --model-path Qwen/Qwen3-VL-2B-Instruct \
        --image-url "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png" \
        --output /tmp/test_embeddings.safetensors

The output file can then be used for inference via the image_url field:
    curl ... -d '{"image_url": {"url": "/tmp/test_embeddings.safetensors"}}'
"""

import argparse
import logging

import torch
from PIL import Image
from safetensors.torch import save_file as safetensors_save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(url: str) -> Image.Image:
    """Load image from URL or local path."""
    if url.startswith(("http://", "https://")):
        import httpx

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            from io import BytesIO

            return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(url).convert("RGB")


def generate_with_encoder(model_path: str, image: Image.Image) -> torch.Tensor:
    """Use TRT-LLM's MultimodalEncoder to generate embeddings."""
    from tensorrt_llm.llmapi import MultimodalEncoder

    encoder = MultimodalEncoder(model=model_path, max_batch_size=1)
    inputs = [
        {
            "prompt_token_ids": [],
            "multi_modal_data": {"image": [image]},
            "mm_processor_kwargs": {},
        }
    ]
    outputs = list(encoder.generate(inputs))
    if not outputs or outputs[0].disaggregated_params is None:
        raise RuntimeError("Encoder produced no output")

    dp = outputs[0].disaggregated_params
    if dp.multimodal_embedding_handles is not None:
        embeddings = dp.multimodal_embedding_handles[0]
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu()

    raise RuntimeError("Could not extract embeddings from encoder output")


def generate_random_embeddings(hidden_size: int = 1024, seq_len: int = 10) -> torch.Tensor:
    """Generate random embeddings for testing (no GPU required)."""
    logger.info(f"Generating random embeddings: shape=({seq_len}, {hidden_size})")
    return torch.randn(seq_len, hidden_size, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Generate safetensors embeddings")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    )
    parser.add_argument("--output", type=str, default="/tmp/test_embeddings.safetensors")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate random embeddings (no GPU/model required)",
    )
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    if args.random:
        embeddings = generate_random_embeddings(args.hidden_size, args.seq_len)
    else:
        logger.info(f"Loading image from {args.image_url}")
        image = load_image(args.image_url)
        logger.info(f"Image size: {image.size}")

        logger.info(f"Running encoder with model: {args.model_path}")
        embeddings = generate_with_encoder(args.model_path, image)

    logger.info(f"Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")

    safetensors_save_file({"embeddings": embeddings}, args.output)
    logger.info(f"Saved safetensors embeddings to: {args.output}")

    # Verify we can load it back
    from safetensors.torch import load_file

    loaded = load_file(args.output)
    assert "embeddings" in loaded
    assert loaded["embeddings"].shape == embeddings.shape
    logger.info("Verification passed: file loads correctly")


if __name__ == "__main__":
    main()
