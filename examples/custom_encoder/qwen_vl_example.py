# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example FullPromptEncoder using Qwen2.5-VL as the vision backbone.

Demonstrates how to implement FullPromptEncoder for a VLM encoder
paired with a text-only LLM as the PD worker.  The encoder loads
everything it needs — ViT, embed_tokens of the target LM, projector —
in load(), then encode() produces a ready-to-transfer tensor.

Architecture:
    Encoder: Qwen2.5-VL ViT → image embeddings (n_tokens, vit_hidden_dim)
    Projector: random matrix → (n_tokens, lm_hidden_dim)   [POC only]
    Text: LM embed_tokens lookup → (n_text, lm_hidden_dim)
    Splice: [<image_embeds> | <text_embeds>]
    PD: receives full (seq_len, lm_hidden_dim) as EmbedsPrompt

NOTE: Output is semantically incorrect (different model families, random
projection, image prepended rather than spliced at placeholder).  This
is a plumbing demonstration, not a production encoder.

Usage (see enc_full_prompt_pd.sh for the complete launch script):

    python -m dynamo.vllm \\
        --multimodal-encode-worker \\
        --model Qwen/Qwen2.5-VL-3B-Instruct \\
        --full-prompt-encoder-class qwen_vl_example.QwenVLExampleEncoder \\
        --served-model-name Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import torch
from transformers import AutoImageProcessor, AutoModel

from dynamo.vllm.multimodal_utils.custom_encoder import FullPromptEncoder
from dynamo.vllm.multimodal_utils.image_loader import ImageLoader

logger = logging.getLogger(__name__)

# The PD model whose embed_tokens we load in load().
# Passed as --served-model-name on the encoder worker.
_PD_MODEL_ENV = "DYN_PD_MODEL"


class QwenVLExampleEncoder(FullPromptEncoder):
    """Wraps Qwen2.5-VL's ViT as a FullPromptEncoder.

    Loads only the vision tower from the VLM checkpoint.
    Also loads the PD LM's embed_tokens from the path in
    DYN_PD_MODEL (or defaults to Qwen/Qwen2.5-1.5B) for
    text embedding lookup.
    """

    def __init__(self) -> None:
        self.vision_encoder: Optional[torch.nn.Module] = None
        self.image_processor = None
        self._image_loader = ImageLoader(cache_size=8)
        self._embed_tokens: Optional[torch.nn.Embedding] = None
        self._device = "cpu"
        self._proj_cache: dict = {}

    def load(self, checkpoint_path: str, device: str) -> None:
        """Load the Qwen2.5-VL vision tower and the PD model's embed_tokens."""
        import os

        self._device = device

        # ── Vision tower (ViT only) ───────────────────────────────────────────
        logger.info(
            "[QwenVLExampleEncoder] loading vision tower from %s", checkpoint_path
        )
        hf = AutoModel.from_pretrained(
            checkpoint_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        # Qwen2/3-VL exposes the vision transformer at .visual
        self.vision_encoder = hf.visual.eval()
        del hf
        self.image_processor = AutoImageProcessor.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        logger.info("[QwenVLExampleEncoder] vision tower loaded")

        # ── PD model embed_tokens ─────────────────────────────────────────────
        # Load only the embedding layer from the PD model checkpoint using
        # safetensors lazy reads (only the shard containing embed_tokens.weight
        # is read from disk).
        import json
        from pathlib import Path

        from safetensors import safe_open
        from transformers.utils import cached_file

        pd_model = os.environ.get(_PD_MODEL_ENV, "Qwen/Qwen2.5-1.5B")
        logger.info("[QwenVLExampleEncoder] loading embed_tokens from %s", pd_model)

        try:
            index_path = cached_file(pd_model, "model.safetensors.index.json")
            model_dir = Path(index_path).parent
            weight_map: dict[str, str] = json.loads(Path(index_path).read_text())[
                "weight_map"
            ]
            embed_key = next(
                (k for k in weight_map if k.endswith("embed_tokens.weight")), None
            )
            if embed_key is None:
                raise FileNotFoundError(
                    f"embed_tokens.weight not in index for {pd_model}"
                )
            shard_path = model_dir / weight_map[embed_key]
        except (OSError, StopIteration):
            shard_path = Path(cached_file(pd_model, "model.safetensors"))
            embed_key = None

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if embed_key is None:
                embed_key = next(
                    (k for k in f.keys() if k.endswith("embed_tokens.weight")), None
                )
            if embed_key is None:
                raise FileNotFoundError(
                    f"embed_tokens.weight not found in {shard_path}"
                )
            weight = f.get_tensor(embed_key).to(dtype=torch.float16)

        self._embed_tokens = (
            torch.nn.Embedding(weight.shape[0], weight.shape[1], _weight=weight)
            .eval()
            .to(device)
        )
        logger.info(
            "[QwenVLExampleEncoder] embed_tokens loaded: vocab=%d hidden=%d",
            weight.shape[0],
            weight.shape[1],
        )

    def encode(
        self,
        image_urls: List[str],
        lm_token_ids: List[int],
    ) -> torch.Tensor:
        """Encode images and splice with text embeddings.

        Steps:
            1. Download and preprocess images.
            2. Run Qwen ViT → image embeddings (n_img_tokens, vit_hidden).
            3. Project vit_hidden → lm_hidden if needed (random matrix, POC only).
            4. Look up text embeddings from the LM's embed_tokens.
            5. Prepend image embeddings before text: [image | text].

        Returns:
            Tensor of shape ``(seq_len, lm_hidden_dim)``.
        """
        assert (
            self.vision_encoder is not None
            and self.image_processor is not None
            and self._embed_tokens is not None
        ), "call load() before encode()"

        # 1. Download images (synchronous — called via asyncio.to_thread)
        loop = asyncio.new_event_loop()
        try:
            images = loop.run_until_complete(
                asyncio.gather(
                    *[self._image_loader.load_image(url) for url in image_urls]
                )
            )
        finally:
            loop.close()

        # 2. Preprocess and run ViT
        inputs = self.image_processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device, dtype=torch.float16)
        grid_thw = inputs.get("image_grid_thw")

        with torch.no_grad():
            if grid_thw is not None:
                image_embeds = self.vision_encoder(
                    pixel_values, grid_thw=grid_thw.tolist()
                )
            else:
                image_embeds = self.vision_encoder(pixel_values)

        # Flatten any batch dim: → (n_img_tokens, vit_hidden)
        image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])

        # 3. Project vit_hidden → lm_hidden if dimensions differ (random POC matrix)
        lm_hidden = self._embed_tokens.weight.shape[1]
        vit_hidden = image_embeds.shape[-1]
        if vit_hidden != lm_hidden:
            key = (vit_hidden, lm_hidden, image_embeds.dtype, image_embeds.device)
            if key not in self._proj_cache:
                rng = torch.Generator(device=image_embeds.device)
                rng.manual_seed(42)
                proj = torch.randn(
                    vit_hidden,
                    lm_hidden,
                    dtype=image_embeds.dtype,
                    device=image_embeds.device,
                    generator=rng,
                ) / (vit_hidden**0.5)
                self._proj_cache[key] = proj
                logger.info(
                    "[QwenVLExampleEncoder] random projection %d→%d "
                    "(POC only — replace with learned projector for production)",
                    vit_hidden,
                    lm_hidden,
                )
            image_embeds = image_embeds @ self._proj_cache[key]

        # 4. Look up text embeddings
        ids_tensor = torch.tensor(
            lm_token_ids, dtype=torch.long, device=self._embed_tokens.weight.device
        )
        with torch.no_grad():
            text_embeds = self._embed_tokens(ids_tensor)  # (n_text, lm_hidden)

        # 5. Splice: [image | text]
        # The LM tokenizer has no image placeholder, so we prepend the image.
        # A production implementation would locate the placeholder position instead.
        image_embeds = image_embeds.to(
            dtype=text_embeds.dtype, device=text_embeds.device
        )
        spliced = torch.cat([image_embeds, text_embeds], dim=0)

        logger.info(
            "[QwenVLExampleEncoder] image=%d text=%d total=%d lm_hidden=%d",
            image_embeds.shape[0],
            text_embeds.shape[0],
            spliced.shape[0],
            lm_hidden,
        )
        return spliced  # (seq_len, lm_hidden_dim)
