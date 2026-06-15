# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example FullPromptEncoder using Qwen2.5-VL as the vision backbone.

This demonstrates how to implement FullPromptEncoder for a VLM encoder
paired with a text-only LLM as the PD worker.

Architecture:
    Encoder: Qwen2.5-VL ViT → image embeddings (n_tokens, vit_hidden_dim)
    Projector: optional linear layer → (n_tokens, lm_hidden_dim)
    Text: LM embed_tokens lookup → (n_text, lm_hidden_dim)
    Splice: [<prefix_text> | <image_embeds> | <suffix_text>]
    PD: receives full (seq_len, lm_hidden_dim) as EmbedsPrompt

NOTE: This example prepends image embeddings before text when the LM
tokenizer has no image placeholder token.  Output is semantically
incorrect (different model families, no learned cross-modal alignment),
but the system does not crash — useful for verifying the plumbing.

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


class QwenVLExampleEncoder(FullPromptEncoder):
    """Wraps Qwen2.5-VL's ViT as a FullPromptEncoder.

    Loads only the vision tower from the VLM checkpoint; the LM head and
    language model layers are discarded, keeping GPU memory usage low.
    """

    def __init__(self) -> None:
        self.vision_encoder: Optional[torch.nn.Module] = None
        self.image_processor = None
        self._image_loader = ImageLoader(cache_size=8)
        self._proj: Optional[torch.nn.Linear] = None
        self._device = "cpu"

    def load(self, checkpoint_path: str, device: str) -> None:
        """Load the Qwen2.5-VL vision tower from ``checkpoint_path``."""
        self._device = device
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

    def encode(
        self,
        image_urls: List[str],
        lm_token_ids: List[int],
        lm_embed_tokens: torch.nn.Embedding,
    ) -> torch.Tensor:
        """Encode images and splice with text embeddings.

        Steps:
            1. Download and preprocess images.
            2. Run Qwen ViT → image embeddings (n_img_tokens, vit_hidden).
            3. Optionally project vit_hidden → lm_hidden (random matrix, POC only).
            4. Look up text embeddings from the LM's embed_tokens.
            5. Prepend image embeddings before text: [image | text].

        Args:
            image_urls: One URL per image in the prompt.
            lm_token_ids: Token IDs as produced by the LM tokenizer.
            lm_embed_tokens: LM embedding layer, loaded by Dynamo from pd_model.

        Returns:
            Tensor of shape ``(seq_len, lm_hidden_dim)``.
        """
        assert (
            self.vision_encoder is not None and self.image_processor is not None
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
        lm_hidden = lm_embed_tokens.weight.shape[1]
        vit_hidden = image_embeds.shape[-1]
        if vit_hidden != lm_hidden:
            if not hasattr(self, "_proj_cache"):
                self._proj_cache: dict = {}
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

        # 4. Look up text embeddings from the LM's embed_tokens
        ids_tensor = torch.tensor(
            lm_token_ids, dtype=torch.long, device=lm_embed_tokens.weight.device
        )
        with torch.no_grad():
            text_embeds = lm_embed_tokens(ids_tensor)  # (n_text, lm_hidden)

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
