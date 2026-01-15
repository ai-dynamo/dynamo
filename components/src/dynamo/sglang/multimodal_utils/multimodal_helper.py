# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared multimodal helper for aggregated and encode-worker paths."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from dynamo.sglang.args import Config
from dynamo.sglang.multimodal_utils.multimodal_encode_utils import (
    encode_image_embeddings,
)
from dynamo.sglang.multimodal_utils.multimodal_image_loader import ImageLoader

logger = logging.getLogger(__name__)


def extract_image_url(request: Dict[str, Any]) -> Optional[str]:
    """Return the image URL from request if present, else None."""
    mm_data = request.get("multi_modal_data")
    if not mm_data:
        return None
    image_urls = mm_data.get("image_url", [])
    if not image_urls:
        return None
    item = image_urls[0]
    if isinstance(item, dict) and "Url" in item:
        return item["Url"]
    if isinstance(item, str):
        return item
    return None


class MultimodalHelper:
    """Shared multimodal encoding and token expansion logic."""

    def __init__(
        self,
        model_path: str,
        served_model_name: str,
        chat_template: str,
        image_loader: Optional[ImageLoader] = None,
    ) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name
        self.image_loader = image_loader or ImageLoader()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vision_model = AutoModel.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Get image token string from chat template
        image_token_str = chat_templates[chat_template].copy().image_token

        # For Qwen2.5-VL, the image token is a sequence of special tokens.
        # Use the image_pad token as the main image token for expansion.
        if image_token_str == "<|vision_start|><|image_pad|><|vision_end|>":
            self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        else:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token_str)

        logger.info("Multimodal helper initialized.")

    @classmethod
    def from_config(cls, config: Config) -> "MultimodalHelper":
        return cls(
            model_path=config.server_args.model_path,
            served_model_name=config.server_args.served_model_name,
            chat_template=config.server_args.chat_template,
        )

    def expand_image_tokens(
        self, token_ids: List[int], num_image_tokens: int
    ) -> List[int]:
        if self.image_token_id not in token_ids:
            raise ValueError(
                f"Image token ID {self.image_token_id} not found in token_ids."
            )
        idx = token_ids.index(self.image_token_id)
        return (
            token_ids[:idx]
            + [self.image_token_id] * num_image_tokens
            + token_ids[idx + 1 :]
        )

    async def encode_image(
        self, image_url: str
    ) -> Tuple[torch.Tensor, Optional[List[int]]]:
        image = await self.image_loader.load_image(image_url)
        image_embeds = self.image_processor(images=image, return_tensors="pt")
        precomputed_embeddings = encode_image_embeddings(
            model_name=self.served_model_name,
            image_embeds=image_embeds,
            vision_encoder=self.vision_model,
            projector=None,
        )
        image_grid_thw = (
            image_embeds["image_grid_thw"].tolist()
            if "image_grid_thw" in image_embeds
            else None
        )
        return precomputed_embeddings, image_grid_thw

    async def prepare_multimodal_inputs(
        self, token_ids: List[int], image_url: str
    ) -> Tuple[List[int], dict]:
        embeddings, image_grid_thw = await self.encode_image(image_url)
        num_image_tokens = embeddings.shape[1]
        expanded_token_ids = self.expand_image_tokens(token_ids, num_image_tokens)
        # Use "processor_output" format with "precomputed_embeddings" key.
        # SGLang's collect_mm_items_from_processor_output handles this correctly,
        # setting item.precomputed_embeddings (not item.feature).
        mm_item = {
            "format": "processor_output",
            "modality": "IMAGE",
            "image_grid_thw": torch.tensor(image_grid_thw) if image_grid_thw else None,
            "precomputed_embeddings": embeddings,
        }
        return expanded_token_ids, mm_item
