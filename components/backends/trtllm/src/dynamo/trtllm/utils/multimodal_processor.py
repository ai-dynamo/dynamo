# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

import torch
from tensorrt_llm.inputs import default_multimodal_input_loader


class MultimodalRequestProcessor:
    """Simple processor for OpenAI format multimodal requests."""

    def __init__(self, model_type: str, model_dir: str):
        self.model_type = model_type
        self.model_dir = model_dir
        self.modality = ""

    def extract_prompt_and_media(
        self, messages: List[Dict]
    ) -> tuple[str, List[str], List[str]]:
        """Extracts text prompt, image URLs, and embedding paths from messages."""
        text_parts = []
        image_urls = []
        embedding_paths = []

        for message in messages:
            for content in message.get("content", []):
                if content.get("type") == "text":
                    text_parts.append(content.get("text", ""))
                elif content.get("type") == "image_url":
                    url = content.get("image_url", {}).get("url", "")
                    if not url:
                        continue
                    self.modality = "image"
                    if url.endswith((".pt", ".pth", ".bin")):
                        embedding_paths.append(url)
                    else:
                        image_urls.append(url)

        return " ".join(text_parts), image_urls, embedding_paths

    async def process_openai_request(self, request: Dict) -> Dict:
        """Process OpenAI request and return with multimodal data."""
        messages = request.get("messages", [])
        text_prompt, image_urls, embedding_paths = self.extract_prompt_and_media(
            messages
        )

        if not image_urls and not embedding_paths:
            # No multimodal content, return original request
            return request

        mm_embeds = None
        if embedding_paths:
            mm_embeds = [torch.load(path) for path in embedding_paths]
            if not image_urls:
                image_urls = ["empty_url"]

        kwargs = {}
        if mm_embeds is not None:
            kwargs["mm_embeddings"] = mm_embeds

        # Process with default_multimodal_input_loader
        processed_inputs = default_multimodal_input_loader(
            tokenizer=None,
            model_dir=self.model_dir,
            model_type=self.model_type,
            modality=self.modality,
            prompts=[text_prompt],
            media=[image_urls],
            image_data_format="pt",
            device="cuda",
            **kwargs,
        )

        # Return modified request
        return {**request, "processed_inputs": processed_inputs}
