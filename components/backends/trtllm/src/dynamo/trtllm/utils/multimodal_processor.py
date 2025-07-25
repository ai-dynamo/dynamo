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

from tensorrt_llm.inputs import default_multimodal_input_loader


class MultimodalRequestProcessor:
    """Simple processor for OpenAI format multimodal requests."""

    def __init__(self, model_type: str, model_dir: str):
        self.model_type = model_type
        self.model_dir = model_dir
        self.modality = ""

    def extract_text_and_images(self, messages: List[Dict]) -> tuple[str, List[str]]:
        """Extract text and image URLs from messages."""
        text_parts = []
        image_urls = []

        for message in messages:
            for content in message.get("content", []):
                if content.get("type") == "text":
                    text_parts.append(content.get("text", ""))
                elif content.get("type") == "image_url":
                    url = content.get("image_url", {}).get("url", "")
                    self.modality = "image"
                    if url:
                        image_urls.append(url)

        return " ".join(text_parts), image_urls

    async def process_openai_request(self, request: Dict) -> Dict:
        """Process OpenAI request and return with multimodal data."""
        messages = request.get("messages", [])
        text_prompt, image_urls = self.extract_text_and_images(messages)

        if not image_urls:
            # No images, return original request
            return request

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
        )

        # Return modified request
        return {**request, "processed_inputs": processed_inputs}
