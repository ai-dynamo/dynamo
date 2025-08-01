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

import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from tensorrt_llm.inputs import default_multimodal_input_loader


class TokenizerProtocol(Protocol):
    """
    A protocol for tokenizers that defines a decode method.

    This is used for type hinting to resolve mypy errors related to
    the tokenizer's decode method not being found on a generic 'object' type.
    """

    def decode(self, token_ids: List[int]) -> str:
        ...


class MultimodalRequestProcessor:
    """Simple processor for OpenAI format multimodal requests."""

    def __init__(
        self,
        model_type: str,
        model_dir: str,
        tokenizer: Optional[TokenizerProtocol] = None,
    ):
        self.model_type = model_type
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.modality = ""

    def extract_prompt_and_media(
        self, messages: List[Dict]
    ) -> Tuple[str, List[str], List[str]]:
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

    async def process_openai_request(self, request: Dict) -> Optional[Any]:
        """Process OpenAI request and return with multimodal data."""
        # Normalize the request to handle OpenAI format
        if "stop_conditions" not in request:
            request["stop_conditions"] = {}
        if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
            request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

        if "sampling_options" not in request:
            request["sampling_options"] = {}
        if (
            "temperature" in request
            and "temperature" not in request["sampling_options"]
        ):
            request["sampling_options"]["temperature"] = request.pop("temperature")

        messages = request.get("messages", [])
        text_prompt, image_urls, embedding_paths = self.extract_prompt_and_media(
            messages
        )

        if not image_urls and not embedding_paths:
            # No multimodal content, return None
            return None

        loader_kwargs = {}
        if embedding_paths:
            mm_embeds = [torch.load(path) for path in embedding_paths]
            loader_kwargs["mm_embeddings"] = mm_embeds
        elif image_urls:
            loader_kwargs["media"] = [image_urls]

        # Process with default_multimodal_input_loader
        processed_inputs = default_multimodal_input_loader(
            tokenizer=None,
            model_dir=self.model_dir,
            model_type=self.model_type,
            modality=self.modality,
            prompts=[text_prompt],
            image_data_format="pt",
            device="cuda",
            **loader_kwargs,
        )

        # Return the first processed input if available
        if processed_inputs:
            return processed_inputs[0]

        return None

    def create_response_chunk(
        self,
        output: Any,
        num_output_tokens_so_far: int,
        request_id: str,
        model_name: str,
    ) -> Dict[str, Any]:
        """Creates a response chunk for multimodal streaming."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided for creating response chunks.")

        new_tokens = output.token_ids[num_output_tokens_so_far:]
        # Decode the new token IDs into a string. This is the incremental piece
        # of text to be sent to the client.
        delta_text = self.tokenizer.decode(new_tokens)
        # Assemble the delta payload for the response chunk.
        delta = {"content": delta_text if delta_text else ""}
        if num_output_tokens_so_far == 0:
            # The first chunk must include the "assistant" role.
            delta["role"] = "assistant"
        choice = {
            "index": 0,
            "delta": delta,
            "finish_reason": output.finish_reason,
        }
        # Wrap the choice in the final response chunk following the OpenAI
        # streaming format.
        return {
            "id": request_id,
            "model": model_name,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "choices": [choice],
        }

    def get_stop_response(self, request_id: str, model_name: str) -> Dict[str, Any]:
        """Creates the final stop response chunk for multimodal streaming."""
        final_choice = {
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }
        return {
            "id": request_id,
            "model": model_name,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "choices": [final_choice],
            "finish_reason": "stop",
        }
