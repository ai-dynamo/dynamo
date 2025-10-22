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

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from tensorrt_llm.inputs import default_multimodal_input_loader

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()


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
        max_file_size_mb: int,
        tokenizer: Optional[TokenizerProtocol] = None,
        allowed_local_media_path: str = "",
    ):
        self.model_type = model_type
        self.model_dir = model_dir
        self.tokenizer = tokenizer
        self.modality = ""
        self.allowed_local_media_path = allowed_local_media_path
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)

    def load_tensor_from_path_or_url(self, path: str) -> torch.Tensor:
        """Load a tensor from either a local file path or a URL."""
        if self.is_url(path):
            # Download directly to memory using BytesIO (no filesystem ops)
            try:
                with urlopen(path) as response:
                    # Read at most max_size + 1 bytes to detect if file exceeds limit
                    data = response.read(self.max_file_size_bytes + 1)
                    if len(data) > self.max_file_size_bytes:
                        raise RuntimeError(
                            f"File size exceeds limit: {len(data) // (1024*1024)}MB > "
                            f"{self.max_file_size_mb}MB "
                        )
                    tensor_stream = BytesIO(data)
                    tensor = torch.load(
                        tensor_stream, map_location="cpu", weights_only=True
                    )
                    return tensor
            except Exception as e:
                # Log actual error for debugging, return generic error to user
                logging.error(f"Failed to download or load tensor from URL: {e}")
                raise RuntimeError("Failed to load tensor")
        else:
            # Restrict local file access to configured directory only
            try:
                # Check if local media path is configured
                if not self.allowed_local_media_path:
                    logging.warning(
                        "Local file access attempted but no allowed path configured"
                    )
                    raise RuntimeError("Failed to load tensor")

                resolved_path = Path(path).resolve()
                allowed_path = Path(self.allowed_local_media_path).resolve()

                # Secure path validation: Check if the resolved path is actually within allowed directory
                try:
                    resolved_path.relative_to(allowed_path)
                except ValueError:
                    logging.warning(
                        f"Blocked access to file outside {self.allowed_local_media_path}: {path}"
                    )
                    raise RuntimeError("Failed to load tensor")

                # Check file size before loading
                if resolved_path.exists():
                    file_size = resolved_path.stat().st_size
                    if file_size > self.max_file_size_bytes:
                        raise RuntimeError(
                            f"File size ({file_size // (1024*1024)}MB) exceeds "
                            f"maximum allowed size ({self.max_file_size_bytes // (1024*1024)}MB)"
                        )
                return torch.load(resolved_path, map_location="cpu", weights_only=True)
            except Exception as e:
                # Log actual error for debugging, return generic error to user
                logging.error(f"Failed to load tensor from local path: {e}")
                raise RuntimeError("Failed to load tensor")

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

    async def process_openai_request(
        self, request: Dict, embeddings: Any, ep_disaggregated_params: Any
    ) -> Optional[Any]:
        """Process OpenAI request and return with multimodal data."""
        self.previous_decoded_text = ""
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

        if not image_urls and not embedding_paths and not ep_disaggregated_params:
            logging.warning("No multimodal content, returning None")
            return None

        # Full EPD flow - check if we have a processed prompt from the encoder
        # For decode worker: ep_disaggregated_params will be None, but _epd_processed_prompt will be present
        processed_prompt_from_encoder = request.get("_epd_processed_prompt")
        
        if processed_prompt_from_encoder is not None or ep_disaggregated_params is not None:
            logging.info("MM PROCESSOR: Full EPD flow - skipping default_multimodal_input_loader")
            
            # Use the processed prompt from encoder if available (includes <image> tokens)
            # The encoder already called default_multimodal_input_loader which adds <image> tokens
            if processed_prompt_from_encoder is not None:
                text_prompt = processed_prompt_from_encoder
                logging.info(f"MM PROCESSOR: Using processed prompt from encoder: {text_prompt}")
            else:
                logging.warning(f"MM PROCESSOR: No _epd_processed_prompt in request, using original text_prompt")
            
            # Return minimal processed_input with prompt containing image placeholders
            # Prefill: multimodal embeddings handled via ep_disaggregated_params
            # Decode: multimodal embeddings already consumed during prefill, just continue generation
            result = {"prompt": text_prompt}
            
            # Include prompt_token_ids from encoder if available (for consistency)
            if "_epd_prompt_token_ids" in request and request["_epd_prompt_token_ids"]:
                result["prompt_token_ids"] = request["_epd_prompt_token_ids"]
                logging.info(f"MM PROCESSOR: Including prompt_token_ids from encoder (length={len(request['_epd_prompt_token_ids'])})")
            
            return result

        # Other flows (NIXL, embedding paths, raw images)
        loader_kwargs = {}
        if embeddings is not None:
            # EPD flow with NIXL (legacy)
            loader_kwargs["mm_embeddings"] = [embeddings]
            logging.debug(f"Using NIXL embeddings in prefill worker: {embeddings}")
        elif image_urls:
            # Image-only flow
            loader_kwargs["media"] = [image_urls]
        elif embedding_paths:
            # PD flow with no NIXL and no encoder
            loader_kwargs["mm_embeddings"] = [
                self.load_tensor_from_path_or_url(path) for path in embedding_paths
            ]
            logging.debug(f"Using embedding paths in prefill worker: {embedding_paths}")

        # Process with default_multimodal_input_loader (only for non-full-EPD flows)
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

        all_tokens = output.token_ids
        current_text = self.tokenizer.decode(
            all_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if num_output_tokens_so_far == 0:
            # First chunk: use all decoded text
            delta_text = current_text
            # Store for next iteration
            self.previous_decoded_text = current_text
        else:
            # Incremental chunk: extract delta using cached previous text
            delta_text = current_text[len(self.previous_decoded_text) :]
            # Update cache for next iteration
            self.previous_decoded_text = current_text
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
