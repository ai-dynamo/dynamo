# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tensorrt_llm.inputs import (
    MultimodalDataTracker,
    add_multimodal_placeholders,
    apply_chat_template as trtllm_apply_chat_template,
)
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory

from dynamo.common.multimodal.image_loader import ImageLoader
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
        self.modality = ""
        self.allowed_local_media_path = allowed_local_media_path
        self.max_file_size_mb = max_file_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        # Used for streaming delta computation in create_response_chunk()
        self.previous_decoded_text = ""

        # Initialize tokenizer ONCE at startup to avoid per-request overhead
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = tokenizer_factory(model_dir)

        self.image_loader = ImageLoader()

        # Cached AutoProcessor for apply_chat_template (loaded once)
        self._processor = None

    def _get_processor(self):
        """Lazily load and cache an AutoProcessor for chat-template application."""
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_dir, use_fast=True, trust_remote_code=True
            )
        return self._processor

    def _build_vlm_prompt(self, text_prompt, num_images):
        """Build a prompt with model-specific vision placeholders and chat template.

        Reuses TRT-LLM's ``MultimodalDataTracker`` and ``add_multimodal_placeholders``
        for inserting the correct model-specific image placeholders (e.g.
        ``<|vision_start|><|image_pad|><|vision_end|>`` for Qwen), then
        ``apply_chat_template`` for wrapping with the model's chat format.

        This mirrors the prompt-building logic inside TRT-LLM's
        ``default_multimodal_input_loader`` without the expensive media-loading
        part.

        Returns:
            The prompt string with vision placeholders and chat wrapping,
            or *None* on failure (caller should fall back to token_ids decode).
        """
        try:
            # 1. Count model-specific image placeholders via TRT-LLM's tracker
            mm_data_tracker = MultimodalDataTracker(self.model_type)
            for _ in range(num_images):
                mm_data_tracker.add_data("image", None)
            mm_placeholder_counts = mm_data_tracker.placeholder_counts()

            # 2. Insert placeholders into text (position is model-specific)
            content = text_prompt
            if mm_placeholder_counts:
                content = add_multimodal_placeholders(
                    self.model_type, text_prompt, mm_placeholder_counts
                )

            # 3. Apply chat template via TRT-LLM (handles edge-case models)
            processor = self._get_processor()
            prompt = trtllm_apply_chat_template(
                model_type=self.model_type,
                tokenizer=self.tokenizer,
                processor=processor,
                conversation=[{"role": "user", "content": content}],
                add_generation_prompt=True,
                mm_placeholder_counts=[mm_placeholder_counts],
            )
            return prompt
        except Exception as e:
            logging.warning(
                "_build_vlm_prompt failed (%s), falling back to token_ids decode", e
            )
            return None

    def is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        parsed = urlparse(path)
        # file:// URLs have scheme but no netloc, treat them as local paths
        if parsed.scheme == "file":
            return False
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

                # Strip file:// prefix if present
                local_path = path.removeprefix("file://")

                resolved_path = Path(local_path).resolve()
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
                if isinstance(content, str):
                    text_parts.append(content)
                else:
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

        return "".join(text_parts), image_urls, embedding_paths

    async def process_openai_request(
        self, request: Dict, embeddings: Any, ep_disaggregated_params: Any
    ) -> Optional[Any]:
        """
        Process OpenAI request and return multimodal data in TokensPrompt format.

        Supports three flows:
        1. EPD Case 1: Encoder fully processed (has _epd_processed_prompt)
        2. EPD Case 2: NIXL embeddings (embeddings parameter is not None)
        3. PD Flow: Rust pre-tokenized with direct media loading

        Returns dict compatible with TRT-LLM's generate_async:
        {
            "prompt_token_ids": List[int],
            "multi_modal_data": Dict[str, List[torch.Tensor]]
        }
        or for EPD Case 1:
        {
            "prompt": str,
            "prompt_token_ids": List[int]
        }
        """
        self.previous_decoded_text = ""

        # EPD Flow Case 1: Encoder has fully processed the prompt
        # The encode worker has done everything: vision encoding, prompt processing, tokenization
        # Return the encoder's processed prompt and tokens directly
        processed_prompt_from_encoder = request.get("_epd_processed_prompt")
        if processed_prompt_from_encoder is not None:
            logging.info("MM: Using fully processed prompt from encoder")
            result = {"prompt": processed_prompt_from_encoder}
            prompt_token_ids = request.get("_epd_prompt_token_ids")
            if prompt_token_ids:
                result["prompt_token_ids"] = prompt_token_ids
            else:
                logging.warning("MM: No prompt_token_ids from encoder")
            return result

        # Get token_ids from request (already tokenized by Rust frontend)
        token_ids = request.get("token_ids")
        if not token_ids:
            logging.warning("No token_ids in request")
            return None

        # Initialize result in TokensPrompt format
        # mm_processor_kwargs must be a dict (not None) for TRT-LLM's processor
        processed_inputs = {"prompt_token_ids": token_ids, "mm_processor_kwargs": {}}

        # Build text prompt with model-specific vision placeholders.
        # VLMs (e.g. Qwen2-VL) need placeholders like
        # <|vision_start|><|image_pad|><|vision_end|> in the prompt.  The Rust
        # frontend may use a different template that omits these, so we rebuild
        # using TRT-LLM's MultimodalDataTracker + add_multimodal_placeholders +
        # apply_chat_template — the same logic as default_multimodal_input_loader.
        prompt_text = None
        if self.tokenizer is not None:
            # Try building VLM prompt with TRT-LLM utilities
            messages = request.get("extra_args", {}).get(
                "messages", request.get("messages", [])
            )
            if messages:
                text_from_msgs, image_urls_from_msgs, _ = (
                    self.extract_prompt_and_media(messages)
                )
                if text_from_msgs and image_urls_from_msgs:
                    prompt_text = self._build_vlm_prompt(
                        text_from_msgs, len(image_urls_from_msgs)
                    )
                    if prompt_text:
                        logging.debug(
                            "MM: built VLM prompt via TRT-LLM utilities"
                        )

            # Fallback: decode Rust token_ids (works when the Rust template
            # already includes vision placeholders, e.g. LLaVA)
            if prompt_text is None:
                prompt_text = self.tokenizer.decode(
                    token_ids, skip_special_tokens=False
                )

            processed_inputs["prompt"] = prompt_text

        # EPD Flow Case 2: Embeddings received via NIXL from encode worker
        # The encode worker computed vision embeddings and transferred them via RDMA/NIXL
        # We need to pass these embeddings directly to TRT-LLM's generate_async
        if embeddings is not None:
            logging.info(
                f"Using NIXL embeddings from encoder: shape={embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}"
            )

            # Structure embeddings in the format TRT-LLM's generate_async expects
            processed_inputs["multi_modal_embeddings"] = embeddings

            return processed_inputs

        # PD Flow: Pre-tokenized by Rust frontend with direct media loading
        # TODO: Add frontend decoding support

        # Handle multimodal data if present
        multi_modal_data = request.get("multi_modal_data")
        if multi_modal_data and isinstance(multi_modal_data, dict):
            processed_mm_data = {}

            # Process images and embedding paths from image_url field
            image_items = multi_modal_data.get("image_url", [])
            if image_items and isinstance(image_items, list):
                # Separate embedding paths from regular image URLs
                # Items come from Rust in format: {"Url": "..."} or {"Decoded": ...}
                embedding_paths = []
                image_urls = []

                for item in image_items:
                    # Extract URL from item (Rust enum serialization uses "Url" with capital U)
                    if isinstance(item, dict) and "Url" in item:
                        url = item["Url"]
                    elif isinstance(item, dict) and "Decoded" in item:
                        # Already decoded data (NIXL) - always treat as image
                        image_urls.append(item)
                        continue
                    elif isinstance(item, str):
                        # Fallback for string URLs (backward compatibility)
                        url = item
                    else:
                        logging.warning(
                            f"Unexpected item format in image_items: {item}"
                        )
                        continue

                    # Check if this is an embedding file based on extension
                    if url.endswith((".pt", ".pth", ".bin")):
                        embedding_paths.append(url)
                    else:
                        # Keep original item format for load_image_batch
                        image_urls.append(
                            item if isinstance(item, dict) else {"Url": item}
                        )

                # Load regular images as PIL Images for TRT-LLM's input processor
                # TRT-LLM will auto-detect this and compute mrope_config
                if image_urls:
                    try:
                        pil_images = await self.image_loader.load_image_batch(
                            image_urls
                        )
                        if pil_images:
                            processed_mm_data["image"] = pil_images
                            logging.info(
                                f"Loaded {len(pil_images)} image(s) as PIL Images"
                            )
                    except Exception as e:
                        logging.error(f"Failed to load images: {e}")
                        return None

                # Load embedding files (.pt, .pth, .bin) for PD flow
                # These are pre-computed vision encoder outputs
                if embedding_paths:
                    try:
                        loaded_embeddings = [
                            self.load_tensor_from_path_or_url(path)
                            for path in embedding_paths
                        ]
                        if loaded_embeddings:
                            processed_mm_data["embedding"] = loaded_embeddings
                            logging.info(
                                f"Loaded {len(loaded_embeddings)} embedding file(s) from paths: {embedding_paths}"
                            )
                    except Exception as e:
                        logging.error(f"Failed to load embeddings: {e}")
                        return None

            # TODO: Add support for video_url, audio_url

            if processed_mm_data:
                processed_inputs["multi_modal_data"] = processed_mm_data

                # IMPORTANT: When multi_modal_data is present, we must NOT include
                # prompt_token_ids in the dict. TRT-LLM's _preprocess dispatches on
                # an if/elif chain where "prompt_token_ids" is checked BEFORE "prompt".
                # If both keys are present, TRT-LLM takes the prompt_token_ids branch
                # and skips multimodal processing entirely (no input_processor runs,
                # no multimodal hashing, no multimodal_lengths). By providing only
                # "prompt" + "multi_modal_data", we ensure TRT-LLM takes the correct
                # multimodal branch that runs the model-specific input processor.
                if prompt_text is not None:
                    processed_inputs.pop("prompt_token_ids", None)
                else:
                    logging.warning(
                        "Cannot remove prompt_token_ids: no tokenizer available "
                        "to decode prompt text. TRT-LLM may skip multimodal processing."
                    )

        return processed_inputs

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
