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
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib.parse import urlparse

import httpx
import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import load_file as safetensors_load_file
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

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
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
        enable_frontend_decoding: bool = False,
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

        self.image_loader = ImageLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )

    def is_url(self, path: str) -> bool:
        """Check if a path is a URL."""
        parsed = urlparse(path)
        # file:// URLs have scheme but no netloc, treat them as local paths
        if parsed.scheme == "file":
            return False
        return bool(parsed.scheme and parsed.netloc)

    def _unwrap_safetensors(
        self, data: Dict[str, torch.Tensor]
    ) -> "torch.Tensor | Dict[str, torch.Tensor]":
        """Return a single tensor when the file has one key, else the full dict.

        Multi-key files (e.g. Maverick/Scout with mm_embeddings +
        image_special_tokens + image_special_token_offsets) need the
        full dict so encode_helper can extract auxiliary data.
        """
        if len(data) == 1:
            return next(iter(data.values()))
        return data

    def load_tensor_from_path_or_url(
        self, path: str
    ) -> "torch.Tensor | Dict[str, torch.Tensor]":
        """Load tensors from a local .safetensors path or URL.

        Returns a single tensor for single-key files (e.g. LLaVA-NeXT),
        or a dict of tensors for multi-key files (e.g. Maverick/Scout).
        Only .safetensors format is accepted.
        """
        parsed = urlparse(path)
        lower_path = parsed.path.lower()
        if lower_path.endswith((".pt", ".pth", ".bin")):
            raise RuntimeError(
                "Unsafe tensor format: .pt/.pth/.bin files are not allowed. "
                "Use .safetensors format instead."
            )
        if not lower_path.endswith(".safetensors"):
            raise RuntimeError("Only .safetensors embedding files are supported.")

        if self.is_url(path):
            if parsed.scheme not in ("http", "https"):
                raise RuntimeError(f"Unsupported URL scheme: {parsed.scheme}")
            try:
                with httpx.Client(timeout=300.0) as client:
                    with client.stream("GET", path) as resp:
                        resp.raise_for_status()
                        content_length = resp.headers.get("content-length")
                        if (
                            content_length
                            and int(content_length) > self.max_file_size_bytes
                        ):
                            raise RuntimeError(
                                f"File size exceeds limit: "
                                f"{int(content_length) // (1024*1024)}MB > "
                                f"{self.max_file_size_mb}MB"
                            )
                        chunks = []
                        downloaded = 0
                        for chunk in resp.iter_bytes():
                            downloaded += len(chunk)
                            if downloaded > self.max_file_size_bytes:
                                raise RuntimeError(
                                    f"File size exceeds limit: "
                                    f"{downloaded // (1024*1024)}MB > "
                                    f"{self.max_file_size_mb}MB"
                                )
                            chunks.append(chunk)
                        content = b"".join(chunks)
                    data = safetensors_load(content)
                    return self._unwrap_safetensors(data)
            except RuntimeError:
                raise
            except Exception as e:
                logging.error(f"Failed to download or load tensor from URL: {e}")
                raise RuntimeError("Failed to load tensor")
        else:
            try:
                if not self.allowed_local_media_path:
                    logging.warning(
                        "Local file access attempted but no allowed path configured"
                    )
                    raise RuntimeError("Failed to load tensor")

                local_path = path.removeprefix("file://")
                resolved_path = Path(local_path).resolve()
                allowed_path = Path(self.allowed_local_media_path).resolve()

                try:
                    resolved_path.relative_to(allowed_path)
                except ValueError:
                    logging.warning(
                        f"Blocked access to file outside {self.allowed_local_media_path}: {path}"
                    )
                    raise RuntimeError("Failed to load tensor")

                if not resolved_path.exists():
                    raise RuntimeError(f"Embedding file not found: {resolved_path}")
                file_size = resolved_path.stat().st_size
                if file_size > self.max_file_size_bytes:
                    raise RuntimeError(
                        f"File size ({file_size // (1024*1024)}MB) exceeds "
                        f"maximum allowed size ({self.max_file_size_bytes // (1024*1024)}MB)"
                    )
                data = safetensors_load_file(str(resolved_path))
                return self._unwrap_safetensors(data)
            except RuntimeError:
                raise
            except Exception as e:
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
                        if url.endswith(".safetensors"):
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

        # Initialize result in TokensPrompt format
        # mm_processor_kwargs must be a dict (not None) for TRT-LLM's processor
        processed_inputs: Dict[str, Any] = {"mm_processor_kwargs": {}}

        # TODO(TRTLLM-11294): Remove the fallback to text_prompt for EPD-NIXL and embeddings cases.
        # This is a temporary workaround to bypass TRT-LLM's bug where token IDs & embeddings
        # are not processed correctly.
        extra_args = request.get("extra_args") or {}
        formatted_prompt_from_frontend = extra_args.get("formatted_prompt")

        # EPD Flow Case 2: Embeddings received via NIXL from encode worker
        # The encode worker computed vision embeddings and transferred them via RDMA/NIXL
        # We need to pass these embeddings directly to TRT-LLM's generate_async
        if embeddings is not None:
            logging.info(
                f"Using NIXL embeddings from encoder: shape={embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}"
            )

            # Same structure as PD flow (TRT-LLM expects dict with "image" key)
            image_embeddings = (
                embeddings if isinstance(embeddings, list) else [embeddings]
            )
            processed_inputs["multi_modal_embeddings"] = {"image": image_embeddings}
            if formatted_prompt_from_frontend:
                processed_inputs["prompt"] = formatted_prompt_from_frontend
            else:
                logging.warning("No formatted prompt from frontend")
                return None
            return processed_inputs

        # PD Flow: Pre-tokenized by Rust frontend with direct media loading
        # TODO: Add frontend decoding support

        # Handle multimodal data if present
        multi_modal_data = request.get("multi_modal_data")
        if multi_modal_data and isinstance(multi_modal_data, dict):
            processed_mm_data = {}
            loaded_embeddings: list[torch.Tensor] = []

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

                    if url.endswith(".safetensors"):
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

                # Load pre-computed vision encoder embeddings (.safetensors) for PD flow
                if embedding_paths:
                    try:
                        raw_loaded = [
                            self.load_tensor_from_path_or_url(path)
                            for path in embedding_paths
                        ]
                        loaded_embeddings = []
                        for item in raw_loaded:
                            if isinstance(item, dict):
                                emb = item.get("mm_embeddings")
                                if emb is None:
                                    logging.error(
                                        "Dictionary embeddings missing 'mm_embeddings' key"
                                    )
                                    return None
                                loaded_embeddings.append(emb)
                            else:
                                loaded_embeddings.append(item)
                        if loaded_embeddings:
                            logging.info(
                                f"Loaded {len(loaded_embeddings)} embedding file(s) from paths: {embedding_paths}"
                            )
                    except Exception as e:
                        logging.error(f"Failed to load embeddings: {e}")
                        return None

            # TODO: Add support for video_url, audio_url

            if loaded_embeddings:
                # For TRT-LLM MM embeddings, the currently
                # supported modality is "image".
                if formatted_prompt_from_frontend:
                    processed_inputs["prompt"] = formatted_prompt_from_frontend
                else:
                    logging.warning("No formatted prompt from frontend")
                    return None

                processed_inputs["multi_modal_embeddings"] = {
                    "image": loaded_embeddings
                }
                return processed_inputs

            if processed_mm_data:
                processed_inputs["multi_modal_data"] = processed_mm_data

        # Get token_ids from request (already tokenized by Rust frontend)
        token_ids = request.get("token_ids")
        if not token_ids:
            logging.warning("No token_ids in request")
            return None
        processed_inputs["prompt_token_ids"] = token_ids

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
