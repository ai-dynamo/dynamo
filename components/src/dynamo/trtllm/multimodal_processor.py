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

import hashlib
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


def apply_mm_hashes(mm_data: Any) -> int:
    """
    Compute a hash for multimodal data.

    This function creates a deterministic hash from multimodal objects (images, videos, etc.)
    to uniquely identify them for KV cache routing. The hash is computed from the
    actual tensor data to ensure identical content produces identical hashes.

    Args:
        mm_data: Multimodal data - can be a torch.Tensor, dict with tensors, or other format

    Returns:
        A 64-bit unsigned integer hash (as Python int)
    """
    try:
        # Handle different input formats
        if isinstance(mm_data, torch.Tensor):
            # For tensor, hash the raw bytes
            data_bytes = mm_data.cpu().numpy().tobytes()
        elif isinstance(mm_data, dict):
            # For dictionary (e.g., with 'mm_embeddings'), hash the main embeddings
            if "mm_embeddings" in mm_data:
                data_bytes = mm_data["mm_embeddings"].cpu().numpy().tobytes()
            else:
                # Fallback: concatenate all tensor values
                all_bytes = b""
                for key in sorted(mm_data.keys()):
                    if isinstance(mm_data[key], torch.Tensor):
                        all_bytes += mm_data[key].cpu().numpy().tobytes()
                data_bytes = all_bytes
        elif isinstance(mm_data, (list, tuple)):
            # For list/tuple of tensors, concatenate all
            all_bytes = b""
            for item in mm_data:
                if isinstance(item, torch.Tensor):
                    all_bytes += item.cpu().numpy().tobytes()
            data_bytes = all_bytes
        else:
            # Fallback: try to convert to bytes
            data_bytes = str(mm_data).encode("utf-8")

        # Use xxHash-like algorithm via Python's built-in hash with SHA256 for consistency
        # We use SHA256 to get a deterministic 64-bit hash across processes
        hash_digest = hashlib.sha256(data_bytes).digest()
        # Take first 8 bytes and convert to uint64
        mm_hash = int.from_bytes(hash_digest[:8], byteorder="little", signed=False)

        logging.debug(f"Computed MM hash: {mm_hash} for data type: {type(mm_data)}")
        return mm_hash

    except Exception as e:
        logging.error(f"Failed to compute MM hash: {e}")
        # Return a deterministic fallback hash
        return hash(str(type(mm_data))) & 0xFFFFFFFFFFFFFFFF


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
        self, request: Dict, embeddings: Any
    ) -> Optional[Any]:
        """
        Process OpenAI request and return with multimodal data.

        Also computes mm_hash for the multimodal objects to enable KV cache routing.
        """
        # Extract messages - check extra_args first (from Rust preprocessor for multimodal)
        # Fall back to direct messages field for backward compatibility
        messages = request.get("extra_args", {}).get(
            "messages", request.get("messages", [])
        )
        text_prompt, image_urls, embedding_paths = self.extract_prompt_and_media(
            messages
        )

        if not image_urls and not embedding_paths:
            logging.warning("No multimodal content, returning None")
            return None

        loader_kwargs = {}
        mm_data_for_hash = None

        if embeddings is not None:
            # EPD flow
            loader_kwargs["mm_embeddings"] = [embeddings]
            mm_data_for_hash = embeddings
            logging.debug(f"Using NIXL embeddings in prefill worker: {embeddings}")
        elif image_urls:
            # Image-only flow
            loader_kwargs["media"] = [image_urls]
            # For image URLs, we'll compute hash after loading in TRTLLM
            # Store URL as fallback for now
            mm_data_for_hash = image_urls[0] if image_urls else None
        elif embedding_paths:
            # PD flow with no NIXL and no encoder
            loaded_embeddings = [
                self.load_tensor_from_path_or_url(path) for path in embedding_paths
            ]
            loader_kwargs["mm_embeddings"] = loaded_embeddings
            mm_data_for_hash = loaded_embeddings[0] if loaded_embeddings else None
            logging.debug(f"Using embedding paths in prefill worker: {embedding_paths}")

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

        # Compute MM hash if we have the data
        mm_hash = None
        if mm_data_for_hash is not None:
            try:
                mm_hash = apply_mm_hashes(mm_data_for_hash)
                logging.debug(f"Computed mm_hash: {mm_hash}")
            except Exception as e:
                logging.warning(f"Failed to compute mm_hash: {e}")

        # Store mm_hash in the request for later use by router
        # This will be used to construct RequestExtraInfo
        if mm_hash is not None:
            if "extra_args" not in request:
                request["extra_args"] = {}
            if not isinstance(request["extra_args"], dict):
                request["extra_args"] = {}
            request["extra_args"]["mm_hash"] = mm_hash

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
