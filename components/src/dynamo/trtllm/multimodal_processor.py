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
from typing import Any, Coroutine, Dict, List, Optional, Protocol, Tuple, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
from tensorrt_llm.inputs import (
    ConversationMessage,
    MultimodalData,
    MultimodalDataTracker,
    add_multimodal_placeholders,
    apply_chat_template,
    load_image,
    load_video,
)
from tensorrt_llm.inputs.utils import HF_CHAT_TEMPLATE_EXCEPTIONS, load_audio
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.llmapi.tokenizer import (
    TokenizerBase,
    TransformersTokenizer,
    tokenizer_factory,
)
from tensorrt_llm.serve.chat_utils import parse_chat_message_content
from transformers import AutoProcessor, ProcessorMixin

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

        # Initialize tokenizer ONCE at startup to avoid per-request overhead
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = tokenizer_factory(model_dir)

        self.processor = None
        if model_type not in HF_CHAT_TEMPLATE_EXCEPTIONS:
            self.processor = AutoProcessor.from_pretrained(
                model_dir, use_fast=True, trust_remote_code=True
            )

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

        return " ".join(text_parts), image_urls, embedding_paths

    async def process_openai_request(
        self, request: Dict, embeddings: Any
    ) -> Optional[Any]:
        """Process OpenAI request and return with multimodal data."""
        # Extract messages - check extra_args first (from Rust preprocessor for multimodal)
        # Fall back to direct messages field for backward compatibility
        messages = request.get("extra_args", {}).get(
            "messages", request.get("messages", [])
        )
        text_prompt, image_urls, embedding_paths = self.extract_prompt_and_media(
            messages
        )

        if not image_urls and not embedding_paths:
            logging.warning("No multimodal content, pure text request detected.")
            processed_inputs = await get_multimodal_inputs(
                messages=messages,
                model_type=self.model_type,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
        else:
            loader_kwargs = {}
            if embeddings is not None:
                # EPD flow
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
                logging.debug(
                    f"Using embedding paths in prefill worker: {embedding_paths}"
                )

            # Process with default_multimodal_input_loader
            # Pass self.tokenizer to reuse the pre-initialized tokenizer instead of
            # creating a new one per request
            processed_inputs = multimodal_input_loader(
                tokenizer=self.tokenizer,
                processor=self.processor,
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


def multimodal_input_loader(
    *,
    tokenizer: Optional[Union[TransformersTokenizer, TokenizerBase]],
    processor: ProcessorMixin,
    model_dir: str,
    model_type: str,
    modality: str,
    prompts: List[str],
    media: Optional[Union[List[str], List[List[str]]]] = None,
    image_data_format: str = "pt",
    num_frames: int = 8,
    mm_embeddings: Optional[Union[List[torch.Tensor], List[List[torch.Tensor]]]] = None,
    device: str = "cpu",
) -> List[dict[str, Union[str, torch.Tensor]]]:
    def convert_to_conversation_message(
        prompt: str,
        media: Union[Any, List[Any]],
        modality: str,
        is_embedding: bool = False,
    ) -> ConversationMessage:
        if isinstance(media, str):
            media = [media]
        if modality in ["image", "multiple_image"]:
            if is_embedding:
                # each mm_embedding corresponds to each image placeholder
                if not isinstance(media, list):
                    media = [media]

                mm_data = [
                    {"modality": modality, "mm_embedding_info": mm} for mm in media
                ]
            else:
                mm_data = [
                    MultimodalData(
                        modality=modality,
                        data=load_image(i, format=image_data_format, device=device),
                    )
                    for i in media
                ]
        elif modality == "video":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for video modality yet."
                )
            mm_data = [
                MultimodalData(
                    modality=modality,
                    data=load_video(
                        i, num_frames, format=image_data_format, device=device
                    ),
                )
                for i in media
            ]
        elif modality == "audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for audio modality yet."
                )
            mm_data = [
                MultimodalData(modality=modality, data=load_audio(i, device=device))
                for i in media
            ]
        elif modality == "image_audio":
            if is_embedding:
                raise ValueError(
                    "External embedding is not supported for image_audio modality yet."
                )
            # Use different load_xxx functions to match the modality.
            mm_data = []
            for m in media:
                data = None
                _modal = None
                if _modal is None:
                    try:
                        data = load_image(m, format=image_data_format, device=device)
                        _modal = "image"
                    except Exception:
                        pass
                if _modal is None:
                    try:
                        data = load_audio(m, device=device)
                        _modal = "audio"
                    except Exception:
                        pass
                if _modal is None:
                    raise ValueError(f"Unknown matching modality: {modality}")
                mm_data.append(MultimodalData(modality=_modal, data=data))
        elif modality == "mixture_text_image":
            mm_data = []
            for m in media:
                if m:
                    mm_data.append(
                        MultimodalData(
                            modality="image",
                            data=load_image(m, format=image_data_format, device=device),
                        )
                    )
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return ConversationMessage(role="user", content=prompt, media=mm_data)

    assert (
        media is not None or mm_embeddings is not None
    ), "Either media or mm_embeddings must be provided."
    assert (
        media is None or mm_embeddings is None
    ), "Either media or mm_embeddings must be provided, not both."
    media_or_embeddings = media if media is not None else mm_embeddings
    is_embedding = mm_embeddings is not None

    if len(media_or_embeddings) > len(prompts) and len(prompts) == 1:
        # 1 prompt + N media
        assert not isinstance(
            media_or_embeddings[0], list
        )  # media cannot be a list of lists in this case
        media_or_embeddings = [media_or_embeddings]
    assert len(media_or_embeddings) == len(prompts)

    if tokenizer is None and model_type not in HF_CHAT_TEMPLATE_EXCEPTIONS:
        tokenizer = ModelLoader.load_hf_tokenizer(model_dir, use_fast=True)

    if processor is None:
        if model_type not in HF_CHAT_TEMPLATE_EXCEPTIONS:
            processor = AutoProcessor.from_pretrained(
                model_dir, use_fast=True, trust_remote_code=True
            )

    inputs = []
    for prompt_idx, (prompt, media) in enumerate(zip(prompts, media_or_embeddings)):
        conv = convert_to_conversation_message(prompt, media, modality, is_embedding)

        mm_data_tracker = MultimodalDataTracker(model_type)
        for mdata in conv["media"]:
            # Check if mdata is a MultimodalData
            if isinstance(mdata, dict) and "modality" in mdata and "data" in mdata:
                mdata_modality = mdata["modality"]
                if modality == "multiple_image":
                    mdata_modality = "image"
                mm_data_tracker.add_data(mdata_modality, mdata["data"])
            else:
                # Add embeddings to the tracker for placeholder handling
                mm_data_tracker.add_data(mdata["modality"], mdata["mm_embedding_info"])
        mm_placeholder_counts = mm_data_tracker.placeholder_counts()
        prompt = conv["content"]
        if mm_placeholder_counts:
            conv["content"] = add_multimodal_placeholders(
                model_type, conv["content"], mm_placeholder_counts
            )
        prompt = apply_chat_template(
            model_type=model_type,
            tokenizer=tokenizer,
            processor=processor,
            conversation=[conv],
            add_generation_prompt=True,
            mm_placeholder_counts=[mm_placeholder_counts],
        )
        input = {"prompt": prompt}
        if mm_placeholder_counts:
            if mm_embeddings is not None:
                input["multi_modal_embeddings"] = mm_data_tracker.retrieve_all_sync()
            else:
                input["multi_modal_data"] = mm_data_tracker.retrieve_all_sync()
        inputs.append(input)

    return inputs


async def get_multimodal_inputs(
    messages: List[Dict],
    model_type: str,
    tokenizer: Optional[Union[TransformersTokenizer, TokenizerBase]],
    processor: ProcessorMixin,
) -> List[Any]:
    conversation: List[ConversationMessage] = []

    conversation, mm_coroutines, mm_placeholder_counts = parse_chat_messages_coroutines(
        messages, model_type
    )

    prompt: str = apply_chat_template(
        model_type=model_type,
        tokenizer=tokenizer,
        processor=processor,
        conversation=conversation,
        mm_placeholder_counts=mm_placeholder_counts,
        add_generation_prompt=True,
    )
    input = {"prompt": prompt}

    mm_data = await mm_coroutines
    if mm_data is not None:
        input["multi_modal_data"] = mm_data
    return [input]


def parse_chat_messages_coroutines(
    messages: List[Any],
    model_type: str,
) -> Tuple[
    List[ConversationMessage],
    Optional[Coroutine[Any, Any, Optional[Dict[str, List[Any]]]]],
]:
    """Parse multiple chat messages and return conversation and coroutine."""
    conversation = []
    mm_placeholder_counts = []
    mm_data_tracker = MultimodalDataTracker(model_type)

    for msg in messages:
        parsed_msg = parse_chat_message_content(msg, mm_data_tracker)
        conversation.append(parsed_msg)
        tmp_tracker = MultimodalDataTracker(model_type)
        if parsed_msg["media"]:
            for mdata in parsed_msg["media"]:
                mm_data_tracker.add_data(mdata["modality"], mdata["data"])
                tmp_tracker.add_data(mdata["modality"], mdata["data"])
        mm_placeholder_count = tmp_tracker.placeholder_counts()
        if mm_placeholder_count:
            parsed_msg["content"] = add_multimodal_placeholders(
                model_type, parsed_msg["content"], mm_placeholder_count
            )
        mm_placeholder_counts.append(mm_placeholder_count)

    return conversation, mm_data_tracker.retrieve_all_async(), mm_placeholder_counts
