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

import asyncio
import dataclasses
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from urllib.parse import urlparse

import torch
from safetensors.torch import load as safetensors_load
from safetensors.torch import load_file as safetensors_load_file
from tensorrt_llm.inputs.multimodal_data import VideoData
from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY
from tensorrt_llm.inputs.utils import async_load_video
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory

from dynamo.common.http import HttpConfigurationError, HttpStatusError, fetch_bytes
from dynamo.common.http.url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    validate_media_url,
)
from dynamo.common.multimodal.codec_errors import (
    MissingMediaDecoderError,
    video_decoder_missing,
)
from dynamo.common.multimodal.image_loader import (
    ImageLoader,
    image_cache_scope_from_request,
)
from dynamo.common.multimodal.media_source import decode_data_uri, describe_media_source
from dynamo.common.multimodal.nvdec_decoder import probe_video_codec, should_use_nvdec
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.utils.token_ids import token_ids_to_list
from dynamo.llm.exceptions import InvalidArgument
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()

# Shortest whole-request budget for one embedding download, in seconds.
_EMBEDDING_FETCH_MIN_TIMEOUT_S = 300.0
# Above the floor, the budget is the size cap divided by this rate (bytes/s).
_EMBEDDING_FETCH_MIN_RATE = 64 * 1024
# A server that sends nothing for this long fails the download, in seconds.
_EMBEDDING_FETCH_READ_TIMEOUT_S = 300.0

# Model types whose TRT-LLM input processor rescales images and videos with
# one flag. Next to float video frames, a PIL image would reach such a model
# unscaled, so image + video requests are refused for them (MiniCPM-V 4.6's
# processor currently refuses the mix itself as well).
# TODO: remove once images go to TRT-LLM as float tensors for models without
# a preferred image format, as trtllm-serve sends them.
_SHARED_RESCALE_FLAG_MODEL_TYPES = frozenset(
    {"qwen2_vl", "qwen2_5_vl", "exaone4_5", "minicpmv4_6"}
)


def _nvdec_video_data(content: bytes, num_frames: int) -> VideoData:
    """Decode H.264/H.265 via NVDEC into TRT-LLM's ``VideoData`` (format="pt").

    Mirrors ``tensorrt_llm.inputs.media_io._load_video_by_cv2``'s "pt" output for
    the pinned v1.3.0rc21: RGB, a list of ``(C, H, W)`` float32 [0,1] tensors,
    with the same metadata keys. Synchronous -- call via ``asyncio.to_thread``.
    """
    from dynamo.common.multimodal.nvdec_decoder import decode_video_nvdec

    frames_np, meta = decode_video_nvdec(content, num_frames)  # (N,H,W,3) uint8 RGB
    stacked = frames_np.astype("float32") * (1.0 / 255.0)
    nchw = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    frames_pt = list(torch.unbind(nchw, dim=0))
    fps = float(meta.get("fps") or 0.0)
    total = int(meta["total_num_frames"])
    metadata = {
        "total_num_frames": total,
        "fps": fps,
        "duration": (total / fps) if fps > 0 else 0.0,
        "frames_indices": meta["frames_indices"],
    }
    return VideoData(frames=frames_pt, metadata=metadata, audio=None)


def _video_frames_to_uint8(video: VideoData) -> VideoData:
    """Return ``video`` with its float [0,1] ``(C, H, W)`` tensor frames as one
    uint8 ``(N, H, W, 3)`` array, TRT-LLM's ``format="np"`` video form.

    Both decode paths scale uint8 frames by 1/255, so rounding back recovers
    the decoded pixels exactly. Other fields are kept.
    """
    stacked = torch.stack(list(video.frames))
    frames_u8 = stacked.mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
    return dataclasses.replace(
        video, frames=frames_u8.permute(0, 2, 3, 1).contiguous().numpy()
    )


def _mm_item_order(
    multi_modal_order: Any,
    counts: Dict[str, int],
    prompt_modality_order: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Prompt-order manifest for TRT-LLM's ``inputs["mm_item_order"]``: one
    ``{"modality", "index"}`` per item, ``index`` being the item's position in
    ``multi_modal_data[modality]``.

    Taken from the frontend's ``multi_modal_order``: the media parts in message
    order, each as ``{"modality": "image_url", "index": i}`` with ``i`` its
    position in ``multi_modal_data["image_url"]``, the order the items are
    loaded in here. As trtllm-serve does, the order is then grouped by modality
    when the model's chat template groups placeholders that way
    (``prompt_modality_order``); the frontend's order does not include that
    step. ``counts`` is the number of loaded items per modality, and the order
    must name each of them once. Otherwise this raises ``InvalidArgument``: a
    400 whose message reaches the client.
    """
    entries = multi_modal_order if isinstance(multi_modal_order, list) else None
    if entries is None:
        raise InvalidArgument(
            "Cannot order the request's media items: the request has no "
            "multi_modal_order"
        )
    order: List[Dict[str, Any]] = []
    for entry in entries:
        modality = entry.get("modality") if isinstance(entry, dict) else None
        index = entry.get("index") if isinstance(entry, dict) else None
        order.append(
            {
                "modality": modality.removesuffix("_url")
                if isinstance(modality, str)
                else None,
                "index": index if type(index) is int else None,
            }
        )
    expected = {(m, i) for m, n in counts.items() for i in range(n)}
    named = {(e["modality"], e["index"]) for e in order}
    if len(order) != len(expected) or named != expected:
        raise InvalidArgument(
            "Cannot order the request's media items: multi_modal_order does not "
            f"name each loaded item exactly once (loaded: {counts})"
        )
    if prompt_modality_order:
        rank = {m: i for i, m in enumerate(prompt_modality_order)}
        order.sort(key=lambda e: (rank.get(e["modality"], len(rank)), e["index"]))
    return order


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


def resolve_mm_processor_kwargs(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Per-request processor overrides, canonical field first.

    Presence-based: an explicit top-level {} must not fall through to extra_args.
    """
    mm_kwargs = request.get("mm_processor_kwargs")
    if mm_kwargs is None:
        mm_kwargs = (request.get("extra_args") or {}).get("mm_processor_kwargs")
    return mm_kwargs


def _is_safetensors_url(url: str) -> bool:
    """True when the URL path (not query) ends with ``.safetensors``."""
    return urlparse(url).path.lower().endswith(".safetensors")


def _urls_from_multi_modal_items(
    items: Any,
) -> Tuple[List[str], List[str]]:
    """Split ``multi_modal_data`` image items into image URLs and embedding paths."""
    image_urls: List[str] = []
    embedding_paths: List[str] = []
    if not isinstance(items, list):
        return image_urls, embedding_paths
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("Url"), str):
            url = item["Url"]
        elif isinstance(item, str):
            url = item
        else:
            continue
        if not url:
            continue
        if _is_safetensors_url(url):
            embedding_paths.append(url)
        else:
            image_urls.append(url)
    return image_urls, embedding_paths


def request_messages(request: Dict[str, Any]) -> List[Dict]:
    extra_args = request.get("extra_args") or {}
    messages = extra_args.get("messages") or request.get("messages") or []
    return messages if isinstance(messages, list) else []


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
            enable_frontend_decoding=enable_frontend_decoding,
            max_bytes=self.max_file_size_bytes,
        )

        # Reuse the shared default so this preprocessor and the vLLM/SGLang
        # backends agree on DYN_MM_VIDEO_NUM_FRAMES.
        self.num_video_frames = max(1, VideoLoader.NUM_FRAMES_DEFAULT)
        self._url_policy = UrlValidationPolicy.from_env()

        # Input processor used to size an omitted max_tokens (see
        # _expanded_prompt_len) and to read the model's preferred video format
        # (see _prefers_np_video). Optional: unavailable for models without a
        # registered processor, in which case max_tokens sizing is left to the
        # engine and videos keep their decoded format.
        self.input_processor = None
        try:
            from tensorrt_llm.inputs import create_input_processor

            self.input_processor = create_input_processor(model_dir, self.tokenizer)
        except Exception as e:
            logging.warning("Input processor unavailable for max_tokens sizing: %s", e)

    def _expanded_prompt_len(
        self, token_ids: List[int], images: Optional[List[Any]]
    ) -> Optional[int]:
        """Post-expansion prompt length: text tokens plus per-image tokens, with
        image placeholders in token_ids replaced by their expanded token counts.

        Returns None when it cannot be computed, so callers fall back to the
        engine default instead of an over/underestimate.
        """
        if not self.input_processor or not images or not token_ids:
            return None
        try:
            mm_ids = self.input_processor.get_mm_token_ids()
            mm_id_set = set(mm_ids.tolist()) if mm_ids is not None else set()
            num_placeholders = sum(1 for t in token_ids if t in mm_id_set)
            image_tokens = sum(
                int(self.input_processor.get_num_tokens_per_image(image=img))
                for img in images
            )
            return len(token_ids) - num_placeholders + image_tokens
        except Exception as e:
            logging.warning("Could not compute expanded prompt length: %s", e)
            return None

    def _prefers_np_video(self) -> bool:
        """True when the model's input processor asks for uint8 numpy video
        (``get_preferred_media_io_kwargs``, which trtllm-serve applies).

        Only multimodal input processors define the method; the processor may
        also be missing (None) or TRT-LLM's text-only DefaultInputProcessor.
        """
        get_prefs = getattr(self.input_processor, "get_preferred_media_io_kwargs", None)
        if get_prefs is None:
            return False
        prefs = get_prefs() or {}
        return prefs.get("video", {}).get("format") == "np"

    def _prompt_modality_order(self) -> Optional[Tuple[str, ...]]:
        """Modality order the model's chat template groups placeholders in
        (TRT-LLM's ``prompt_modality_order``), or None when the template keeps
        the message order. Looked up by the model's ``model_type``, as
        trtllm-serve does."""
        config = getattr(self.input_processor, "config", None)
        model_type = getattr(config, "model_type", None)
        if not isinstance(model_type, str):
            return None
        return MULTIMODAL_PLACEHOLDER_REGISTRY.get_prompt_modality_order(model_type)

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

    async def load_tensor_from_path_or_url(
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
            # One budget for the whole download: at least 300 s, and 800 s at
            # the default 50 MiB cap. A server that sends nothing for 300 s
            # fails sooner.
            timeout = max(
                _EMBEDDING_FETCH_MIN_TIMEOUT_S,
                self.max_file_size_bytes / _EMBEDDING_FETCH_MIN_RATE,
            )
            try:
                # The shared client checks self._url_policy on the URL and on
                # each redirect hop, filters blocked addresses again when it
                # connects, and stops reading past the size cap.
                content = await fetch_bytes(
                    path,
                    timeout,
                    policy=self._url_policy,
                    max_bytes=self.max_file_size_bytes,
                    read_timeout=_EMBEDDING_FETCH_READ_TIMEOUT_S,
                )
                data = safetensors_load(content)
                return self._unwrap_safetensors(data)
            except RuntimeError:
                raise
            except (UrlValidationError, HttpStatusError, HttpConfigurationError):
                # Keep the type, so that the callers can tell a rejected URL
                # (a client error) from a proxy configuration fault (a server
                # error).
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
                        if _is_safetensors_url(url):
                            embedding_paths.append(url)
                        else:
                            image_urls.append(url)

        return "".join(text_parts), image_urls, embedding_paths

    def extract_prompt_and_media_from_request(
        self, request: Dict[str, Any]
    ) -> Tuple[str, List[str], List[str]]:
        """Extract text and media URLs, preferring ``multi_modal_data``.

        The frontend strips inline ``data:`` payloads from
        ``extra_args.messages`` so the request plane carries a single copy of
        the media in ``multi_modal_data``. Chat-template structure still lives
        in ``extra_args.messages``.
        """
        text, image_urls, embedding_paths = self.extract_prompt_and_media(
            request_messages(request)
        )
        mm_data = request.get("multi_modal_data")
        if isinstance(mm_data, dict):
            mm_urls, mm_emb = _urls_from_multi_modal_items(mm_data.get("image_url"))
            if mm_urls:
                image_urls = mm_urls
            if mm_emb:
                embedding_paths = mm_emb
        return text, image_urls, embedding_paths

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
        extra_args = request.get("extra_args") or {}
        mm_kwargs = resolve_mm_processor_kwargs(request)
        if mm_kwargs is not None and not isinstance(mm_kwargs, dict):
            raise HttpStatusError(
                400,
                "Malformed mm_processor_kwargs field: expected an object",
                str(mm_kwargs),
            )
        processed_inputs: Dict[str, Any] = {
            "mm_processor_kwargs": mm_kwargs if mm_kwargs is not None else {}
        }

        # TODO(TRTLLM-11294): Remove the fallback to text_prompt for EPD-NIXL and embeddings cases.
        # This is a temporary workaround to bypass TRT-LLM's bug where token IDs & embeddings
        # are not processed correctly.
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

                    if _is_safetensors_url(url):
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
                            image_urls,
                            cache_scope=image_cache_scope_from_request(request),
                        )
                        if pil_images:
                            processed_mm_data["image"] = pil_images
                            logging.info(
                                f"Loaded {len(pil_images)} image(s) as PIL Images"
                            )
                    except (UrlValidationError, HttpStatusError):
                        # Client errors: let them reach the frontend as a 4xx
                        # instead of the generic catch below swallowing them to None.
                        raise
                    except Exception as e:
                        logging.error(f"Failed to load images: {e}")
                        return None

                # Load pre-computed vision encoder embeddings (.safetensors) for PD flow
                if embedding_paths:
                    try:
                        raw_loaded = [
                            await self.load_tensor_from_path_or_url(path)
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
                    except (
                        UrlValidationError,
                        HttpStatusError,
                        HttpConfigurationError,
                    ):
                        # Keep the type: a rejected URL is a client error (4xx)
                        # and a proxy configuration fault is a server error
                        # (5xx). A None return makes both a generic 500.
                        raise
                    except Exception as e:
                        logging.error(f"Failed to load embeddings: {e}")
                        return None

            # Video is forwarded as raw URLs ({"Url": ...}); reject local-file
            # schemes for the same SSRF reason as the image path above.
            video_items = multi_modal_data.get("video_url") or []
            if not isinstance(video_items, list):
                raise HttpStatusError(
                    400, "Malformed video_url field: expected a list", str(video_items)
                )
            # Videos go over as uint8 when the model asks for that format, as
            # Qwen3-VL does, with or without an image: trtllm-serve sends them
            # that way, and the same clip then gets the same multimodal hash
            # in both kinds of request. Images reach the model as PIL (not
            # rescaled) and both video decode paths give float [0,1] tensors
            # (rescaled). A processor that takes one rescale flag for both
            # modalities cannot take that mix: Qwen3-VL's rejects it, and
            # Qwen2/2.5-VL's would leave the image unscaled. For the models in
            # _SHARED_RESCALE_FLAG_MODEL_TYPES, which ask for no format, image +
            # video requests are refused before any video is fetched; other
            # models get the media as decoded.
            image_and_video = bool(video_items) and "image" in processed_mm_data
            if image_and_video and self.model_type in _SHARED_RESCALE_FLAG_MODEL_TYPES:
                raise InvalidArgument(
                    "This model cannot take an image and a video in the same "
                    "request yet; send them in separate requests"
                )
            videos = []
            for item in video_items:
                url = item.get("Url") if isinstance(item, dict) else item
                # Everything user-supplied that can reach an error message or a
                # log line goes through this bounded label: a data: URI carries
                # the entire media payload inline, so echoing one back would
                # serialize megabytes of base64 to the client and to every log
                # sink that records the failure.
                source = describe_media_source(
                    url if isinstance(url, str) else str(item)
                )
                if not isinstance(url, str):
                    raise HttpStatusError(
                        400, f"Unsupported video item: {source}", source
                    )
                if urlparse(url).scheme in ("", "file"):
                    raise HttpStatusError(
                        400, "Local file access is not allowed for video", source
                    )
                try:
                    normalized_url = await validate_media_url(url, self._url_policy)
                    scheme = urlparse(normalized_url).scheme
                    if scheme in ("http", "https", "data"):
                        if scheme == "data":
                            content = decode_data_uri(
                                normalized_url, max_bytes=self.max_file_size_bytes
                            )
                        else:
                            content = await fetch_bytes(
                                normalized_url,
                                30.0,
                                policy=self._url_policy,
                                max_bytes=self.max_file_size_bytes,
                            )
                        # Dual decode path: H.264/H.265 via NVDEC (hardware); other
                        # codecs via the vendor cv2 loader. NVDEC failure falls back.
                        nvdec_video = None
                        codec = probe_video_codec(content)
                        if should_use_nvdec(codec):
                            try:
                                nvdec_video = await asyncio.to_thread(
                                    _nvdec_video_data, content, self.num_video_frames
                                )
                            except Exception as exc:  # noqa: BLE001 - fall back
                                logging.warning(
                                    "NVDEC decode failed (%s); using the vendor "
                                    "video decoder",
                                    exc,
                                )
                        if nvdec_video is not None:
                            videos.append(nvdec_video)
                        else:
                            with tempfile.NamedTemporaryFile(
                                suffix=".mp4"
                            ) as video_file:
                                await asyncio.to_thread(video_file.write, content)
                                await asyncio.to_thread(video_file.flush)
                                try:
                                    videos.append(
                                        await async_load_video(
                                            video_file.name, self.num_video_frames
                                        )
                                    )
                                except ImportError as exc:
                                    # The vendor loader needs cv2, which the
                                    # image deliberately omits; its bare error
                                    # names neither codec nor remedy. Carry its
                                    # text as the cause so the underlying
                                    # reason still reaches the client.
                                    raise video_decoder_missing(
                                        "trtllm",
                                        "opencv-python-headless",
                                        "cv2",
                                        codec,
                                        cause=str(exc),
                                    ) from exc
                    else:
                        try:
                            videos.append(
                                await async_load_video(
                                    normalized_url, self.num_video_frames
                                )
                            )
                        except ImportError as exc:
                            # No bytes fetched on this branch, so no codec probe.
                            raise video_decoder_missing(
                                "trtllm",
                                "opencv-python-headless",
                                "cv2",
                                None,
                                cause=str(exc),
                            ) from exc
                except UrlValidationError as e:
                    raise HttpStatusError(400, str(e), source) from e
                except (HttpStatusError, HttpConfigurationError):
                    # An untrusted egress proxy is deployment configuration,
                    # not a bad request. It carries no status, so the generic
                    # handler below would make it a 400.
                    raise
                except MissingMediaDecoderError as e:
                    # A missing decoder is deployment configuration, not a bad
                    # request: 500, not the 400 the generic handler below
                    # assigns. The actionable text (codec, bounded spec,
                    # installer command, vendor cause) is the message.
                    raise HttpStatusError(
                        500, f"Failed to load video ({source}): {e}", source
                    ) from e
                except Exception as e:
                    status = getattr(e, "status", None) or getattr(e, "code", None)
                    raise HttpStatusError(
                        status if isinstance(status, int) and status >= 400 else 400,
                        f"Failed to load video ({source}): {e}",
                        source,
                    ) from e
            if videos:
                if self._prefers_np_video():
                    # CPU work that grows with the frame count: run it off the
                    # event loop, as the decode is.
                    videos = [
                        (await asyncio.to_thread(_video_frames_to_uint8, v))
                        if isinstance(v.frames[0], torch.Tensor)
                        else v
                        for v in videos
                    ]
                processed_mm_data["video"] = videos
                logging.info("Loaded %d video(s)", len(videos))

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
                if len(processed_mm_data) > 1:
                    # With more than one modality, TRT-LLM needs the items'
                    # prompt order to give each item its own tokens and
                    # embeddings. Without it, its multimodal hashing counts
                    # the first modality's tokens only, and the assertion
                    # that catches this stops the engine.
                    processed_inputs["mm_item_order"] = _mm_item_order(
                        request.get("multi_modal_order"),
                        {m: len(items) for m, items in processed_mm_data.items()},
                        self._prompt_modality_order(),
                    )

                # TRT-LLM echoes these UUIDs into its KV-reuse events, so the
                # router's per-image identity matches what the worker caches.
                images = processed_mm_data.get("image")
                mm_hashes = extra_args.get("mm_hashes")
                if (
                    images
                    and isinstance(mm_hashes, list)
                    and len(mm_hashes) == len(images)
                ):
                    processed_inputs["multi_modal_uuids"] = {"image": list(mm_hashes)}

        # Get token_ids from request (already tokenized by Rust frontend)
        token_ids = token_ids_to_list(request.get("token_ids"))
        if not token_ids:
            logging.warning("No token_ids in request")
            return None
        processed_inputs["prompt_token_ids"] = token_ids

        # Post-expansion prompt length, so an omitted max_tokens can be sized
        # against the real context usage rather than the unexpanded placeholders.
        mm_data = processed_inputs.get("multi_modal_data")
        # Skipped when the request overrides the processor: the sizing
        # calculator is not override-aware (Qwen2-VL ignores the kwargs while
        # counting) and is not guaranteed non-mutating (Gemma-4 writes them
        # into class-level defaults). Falling back to the engine default beats
        # a stale or leaked estimate. Also skipped when the request has video:
        # the estimate counts image tokens only, and would drop the video's.
        # Video-only requests already get no estimate.
        expanded_len = (
            None
            if processed_inputs.get("mm_processor_kwargs")
            else self._expanded_prompt_len(
                token_ids,
                mm_data.get("image") if mm_data and not mm_data.get("video") else None,
            )
        )
        if expanded_len is not None:
            processed_inputs["expanded_prompt_len"] = expanded_len

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
