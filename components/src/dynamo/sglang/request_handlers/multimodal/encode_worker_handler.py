# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import torch
from blake3 import blake3

# MMEncoder chain imports compiled CUDA ops; may fail in CPU-only environments.
try:
    from sglang.srt.disaggregation.encode_server import MMEncoder
    from sglang.srt.managers.schedule_batch import Modality
except (ImportError, OSError):
    MMEncoder = None  # type: ignore[assignment]
    Modality = None  # type: ignore[assignment]
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

from dynamo._core import Client, Context
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    CachedEmbedding,
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal import (
    EMBEDDING_SENDER_FACTORIES,
    ImageLoader,
    VideoLoader,
)
from dynamo.common.multimodal.image_loader import DECODED_VARIANT_KEY, URL_VARIANT_KEY
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.llm import MultimodalEmbeddingCachePublisher
from dynamo.sglang._compat import ensure_sglang_frontend_decoded_video_support
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SglangMultimodalRequest,
)
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler
from dynamo.sglang.request_handlers.llm.mm_disagg_utils import (
    IMAGE_URL_KEY,
    VIDEO_URL_KEY,
    extract_mm_hashes,
)
from dynamo.sglang.request_handlers.multimodal.video_input import as_sglang_video

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CONTENT_HASH_KEY = "content_hash"
MM_REPLACEMENTS_BY_MODALITY_KEY = "mm_replacements_by_modality"


@dataclass(frozen=True)
class _TokenReplacement:
    placeholder_token_id: int
    target_tokens: list[int]
    replacement_tokens: list[int]


@dataclass(frozen=True)
class _ModalityBatch:
    name: str
    media_inputs: list[Any]
    cache_keys: list[Optional[str]]
    prechecked_entries: dict[int, Optional[CachedEmbedding]]
    modality: Any
    token_id: Optional[int]
    grid_attr: str
    url_attr: str


class MultimodalEncodeWorkerHandler(BaseWorkerHandler[SglangMultimodalRequest, str]):
    """
    Handler for multimodal encode worker component that processes images/videos
    and forwards them to the downstream worker.

    Receives pre-tokenized requests from the Rust frontend (ModelInput.Tokens)
    with token_ids and multi_modal_data containing image/video URLs or
    frontend-decoded media. Encodes media via MMEncoder, expands placeholder
    tokens, transfers embeddings, and forwards to the PD worker.
    """

    def __init__(
        self,
        config: Config,
        pd_worker_client: Client,
        cache_publisher: MultimodalEmbeddingCachePublisher | None = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        super().__init__(engine=None, config=config, shutdown_event=shutdown_event)
        self.pd_worker_client = pd_worker_client
        self._cache_publisher = cache_publisher
        self.model = config.server_args.model_path
        self._missing_video_cache_key_config_warned = False
        self._decoded_content_hash_warning_emitted = False
        self._image_loader: Optional[ImageLoader] = (
            ImageLoader(enable_frontend_decoding=True)
            if config.dynamo_args.frontend_decoding
            else None
        )
        self._video_loader: Optional[VideoLoader] = (
            VideoLoader(enable_frontend_decoding=True)
            if config.dynamo_args.frontend_decoding
            else None
        )

        if MMEncoder is None:
            raise RuntimeError(
                "MMEncoder is not available. "
                "Multimodal encode worker requires a CUDA environment."
            )

        if self._video_loader is not None:
            ensure_sglang_frontend_decoded_video_support()

        # torch.distributed requires a dist_init_method even for tp=1;
        # port 0 lets the OS assign a free port.
        self.encoder = MMEncoder(
            server_args=config.server_args,
            dist_init_method="tcp://127.0.0.1:0",
            rank=0,
        )

        # Load tokenizer to convert image token string to integer ID
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=config.server_args.trust_remote_code
        )

        # Get image/video token strings and resolve them to integer IDs
        template = chat_templates[getattr(config.server_args, "chat_template")].copy()
        image_token_str = template.image_token

        image_token_id = self._resolve_mm_token_id(
            image_token_str, preferred_token="<|image_pad|>"
        )
        if image_token_id is None:
            raise ValueError("image token is not defined in chat template")
        self.image_token_id = image_token_id

        self.video_token_id: Optional[int] = self._resolve_mm_token_id(
            getattr(template, "video_token", None), preferred_token="<|video_pad|>"
        )

        self.min_workers = 1

        sender = EMBEDDING_SENDER_FACTORIES.get(
            config.dynamo_args.embedding_transfer_mode
        )
        if sender is None:
            raise ValueError(
                "Invalid embedding transfer mode: "
                f"{config.dynamo_args.embedding_transfer_mode}"
            )
        self.embedding_sender = sender()

        # Optional CPU-side LRU embedding cache
        self._embedding_cache: MultimodalEmbeddingCacheManager | None = None
        capacity_gb = config.dynamo_args.multimodal_embedding_cache_capacity_gb
        if capacity_gb > 0:
            capacity_bytes = int(capacity_gb * 1024**3)
            self._embedding_cache = MultimodalEmbeddingCacheManager(capacity_bytes)
            logger.info("Multimodal embedding cache enabled: %.2f GB", capacity_gb)

    def _publish_cache_delta(
        self, added_keys: list[str], removed_keys: list[str]
    ) -> None:
        if self._cache_publisher is None or (not added_keys and not removed_keys):
            return
        try:
            self._cache_publisher.publish_delta(added_keys, removed_keys)
        except Exception:
            logger.warning(
                "Failed to publish embedding cache delta; "
                "routing cache state may be stale",
                exc_info=True,
            )

    def cleanup(self) -> None:
        pass

    @staticmethod
    def _url_hash(url: str) -> str:
        """Stable blake3 hash of a media URL, used as embedding cache key."""
        return blake3(url.encode()).hexdigest()

    @classmethod
    def _normalize_cache_key_value(cls, value: Any) -> Any:
        """Convert nested config values into a stable JSON-serializable form."""
        if isinstance(value, dict):
            return {
                str(key): cls._normalize_cache_key_value(nested_value)
                for key, nested_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_cache_key_value(item) for item in value]
        if isinstance(value, torch.Tensor):
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _media_cache_key(self, url: str, modality: Any, encoder: Any) -> str:
        """Build a cache key that is URL-stable for images and config-aware for video."""
        if modality != Modality.VIDEO:
            return self._url_hash(url)

        video_config = {}
        missing_config_fields: list[str] = []
        vision_config = getattr(encoder, "vision_config", None)
        if isinstance(vision_config, dict):
            video_config = self._normalize_cache_key_value(
                vision_config.get("video", {})
            )
        else:
            missing_config_fields.append("vision_config")

        if missing_config_fields and not getattr(
            self, "_missing_video_cache_key_config_warned", False
        ):
            logger.warning(
                "Video embedding cache key could not include encoder %s; "
                "cache reuse may not reflect all video processor settings.",
                ", ".join(missing_config_fields),
            )
            self._missing_video_cache_key_config_warned = True

        cache_key_payload = {
            "url": url,
            "video_config": video_config,
        }
        return self._url_hash(
            json.dumps(cache_key_payload, sort_keys=True, separators=(",", ":"))
        )

    def _resolve_mm_token_id(
        self, token_str: Optional[str], preferred_token: Optional[str] = None
    ) -> Optional[int]:
        if not token_str:
            return None

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        token_id = self.tokenizer.convert_tokens_to_ids(token_str)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            return token_id

        # For templates like qwen2-vl, modality placeholders are composite
        # strings and need to be resolved to inner pad-token IDs.
        candidates: list[str] = []
        if preferred_token and preferred_token in token_str:
            candidates.append(preferred_token)

        for marker in ("<|image_pad|>", "<|video_pad|>"):
            if marker in token_str and marker not in candidates:
                candidates.append(marker)

        for candidate in candidates:
            candidate_id = self.tokenizer.convert_tokens_to_ids(candidate)
            if isinstance(candidate_id, int) and candidate_id >= 0:
                return candidate_id

        return None

    @staticmethod
    def _ensure_batched_grid(grid_dim: Any, item_count: int) -> list:
        grid_list = (
            grid_dim.tolist() if isinstance(grid_dim, torch.Tensor) else grid_dim
        )
        if (
            item_count == 1
            and isinstance(grid_list, list)
            and len(grid_list) == 3
            and not isinstance(grid_list[0], list)
        ):
            # SGLang may squeeze the batch dimension for a single media item.
            # Normalize that flat THW grid to the batched shape used below.
            return [grid_list]
        return grid_list

    @staticmethod
    def _grid_units(grid_item: Any, modality: str) -> int:
        if modality not in ("IMAGE", "VIDEO"):
            raise ValueError(f"Unsupported modality for grid units: {modality}")
        if not isinstance(grid_item, list) or len(grid_item) != 3:
            raise ValueError(f"Invalid {modality.lower()} grid: {grid_item}")
        return int(grid_item[0] * grid_item[1] * grid_item[2])

    def _split_token_counts(
        self, grid_list: list, total_tokens: int, modality: str
    ) -> list[int]:
        """Compute per-item token counts for a modality from encoder grid metadata."""
        if total_tokens <= 0:
            raise ValueError("Invalid token count for embeddings")

        grid_sizes = [self._grid_units(item, modality) for item in grid_list]
        total_grid_tokens = sum(grid_sizes)
        if total_grid_tokens <= 0:
            raise ValueError("Invalid grid statistics for embeddings")
        if total_grid_tokens % total_tokens != 0:
            raise ValueError(
                "Cannot infer merge factor: grid token total is not divisible "
                "by embedding token total"
            )
        merge_factor = total_grid_tokens // total_tokens
        token_counts = []
        for grid_count in grid_sizes:
            if grid_count % merge_factor != 0:
                raise ValueError(
                    "Cannot split embeddings: per-item grid token count not "
                    "divisible by inferred merge factor"
                )
            token_counts.append(grid_count // merge_factor)
        if sum(token_counts) != total_tokens:
            raise ValueError(
                "Cannot split embeddings: per-item token counts do not match "
                "embedding token total"
            )
        return token_counts

    @staticmethod
    def _jsonable_media_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.item() if value.ndim == 0 else value.tolist()
        return value

    @classmethod
    def _aux_value_for_item(
        cls,
        value: Any,
        index: int,
        item_count: int,
    ) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            if len(value) == item_count:
                return cls._jsonable_media_value(value[index])
            if item_count == 1:
                return cls._jsonable_media_value(value)
            raise ValueError(
                "Auxiliary media metadata length mismatch: "
                f"expected {item_count} items, got {len(value)}"
            )
        return cls._jsonable_media_value(value)

    async def _encode_with_cache(
        self,
        media_inputs: list[Any],
        cache_keys: list[Optional[str]],
        modality: Any,
        prechecked_entries: Optional[dict[int, Optional[CachedEmbedding]]] = None,
    ) -> tuple[Any, torch.Tensor, list[CachedEmbedding]]:
        """Cache-aware multimodal encoding.

        Cache keys are computed before this method so URL inputs and
        frontend-decoded pixels can share the same encoding path. Items without
        a key are encoded normally and omitted from the cache.
        """
        cache = self._embedding_cache
        if cache is None:
            raise RuntimeError("_encode_with_cache requires an enabled embedding cache")
        if len(media_inputs) != len(cache_keys):
            raise ValueError(
                "Media input/cache key count mismatch: "
                f"{len(media_inputs)} inputs, {len(cache_keys)} keys"
            )

        modality_name = getattr(modality, "name", str(modality))
        cached: dict[int, CachedEmbedding] = {}
        prechecked_entries = prechecked_entries or {}
        uncached_indices: list[int] = []
        uncached_inputs: list[Any] = []

        for i, (media_input, cache_key) in enumerate(zip(media_inputs, cache_keys)):
            hit = (
                prechecked_entries[i]
                if i in prechecked_entries
                else cache.get(cache_key)
                if cache_key is not None
                else None
            )
            if hit is not None:
                logger.info("Embedding cache hit for %s index %d", modality_name, i)
                cached[i] = hit
            else:
                if media_input is None:
                    raise RuntimeError(
                        f"{modality_name} cache miss has no materialized media input"
                    )
                uncached_indices.append(i)
                uncached_inputs.append(media_input)

        new_entries: dict[int, CachedEmbedding] = {}
        # SGLang's _encode outputs are already on CPU; use CPU as target for consistency
        target_device = torch.device("cpu")
        if uncached_inputs:
            grid_dim, new_embeddings, aux_data = await self.encoder._encode(
                uncached_inputs, modality
            )
            # Verify SGLang output is on CPU as expected
            if new_embeddings.device != target_device:
                logger.warning(
                    f"SGLang _encode returned embeddings on {new_embeddings.device}, "
                    f"expected CPU. Moving to CPU."
                )
                new_embeddings = new_embeddings.to(target_device)
            grid_list = self._ensure_batched_grid(grid_dim, len(uncached_inputs))
            if not (
                isinstance(new_embeddings, torch.Tensor) and new_embeddings.ndim == 2
            ):
                raise ValueError(
                    f"Unsupported embeddings type from encoder: {type(new_embeddings)}"
                )
            token_counts = self._split_token_counts(
                grid_list, new_embeddings.shape[0], modality_name
            )
            split_tensors = torch.split(new_embeddings, token_counts, dim=0)
            item_count = len(grid_list)
            for local_idx, (orig_idx, tensor, grid_thw) in enumerate(
                zip(uncached_indices, split_tensors, grid_list)
            ):
                entry_kwargs: dict[str, Any] = {"tensor": tensor.contiguous()}
                if modality_name == "IMAGE":
                    entry_kwargs["image_grid_thw"] = grid_thw
                elif modality_name == "VIDEO":
                    entry_kwargs["video_grid_thw"] = grid_thw
                    if aux_data:
                        entry_kwargs["second_per_grid_ts"] = self._aux_value_for_item(
                            aux_data.get("second_per_grid_ts"),
                            local_idx,
                            item_count,
                        )
                        entry_kwargs["video_timestamps"] = self._aux_value_for_item(
                            aux_data.get("video_timestamps"),
                            local_idx,
                            item_count,
                        )
                else:
                    raise ValueError(f"Unsupported multimodal modality: {modality}")
                entry = CachedEmbedding(**entry_kwargs)
                cache_key = cache_keys[orig_idx]
                if cache_key is not None:
                    mutation = cache.set_with_delta(cache_key, entry)
                    self._publish_cache_delta(
                        mutation.added_keys, mutation.removed_keys
                    )
                new_entries[orig_idx] = entry

        # Reassemble results in original input order.
        all_grid_thw: list = []
        all_entries: list[CachedEmbedding] = []
        embedding_parts: list[torch.Tensor] = []
        for i in range(len(media_inputs)):
            entry = cached[i] if i in cached else new_entries[i]
            grid_thw = (
                entry.image_grid_thw
                if modality_name == "IMAGE"
                else entry.video_grid_thw
            )
            if grid_thw is None:
                raise ValueError(
                    f"{modality_name.lower()}_grid_thw is required for cached item"
                )
            all_grid_thw.append(grid_thw)
            all_entries.append(entry)
            embedding_parts.append(entry.tensor)

        full_embeddings = torch.cat(embedding_parts, dim=0)
        return torch.tensor(all_grid_thw), full_embeddings, all_entries

    def _extract_media_inputs(
        self, request: Dict[str, Any]
    ) -> tuple[list[Any], list[Any]]:
        """
        Extract image and video wire items from a PreprocessedRequest.

        The Rust frontend populates multi_modal_data with the format:
            {"image_url": [{"Url": "https://..."} | {"Decoded": {...}}, ...],
             "video_url": [{"Url": "https://..."} | {"Decoded": {...}}, ...]}

        Returns:
            Tuple of image and video wire items. Decoded variants are loaded
            asynchronously by the corresponding preparation methods.
        """
        mm_data = request.get("multi_modal_data")
        if not mm_data:
            raise ValueError("multi_modal_data is required for the encode worker.")
        if not isinstance(mm_data, dict):
            raise ValueError("multi_modal_data must be an object.")

        image_items = mm_data.get(IMAGE_URL_KEY, [])
        if not isinstance(image_items, list):
            raise ValueError("multi_modal_data.image_url must be a list.")
        video_items = mm_data.get(VIDEO_URL_KEY, [])
        if not isinstance(video_items, list):
            raise ValueError("multi_modal_data.video_url must be a list.")

        if not image_items and not video_items:
            raise ValueError(
                "multi_modal_data must contain image_url or video_url entries."
            )

        return list(image_items), list(video_items)

    @staticmethod
    def _parse_media_item(item: Any, media_name: str) -> tuple[str, Any]:
        """Return the single wire variant and value for one media item."""
        if isinstance(item, str):
            return URL_VARIANT_KEY, item
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported {media_name} data variant: {item}")

        variants = [
            key for key in (URL_VARIANT_KEY, DECODED_VARIANT_KEY) if key in item
        ]
        if len(variants) != 1:
            raise ValueError(f"Unsupported {media_name} data variant: {item}")
        variant = variants[0]
        return variant, item[variant]

    @staticmethod
    def _decoded_content_cache_key(metadata: Any) -> Optional[str]:
        """Read the canonical key computed by the Rust media decoder."""
        if not isinstance(metadata, dict):
            return None
        key = metadata.get(CONTENT_HASH_KEY)
        if not isinstance(key, str) or len(key) != 16:
            return None
        if any(char not in "0123456789abcdef" for char in key):
            return None
        return key

    def _warn_invalid_decoded_content_hash(self, media_name: str) -> None:
        if self._decoded_content_hash_warning_emitted:
            return
        logger.warning(
            "Frontend-decoded %s descriptor has a missing or invalid canonical "
            "content_hash; this item will bypass the Dynamo embedding cache. "
            "Ensure the frontend and encode worker use compatible Dynamo "
            "versions and the descriptor is not corrupted.",
            media_name,
        )
        self._decoded_content_hash_warning_emitted = True

    async def _prepare_image_inputs(
        self, image_items: list[Any]
    ) -> tuple[list[Any], list[Optional[str]], dict[int, Optional[CachedEmbedding]],]:
        """Prepare MMEncoder inputs and aligned embedding-cache keys.

        URL variants stay as strings so the existing SGLang loading path is
        unchanged. Decoded variants are read from NIXL and become PIL Images.
        Their cache keys come from the canonical content hash serialized by the
        Rust media decoder.
        """
        if not image_items:
            return [], [], {}

        encoder_inputs: list[Any] = [None] * len(image_items)
        cache_keys: list[Optional[str]] = [None] * len(image_items)
        prechecked_entries: dict[int, Optional[CachedEmbedding]] = {}
        decoded_items: list[Dict[str, Any]] = []
        decoded_indices: list[int] = []
        cache = self._embedding_cache
        image_loader = self._image_loader

        for index, item in enumerate(image_items):
            variant, value = self._parse_media_item(item, "image")
            if variant == URL_VARIANT_KEY:
                url = value
                if not isinstance(url, str):
                    raise ValueError(f"Unsupported image data variant: {item}")
                encoder_inputs[index] = url
                if cache is not None:
                    cache_keys[index] = self._url_hash(url)
                continue

            if not isinstance(value, dict):
                raise ValueError(f"Unsupported image data variant: {item}")
            if image_loader is None:
                raise ValueError(
                    "Received frontend-decoded images but --frontend-decoding "
                    "is not enabled on the multimodal encode worker."
                )

            if cache is not None:
                cache_key = self._decoded_content_cache_key(value)
                cache_keys[index] = cache_key
                if cache_key is None:
                    self._warn_invalid_decoded_content_hash("image")
                else:
                    cached_entry = cache.get(cache_key)
                    prechecked_entries[index] = cached_entry
                    if cached_entry is not None:
                        continue

            decoded_items.append({DECODED_VARIANT_KEY: value})
            decoded_indices.append(index)

        if decoded_items:
            if image_loader is None:
                raise RuntimeError("Frontend image loader is not initialized")
            decoded_images = await image_loader.load_image_batch(decoded_items)
            if len(decoded_images) != len(decoded_indices):
                raise ValueError(
                    "Decoded image count mismatch: "
                    f"expected {len(decoded_indices)}, got {len(decoded_images)}"
                )
            for index, image in zip(decoded_indices, decoded_images):
                encoder_inputs[index] = image

        return encoder_inputs, cache_keys, prechecked_entries

    async def _prepare_video_inputs(
        self, video_items: list[Any]
    ) -> tuple[list[Any], list[Optional[str]], dict[int, Optional[CachedEmbedding]]]:
        """Prepare URL or frontend-decoded videos for SGLang's MMEncoder."""
        if not video_items:
            return [], [], {}

        encoder_inputs: list[Any] = [None] * len(video_items)
        cache_keys: list[Optional[str]] = [None] * len(video_items)
        prechecked_entries: dict[int, Optional[CachedEmbedding]] = {}
        decoded_items: list[Dict[str, Any]] = []
        decoded_indices: list[int] = []
        cache = self._embedding_cache
        video_loader = self._video_loader

        for index, item in enumerate(video_items):
            variant, value = self._parse_media_item(item, "video")
            if variant == URL_VARIANT_KEY:
                url = value
                if not isinstance(url, str):
                    raise ValueError(f"Unsupported video data variant: {item}")
                encoder_inputs[index] = url
                if cache is not None:
                    cache_keys[index] = self._media_cache_key(
                        url, Modality.VIDEO, self.encoder
                    )
                continue

            if not isinstance(value, dict):
                raise ValueError(f"Unsupported video data variant: {item}")
            if video_loader is None:
                raise ValueError(
                    "Received frontend-decoded videos but --frontend-decoding "
                    "is not enabled on the multimodal encode worker."
                )

            if cache is not None:
                cache_key = self._decoded_content_cache_key(value)
                cache_keys[index] = cache_key
                if cache_key is None:
                    self._warn_invalid_decoded_content_hash("video")
                else:
                    cached_entry = cache.get(cache_key)
                    prechecked_entries[index] = cached_entry
                    if cached_entry is not None:
                        continue

            decoded_items.append({DECODED_VARIANT_KEY: value})
            decoded_indices.append(index)

        if decoded_items:
            if video_loader is None:
                raise RuntimeError("Frontend video loader is not initialized")
            decoded_videos = await video_loader.load_video_batch(decoded_items)
            if len(decoded_videos) != len(decoded_indices):
                raise ValueError(
                    "Decoded video count mismatch: "
                    f"expected {len(decoded_indices)}, got {len(decoded_videos)}"
                )
            for index, (frames, metadata) in zip(decoded_indices, decoded_videos):
                encoder_inputs[index] = as_sglang_video(frames, metadata)

        return encoder_inputs, cache_keys, prechecked_entries

    @staticmethod
    def _extract_video_replacements(
        request: Dict[str, Any],
    ) -> list[_TokenReplacement] | None:
        """Parse exact worker token replacements produced by the Rust frontend."""
        extra_args = request.get("extra_args")
        if not isinstance(extra_args, dict):
            return None
        grouped = extra_args.get(MM_REPLACEMENTS_BY_MODALITY_KEY)
        if grouped is None:
            return None
        if not isinstance(grouped, dict) or not isinstance(grouped.get("video"), list):
            logger.warning(
                "Ignoring malformed extra_args.%s video replacements",
                MM_REPLACEMENTS_BY_MODALITY_KEY,
            )
            return None

        replacements: list[_TokenReplacement] = []
        for item in grouped["video"]:
            if not isinstance(item, dict):
                logger.warning("Ignoring malformed frontend video token replacement")
                return None
            placeholder_token_id = item.get("placeholder_token_id")
            target_tokens = item.get("target_tokens")
            replacement_tokens = item.get("replacement_tokens")
            if (
                type(placeholder_token_id) is not int
                or not isinstance(target_tokens, list)
                or not target_tokens
                or not all(type(token) is int for token in target_tokens)
                or not isinstance(replacement_tokens, list)
                or not replacement_tokens
                or not all(type(token) is int for token in replacement_tokens)
            ):
                logger.warning("Ignoring malformed frontend video token replacement")
                return None
            replacements.append(
                _TokenReplacement(
                    placeholder_token_id=placeholder_token_id,
                    target_tokens=target_tokens,
                    replacement_tokens=replacement_tokens,
                )
            )
        return replacements or None

    @staticmethod
    def _find_token_sequence(
        token_ids: list[int], target: list[int], start: int
    ) -> int | None:
        end = len(token_ids) - len(target) + 1
        for index in range(start, end):
            if token_ids[index : index + len(target)] == target:
                return index
        return None

    @classmethod
    def _apply_video_replacements(
        cls,
        token_ids: list[int],
        replacements: list[_TokenReplacement],
        token_counts: list[int],
        video_token_id: int,
    ) -> list[int] | None:
        """Apply all video replacements atomically, returning None on mismatch."""
        if len(replacements) != len(token_counts):
            logger.warning(
                "Frontend video replacement count (%d) does not match video count (%d); "
                "falling back to SGLang's legacy placeholder expansion",
                len(replacements),
                len(token_counts),
            )
            return None

        rebuilt: list[int] = []
        cursor = 0
        for replacement, expected_tokens in zip(replacements, token_counts):
            if replacement.placeholder_token_id != video_token_id:
                logger.warning(
                    "Frontend video replacement token id %d does not match worker token "
                    "id %d; falling back to legacy placeholder expansion",
                    replacement.placeholder_token_id,
                    video_token_id,
                )
                return None
            actual_tokens = replacement.replacement_tokens.count(video_token_id)
            if actual_tokens != expected_tokens:
                logger.warning(
                    "Frontend video replacement has %d embedding tokens but SGLang "
                    "encoded %d; falling back to legacy placeholder expansion",
                    actual_tokens,
                    expected_tokens,
                )
                return None

            target_index = cls._find_token_sequence(
                token_ids, replacement.target_tokens, cursor
            )
            if target_index is None:
                logger.warning(
                    "Frontend video replacement target is absent from worker token ids; "
                    "falling back to legacy placeholder expansion"
                )
                return None
            rebuilt.extend(token_ids[cursor:target_index])
            rebuilt.extend(replacement.replacement_tokens)
            cursor = target_index + len(replacement.target_tokens)

        rebuilt.extend(token_ids[cursor:])
        return rebuilt

    @_nvtx.range_decorator("mm:enc:generate", color="blue")
    async def generate(
        self, raw_request: Dict[str, Any], context: Context
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Encode media from a pre-tokenized multimodal request, expand placeholder
        tokens, transfer embeddings via NIXL, and stream PD worker responses.

        The Rust frontend (ModelInput.Tokens) sends a PreprocessedRequest dict
        with token_ids and multi_modal_data. This handler:
        1. Extracts URL inputs and reads frontend-decoded media from NIXL.
        2. Runs vision encoding via MMEncoder.
        3. Expands image placeholder tokens to match patch counts.
        4. Creates a NIXL descriptor for embedding transfer.
        5. Forwards the request to the PD worker and streams responses back.

        Args:
            raw_request: PreprocessedRequest dict from the Rust frontend.
            context: Context object for cancellation handling.
        """
        if isinstance(raw_request, str):
            raw_request = json.loads(raw_request)

        # Keep URL inputs on SGLang's existing loading path and materialize only
        # frontend-decoded media received through NIXL.
        image_items, video_items = self._extract_media_inputs(raw_request)
        (
            image_inputs,
            image_cache_keys,
            image_prechecked_entries,
        ) = await self._prepare_image_inputs(image_items)
        (
            video_inputs,
            video_cache_keys,
            video_prechecked_entries,
        ) = await self._prepare_video_inputs(video_items)

        routing_hashes = extract_mm_hashes(raw_request)
        video_replacements = self._extract_video_replacements(raw_request)

        # Build MultiModalGroup objects for the downstream SglangMultimodalRequest.
        multimodal_groups = [
            MultiModalGroup(
                multimodal_input=MultiModalInput(
                    image_url=value if isinstance(value, str) else None
                )
            )
            for value in image_inputs
        ] + [
            MultiModalGroup(
                multimodal_input=MultiModalInput(
                    video_url=value if isinstance(value, str) else None
                )
            )
            for value in video_inputs
        ]
        preprocessed_request = PreprocessedRequest.model_validate(raw_request)

        forwarded_hashes = routing_hashes
        if forwarded_hashes is not None and len(forwarded_hashes) != len(
            multimodal_groups
        ):
            logger.warning(
                "Frontend MM hash count (%d) does not match encoded media count (%d); "
                "letting SGLang recompute hashes",
                len(forwarded_hashes),
                len(multimodal_groups),
            )
            forwarded_hashes = None
        if video_inputs and forwarded_hashes is not None and video_replacements is None:
            logger.warning(
                "Frontend video hashes arrived without exact token replacements; "
                "letting SGLang recompute hashes"
            )
            forwarded_hashes = None

        # Build SglangMultimodalRequest from the pre-tokenized request
        request = SglangMultimodalRequest(
            request=preprocessed_request,
            multimodal_inputs=multimodal_groups,
        )

        try:
            transfer_future = None
            combined_embeddings_parts: list[torch.Tensor] = []

            # Build modality-local metadata in the same order as multimodal_groups.
            modality_batches = [
                _ModalityBatch(
                    name="IMAGE",
                    media_inputs=image_inputs,
                    cache_keys=image_cache_keys,
                    prechecked_entries=image_prechecked_entries,
                    modality=Modality.IMAGE,
                    token_id=self.image_token_id,
                    grid_attr="image_grid_thw",
                    url_attr="image_url",
                ),
                _ModalityBatch(
                    name="VIDEO",
                    media_inputs=video_inputs,
                    cache_keys=video_cache_keys,
                    prechecked_entries=video_prechecked_entries,
                    modality=Modality.VIDEO,
                    token_id=self.video_token_id,
                    grid_attr="video_grid_thw",
                    url_attr="video_url",
                ),
            ]

            group_offset = 0
            for batch in modality_batches:
                modality_name = batch.name
                media_inputs = batch.media_inputs
                modality_enum = batch.modality
                token_id = batch.token_id
                if not media_inputs:
                    continue
                if token_id is None:
                    raise ValueError(
                        f"{modality_name.lower()} token is not defined in chat template"
                    )

                aux_data: dict[str, Any] | None = None
                cached_entries: list[CachedEmbedding] | None = None
                with _nvtx.annotate("mm:enc:vision_encode", color="red"):
                    if self._embedding_cache is not None:
                        (
                            grid_dim,
                            embeddings,
                            cached_entries,
                        ) = await self._encode_with_cache(
                            media_inputs,
                            batch.cache_keys,
                            modality_enum,
                            prechecked_entries=batch.prechecked_entries,
                        )
                    else:
                        grid_dim, embeddings, aux_data = await self.encoder._encode(
                            list(media_inputs), modality_enum
                        )

                grid_list = self._ensure_batched_grid(grid_dim, len(media_inputs))

                if not isinstance(grid_list, list) or len(grid_list) != len(
                    media_inputs
                ):
                    raise ValueError(
                        f"{modality_name.lower()} grid size mismatch: "
                        f"expected {len(media_inputs)} items, got {grid_list}"
                    )

                if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                    raise ValueError(
                        f"Unsupported embeddings type from encoder: {type(embeddings)}"
                    )

                token_counts = self._split_token_counts(
                    grid_list, embeddings.shape[0], modality_name
                )

                placeholder_count = request.request.token_ids.count(token_id)
                if placeholder_count < len(media_inputs):
                    raise ValueError(
                        f"Not enough {modality_name.lower()} placeholders in token_ids"
                    )

                group_slice = multimodal_groups[
                    group_offset : group_offset + len(media_inputs)
                ]
                for idx, (mm_group, grid_item, token_count) in enumerate(
                    zip(group_slice, grid_list, token_counts)
                ):
                    setattr(mm_group, batch.grid_attr, grid_item)
                    mm_group.num_mm_tokens = int(token_count)
                    if modality_name == "VIDEO":
                        if cached_entries is not None:
                            mm_group.second_per_grid_ts = cached_entries[
                                idx
                            ].second_per_grid_ts
                            mm_group.video_timestamps = cached_entries[
                                idx
                            ].video_timestamps
                        elif aux_data:
                            mm_group.second_per_grid_ts = self._aux_value_for_item(
                                aux_data.get("second_per_grid_ts"),
                                idx,
                                len(media_inputs),
                            )
                            mm_group.video_timestamps = self._aux_value_for_item(
                                aux_data.get("video_timestamps"),
                                idx,
                                len(media_inputs),
                            )
                    if mm_group.multimodal_input is not None:
                        setattr(mm_group.multimodal_input, batch.url_attr, None)

                exact_video_tokens = None
                if modality_name == "VIDEO" and video_replacements is not None:
                    exact_video_tokens = self._apply_video_replacements(
                        request.request.token_ids,
                        video_replacements,
                        token_counts,
                        token_id,
                    )

                if exact_video_tokens is not None:
                    request.request.token_ids = exact_video_tokens
                else:
                    if modality_name == "VIDEO":
                        forwarded_hashes = None
                    search_start = 0
                    for num_tokens in token_counts:
                        try:
                            token_index = request.request.token_ids.index(
                                token_id, search_start
                            )
                        except ValueError as e:
                            raise ValueError(
                                f"Not enough {modality_name.lower()} tokens found "
                                "for provided inputs"
                            ) from e

                        request.request.token_ids = (
                            request.request.token_ids[:token_index]
                            + [token_id] * num_tokens
                            + request.request.token_ids[token_index + 1 :]
                        )
                        search_start = token_index + num_tokens

                combined_embeddings_parts.append(embeddings)
                group_offset += len(media_inputs)

            # _ModalityBatch shares this list, so clearing it releases decoded
            # media buffers before the generator awaits the downstream stream.
            image_inputs.clear()
            image_prechecked_entries.clear()
            video_inputs.clear()
            video_prechecked_entries.clear()

            if forwarded_hashes is not None:
                request.mm_hashes = forwarded_hashes

            if combined_embeddings_parts:
                precomputed_embeddings = torch.cat(combined_embeddings_parts, dim=0)
                request.embeddings_shape = tuple(precomputed_embeddings.shape)  # type: ignore[assignment]
                request.transfer_payload = None

                with _nvtx.annotate("mm:enc:embedding_transfer", color="purple"):
                    (
                        transfer_request,
                        transfer_future,
                    ) = await self.embedding_sender.send_embeddings(
                        precomputed_embeddings
                    )
                    request.transfer_payload = transfer_request
            # Get the response generator from downstream worker
            payload = request.model_dump_json()
            response_generator = await self.pd_worker_client.round_robin(
                payload, context=context
            )

            # Parse PD worker responses and yield as LLMEngineOutput-
            # compatible dicts for the Rust frontend to post-process.
            async for response in response_generator:
                raw = response.data() if hasattr(response, "data") else str(response)
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    logger.warning("Non-JSON response from PD worker: %r", raw[:200])
                    data = {"token_ids": [], "text": raw}
                # Strip the internal 'finished' flag — the Rust frontend
                # uses 'finish_reason' (present when finished=True).
                data.pop("finished", None)
                # Remove empty 'text' so the Rust frontend detokenizes
                # from token_ids instead of using the empty string.
                if not data.get("text"):
                    data.pop("text", None)
                yield data

            if transfer_future is not None:
                await transfer_future

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
