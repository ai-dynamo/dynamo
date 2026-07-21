# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
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
from dynamo.common.multimodal import EMBEDDING_SENDER_FACTORIES
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.llm import MultimodalEmbeddingCachePublisher
from dynamo.sglang.args import Config
from dynamo.sglang.protocol import (
    MultiModalGroup,
    MultiModalInput,
    PreprocessedRequest,
    SglangMultimodalRequest,
)
from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

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

IMAGE_URL_KEY = "image_url"
VIDEO_URL_KEY = "video_url"


def _ensure_video_decoder_available() -> None:
    """Fail fast with actionable guidance when the SGLang video decoder is absent.

    SGLang decodes video with ``decord`` (via ``MMEncoder``). The runtime image
    ships without it for media-licensing reasons, so a video request would
    otherwise fail deep inside SGLang with a bare ``No module named 'decord'``.
    """
    import importlib.util

    if importlib.util.find_spec("decord") is None:
        raise RuntimeError(
            "Video decoding is unavailable in this image: the SGLang video path "
            "requires the 'decord' decoder, which is not installed. Install it "
            "with `pip install decord2` (it provides the `decord` module), or "
            "send images instead of video."
        )


class MultimodalEncodeWorkerHandler(BaseWorkerHandler[SglangMultimodalRequest, str]):
    """
    Handler for multimodal encode worker component that processes images/videos
    and forwards them to the downstream worker.

    Receives pre-tokenized requests from the Rust frontend (ModelInput.Tokens)
    with token_ids and multi_modal_data containing image URLs. Encodes images
    via MMEncoder, expands placeholder tokens, transfers embeddings via NIXL,
    and forwards to the PD worker.
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

        if MMEncoder is None:
            raise RuntimeError(
                "MMEncoder is not available. "
                "Multimodal encode worker requires a CUDA environment."
            )

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
        self, media_urls: list[str], modality: Any
    ) -> tuple[Any, torch.Tensor, list[CachedEmbedding]]:
        """Cache-aware multimodal encoding.

        Checks the CPU LRU cache per media URL. Uncached URLs are batch-encoded,
        split per item, stored in cache, then reassembled with the cached hits in
        the original URL order.
        """
        assert self._embedding_cache is not None

        modality_name = getattr(modality, "name", str(modality))
        cached: dict[int, CachedEmbedding] = {}
        uncached_indices: list[int] = []
        uncached_urls: list[str] = []

        cache_keys = [
            self._media_cache_key(url, modality, self.encoder) for url in media_urls
        ]

        for i, (url, cache_key) in enumerate(zip(media_urls, cache_keys)):
            hit = self._embedding_cache.get(cache_key)
            if hit is not None:
                logger.info("Embedding cache hit for %s URL index %d", modality_name, i)
                cached[i] = hit
            else:
                uncached_indices.append(i)
                uncached_urls.append(url)

        new_entries: dict[int, CachedEmbedding] = {}
        # SGLang's _encode outputs are already on CPU; use CPU as target for consistency
        target_device = torch.device("cpu")
        if uncached_urls:
            grid_dim, new_embeddings, aux_data = await self.encoder._encode(
                uncached_urls, modality
            )
            # Verify SGLang output is on CPU as expected
            if new_embeddings.device != target_device:
                logger.warning(
                    f"SGLang _encode returned embeddings on {new_embeddings.device}, "
                    f"expected CPU. Moving to CPU."
                )
                new_embeddings = new_embeddings.to(target_device)
            grid_list = self._ensure_batched_grid(grid_dim, len(uncached_urls))
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
            for local_idx, (orig_idx, _url, tensor, grid_thw) in enumerate(
                zip(uncached_indices, uncached_urls, split_tensors, grid_list)
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
                mutation = self._embedding_cache.set_with_delta(
                    cache_keys[orig_idx], entry
                )
                self._publish_cache_delta(mutation.added_keys, mutation.removed_keys)
                new_entries[orig_idx] = entry

        # Reassemble results in original URL order
        all_grid_thw: list = []
        all_entries: list[CachedEmbedding] = []
        embedding_parts: list[torch.Tensor] = []
        for i in range(len(media_urls)):
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

    def _extract_media_urls(
        self, request: Dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """
        Extract image/video URLs from the multi_modal_data field of a PreprocessedRequest.

        The Rust frontend populates multi_modal_data with the format:
            {"image_url": [{"Url": "https://..."}, ...], "video_url": [{"Url": "https://..."}, ...]}

        Returns:
            Tuple of (image_urls, video_urls) lists.
        """
        mm_data = request.get("multi_modal_data")
        if not mm_data:
            raise ValueError("multi_modal_data is required for the encode worker.")

        image_items = mm_data.get(IMAGE_URL_KEY, [])
        video_items = mm_data.get(VIDEO_URL_KEY, [])

        if not image_items and not video_items:
            raise ValueError(
                "multi_modal_data must contain image_url or video_url entries."
            )

        image_urls: list[str] = []
        video_urls: list[str] = []

        # Extract image URLs
        for item in image_items:
            if isinstance(item, str):
                image_urls.append(item)
            elif isinstance(item, dict) and "Url" in item:
                image_urls.append(item["Url"])
            elif isinstance(item, dict) and "Decoded" in item:
                raise ValueError(
                    "Frontend-decoded media (Decoded variant) is incompatible "
                    "with the multimodal encode worker. The encode worker "
                    "requires image URLs to run vision encoding via MMEncoder. "
                    "Disable --frontend-decoding when using EPD serving."
                )
            else:
                raise ValueError(f"Unsupported image data variant: {item}")

        # Extract video URLs
        for item in video_items:
            if isinstance(item, str):
                video_urls.append(item)
            elif isinstance(item, dict) and "Url" in item:
                video_urls.append(item["Url"])
            elif isinstance(item, dict) and "Decoded" in item:
                raise ValueError(
                    "Frontend-decoded media (Decoded variant) is incompatible "
                    "with the current SGLang EPD video path. Video inputs are "
                    "URL passthrough only in EPD mode and do not accept Decoded "
                    "payloads in the encode worker. Disable --frontend-decoding "
                    "or use URL-based video_url inputs."
                )
            else:
                raise ValueError(f"Unsupported video data variant: {item}")

        # Only after the payload is validated (malformed / Decoded-variant items
        # already rejected above) preflight the video decoder, so a genuine video
        # URL request fails with actionable guidance instead of a bare
        # "No module named 'decord'" from SGLang.
        if video_urls:
            _ensure_video_decoder_available()

        return image_urls, video_urls

    @_nvtx.range_decorator("mm:enc:generate", color="blue")
    async def generate(
        self, raw_request: Dict[str, Any], context: Context
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Encode images from a pre-tokenized multimodal request, expand placeholder
        tokens, transfer embeddings via NIXL, and stream PD worker responses.

        The Rust frontend (ModelInput.Tokens) sends a PreprocessedRequest dict
        with token_ids and multi_modal_data. This handler:
        1. Extracts image URLs from multi_modal_data.
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

        # Extract image/video URLs from the frontend's multi_modal_data
        image_urls, video_urls = self._extract_media_urls(raw_request)

        # Build MultiModalGroup objects for the downstream SglangMultimodalRequest.
        multimodal_groups = [
            MultiModalGroup(multimodal_input=MultiModalInput(image_url=url))
            for url in image_urls
        ] + [
            MultiModalGroup(multimodal_input=MultiModalInput(video_url=url))
            for url in video_urls
        ]
        preprocessed_request = PreprocessedRequest.model_validate(raw_request)

        # Build SglangMultimodalRequest from the pre-tokenized request
        request = SglangMultimodalRequest(
            request=preprocessed_request,
            multimodal_inputs=multimodal_groups,
        )

        try:
            transfer_future = None
            combined_embeddings_parts: list[torch.Tensor] = []

            # Build modality-local metadata in the same order as multimodal_groups.
            modality_specs = [
                (
                    "IMAGE",
                    image_urls,
                    Modality.IMAGE,
                    self.image_token_id,
                    "image_grid_thw",
                    "image_url",
                ),
                (
                    "VIDEO",
                    video_urls,
                    Modality.VIDEO,
                    self.video_token_id,
                    "video_grid_thw",
                    "video_url",
                ),
            ]

            group_offset = 0
            for (
                modality_name,
                urls,
                modality_enum,
                token_id,
                grid_attr,
                url_attr,
            ) in modality_specs:
                if not urls:
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
                        ) = await self._encode_with_cache(urls, modality_enum)
                    else:
                        grid_dim, embeddings, aux_data = await self.encoder._encode(
                            urls, modality_enum
                        )

                grid_list = self._ensure_batched_grid(grid_dim, len(urls))

                if not isinstance(grid_list, list) or len(grid_list) != len(urls):
                    raise ValueError(
                        f"{modality_name.lower()} grid size mismatch: "
                        f"expected {len(urls)} items, got {grid_list}"
                    )

                if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
                    raise ValueError(
                        f"Unsupported embeddings type from encoder: {type(embeddings)}"
                    )

                token_counts = self._split_token_counts(
                    grid_list, embeddings.shape[0], modality_name
                )

                placeholder_count = request.request.token_ids.count(token_id)
                if placeholder_count < len(urls):
                    raise ValueError(
                        f"Not enough {modality_name.lower()} placeholders in token_ids"
                    )

                group_slice = multimodal_groups[group_offset : group_offset + len(urls)]
                for idx, (mm_group, grid_item, token_count) in enumerate(
                    zip(group_slice, grid_list, token_counts)
                ):
                    setattr(mm_group, grid_attr, grid_item)
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
                                len(urls),
                            )
                            mm_group.video_timestamps = self._aux_value_for_item(
                                aux_data.get("video_timestamps"),
                                idx,
                                len(urls),
                            )
                    if mm_group.multimodal_input is not None:
                        setattr(mm_group.multimodal_input, url_attr, None)

                search_start = 0
                for num_tokens in token_counts:
                    try:
                        token_index = request.request.token_ids.index(
                            token_id, search_start
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Not enough {modality_name.lower()} tokens found for provided inputs"
                        ) from e

                    request.request.token_ids = (
                        request.request.token_ids[:token_index]
                        + [token_id] * num_tokens
                        + request.request.token_ids[token_index + 1 :]
                    )
                    search_start = token_index + num_tokens

                combined_embeddings_parts.append(embeddings)
                group_offset += len(urls)

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
