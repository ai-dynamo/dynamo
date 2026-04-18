# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import torch

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
from dynamo.sglang._compat import mm_encode
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


class MultimodalEncodeWorkerHandler(BaseWorkerHandler[SglangMultimodalRequest, str]):
    """
    Handler for multimodal encode worker component that processes image/video
    and forwards them to the downstream worker.

    Receives pre-tokenized requests from the Rust frontend (ModelInput.Tokens)
    with token_ids and multi_modal_data containing media URLs. Encodes inputs
    via MMEncoder, expands placeholder tokens, transfers embeddings via NIXL,
    and forwards them to the PD worker.
    """

    def __init__(
        self,
        config: Config,
        pd_worker_client: Client,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        super().__init__(engine=None, config=config, shutdown_event=shutdown_event)
        self.pd_worker_client = pd_worker_client
        self.model = config.server_args.model_path

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

        # Load tokenizer to convert multimodal token strings to integer IDs
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )
        template_name = getattr(config.server_args, "chat_template", None)
        template = chat_templates.get(template_name)
        if template is None:
            raise ValueError(
                "Multimodal encode worker requires a chat template registered in "
                f"sglang.srt.parser.conversation.chat_templates; got {template_name!r}"
            )
        template = template.copy()
        self.image_token_id = self._resolve_media_token_id(
            template,
            token_attr="image_token",
            fallback_token="<|image_pad|>",
            required=True,
        )
        self.video_token_id = self._resolve_media_token_id(
            template,
            token_attr="video_token",
            fallback_token="<|video_pad|>",
            required=False,
        )
        self.media_token_ids = {Modality.IMAGE: self.image_token_id}
        if self.video_token_id is not None:
            self.media_token_ids[Modality.VIDEO] = self.video_token_id

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

    def cleanup(self) -> None:
        pass

    @staticmethod
    def _url_hash(url: str) -> str:
        """Stable blake2b hash of an image URL, used as embedding cache key."""
        return hashlib.blake2b(url.encode(), digest_size=32).hexdigest()

    @staticmethod
    def _split_token_counts(grid_list: list, total_tokens: int) -> list[int]:
        """Compute per-item embedding token counts from grid-shape metadata.

        The encoder returns one concatenated embedding tensor for all items in the
        request. We infer the shared merge factor from the ratio of total grid
        elements to total embedding tokens, then split the embedding tensor back
        into per-item segments.
        """
        if total_tokens <= 0:
            raise ValueError("Invalid token count for embeddings")
        grid_sizes = []
        for grid_shape in grid_list:
            if not isinstance(grid_shape, list) or len(grid_shape) < 2:
                raise ValueError(f"Invalid multimodal grid shape: {grid_shape}")
            grid_tensor = torch.as_tensor(grid_shape, dtype=torch.long)
            grid_sizes.append(int(torch.prod(grid_tensor).item()))
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
                    "Cannot split embeddings: per-image grid token count not "
                    "divisible by inferred merge factor"
                )
            token_counts.append(grid_count // merge_factor)
        if sum(token_counts) != total_tokens:
            raise ValueError(
                "Cannot split embeddings: per-image token counts do not match "
                "embedding token total"
            )
        return token_counts

    async def _encode_with_cache(
        self, image_urls: list[str]
    ) -> tuple[Any, torch.Tensor]:
        """Cache-aware vision encoding.

        Checks the CPU LRU cache per URL. Uncached URLs are batch-encoded,
        split per image, stored in cache, then reassembled with the cached
        hits in the original URL order.

        Returns the same (image_grid_dim, embeddings) shape as
        ``self.encoder._encode()``.
        """
        assert self._embedding_cache is not None

        cached: dict[int, CachedEmbedding] = {}
        uncached_indices: list[int] = []
        uncached_urls: list[str] = []

        for i, url in enumerate(image_urls):
            hit = self._embedding_cache.get(self._url_hash(url))
            if hit is not None:
                logger.debug("Embedding cache hit for URL index %d", i)
                cached[i] = hit
            else:
                uncached_indices.append(i)
                uncached_urls.append(url)

        new_entries: dict[int, CachedEmbedding] = {}
        # SGLang's _encode outputs are already on CPU; use CPU as target for consistency
        target_device = torch.device("cpu")
        if uncached_urls:
            grid_dim, new_embeddings, _aux = await mm_encode(
                self.encoder, uncached_urls, Modality.IMAGE
            )
            # Verify SGLang output is on CPU as expected
            if new_embeddings.device != target_device:
                logger.warning(
                    f"SGLang _encode returned embeddings on {new_embeddings.device}, "
                    f"expected CPU. Moving to CPU."
                )
                new_embeddings = new_embeddings.to(target_device)
            grid_list: list = (
                grid_dim.tolist() if isinstance(grid_dim, torch.Tensor) else grid_dim
            )
            if not (
                isinstance(new_embeddings, torch.Tensor) and new_embeddings.ndim == 2
            ):
                raise ValueError(
                    f"Unsupported embeddings type from encoder: {type(new_embeddings)}"
                )
            token_counts = self._split_token_counts(grid_list, new_embeddings.shape[0])
            split_tensors = torch.split(new_embeddings, token_counts, dim=0)
            for orig_idx, url, tensor, grid_thw in zip(
                uncached_indices, uncached_urls, split_tensors, grid_list
            ):
                entry = CachedEmbedding(
                    tensor=tensor.contiguous(),
                    image_grid_thw=grid_thw,
                )
                self._embedding_cache.set(self._url_hash(url), entry)
                new_entries[orig_idx] = entry

        # Reassemble results in original URL order
        all_grid_thw: list = []
        embedding_parts: list[torch.Tensor] = []
        for i in range(len(image_urls)):
            entry = cached[i] if i in cached else new_entries[i]
            all_grid_thw.append(entry.image_grid_thw)
            embedding_parts.append(entry.tensor)

        full_embeddings = torch.cat(embedding_parts, dim=0)
        return torch.tensor(all_grid_thw), full_embeddings

    def _resolve_media_token_id(
        self,
        template: Any,
        token_attr: str,
        fallback_token: str,
        required: bool,
    ) -> int | None:
        """Resolve the single placeholder token ID used for multimodal expansion."""
        token_str = getattr(template, token_attr, None)
        candidates: list[str] = []
        if token_str and fallback_token in token_str:
            candidates.append(fallback_token)
        if token_str:
            candidates.append(token_str)
        elif fallback_token:
            candidates.append(fallback_token)

        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        for candidate in dict.fromkeys(candidates):
            token_id = self.tokenizer.convert_tokens_to_ids(candidate)
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and (unk_token_id is None or token_id != unk_token_id)
            ):
                return token_id

            encoded = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                return encoded[0]

        if required:
            raise ValueError(f"Unable to resolve placeholder token for {token_attr}")
        return None

    @staticmethod
    def _jsonable_media_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.item() if value.ndim == 0 else value.tolist()
        return value

    @classmethod
    def _build_group_model_specific_data(
        cls,
        modality: Any,
        grid_shape: list[Any],
        aux_data: dict[str, Any] | None,
        index: int,
    ) -> dict[str, Any]:
        model_specific_data: dict[str, Any] = {}
        if modality == Modality.IMAGE:
            model_specific_data["image_grid_thw"] = grid_shape
        elif modality == Modality.VIDEO:
            model_specific_data["video_grid_thw"] = grid_shape
            for key, value in (aux_data or {}).items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple, torch.Tensor)):
                    model_specific_data[key] = cls._jsonable_media_value(value[index])
                else:
                    model_specific_data[key] = cls._jsonable_media_value(value)
        else:
            raise ValueError(f"Unsupported multimodal modality: {modality}")
        return model_specific_data

    @staticmethod
    def _expand_placeholder_tokens(
        token_ids: list[int],
        placeholder_token_id: int,
        token_counts: list[int],
        modality_name: str,
    ) -> list[int]:
        expanded_token_ids = token_ids
        search_start = 0
        for num_mm_tokens in token_counts:
            try:
                placeholder_index = expanded_token_ids.index(
                    placeholder_token_id, search_start
                )
            except ValueError as e:
                raise ValueError(
                    f"Not enough {modality_name} tokens found for provided inputs"
                ) from e

            expanded_token_ids = (
                expanded_token_ids[:placeholder_index]
                + [placeholder_token_id] * num_mm_tokens
                + expanded_token_ids[placeholder_index + 1 :]
            )
            search_start = placeholder_index + num_mm_tokens

        return expanded_token_ids

    def _extract_multimodal_inputs(
        self, request: Dict[str, Any]
    ) -> tuple[Any, list[str]]:
        """
        Extract a single supported multimodal input type from multi_modal_data.

        The Rust frontend populates multi_modal_data with the format:
            {"image_url": [{"Url": "https://..."}, ...]}
        """
        mm_data = request.get("multi_modal_data")
        if not mm_data:
            raise ValueError("multi_modal_data is required for the encode worker.")

        image_items = mm_data.get(IMAGE_URL_KEY) or []
        video_items = mm_data.get(VIDEO_URL_KEY) or []
        if image_items and video_items:
            raise ValueError(
                "Mixed image_url and video_url inputs are not supported by the "
                "multimodal encode worker."
            )

        if image_items:
            modality = Modality.IMAGE
            media_items = image_items
        elif video_items:
            modality = Modality.VIDEO
            media_items = video_items
        else:
            raise ValueError(
                "multi_modal_data must contain either image_url or video_url entries."
            )

        media_urls: list[str] = []
        for item in media_items:
            if isinstance(item, str):
                media_urls.append(item)
            elif isinstance(item, dict) and "Url" in item:
                media_urls.append(item["Url"])
            elif isinstance(item, dict) and "Decoded" in item:
                raise ValueError(
                    "Frontend-decoded media (Decoded variant) is incompatible "
                    "with the multimodal encode worker. The encode worker "
                    "requires media URLs to run MMEncoder. Disable "
                    "--frontend-decoding when using EPD serving."
                )
            else:
                raise ValueError(f"Unsupported multimodal data variant: {item}")

        return modality, media_urls

    @_nvtx.range_decorator("mm:enc:generate", color="blue")
    async def generate(
        self, raw_request: Dict[str, Any], context: Context
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Encode multimodal inputs from a pre-tokenized request, expand placeholder
        tokens, transfer embeddings via NIXL, and stream PD worker responses.

        The Rust frontend (ModelInput.Tokens) sends a PreprocessedRequest dict
        with token_ids and multi_modal_data. This handler:
        1. Extracts image/video URLs from multi_modal_data.
        2. Runs vision encoding via MMEncoder.
        3. Expands multimodal placeholder tokens to match patch counts.
        4. Creates a NIXL descriptor for embedding transfer.
        5. Forwards the request to the PD worker and streams responses back.

        Args:
            raw_request: PreprocessedRequest dict from the Rust frontend.
            context: Context object for cancellation handling.
        """
        if isinstance(raw_request, str):
            raw_request = json.loads(raw_request)

        # Extract image/video URLs from the frontend's multi_modal_data
        modality, media_urls = self._extract_multimodal_inputs(raw_request)

        # Build MultiModalGroup objects for the downstream SglangMultimodalRequest
        multimodal_groups = []
        for url in media_urls:
            mm_input = MultiModalInput()
            if modality == Modality.IMAGE:
                mm_input.image_url = url
            elif modality == Modality.VIDEO:
                mm_input.video_url = url
            else:
                raise ValueError(f"Unsupported multimodal modality: {modality}")
            multimodal_groups.append(MultiModalGroup(multimodal_input=mm_input))
        preprocessed_request = PreprocessedRequest.model_validate(raw_request)

        # Build SglangMultimodalRequest from the pre-tokenized request
        request = SglangMultimodalRequest(
            request=preprocessed_request,
            multimodal_inputs=multimodal_groups,
        )

        try:
            aux_data: dict[str, Any] | None = None
            with _nvtx.annotate("mm:enc:vision_encode", color="red"):
                if modality == Modality.IMAGE and self._embedding_cache is not None:
                    (
                        media_grid_dim,
                        precomputed_embeddings,
                    ) = await self._encode_with_cache(media_urls)
                else:
                    (
                        media_grid_dim,
                        precomputed_embeddings,
                        aux_data,
                    ) = await mm_encode(self.encoder, media_urls, modality)

            grid_shape_list = (
                media_grid_dim.tolist()
                if isinstance(media_grid_dim, torch.Tensor)
                else media_grid_dim
            )

            if len(grid_shape_list) != len(multimodal_groups):
                raise ValueError("multimodal grid size mismatch")

            if isinstance(precomputed_embeddings, torch.Tensor):
                if precomputed_embeddings.ndim != 2:
                    raise ValueError(
                        "Unsupported embeddings tensor rank from encoder: "
                        f"{precomputed_embeddings.ndim}. Expected 2D [tokens, hidden]."
                    )
                token_counts = self._split_token_counts(
                    grid_shape_list, precomputed_embeddings.shape[0]
                )
            else:
                raise ValueError(
                    "Unsupported embeddings type from encoder: "
                    f"{type(precomputed_embeddings)}"
                )

            placeholder_token_id = self.media_token_ids.get(modality)
            if placeholder_token_id is None:
                raise ValueError(
                    f"No placeholder token configured for modality {modality.name}"
                )

            placeholder_count = request.request.token_ids.count(
                placeholder_token_id
            )
            if placeholder_count < len(multimodal_groups):
                raise ValueError(
                    f"Not enough {modality.name.lower()} placeholders in token_ids "
                    "for provided inputs"
                )

            # Keep per-item metadata in request groups for worker-side mm_item.
            for idx, (mm_group, grid_shape) in enumerate(
                zip(multimodal_groups, grid_shape_list)
            ):
                mm_group.modality = modality.name
                mm_group.model_specific_data = self._build_group_model_specific_data(
                    modality, grid_shape, aux_data, idx
                )
                if mm_group.multimodal_input is not None:
                    mm_group.multimodal_input.image_url = None
                    mm_group.multimodal_input.video_url = None

            # Store shared tensor transfer metadata at request level.
            request.embeddings_shape = tuple(precomputed_embeddings.shape)  # type: ignore[assignment]
            request.transfer_payload = None

            request.request.token_ids = self._expand_placeholder_tokens(
                request.request.token_ids,
                placeholder_token_id,
                token_counts,
                modality.name.lower(),
            )

            with _nvtx.annotate("mm:enc:embedding_transfer", color="purple"):
                (
                    transfer_request,
                    transfer_future,
                ) = await self.embedding_sender.send_embeddings(precomputed_embeddings)
                request.transfer_payload = transfer_request
                logger.debug(f"Request: {request.model_dump_json()}")

            # Get the response generator from downstream worker
            response_generator = await self.pd_worker_client.round_robin(
                request.model_dump_json(), context=context
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

            await transfer_future

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
