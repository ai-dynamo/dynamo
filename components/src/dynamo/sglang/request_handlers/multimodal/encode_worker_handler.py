# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import torch

# MMEncoder chain imports compiled CUDA ops; may fail in CPU-only environments.
try:
    from sglang.srt.disaggregation.encode_server import MMEncoder
except (ImportError, OSError):
    MMEncoder = None  # type: ignore[assignment]
from sglang.srt.parser.conversation import chat_templates
from transformers import AutoTokenizer

from dynamo._core import Client, Context
from dynamo.common.multimodal import EMBEDDING_SENDER_FACTORIES
from dynamo.common.utils import nvtx_utils as _nvtx
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

        # Load tokenizer to convert image token string to integer ID
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )

        # Get image token string and handle it properly
        image_token_str = (
            chat_templates[getattr(config.server_args, "chat_template")]
            .copy()
            .image_token
        )

        # For Qwen2.5-VL, the image token might be multiple tokens
        if image_token_str == "<|vision_start|><|image_pad|><|vision_end|>":
            # These are likely the individual special tokens for Qwen2.5-VL
            image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            assert isinstance(
                image_pad_id, int
            ), f"Expected int token id, got {type(image_pad_id)}"

            # Use the image_pad token as the main image token
            self.image_token_id: int = image_pad_id
        else:
            # Fallback for other models
            token_id = self.tokenizer.convert_tokens_to_ids(image_token_str)
            assert isinstance(
                token_id, int
            ), f"Expected int token id, got {type(token_id)}"
            self.image_token_id = token_id

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

    def cleanup(self) -> None:
        pass

    def _extract_image_urls(self, request: Dict[str, Any]) -> list[str]:
        """
        Extract image URLs from the multi_modal_data field of a PreprocessedRequest.

        The Rust frontend populates multi_modal_data with the format:
            {"image_url": [{"Url": "https://..."}, ...]}
        """
        mm_data = request.get("multi_modal_data")
        if not mm_data:
            raise ValueError("multi_modal_data is required for the encode worker.")

        image_items = mm_data.get(IMAGE_URL_KEY)
        if not image_items:
            raise ValueError("multi_modal_data must contain image_url entries.")

        image_urls: list[str] = []
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
                raise ValueError(f"Unsupported multimodal data variant: {item}")

        return image_urls

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

        # Extract image URLs from the frontend's multi_modal_data
        image_urls = self._extract_image_urls(raw_request)

        # Build MultiModalGroup objects for the downstream SglangMultimodalRequest
        multimodal_groups = [
            MultiModalGroup(multimodal_input=MultiModalInput(image_url=url))
            for url in image_urls
        ]

        # Build SglangMultimodalRequest from the pre-tokenized request
        request = SglangMultimodalRequest(
            request=PreprocessedRequest(**raw_request),
            multimodal_inputs=multimodal_groups,
        )

        try:
            with _nvtx.annotate("mm:enc:vision_encode", color="red"):
                image_grid_dim, precomputed_embeddings = await self.encoder._encode(
                    image_urls
                )

            image_grid_thw_list = (
                image_grid_dim.tolist()
                if isinstance(image_grid_dim, torch.Tensor)
                else image_grid_dim
            )

            if len(image_grid_thw_list) != len(multimodal_groups):
                raise ValueError("image_grid_thw size mismatch")

            def _build_token_counts(total_tokens: int) -> list[int]:
                if total_tokens <= 0:
                    raise ValueError("Invalid token statistics for embeddings")

                # image_grid_thw is [t, h, w]. We derive per-item relative sizes
                # from spatial grid (h * w), then infer merge factor
                # from the total embedding token count.
                grid_sizes = []
                for image_grid_thw in image_grid_thw_list:
                    if not isinstance(image_grid_thw, list) or len(image_grid_thw) != 3:
                        raise ValueError(
                            "Cannot split embeddings: invalid image_grid_thw"
                        )
                    grid_sizes.append(int(image_grid_thw[1] * image_grid_thw[2]))

                total_grid_tokens = sum(grid_sizes)
                if total_grid_tokens <= 0:
                    raise ValueError("Invalid grid statistics for embeddings")

                if total_grid_tokens % total_tokens != 0:
                    raise ValueError(
                        "Cannot infer merge factor: grid token total is not divisible by embedding token total"
                    )

                merge_factor = total_grid_tokens // total_tokens
                token_counts = []
                for grid_count in grid_sizes:
                    if grid_count % merge_factor != 0:
                        raise ValueError(
                            "Cannot split embeddings: per-image grid token count not divisible by inferred merge factor"
                        )
                    token_counts.append(grid_count // merge_factor)

                if sum(token_counts) != total_tokens:
                    raise ValueError(
                        "Cannot split embeddings: per-image token counts do not match embedding token total"
                    )

                return token_counts

            if isinstance(precomputed_embeddings, torch.Tensor):
                if precomputed_embeddings.ndim != 2:
                    raise ValueError(
                        "Unsupported embeddings tensor rank from encoder: "
                        f"{precomputed_embeddings.ndim}. Expected 2D [tokens, hidden]."
                    )

                token_counts = _build_token_counts(precomputed_embeddings.shape[0])
            else:
                raise ValueError(
                    "Unsupported embeddings type from encoder: "
                    f"{type(precomputed_embeddings)}"
                )

            image_placeholder_count = request.request.token_ids.count(
                self.image_token_id
            )
            if image_placeholder_count < len(multimodal_groups):
                raise ValueError(
                    "Not enough image placeholders in token_ids for provided images"
                )

            # Keep per-image grid metadata in request groups for worker-side mm_item.
            for idx, (mm_group, image_grid_thw) in enumerate(
                zip(multimodal_groups, image_grid_thw_list)
            ):
                mm_group.image_grid_thw = image_grid_thw
                if mm_group.multimodal_input is not None:
                    mm_group.multimodal_input.image_url = None

            # Store shared tensor transfer metadata at request level.
            request.embeddings_shape = tuple(precomputed_embeddings.shape)  # type: ignore[assignment]
            request.transfer_payload = None

            search_start = 0
            for num_image_tokens in token_counts:
                try:
                    image_token_id_index = request.request.token_ids.index(
                        self.image_token_id, search_start
                    )
                except ValueError as e:
                    raise ValueError(
                        "Not enough image tokens found for provided images"
                    ) from e

                request.request.token_ids = (
                    request.request.token_ids[:image_token_id_index]
                    + [self.image_token_id] * num_image_tokens
                    + request.request.token_ids[image_token_id_index + 1 :]
                )
                search_start = image_token_id_index + num_image_tokens

            with _nvtx.annotate("mm:enc:embedding_transfer", color="purple"):
                (
                    transfer_request,
                    transfer_future,
                ) = await self.embedding_sender.send_embeddings(precomputed_embeddings)
                request.transfer_payload = transfer_request
                logger.debug(f"Request: {request.model_dump_json()}")

            # Get the response generator from downstream worker
            response_generator = await self.pd_worker_client.round_robin(
                request.model_dump_json()
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
