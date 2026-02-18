# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import uuid
from collections import defaultdict
from typing import Any

import torch
from vllm.inputs.data import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

import dynamo.nixl_connect as connect
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.runtime import Client, Component, DistributedRuntime

from ..args import Config
from ..handlers import BaseWorkerHandler, build_sampling_params
from ..multimodal_utils import (
    MultiModalGroup,
    MyRequestOutput,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)
from ..multimodal_utils.model import is_qwen_vl_model
from ..multimodal_utils.prefill_worker_utils import (
    IMAGE_URL_KEY,
    accumulate_embeddings,
    fetch_embeddings_from_encode_workers,
    load_embeddings,
)

logger = logging.getLogger(__name__)


class MultimodalPDWorkerHandler(BaseWorkerHandler):
    """Prefill/Decode or Prefill-only worker for multimodal serving"""

    def __init__(
        self,
        runtime,
        component: Component,
        engine_client: AsyncLLM,
        config: Config,
        encode_worker_client: Client | None = None,
        decode_worker_client: Client | None = None,
        shutdown_event=None,
    ):
        # Get default_sampling_params from config
        default_sampling_params = (
            config.engine_args.create_model_config().get_diff_sampling_param()
        )

        # Call BaseWorkerHandler.__init__ with proper parameters
        super().__init__(
            runtime,
            component,
            engine_client,
            default_sampling_params,
            enable_multimodal=config.enable_multimodal,
            shutdown_event=shutdown_event,
        )

        self.config = config
        self.encode_worker_client = encode_worker_client
        self.decode_worker_client = decode_worker_client
        self.enable_disagg = config.is_prefill_worker
        self.embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None
        if config.multimodal_embedding_cache_capacity_gb > 0:
            capacity_bytes = int(
                config.multimodal_embedding_cache_capacity_gb * 1024**3
            )
            self.embedding_cache_manager = MultimodalEmbeddingCacheManager(
                capacity_bytes
            )

        # Initialize multimodal-specific components
        logger.info("Multimodal PD Worker startup started.")

        if "video" in self.config.model.lower():
            self.EMBEDDINGS_DTYPE = torch.uint8
        else:
            self.EMBEDDINGS_DTYPE = torch.float16

        self.EMBEDDINGS_DEVICE = "cpu"

        # Create and initialize a dynamo connector for this worker.
        # We'll need this to move data between this worker and remote workers efficiently.
        # Note: This is synchronous initialization, async initialization happens in async_init
        self._connector: connect.Connector | None = (
            None  # Will be initialized in async_init
        )

        logger.info("Multimodal PD Worker has been initialized")

    async def async_init(self, runtime: DistributedRuntime):
        """Async initialization for connector that requires async setup"""
        # Initialize the connector asynchronously
        self._connector = connect.Connector()
        logger.info("Multimodal PD Worker async initialization completed.")

    async def _build_request_from_frontend(
        self, raw_request: dict
    ) -> vLLMMultimodalRequest:
        """Convert a raw frontend dict into a vLLMMultimodalRequest.

        When the PD worker is the direct frontend endpoint (no separate
        processor), the Rust frontend sends a dict representation of PreprocessedRequest.
        This method extracts image URLs, routes them to encode workers if available,
        and assembles the standard request object that the rest of ``generate()`` expects.
        """
        request_id = str(uuid.uuid4().hex)

        # Extract image URLs from the raw frontend dict
        image_urls: list[str] = []
        mm_data = raw_request.get("multi_modal_data")
        if mm_data is not None:
            for item in mm_data.get(IMAGE_URL_KEY, []):
                if isinstance(item, dict) and "Url" in item:
                    image_urls.append(item["Url"])

        multimodal_groups: list[MultiModalGroup] = []
        if self.encode_worker_client and image_urls:
            multimodal_groups = await fetch_embeddings_from_encode_workers(
                self.encode_worker_client,
                image_urls,
                request_id,
            )

        sampling_params = build_sampling_params(
            raw_request, self.default_sampling_params
        )

        return vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(
                prompt_token_ids=raw_request["token_ids"]
            ),
            sampling_params=sampling_params,
            request_id=request_id,
            multimodal_inputs=multimodal_groups,
        )

    # ── Request parsing ────────────────────────────────────────────────

    async def _parse_request(self, request) -> vLLMMultimodalRequest:
        """Normalize any incoming format into a validated vLLMMultimodalRequest.

        Handles three input shapes:
        1. Raw frontend dict  (has ``token_ids`` + ``multi_modal_data``)
        2. JSON string         (from encode worker or other serializers)
        3. Plain dict          (Pydantic-compatible mapping)
        """
        if isinstance(request, dict) and "token_ids" in request:
            return await self._build_request_from_frontend(request)

        if type(request) is vLLMMultimodalRequest:
            return request

        if type(request) is str:
            return vLLMMultimodalRequest.model_validate_json(request)

        return vLLMMultimodalRequest.model_validate(request)

    # ── Multimodal data loading ──────────────────────────────────────

    async def _load_multimodal_data(
        self, request: vLLMMultimodalRequest
    ) -> dict[str, Any]:
        """Load pre-computed embeddings into an engine-ready dict.

        Each ``MultiModalGroup`` carries embeddings from encode workers,
        loaded via NIXL RDMA or local safetensors.

        No-op when --route-to-encoder is not set.
        """
        multimodal_inputs: list[MultiModalGroup] = request.multimodal_inputs or []
        multi_modal_data: dict[str, Any] = defaultdict(list)

        for mi in multimodal_inputs:
            embeddings = await load_embeddings(
                mi,
                self.EMBEDDINGS_DTYPE,
                self.EMBEDDINGS_DEVICE,
                self._connector,
            )
            accumulate_embeddings(
                multi_modal_data,
                self.config.model,
                self.EMBEDDINGS_DTYPE,
                embeddings,
                mi.image_grid_thw,
            )

        return multi_modal_data

    # ── Request metadata finalization ────────────────────────────────

    def _finalize_request_metadata(
        self,
        request: vLLMMultimodalRequest,
        multi_modal_data: dict[str, Any],
    ) -> None:
        """Attach model-specific metadata and strip heavy fields from request.

        For Qwen VL (mRoPE) models, captures image grid dimensions and
        embedding shapes so the decode worker can reconstruct
        ``multi_modal_data`` consistently for multiple images.

        Also clears ``multimodal_inputs`` — the raw embeddings / URLs are no
        longer needed once ``multi_modal_data`` is built.
        """
        if is_qwen_vl_model(self.config.model) and isinstance(
            multi_modal_data.get("image"), dict
        ):
            image_data = multi_modal_data["image"]
            image_grid_thw = image_data.get("image_grid_thw")
            image_embeds = image_data.get("image_embeds")
            if image_grid_thw is not None:
                request.image_grid_thw = (
                    image_grid_thw.tolist()
                    if isinstance(image_grid_thw, torch.Tensor)
                    else image_grid_thw
                )
            if image_embeds is not None:
                request.embeddings_shape = list(image_embeds.shape)

        # Use empty list instead of None to satisfy Pydantic validation
        # on decode worker after vllm upgrade.
        request.multimodal_inputs = []

        logger.info(f"Prepared multimodal data size: {len(multi_modal_data['image'])}")
        logger.debug("Multimodal data keys: %s", list(multi_modal_data.keys()))

    # ── Response serialization ───────────────────────────────────────

    @staticmethod
    def _serialize_response(response) -> str:
        """Build a JSON-serialized ``MyRequestOutput`` from an engine response."""
        return MyRequestOutput(
            request_id=response.request_id,
            prompt=response.prompt,
            prompt_token_ids=response.prompt_token_ids,
            prompt_logprobs=response.prompt_logprobs,
            outputs=response.outputs,
            finished=response.finished,
            metrics=response.metrics,
            kv_transfer_params=response.kv_transfer_params,
        ).model_dump_json()

    @staticmethod
    def _format_engine_output(
        response, num_output_tokens_so_far: int
    ) -> dict[str, Any]:
        """Format a vLLM RequestOutput as an LLMEngineOutput-compatible dict.

        This produces the same incremental dict format that the regular
        (non-multimodal) handler yields, which the Rust frontend expects
        after model registration.
        """
        if not response.outputs:
            return {
                "finish_reason": "error: No outputs from vLLM engine",
                "token_ids": [],
            }

        output = response.outputs[0]
        out: dict[str, Any] = {
            "token_ids": output.token_ids[num_output_tokens_so_far:],
        }

        if output.finish_reason:
            # Inline normalization: map vLLM's "abort" to Dynamo's "cancelled"
            finish_reason = output.finish_reason
            if finish_reason.startswith("abort"):
                finish_reason = "cancelled"
            out["finish_reason"] = finish_reason
            out["completion_usage"] = BaseWorkerHandler._build_completion_usage(
                request_output=response,
            )
        if output.stop_reason:
            out["stop_reason"] = output.stop_reason

        return out

    # ── Aggregated generation (prefill + decode locally) ─────────────

    async def _generate_agg(
        self,
        request: vLLMMultimodalRequest,
        multi_modal_data: dict[str, Any],
    ):
        """Run prefill and decode on this worker (aggregated mode)."""
        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

        num_output_tokens_so_far = 0
        async for response in gen:
            logger.debug(f"Response kv_transfer_params: {response.kv_transfer_params}")
            logger.debug(
                f"length of expanded prompt ids: {len(response.prompt_token_ids)}"
            )
            yield self._format_engine_output(response, num_output_tokens_so_far)
            if response.outputs:
                num_output_tokens_so_far = len(response.outputs[0].token_ids)

    # ── Disaggregated generation (prefill here, decode remote) ───────

    async def _generate_disagg(
        self,
        request: vLLMMultimodalRequest,
        multi_modal_data: dict[str, Any],
    ):
        """Prefill locally, then forward to a remote decode worker."""
        # Prepare prefill-only request
        prefill_only_request = copy.deepcopy(request)
        extra_args = prefill_only_request.sampling_params.extra_args or {}
        extra_args["kv_transfer_params"] = {"do_remote_decode": True}
        prefill_only_request.sampling_params.extra_args = extra_args
        prefill_only_request.sampling_params.max_tokens = 1
        prefill_only_request.sampling_params.min_tokens = 1
        logger.debug("Prefill request: %s", prefill_only_request)

        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=prefill_only_request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=prefill_only_request.sampling_params,
            request_id=prefill_only_request.request_id,
        )

        # Drain prefill generator (max_tokens=1, expect a single response)
        async for prefill_response in gen:
            pass

        # Qwen VL (mRoPE): keep the ORIGINAL unexpanded prompt.
        # The decode worker passes multi_modal_data which causes vLLM to
        # expand the prompt identically to prefill, ensuring block counts match.
        #
        # Other models: use the expanded prompt from prefill response.
        # They don't pass multi_modal_data in decode, so they need the
        # already-expanded prompt to match the KV cache layout.
        if not is_qwen_vl_model(self.config.model):
            request.engine_prompt[
                "prompt_token_ids"
            ] = prefill_response.prompt_token_ids

        logger.debug(
            f"Prefill response kv_transfer_params: {prefill_response.kv_transfer_params}"
        )
        extra_args = request.sampling_params.extra_args or {}
        extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
        extra_args.pop("serialized_request", None)
        request.sampling_params.extra_args = extra_args
        logger.debug("Decode request: %s", request)

        # Serialized request is lightweight: token IDs, sampling params with
        # kv_transfer_params, and small Qwen metadata (image_grid_thw,
        # embeddings_shape).  Heavy multimodal data was consumed locally by
        # engine_client.generate() and multimodal_inputs was cleared by
        # `_finalize_request_metadata`.
        async for (
            decode_response
        ) in await self.decode_worker_client.round_robin(  # type: ignore[union-attr]
            request.model_dump_json()
        ):
            output = MyRequestOutput.model_validate_json(decode_response.data())  # type: ignore[attr-defined]
            yield self._serialize_response(output)

    # ── Public entry point ───────────────────────────────────────────

    async def generate(self, request, context):
        """Parse the request, load multimodal data, and run inference."""
        logger.debug(f"Got raw request: {request}")

        request = await self._parse_request(request)
        logger.debug(f"Received PD request: {{ id: {request.request_id} }}.")

        multi_modal_data = await self._load_multimodal_data(request)
        self._finalize_request_metadata(request, multi_modal_data)

        logger.info(
            f"Prepared multimodal data size: {len(multi_modal_data.get('image', []))}"
        )
        logger.debug(f"{multi_modal_data}")

        if self.enable_disagg and self.decode_worker_client:
            async for chunk in self._generate_disagg(request, multi_modal_data):
                yield chunk
        else:
            async for chunk in self._generate_agg(request, multi_modal_data):
                yield chunk
