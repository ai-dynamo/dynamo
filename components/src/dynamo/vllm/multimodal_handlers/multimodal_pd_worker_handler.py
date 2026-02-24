# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
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
from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    NixlPersistentEmbeddingReceiver,
)
from dynamo.runtime import Client, DistributedRuntime

from ..args import Config
from ..constants import DisaggregationMode
from ..handlers import BaseWorkerHandler, build_sampling_params
from ..multimodal_utils import (
    MyRequestOutput,
    PatchedTokensPrompt,
    vLLMMultimodalRequest,
)
from ..multimodal_utils.model import is_qwen_vl_model
from ..multimodal_utils.prefill_worker_utils import load_multimodal_embeddings

logger = logging.getLogger(__name__)

IMAGE_URL_KEY = "image_url"
TRANSFER_LOCAL = int(os.getenv("TRANSFER_LOCAL", 1))


class MultimodalPDWorkerHandler(BaseWorkerHandler):
    """Prefill/Decode or Prefill-only worker for multimodal serving"""

    def __init__(
        self,
        runtime,
        engine_client: AsyncLLM,
        config: Config,
        encode_worker_client: Client | None = None,
        decode_worker_client: Client | None = None,
        shutdown_event=None,
        generate_endpoint=None,
    ):
        # Get default_sampling_params from config
        default_sampling_params = (
            config.engine_args.create_model_config().get_diff_sampling_param()
        )

        # Call BaseWorkerHandler.__init__ with proper parameters
        super().__init__(
            runtime,
            engine_client,
            default_sampling_params,
            enable_multimodal=config.enable_multimodal,
            generate_endpoint=generate_endpoint,
            config=config,
            shutdown_event=shutdown_event,
        )

        self.config = config
        self.encode_worker_client = encode_worker_client
        self.decode_worker_client = decode_worker_client
        self.enable_disagg = config.disaggregation_mode == DisaggregationMode.PREFILL
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

        # Create and initialize a dynamo connector for this worker.
        # We'll need this to move data between this worker and remote workers efficiently.
        # Note: This is synchronous initialization, async initialization happens in async_init
        self._connector: connect.Connector | None = (
            None  # Will be initialized in async_init
        )
        # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
        # to be at matching size, need to overwrite nixl connect library
        self.embedding_receiver = (
            LocalEmbeddingReceiver()
            if TRANSFER_LOCAL
            else NixlPersistentEmbeddingReceiver(max_items=0)
        )

        logger.info("Multimodal PD Worker has been initialized")

    async def async_init(self, runtime: DistributedRuntime):
        """Async initialization for connector that requires async setup"""
        # Initialize the connector asynchronously
        self._connector = connect.Connector()
        logger.info("Multimodal PD Worker async initialization completed.")

    def _parse_frontend_request(
        self, raw_request: dict
    ) -> tuple[vLLMMultimodalRequest, list[str]]:
        """Parse a raw frontend dict into a vLLMMultimodalRequest and image URLs.

        The Rust frontend sends a dict with ``token_ids`` and
        ``multi_modal_data`` (containing image URLs). This method extracts
        those fields into a structured request. No I/O is performed here;
        embedding fetching is handled separately by ``_load_multimodal_data``.
        """
        request_id = str(uuid.uuid4().hex)

        image_urls: list[str] = []
        mm_data = raw_request.get("multi_modal_data")
        if mm_data is not None:
            for item in mm_data.get(IMAGE_URL_KEY, []):
                if isinstance(item, dict) and "Url" in item:
                    image_urls.append(item["Url"])

        sampling_params = build_sampling_params(
            raw_request, self.default_sampling_params
        )

        request = vLLMMultimodalRequest(
            engine_prompt=PatchedTokensPrompt(
                prompt_token_ids=raw_request["token_ids"]
            ),
            sampling_params=sampling_params,
            request_id=request_id,
            model=raw_request.get("model"),
        )

        return request, image_urls

    # ── Multimodal data loading ──────────────────────────────────────

    async def _load_multimodal_data(
        self, image_urls: list[str], request_id: str
    ) -> dict[str, Any]:
        """Fetch embeddings from encode workers and load into an engine-ready dict.

        Returns an empty dict when no encode worker is configured or no images
        are present.
        """
        if not self.encode_worker_client or not image_urls:
            return defaultdict(list)

        return await load_multimodal_embeddings(
            self.encode_worker_client,  # type: ignore[arg-type]
            image_urls,
            request_id,
            self.embedding_receiver,
            model=self.config.model,
            embeddings_dtype=self.EMBEDDINGS_DTYPE,
            cache=self.embedding_cache_manager,
        )

    # ── Request metadata finalization ────────────────────────────────

    def _finalize_request_metadata(
        self,
        request: vLLMMultimodalRequest,
        multi_modal_data: dict[str, Any],
    ) -> None:
        """Attach model-specific metadata to the request for the decode worker.

        For Qwen VL (mRoPE) models, captures image grid dimensions and
        embedding shapes so the decode worker can reconstruct
        ``multi_modal_data`` consistently for multiple images.
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

        logger.debug(f"Prepared multimodal data size: {len(multi_modal_data['image'])}")
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
        lora_request = self._resolve_lora_request(request.model)
        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            lora_request=lora_request,
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

        lora_request = self._resolve_lora_request(request.model)
        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=prefill_only_request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=prefill_only_request.sampling_params,
            request_id=prefill_only_request.request_id,
            lora_request=lora_request,
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
        #
        # request.model (LoRA name) is preserved in the serialized request
        # so the decode worker can resolve the same LoRA adapter.
        if lora_request and request.model:
            logger.debug(
                f"Forwarding disaggregated decode with LoRA '{request.model}' "
                f"— ensure the same adapter is loaded on the decode worker."
            )

        async for (
            decode_response
        ) in await self.decode_worker_client.round_robin(  # type: ignore[union-attr]
            request.model_dump_json()
        ):
            output = MyRequestOutput.model_validate_json(decode_response.data())  # type: ignore[attr-defined]
            yield self._serialize_response(output)

    # ── Public entry point ───────────────────────────────────────────

    async def generate(self, raw_request: dict, context):
        """Parse the request, load multimodal data, and run inference."""
        request, image_urls = self._parse_frontend_request(raw_request)
        logger.debug(f"Received PD request: {{ id: {request.request_id} }}.")

        multi_modal_data = await self._load_multimodal_data(
            image_urls, request.request_id
        )
        self._finalize_request_metadata(request, multi_modal_data)

        if self.enable_disagg and self.decode_worker_client:
            async for chunk in self._generate_disagg(request, multi_modal_data):
                yield chunk
        else:
            async for chunk in self._generate_agg(request, multi_modal_data):
                yield chunk
