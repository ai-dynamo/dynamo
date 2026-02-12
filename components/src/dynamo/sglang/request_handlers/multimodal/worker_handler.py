# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import sglang as sgl
import torch

import dynamo.nixl_connect as connect
from dynamo._core import Client, Component, Context
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.sglang.args import Config, DisaggregationMode
from dynamo.sglang.protocol import (
    DisaggSglangMultimodalRequest,
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


class MultimodalConfig:
    """Configuration specific to multimodal processing"""

    EMBEDDINGS_DTYPE = torch.float16
    EMBEDDINGS_DEVICE = "cpu"


class SglangUtils:
    """General SGLang utilities (not multimodal-specific)"""

    @staticmethod
    def build_sampling_params(request: SglangMultimodalRequest) -> dict:
        """Build sampling parameters for SGLang engine (generic functionality)"""
        sampling_params = {}

        # Extract sampling options from request
        sampling_options = request.request.sampling_options
        stop_conditions = request.request.stop_conditions

        if sampling_options.temperature is not None:
            sampling_params["temperature"] = sampling_options.temperature
        if sampling_options.top_p is not None:
            sampling_params["top_p"] = sampling_options.top_p
        if sampling_options.top_k is not None:
            sampling_params["top_k"] = sampling_options.top_k
        if stop_conditions.max_tokens:
            sampling_params["max_new_tokens"] = stop_conditions.max_tokens
        if stop_conditions.ignore_eos:
            sampling_params["ignore_eos"] = stop_conditions.ignore_eos

        logger.debug(f"Sampling params: {sampling_params}")
        return sampling_params


class EmbeddingsProcessor:
    """Handles multimodal embeddings processing and multimodal item creation"""

    def __init__(self):
        self._connector = None

    async def initialize(self):
        """Initialize the connector for embeddings processing"""
        self._connector = connect.Connector()

    async def process_embeddings(self, request: SglangMultimodalRequest):
        """Process embeddings from serialized request"""

        logger.debug(
            "Processing embeddings with shapes: "
            f"{[group.embeddings_shape for group in request.multimodal_inputs]}"
        )

        if self._connector is None:
            logger.warning(
                "Connector is None - this should not happen after initialization"
            )
            self._connector = connect.Connector()

        multimodal_groups = request.multimodal_inputs

        if not multimodal_groups:
            raise ValueError("multimodal_inputs is required")

        embeddings_list = []
        descriptor_list = []

        for mm_group in multimodal_groups:
            serialized_request = mm_group.serialized_request
            embeddings_shape = mm_group.embeddings_shape
            if serialized_request is None:
                raise ValueError("serialized_request is required")
            if embeddings_shape is None:
                raise ValueError("embeddings_shape is required")
            if len(embeddings_shape) < 2:
                raise ValueError(f"Invalid embeddings shape: {embeddings_shape}")

            embeddings = torch.empty(
                embeddings_shape,
                dtype=MultimodalConfig.EMBEDDINGS_DTYPE,
                device=MultimodalConfig.EMBEDDINGS_DEVICE,
            )

            descriptor = connect.Descriptor(embeddings)
            if descriptor is None:
                raise RuntimeError("Descriptor is None - cannot process embeddings")

            read_op = await self._connector.begin_read(serialized_request, descriptor)
            await read_op.wait_for_completion()

            embeddings_list.append(embeddings)
            descriptor_list.append(descriptor)

        return embeddings_list, descriptor_list

    @staticmethod
    def concat_embeddings(embeddings_list: list[torch.Tensor]) -> torch.Tensor:
        """Normalize per-image embeddings to 2D and concatenate along token axis."""
        normalized = []
        for idx, embeddings in enumerate(embeddings_list):
            if embeddings.dim() == 3 and embeddings.shape[0] == 1:
                embeddings = embeddings.squeeze(0)
            if embeddings.dim() != 2:
                raise ValueError(
                    f"Expected 2D embeddings for image {idx}, got shape {tuple(embeddings.shape)}"
                )
            normalized.append(embeddings)

        if not normalized:
            raise ValueError("No embeddings to concatenate")

        return torch.cat(normalized, dim=0)

    @staticmethod
    def create_multimodal_item(embeddings: torch.Tensor, image_grid_thw) -> dict:
        """
        Create multimodal item for SGLang generation.

        Uses format="precomputed_embedding" since Dynamo's Encoder has already
        run the vision encoder. SGLang expects 2D embeddings (num_patches, hidden_dim).
        """
        precomputed = embeddings.to(MultimodalConfig.EMBEDDINGS_DTYPE)

        # SGLang expects 2D tensor for precomputed_embedding format
        # Encoder outputs 3D (1, num_patches, hidden_dim) for internal consistency
        # Squeeze batch dimension at SGLang boundary
        if precomputed.dim() == 3 and precomputed.shape[0] == 1:
            precomputed = precomputed.squeeze(0)

        grid_thw_tensor = torch.tensor(image_grid_thw)

        mm_item = {
            "format": "precomputed_embedding",
            "feature": precomputed,
            "image_grid_thw": grid_thw_tensor,
            "modality": "IMAGE",
        }

        return mm_item


class StreamProcessor:
    """Unified stream processing for SGLang responses"""

    @staticmethod
    async def process_sglang_stream(stream_source) -> AsyncIterator[str]:
        """Process SGLang stream output.

        With stream_output=True (enforced by Dynamo), SGLang sends disjoint segments
        containing only new tokens since the last output. We pass these through directly.
        """
        try:
            async for res in stream_source:
                try:
                    # With stream_output=True, output_ids contains only new tokens (disjoint)
                    output = {
                        "token_ids": res["output_ids"],
                        "text": res.get("text", ""),
                        "finished": False,
                    }

                    # Check for finish reason
                    finish_reason = res.get("meta_info", {}).get("finish_reason")
                    if finish_reason:
                        output.update(
                            {
                                "finish_reason": normalize_finish_reason(
                                    finish_reason.get("type", "stop")
                                ),
                                "finished": True,
                            }
                        )
                        yield json.dumps(output)
                        break

                    yield json.dumps(output)

                except KeyError as e:
                    logger.error(
                        f"Missing key in SGLang response: {e}, available keys: {list(res.keys())}"
                    )
                    error_output = {
                        "token_ids": [],
                        "finish_reason": "error",
                        "error": f"Missing key: {e}",
                        "finished": True,
                    }
                    yield json.dumps(error_output)
                    break
                except Exception as e:
                    logger.error(f"Error processing SGLang response: {e}")
                    error_output = {
                        "token_ids": [],
                        "finish_reason": "error",
                        "error": str(e),
                        "finished": True,
                    }
                    yield json.dumps(error_output)
                    break

        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            error_output = {
                "token_ids": [],
                "finish_reason": "error",
                "error": str(e),
                "finished": True,
            }
            yield json.dumps(error_output)

    @staticmethod
    def create_bootstrap_info(
        bootstrap_host: str, bootstrap_port: int, bootstrap_room: int
    ) -> dict:
        """Create bootstrap info dictionary"""
        return {
            "bootstrap_host": bootstrap_host,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": bootstrap_room,
        }


class ErrorResponseBuilder:
    """Standardized error response builder"""

    @staticmethod
    def build_error_response(error: Exception, extra_fields=None) -> str:
        """Build standardized error response"""
        response = {
            "token_ids": [],
            "finish_reason": "error",
            "error": str(error),
            "finished": True,
        }
        if extra_fields:
            response.update(extra_fields)
        return json.dumps(response)


async def _build_mm_items(
    request: SglangMultimodalRequest, embeddings_processor: EmbeddingsProcessor
) -> tuple[list[dict], torch.Tensor]:
    """Process embeddings and build a single multimodal item for SGLang."""
    embeddings_list, _ = await embeddings_processor.process_embeddings(request)

    image_grid_thw_list = [group.image_grid_thw for group in request.multimodal_inputs]
    if any(item is None for item in image_grid_thw_list):
        raise ValueError("image_grid_thw is required")
    if len(embeddings_list) != len(image_grid_thw_list):
        raise ValueError("image_grid_thw and embeddings count mismatch")

    combined_embeddings = embeddings_processor.concat_embeddings(embeddings_list)
    mm_items = [
        embeddings_processor.create_multimodal_item(
            combined_embeddings, image_grid_thw_list
        )
    ]

    return mm_items, combined_embeddings


class MultimodalWorkerHandler(BaseWorkerHandler):
    """
    Multimodal worker handler for LLM inference with multimodal data.
    Handles both aggregated and disaggregated modes.
    """

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        prefill_client: Client = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(component, engine, config, None, None, shutdown_event)

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor()

        # Store serving mode and prefill client (like regular SGLang)
        self.serving_mode = config.serving_mode
        self.prefill_client = prefill_client

        # Validate prefill client for disaggregated mode
        if self.serving_mode == DisaggregationMode.DECODE:
            if self.prefill_client is None:
                raise ValueError(
                    "prefill_client must be provided when serving_mode is decode"
                )
            logger.info("Multimodal decode worker handler initialized")
        else:
            logger.info("Multimodal aggregated worker handler initialized")

    async def async_init(self):
        """Initialize async components"""
        await self.embeddings_processor.initialize()

    def _validate_and_parse_request(self, request) -> SglangMultimodalRequest:
        """Validate and parse incoming request"""
        if type(request) is not SglangMultimodalRequest:
            if type(request) is str:
                request_payload = json.loads(request)
                request_payload = self._normalize_embeddings_shape(request_payload)
                request = SglangMultimodalRequest.model_validate(request_payload)
            else:
                request_payload = self._normalize_embeddings_shape(request)
                request = SglangMultimodalRequest.model_validate(request_payload)
        return request

    def _normalize_embeddings_shape(self, request_payload: dict) -> dict:
        """Normalize 2-element embeddings_shape to 3-element by prepending batch dim=1.

        JSON round-trip can strip the leading batch dimension from 3D shapes
        (e.g., [tokens, hidden] should be [1, tokens, hidden]).
        """
        multimodal_inputs = request_payload.get("multimodal_inputs", [])
        for group in multimodal_inputs:
            embeddings_shape = group.get("embeddings_shape")
            if isinstance(embeddings_shape, list) and len(embeddings_shape) == 2:
                group["embeddings_shape"] = [
                    1,
                    embeddings_shape[0],
                    embeddings_shape[1],
                ]
        return request_payload

    async def generate(
        self, request: SglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Generate response using SGLang with multimodal data
        Handles both aggregated and disaggregated modes (following regular SGLang DecodeWorkerHandler pattern)

        Args:
            request: Multimodal request with input and parameters.
            context: Context object for cancellation handling.
        """
        try:
            request = self._validate_and_parse_request(request)

            # Route to appropriate generation method based on serving mode
            if self.serving_mode == DisaggregationMode.DECODE:
                async for output in self._generate_disaggregated(request):
                    yield output
            else:
                async for output in self._generate_aggregated(request):
                    yield output

        except Exception as e:
            logger.error(f"Error in multimodal generation: {e}", exc_info=True)
            yield ErrorResponseBuilder.build_error_response(e)

    async def _generate_disaggregated(
        self, request: SglangMultimodalRequest
    ) -> AsyncIterator[str]:
        """Handle disaggregated mode generation"""
        input_ids = request.request.token_ids
        if not input_ids:
            raise ValueError("input_ids is required")

        sampling_params = SglangUtils.build_sampling_params(request)

        # Request bootstrap info from prefill worker
        bootstrap_info = await self._get_bootstrap_from_prefill(
            request, sampling_params
        )

        # Start decode generation with bootstrap info (no image data needed)
        decode_stream = await self.engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=True,
            bootstrap_host=bootstrap_info["bootstrap_host"],
            bootstrap_port=bootstrap_info["bootstrap_port"],
            bootstrap_room=bootstrap_info["bootstrap_room"],
        )

        async for output in StreamProcessor.process_sglang_stream(decode_stream):
            yield output

    async def _generate_aggregated(
        self, request: SglangMultimodalRequest
    ) -> AsyncIterator[str]:
        """Handle aggregated mode generation"""
        input_ids = request.request.token_ids
        if not input_ids:
            raise ValueError("input_ids is required")

        try:
            sampling_params = SglangUtils.build_sampling_params(request)
            mm_items, combined_embeddings = await _build_mm_items(
                request, self.embeddings_processor
            )
            logger.debug(
                "Generated combined multimodal item with embeddings shape: "
                f"{combined_embeddings.shape}"
            )

            logger.debug(f"Input token sequence length: {len(input_ids)}")

            agg_stream = await self.engine.async_generate(
                input_ids=input_ids,
                image_data=mm_items,
                sampling_params=sampling_params,
                stream=True,
            )

            async for output in StreamProcessor.process_sglang_stream(agg_stream):
                yield output

        except RuntimeError as e:
            if "shape mismatch" in str(e):
                logger.error(
                    "Shape mismatch error - this likely indicates a tokenization/embedding alignment issue"
                )
                logger.error(f"Request token IDs length: {len(input_ids)}")
                logger.error(
                    "Embeddings shape: "
                    f"{[group.embeddings_shape for group in request.multimodal_inputs]}"
                )
                logger.error(f"Token sequence preview: {input_ids[:20]}...")
                error_msg = (
                    f"Multimodal embedding alignment error: {str(e)}. "
                    f"This usually happens when the tokenization changes between requests. "
                    "Token count: "
                    f"{len(input_ids)}, Embedding shape: "
                    f"{[group.embeddings_shape for group in request.multimodal_inputs]}"
                )
                yield ErrorResponseBuilder.build_error_response(RuntimeError(error_msg))
            else:
                yield ErrorResponseBuilder.build_error_response(e)

    async def _get_bootstrap_from_prefill(
        self, request: SglangMultimodalRequest, sampling_params: dict
    ) -> dict:
        """Get bootstrap info from prefill worker"""
        prefill_stream = await self.prefill_client.generate(
            DisaggSglangMultimodalRequest(
                request=request,
                sampling_params=sampling_params,
            ).model_dump_json()
        )

        bootstrap_info = None
        async for info in prefill_stream:
            bootstrap_data = info.data() if hasattr(info, "data") else info
            if isinstance(bootstrap_data, str):
                bootstrap_info = json.loads(bootstrap_data)
            else:
                bootstrap_info = bootstrap_data
            break

        if not bootstrap_info:
            raise RuntimeError("No bootstrap info received from prefill worker")

        return bootstrap_info

    def cleanup(self):
        super().cleanup()
        self.engine.shutdown()
        logger.info("Multimodal worker engine shutdown")


class MultimodalPrefillWorkerHandler(BaseWorkerHandler):
    """
    Multimodal prefill worker handler for disaggregated inference
    Processes multimodal inputs and coordinates with decode worker.
    """

    def __init__(
        self,
        component: Component,
        engine: sgl.Engine,
        config: Config,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(component, engine, config, None, None, shutdown_event)

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor()

        # Get bootstrap info using BootstrapManager
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info(engine)

        logger.info(
            f"Multimodal prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    async def async_init(self):
        """Initialize async components like connector"""
        await self.embeddings_processor.initialize()

    async def generate(
        self, disagg_request: DisaggSglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Handle prefill phase: process multimodal input and provide bootstrap info

        Args:
            disagg_request: Disaggregated multimodal request.
            context: Context object for cancellation handling.
        """
        bootstrap_room = None
        try:
            # Validate and parse request
            disagg_request = self._validate_and_parse_disagg_request(disagg_request)

            # Generate and return bootstrap info first (like regular SGLang)
            bootstrap_room = self._generate_bootstrap_room()
            bootstrap_info = {
                "bootstrap_host": self.bootstrap_host,
                "bootstrap_port": self.bootstrap_port,
                "bootstrap_room": bootstrap_room,
            }

            yield bootstrap_info

            # Process prefill generation
            await self._process_prefill_generation(disagg_request, bootstrap_room)

        except Exception as e:
            logger.error(f"Error in prefill generation: {e}", exc_info=True)
            extra_fields = (
                {"bootstrap_room": bootstrap_room} if bootstrap_room is not None else {}
            )
            yield ErrorResponseBuilder.build_error_response(e, extra_fields)

    def _validate_and_parse_disagg_request(
        self, disagg_request
    ) -> DisaggSglangMultimodalRequest:
        """Validate and parse disaggregated request"""
        if type(disagg_request) is not DisaggSglangMultimodalRequest:
            if type(disagg_request) is str:
                disagg_request = DisaggSglangMultimodalRequest.model_validate_json(
                    disagg_request
                )
            else:
                disagg_request = DisaggSglangMultimodalRequest.model_validate(
                    disagg_request
                )
        return disagg_request

    async def _process_prefill_generation(
        self, disagg_request: DisaggSglangMultimodalRequest, bootstrap_room: int
    ):
        """Process multimodal input and start prefill generation"""
        # Get the SglangMultimodalRequest from the DisaggSglangMultimodalRequest
        request = disagg_request.request
        input_ids = request.request.token_ids
        sampling_params = disagg_request.sampling_params

        # Process embeddings from encode worker using our embeddings processor
        mm_items, _ = await _build_mm_items(request, self.embeddings_processor)

        # Start SGLang prefill generation (like regular SGLang)
        results = await self.engine.async_generate(
            input_ids=input_ids,
            image_data=mm_items,
            sampling_params=sampling_params,
            stream=True,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
            bootstrap_room=bootstrap_room,
        )

        # Consume results without yielding (prefill doesn't return text, just coordinates)
        asyncio.create_task(self._consume_results(results))

    async def _consume_results(self, results):
        """Consume prefill results without returning them (like regular SGLang)"""
        async for _ in results:
            pass

    def cleanup(self):
        super().cleanup()
        self.engine.shutdown()
        logger.info("Multimodal prefill engine shutdown")
