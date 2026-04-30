# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Optional

import sglang as sgl
import torch

from dynamo._core import Client, Context
from dynamo.common.constants import DisaggregationMode, EmbeddingTransferMode
from dynamo.common.multimodal import EMBEDDING_RECEIVER_FACTORIES, TransferRequest
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.otel_tracing import build_trace_headers
from dynamo.sglang.args import Config
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
        if sampling_options.n is not None:
            sampling_params["n"] = sampling_options.n
        if stop_conditions.max_tokens:
            sampling_params["max_new_tokens"] = stop_conditions.max_tokens
        if stop_conditions.ignore_eos:
            sampling_params["ignore_eos"] = stop_conditions.ignore_eos

        logger.debug(f"Sampling params: {sampling_params}")
        return sampling_params


class EmbeddingsProcessor:
    """Handles multimodal embeddings processing and multimodal item creation"""

    def __init__(self, embedding_transfer_mode: EmbeddingTransferMode):
        receiver = EMBEDDING_RECEIVER_FACTORIES.get(embedding_transfer_mode)
        if receiver is None:
            raise ValueError(
                f"Invalid embedding transfer mode: {embedding_transfer_mode}"
            )
        self.embedding_receiver = receiver()

    async def process_embeddings(
        self, request: SglangMultimodalRequest
    ) -> tuple[torch.Tensor, int]:
        """Process one concatenated embedding tensor from serialized request."""
        logger.debug("Processing embeddings with shape: " f"{request.embeddings_shape}")

        multimodal_groups = request.multimodal_inputs
        if not multimodal_groups:
            raise ValueError("multimodal_inputs is required")

        transfer_request = request.transfer_payload
        if transfer_request is None:
            raise ValueError("transfer_payload is required on request")

        if not isinstance(transfer_request, TransferRequest):
            transfer_request = TransferRequest.model_validate(transfer_request)

        embeddings_shape = request.embeddings_shape or tuple(
            transfer_request.embeddings_shape
        )
        if len(embeddings_shape) < 2:
            raise ValueError(f"Invalid embeddings shape: {embeddings_shape}")

        tensor_id, embeddings = await self.embedding_receiver.receive_embeddings(
            transfer_request
        )
        return embeddings, tensor_id

    def release_embeddings(self, tensor_id: int) -> None:
        self.embedding_receiver.release_tensor(tensor_id)

    @staticmethod
    def create_multimodal_item(embeddings: torch.Tensor, image_grid_thw) -> dict:
        """Create mm_item dict for SGLang's engine.async_generate(image_data=[...]).

        Uses format="processor_output" with precomputed_embeddings so SGLang
        bypasses get_image_feature() entirely (model-agnostic path).
        """
        precomputed = embeddings.to(MultimodalConfig.EMBEDDINGS_DTYPE)

        mm_item: dict[str, Any] = {"image_grid_thw": torch.tensor(image_grid_thw)}
        mm_item.update(
            {
                "format": "processor_output",
                "precomputed_embeddings": precomputed,
                "modality": "IMAGE",
            }
        )

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
                    output_ids = res.get("output_ids", [])
                    finish_reason = res.get("meta_info", {}).get("finish_reason")

                    # Empty, non-final chunks can happen during scheduler idle ticks.
                    # Keep waiting for the next chunk.
                    if not output_ids and not finish_reason:
                        continue

                    output = {
                        "token_ids": output_ids,
                        # Preserve SGLang's choice index for n>1 multimodal
                        # streams; older/non-n chunks are choice 0.
                        "index": res.get("index") or 0,
                        "text": res.get("text", ""),
                        "finished": False,
                    }

                    if finish_reason:
                        # For n > 1, choices can finish independently and SGLang
                        # may continue emitting chunks for other choice indices.
                        output.update(
                            {
                                "finish_reason": normalize_finish_reason(
                                    finish_reason.get("type", "stop")
                                ),
                                "finished": True,
                            }
                        )

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
) -> tuple[list[dict], torch.Tensor, int]:
    """Process embeddings and build a single multimodal item for SGLang."""
    embeddings, tensor_id = await embeddings_processor.process_embeddings(request)

    image_grid_thw_list = [group.image_grid_thw for group in request.multimodal_inputs]
    if any(item is None for item in image_grid_thw_list):
        raise ValueError("image_grid_thw is required")

    mm_items = [
        embeddings_processor.create_multimodal_item(embeddings, image_grid_thw_list)
    ]

    return mm_items, embeddings, tensor_id


class MultimodalWorkerHandler(BaseWorkerHandler[SglangMultimodalRequest, str]):
    """
    Multimodal worker handler for LLM inference with multimodal data.
    Handles both aggregated and disaggregated modes.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        prefill_client: Client | None = None,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(engine, config, None, None, shutdown_event)

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor(
            config.dynamo_args.embedding_transfer_mode
        )

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

    def _validate_and_parse_request(self, request) -> SglangMultimodalRequest:
        """Validate and parse incoming request"""
        if type(request) is not SglangMultimodalRequest:
            if type(request) is str:
                request = SglangMultimodalRequest.model_validate_json(request)
            else:
                request = SglangMultimodalRequest.model_validate(request)
        return request

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
        rng_pd = _nvtx.start_range("mm:pd:generate", color="green")
        rng_ttft = _nvtx.start_range("mm:pd:ttft", color="yellow")
        ttft_ended = False

        def _end_ttft() -> None:
            nonlocal ttft_ended
            if not ttft_ended:
                _nvtx.end_range(rng_ttft)
                ttft_ended = True

        try:
            request = self._validate_and_parse_request(request)

            # Route to appropriate generation method based on serving mode
            if self.serving_mode == DisaggregationMode.DECODE:
                rng_disagg = _nvtx.start_range("mm:pd:generate_disagg", color="red")
                try:
                    async for output in self._generate_disaggregated(
                        request, _end_ttft, context=context
                    ):
                        yield output
                finally:
                    _nvtx.end_range(rng_disagg)
            else:
                rng_agg = _nvtx.start_range("mm:pd:generate_agg", color="red")
                try:
                    async for output in self._generate_aggregated(
                        request, _end_ttft, context=context
                    ):
                        yield output
                finally:
                    _nvtx.end_range(rng_agg)

        except Exception as e:
            logger.error(f"Error in multimodal generation: {e}", exc_info=True)
            yield ErrorResponseBuilder.build_error_response(e)
        finally:
            _end_ttft()
            _nvtx.end_range(rng_pd)

    async def _generate_disaggregated(
        self,
        request: SglangMultimodalRequest,
        end_ttft: Callable[[], None],
        context=None,
    ) -> AsyncIterator[str]:
        """Handle disaggregated mode generation"""
        input_ids = request.request.token_ids
        if not input_ids:
            raise ValueError("input_ids is required")

        sampling_params = SglangUtils.build_sampling_params(request)

        # Request bootstrap info from prefill worker
        bootstrap_info = await self._get_bootstrap_from_prefill(
            request, sampling_params, context=context
        )

        trace_header = (
            build_trace_headers(context) if context and self.enable_trace else None
        )

        # Start decode generation with bootstrap info (no image data needed)
        decode_stream = await self.engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            stream=True,
            bootstrap_host=bootstrap_info["bootstrap_host"],
            bootstrap_port=bootstrap_info["bootstrap_port"],
            bootstrap_room=bootstrap_info["bootstrap_room"],
            external_trace_header=trace_header,
            rid=context.trace_id if context else None,
        )

        rng_first = _nvtx.start_range("mm:dec:first_token", color="purple")
        first_token = True
        try:
            async for output in StreamProcessor.process_sglang_stream(decode_stream):
                if first_token:
                    end_ttft()
                    _nvtx.end_range(rng_first)
                    first_token = False
                yield output
        finally:
            if first_token:
                end_ttft()
                _nvtx.end_range(rng_first)

    async def _generate_aggregated(
        self,
        request: SglangMultimodalRequest,
        end_ttft: Callable[[], None],
        context=None,
    ) -> AsyncIterator[str]:
        """Handle aggregated mode generation"""
        input_ids = request.request.token_ids
        if not input_ids:
            raise ValueError("input_ids is required")
        tensor_id: int | None = None
        try:
            sampling_params = SglangUtils.build_sampling_params(request)
            with _nvtx.annotate("mm:pd:load_multimodal", color="cyan"):
                mm_items, combined_embeddings, tensor_id = await _build_mm_items(
                    request, self.embeddings_processor
                )

            logger.debug(
                "Generated combined multimodal item with embeddings shape: "
                f"{combined_embeddings.shape}"
            )
            logger.debug(f"Input token sequence length: {len(input_ids)}")

            trace_header = (
                build_trace_headers(context) if context and self.enable_trace else None
            )

            agg_stream = await self.engine.async_generate(
                input_ids=input_ids,
                image_data=mm_items,
                sampling_params=sampling_params,
                stream=True,
                external_trace_header=trace_header,
                rid=context.trace_id if context else None,
            )

            rng_first = _nvtx.start_range("mm:dec:first_token", color="purple")
            first_token = True
            try:
                async for output in StreamProcessor.process_sglang_stream(agg_stream):
                    if first_token:
                        if tensor_id is not None:
                            self.embeddings_processor.release_embeddings(tensor_id)
                            tensor_id = None
                        end_ttft()
                        _nvtx.end_range(rng_first)
                        first_token = False
                    yield output
            finally:
                if first_token:
                    end_ttft()
                    _nvtx.end_range(rng_first)

        except RuntimeError as e:
            if "shape mismatch" in str(e):
                logger.error(
                    "Shape mismatch error - this likely indicates a tokenization/embedding alignment issue"
                )
                logger.error(f"Request token IDs length: {len(input_ids)}")
                logger.error("Embeddings shape: " f"{request.embeddings_shape}")
                logger.error(f"Token sequence preview: {input_ids[:20]}...")
                error_msg = (
                    f"Multimodal embedding alignment error: {str(e)}. "
                    f"This usually happens when the tokenization changes between requests. "
                    "Token count: "
                    f"{len(input_ids)}, Embedding shape: "
                    f"{request.embeddings_shape}"
                )
                yield ErrorResponseBuilder.build_error_response(RuntimeError(error_msg))
            else:
                yield ErrorResponseBuilder.build_error_response(e)
        finally:
            if tensor_id is not None:
                self.embeddings_processor.release_embeddings(tensor_id)

    async def _get_bootstrap_from_prefill(
        self, request: SglangMultimodalRequest, sampling_params: dict, context=None
    ) -> dict:
        """Get bootstrap info from prefill worker"""
        assert self.prefill_client is not None
        prefill_stream = await self.prefill_client.generate(
            DisaggSglangMultimodalRequest(
                request=request,
                sampling_params=sampling_params,
            ).model_dump_json(),
            context=context,
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


class MultimodalPrefillWorkerHandler(
    BaseWorkerHandler[DisaggSglangMultimodalRequest, str]
):
    """
    Multimodal prefill worker handler for disaggregated inference
    Processes multimodal inputs and coordinates with decode worker.
    """

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
        shutdown_event: Optional[asyncio.Event] = None,
    ):
        super().__init__(engine, config, None, None, shutdown_event)

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor(
            config.dynamo_args.embedding_transfer_mode
        )

        # Get bootstrap info using BootstrapManager
        self.bootstrap_host, self.bootstrap_port = self._get_bootstrap_info(engine)

        logger.info(
            f"Multimodal prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    async def generate(
        self, disagg_request: DisaggSglangMultimodalRequest, context: Context
    ) -> AsyncIterator[str]:
        """
        Handle prefill phase: process multimodal input and provide bootstrap info

        Args:
            disagg_request: Disaggregated multimodal request.
            context: Context object for cancellation handling.
        """
        rng_bootstrap = _nvtx.start_range("mm:prefill:bootstrap", color="yellow")
        bootstrap_ended = False

        def _end_bootstrap() -> None:
            nonlocal bootstrap_ended
            if not bootstrap_ended:
                _nvtx.end_range(rng_bootstrap)
                bootstrap_ended = True

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

            _end_bootstrap()
            yield json.dumps(bootstrap_info)

            # Process prefill generation
            await self._process_prefill_generation(
                disagg_request, bootstrap_room, context=context
            )

        except Exception as e:
            logger.error(f"Error in prefill generation: {e}", exc_info=True)
            extra_fields = (
                {"bootstrap_room": bootstrap_room} if bootstrap_room is not None else {}
            )
            yield ErrorResponseBuilder.build_error_response(e, extra_fields)
        finally:
            _end_bootstrap()

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
        self,
        disagg_request: DisaggSglangMultimodalRequest,
        bootstrap_room: int,
        context=None,
    ):
        """Process multimodal input and start prefill generation"""
        # Get the SglangMultimodalRequest from the DisaggSglangMultimodalRequest
        request = disagg_request.request
        input_ids = request.request.token_ids
        sampling_params = disagg_request.sampling_params
        tensor_id: int | None = None

        # Process embeddings from encode worker using our embeddings processor
        with _nvtx.annotate("mm:prefill:load_multimodal", color="cyan"):
            mm_items, _, tensor_id = await _build_mm_items(
                request, self.embeddings_processor
            )

        trace_header = (
            build_trace_headers(context) if context and self.enable_trace else None
        )

        # Start SGLang prefill generation (like regular SGLang)
        with _nvtx.annotate("mm:prefill:engine_async_generate", color="blue"):
            results = await self.engine.async_generate(
                input_ids=input_ids,
                image_data=mm_items,
                sampling_params=sampling_params,
                stream=True,
                bootstrap_host=self.bootstrap_host,
                bootstrap_port=self.bootstrap_port,
                bootstrap_room=bootstrap_room,
                external_trace_header=trace_header,
                rid=context.trace_id if context else None,
            )

        # Consume results without yielding (prefill doesn't return text, just coordinates)
        asyncio.create_task(self._consume_results(results, tensor_id))

    async def _consume_results(self, results, tensor_id: int):
        """Consume prefill results without returning them (like regular SGLang)"""
        released = False
        try:
            async for _ in results:
                if not released:
                    self.embeddings_processor.release_embeddings(tensor_id)
                    released = True
        finally:
            if not released:
                self.embeddings_processor.release_embeddings(tensor_id)

    def cleanup(self):
        super().cleanup()
        self.engine.shutdown()
        logger.info("Multimodal prefill engine shutdown")
