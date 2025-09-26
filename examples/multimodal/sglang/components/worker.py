# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import logging
import random
import signal
import socket
from abc import ABC, abstractmethod
from typing import AsyncIterator

import sglang as sgl
import torch
import uvloop
from sglang.srt.utils import get_ip
from utils.args import Config, DisaggregationMode, parse_args, parse_endpoint
from utils.protocol import DisaggSglangMultimodalRequest, SglangMultimodalRequest

import dynamo.nixl_connect as connect
from dynamo._core import Client
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
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


class MultimodalConfig:
    """Configuration specific to multimodal processing"""

    EMBEDDINGS_DTYPE = torch.float16
    EMBEDDINGS_DEVICE = "cpu"


class EmbeddingsProcessor:
    """Handles multimodal embeddings processing and multimodal item creation"""

    def __init__(self):
        self._connector = None

    async def initialize(self):
        """Initialize the connector for embeddings processing"""
        self._connector = connect.Connector()
        await self._connector.initialize()

    async def process_embeddings(self, request: SglangMultimodalRequest):
        """Process embeddings from serialized request"""
        logger.debug(f"Processing embeddings with shape: {request.embeddings_shape}")

        # Validate embeddings shape
        if request.embeddings_shape is None or len(request.embeddings_shape) < 2:
            raise ValueError(f"Invalid embeddings shape: {request.embeddings_shape}")

        embeddings = torch.empty(
            request.embeddings_shape,
            dtype=MultimodalConfig.EMBEDDINGS_DTYPE,
            device=MultimodalConfig.EMBEDDINGS_DEVICE,
        )

        descriptor = connect.Descriptor(embeddings)
        if descriptor is None:
            raise RuntimeError("Descriptor is None - cannot process embeddings")

        if self._connector is None:
            logger.warning(
                "Connector is None - this should not happen after initialization"
            )
            self._connector = connect.Connector()
            await self._connector.initialize()

        read_op = await self._connector.begin_read(
            request.serialized_request, descriptor
        )
        await read_op.wait_for_completion()

        return embeddings, descriptor

    @staticmethod
    def create_multimodal_item(
        embeddings: torch.Tensor, request: SglangMultimodalRequest
    ) -> dict:
        """Create multimodal item for SGLang generation"""

        precomputed_embeddings = embeddings.to(MultimodalConfig.EMBEDDINGS_DTYPE)
        grid_thw_tensor = torch.tensor(request.image_grid_thw)

        mm_item = dict(
            modality="IMAGE",
            image_grid_thw=grid_thw_tensor,
            precomputed_embeddings=precomputed_embeddings,
        )

        return mm_item


class StreamProcessor:
    """Unified stream processing for SGLang responses"""

    @staticmethod
    async def process_sglang_stream(stream_source) -> AsyncIterator[str]:
        """Process SGLang stream output following backend pattern"""
        num_output_tokens_so_far = 0

        try:
            async for res in stream_source:
                try:
                    next_total_toks = len(res["output_ids"])

                    # Return incremental tokens
                    output = {
                        "token_ids": res["output_ids"][num_output_tokens_so_far:],
                        "text": res.get("text", ""),
                        "finished": False,
                    }
                    num_output_tokens_so_far = next_total_toks

                    # Check for finish reason
                    finish_reason = res.get("meta_info", {}).get("finish_reason")
                    if finish_reason:
                        output.update(
                            {
                                "token_ids": res["output_ids"][
                                    num_output_tokens_so_far:
                                ],
                                "finish_reason": finish_reason.get("type", "stop"),
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


class BootstrapManager:
    """Handles bootstrap coordination for disaggregated mode"""

    @staticmethod
    def generate_bootstrap_room() -> int:
        """Generate a unique bootstrap room ID"""
        return random.randint(0, 2**63 - 1)

    @staticmethod
    def get_bootstrap_info(engine: sgl.Engine) -> tuple[str, int]:
        """Extract bootstrap info from SGLang engine"""
        inner_tm = engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_ip()

        return bootstrap_host, bootstrap_port

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


class BaseSglangWorkerHandler(ABC):
    """Base handler following SGLang backend pattern"""

    def __init__(
        self,
        engine: sgl.Engine,
        config: Config,
    ):
        self.engine = engine
        self.config = config
        self.model = config.model

    @abstractmethod
    def generate(self, request: SglangMultimodalRequest) -> AsyncIterator[str]:
        pass

    def cleanup(self):
        if self.engine:
            self.engine.shutdown()
            logger.info("SGLang engine shutdown")


class SglangPDWorkerHandler(BaseSglangWorkerHandler):
    """
    SGLang multimodal worker handler following the official backend pattern
    Handles both aggregated and disaggregated multimodal generation
    """

    def __init__(
        self, engine: sgl.Engine, config: Config, prefill_client: Client = None
    ):
        super().__init__(engine, config)

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

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize async components"""
        await self.embeddings_processor.initialize()

    def _validate_and_parse_request(self, request) -> SglangMultimodalRequest:
        """Validate and parse incoming request"""
        if type(request) is not SglangMultimodalRequest:
            if type(request) is str:
                request = SglangMultimodalRequest.model_validate_json(request)
            else:
                request = SglangMultimodalRequest.model_validate(request)
        return request

    async def generate(self, request: SglangMultimodalRequest) -> AsyncIterator[str]:
        """
        Generate response using SGLang with multimodal data
        Handles both aggregated and disaggregated modes (following regular SGLang DecodeWorkerHandler pattern)
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
            embeddings, descriptor = await self.embeddings_processor.process_embeddings(
                request
            )

            # Create multimodal item
            mm_item = self.embeddings_processor.create_multimodal_item(
                embeddings, request
            )

            logger.debug(
                f"Generated multimodal item with embeddings shape: {embeddings.shape}"
            )
            logger.debug(f"Input token sequence length: {len(input_ids)}")

            agg_stream = await self.engine.async_generate(
                input_ids=input_ids,
                image_data=[mm_item],
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
                logger.error(f"Embeddings shape: {request.embeddings_shape}")
                logger.error(f"Token sequence preview: {input_ids[:20]}...")
                error_msg = (
                    f"Multimodal embedding alignment error: {str(e)}. "
                    f"This usually happens when the tokenization changes between requests. "
                    f"Token count: {len(input_ids)}, Embedding shape: {request.embeddings_shape}"
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


class SglangPrefillWorkerHandler(BaseSglangWorkerHandler):
    """
    Multimodal prefill worker handler for disaggregated inference
    Processes multimodal inputs and coordinates with decode worker (following regular SGLang pattern)
    """

    def __init__(self, engine: sgl.Engine, config: Config):
        super().__init__(engine, config)

        # Initialize processors
        self.embeddings_processor = EmbeddingsProcessor()

        # Get bootstrap info using BootstrapManager
        self.bootstrap_host, self.bootstrap_port = BootstrapManager.get_bootstrap_info(
            engine
        )

        logger.info(
            f"Multimodal prefill worker handler initialized - bootstrap host: {self.bootstrap_host}, bootstrap port: {self.bootstrap_port}"
        )

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize async components like connector"""
        await self.embeddings_processor.initialize()

    async def generate(
        self, disagg_request: DisaggSglangMultimodalRequest
    ) -> AsyncIterator[str]:
        """
        Handle prefill phase: process multimodal input and provide bootstrap info
        """
        bootstrap_room = None
        try:
            # Validate and parse request
            disagg_request = self._validate_and_parse_disagg_request(disagg_request)

            # Generate and return bootstrap info first (like regular SGLang)
            bootstrap_room = BootstrapManager.generate_bootstrap_room()
            bootstrap_info = BootstrapManager.create_bootstrap_info(
                self.bootstrap_host, self.bootstrap_port, bootstrap_room
            )

            yield json.dumps(bootstrap_info)

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
        embeddings, descriptor = await self.embeddings_processor.process_embeddings(
            request
        )

        # Create multimodal item for prefill generation
        mm_item = self.embeddings_processor.create_multimodal_item(embeddings, request)

        # Start SGLang prefill generation (like regular SGLang)
        results = await self.engine.async_generate(
            input_ids=input_ids,
            image_data=[mm_item],
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


def parse_worker_args() -> Config:
    """Parse command line arguments for worker"""
    return parse_args(component="worker")


async def graceful_shutdown(runtime):
    """
    By calling `runtime.shutdown()`, the endpoints will immediately be unavailable.
    However, in-flight requests will still be processed until they are finished.
    After all in-flight requests are finished, the `serve_endpoint` functions will return
    and the engine will be shutdown by Python's garbage collector.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    # Runtime setup
    # Set up signal handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    # worker setup
    config = parse_worker_args()

    # Choose init function based on serving mode (like regular SGLang)
    if config.serving_mode == DisaggregationMode.PREFILL:
        await init_prefill(runtime, config)
    else:
        # Handle both aggregated and decode modes
        await init(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    """
    Initialize worker for aggregated or decode mode (following regular SGLang pattern)
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    # Initialize SGLang engine
    engine = sgl.Engine(server_args=config.server_args)

    # Setup handler based on serving mode
    if config.serving_mode == DisaggregationMode.DECODE:
        # Decode mode: create prefill client (like regular SGLang)
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.downstream_endpoint
        )

        logger.info("Initializing prefill client for decode worker")
        prefill_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )
        handler = SglangPDWorkerHandler(engine, config, prefill_client)
    else:
        # Aggregated mode: no prefill client needed
        handler = SglangPDWorkerHandler(engine, config)

    # Initialize async components
    await handler.async_init(runtime)

    logger.info(
        f"Starting to serve the {config.endpoint} endpoint in {config.serving_mode.value} mode..."
    )

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            metrics_labels=[("model", config.model)],
            graceful_shutdown=True,
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Initialize prefill worker for disaggregated mode (following regular SGLang pattern)
    """
    component = runtime.namespace(config.namespace).component(config.component)
    await component.create_service()

    generate_endpoint = component.endpoint(config.endpoint)

    # Initialize SGLang engine
    engine = sgl.Engine(server_args=config.server_args)

    # Create prefill handler
    handler = SglangPrefillWorkerHandler(engine, config)

    # Initialize async components for prefill handler
    await handler.async_init(runtime)

    logger.info(f"Starting to serve the {config.endpoint} endpoint in prefill mode...")

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            metrics_labels=[("model", config.model)],
            graceful_shutdown=True,
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
