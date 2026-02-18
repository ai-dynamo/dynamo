# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization factory for vLLM workers."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.llm import ModelInput
from dynamo.runtime import DistributedRuntime

from .args import Config
from .multimodal_handlers import (
    EncodeWorkerHandler,
    MultimodalDecodeWorkerHandler,
    MultimodalPDWorkerHandler,
)

logger = logging.getLogger(__name__)

# (engine_client, vllm_config, default_sampling_params, prometheus_temp_dir)
EngineSetupResult = tuple[Any, Any, Any, Any]

SetupVllmEngineFn = Callable[..., EngineSetupResult]
SetupKvEventPublisherFn = Callable[..., Optional[Any]]
RegisterVllmModelFn = Callable[..., Awaitable[None]]


class WorkerFactory:
    """Factory for creating and initializing multimodal vLLM workers."""

    def __init__(
        self,
        setup_vllm_engine_fn: SetupVllmEngineFn,
        setup_kv_event_publisher_fn: SetupKvEventPublisherFn,
        register_vllm_model_fn: RegisterVllmModelFn,
    ):
        self.setup_vllm_engine = setup_vllm_engine_fn
        self.setup_kv_event_publisher = setup_kv_event_publisher_fn
        self.register_vllm_model = register_vllm_model_fn

    @staticmethod
    def handles(config: Config) -> bool:
        """Return True if this factory handles the given config."""
        return bool(
            config.multimodal_encode_worker
            or config.multimodal_worker
            or config.multimodal_decode_worker
        )

    async def create(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        pre_created_engine: Optional[EngineSetupResult] = None,
    ) -> None:
        """Create the appropriate multimodal worker based on config flags."""
        if config.multimodal_encode_worker:
            await self._create_multimodal_encode_worker(runtime, config, shutdown_event)
        elif config.multimodal_worker or config.multimodal_decode_worker:
            await self._create_multimodal_worker(
                runtime, config, shutdown_event, pre_created_engine=pre_created_engine
            )
        else:
            raise ValueError(
                "WorkerFactory.create() called but no multimodal worker type set in config"
            )

    async def _create_multimodal_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        pre_created_engine: Optional[EngineSetupResult] = None,
    ) -> None:
        """
        Initialize multimodal worker component.

        Supports:
        - --multimodal-worker: PD worker that may receive embeddings from encoder
        - --multimodal-decode-worker: Decode-only worker

        Modes:
        - Aggregated (P+D): Prefill and decode on same worker
        - Disaggregated (P→D): Prefill forwards to separate decode worker
        """
        component = runtime.namespace(config.namespace).component(config.component)

        generate_endpoint = component.endpoint(config.endpoint)
        clear_endpoint = component.endpoint("clear_kv_blocks")

        # Use pre-created engine if provided (checkpoint mode), otherwise create new
        if pre_created_engine is not None:
            (
                engine_client,
                vllm_config,
                _default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = pre_created_engine
        else:
            (
                engine_client,
                vllm_config,
                _default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = self.setup_vllm_engine(config)

        # Set up encode worker client when routing to encoder is enabled
        encode_worker_client = None
        if config.route_to_encoder:
            encode_worker_client = (
                await runtime.namespace(config.namespace)
                .component("encoder")
                .endpoint("generate")
                .client()
            )
            logger.info("Waiting for Encoder Worker Instances ...")
            await encode_worker_client.wait_for_instances()
            logger.info("Connected to encoder workers")

        # Set up decode worker client for disaggregated mode
        decode_worker_client = None
        if config.is_prefill_worker:
            decode_worker_client = (
                await runtime.namespace(config.namespace)
                .component("decoder")
                .endpoint("generate")
                .client()
            )
            await decode_worker_client.wait_for_instances()
            logger.info("Connected to decode worker for disaggregated mode")

        # Choose handler based on worker type
        if config.multimodal_decode_worker:
            handler = MultimodalDecodeWorkerHandler(
                runtime, component, engine_client, config, shutdown_event
            )
        else:
            handler = MultimodalPDWorkerHandler(
                runtime,
                component,
                engine_client,
                config,
                encode_worker_client,
                decode_worker_client,
                shutdown_event,
            )
        handler.add_temp_dir(prometheus_temp_dir)

        await handler.async_init(runtime)

        # Set up KV event publisher for prefix caching if enabled
        kv_publisher = self.setup_kv_event_publisher(
            config, component, generate_endpoint, vllm_config
        )
        if kv_publisher:
            handler.kv_publisher = kv_publisher

        # Register model with the frontend so it can route requests
        model_type = parse_endpoint_types(config.endpoint_types)
        model_input = (
            ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens
        )
        await self.register_vllm_model(
            model_input,
            model_type,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
        )

        metrics_labels = [("model", config.served_model_name or config.model)]
        try:
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=metrics_labels,
                ),
                clear_endpoint.serve_endpoint(
                    handler.clear_kv_blocks,
                    metrics_labels=metrics_labels,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to serve endpoints: {e}")
            raise
        finally:
            handler.cleanup()

    async def _create_multimodal_encode_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
    ) -> None:
        """Initialize standalone multimodal encode worker."""
        component = runtime.namespace(config.namespace).component(config.component)
        generate_endpoint = component.endpoint(config.endpoint)

        handler = EncodeWorkerHandler(config.engine_args)
        await handler.async_init(runtime)
        logger.info("Starting to serve the encode worker endpoint...")

        try:
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate, metrics_labels=[("model", config.model)]
                ),
            )
        except Exception as e:
            logger.error(f"Failed to serve encode worker endpoint: {e}")
            raise
        finally:
            handler.cleanup()
