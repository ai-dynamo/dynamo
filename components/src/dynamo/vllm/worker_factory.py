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
from .constants import DisaggregationMode
from .multimodal_handlers import (
    EncodeWorkerHandler,
    MultimodalDecodeWorkerHandler,
    MultimodalPDWorkerHandler,
)

logger = logging.getLogger(__name__)

# (engine_client, vllm_config, default_sampling_params, prometheus_temp_dir, component_gauges)
EngineSetupResult = tuple[Any, Any, Any, Any, Any]

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
        shutdown_endpoints: list,
        pre_created_engine: Optional[EngineSetupResult] = None,
    ) -> None:
        """Create the appropriate multimodal worker based on config flags."""

        if config.multimodal_encode_worker:
            await self._create_multimodal_encode_worker(
                runtime, config, shutdown_event, shutdown_endpoints
            )
        elif config.multimodal_worker or config.multimodal_decode_worker:
            await self._create_multimodal_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                pre_created_engine=pre_created_engine,
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
        shutdown_endpoints: list,  # mutated in place
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
        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        clear_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.clear_kv_blocks"
        )
        shutdown_endpoints[:] = [generate_endpoint, clear_endpoint]

        lora_enabled = config.engine_args.enable_lora
        if lora_enabled:
            load_lora_endpoint = runtime.endpoint(
                f"{config.namespace}.{config.component}.load_lora"
            )
            unload_lora_endpoint = runtime.endpoint(
                f"{config.namespace}.{config.component}.unload_lora"
            )
            list_loras_endpoint = runtime.endpoint(
                f"{config.namespace}.{config.component}.list_loras"
            )
            shutdown_endpoints.extend(
                [load_lora_endpoint, unload_lora_endpoint, list_loras_endpoint]
            )
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
            encode_worker_client = await runtime.endpoint(
                f"{config.namespace}.encoder.generate"
            ).client()
            logger.info("Waiting for Encoder Worker Instances ...")
            await encode_worker_client.wait_for_instances()
            logger.info("Connected to encoder workers")

        # Set up decode worker client for disaggregated mode
        decode_worker_client = None
        if config.disaggregation_mode == DisaggregationMode.PREFILL:
            decode_worker_client = await runtime.endpoint(
                f"{config.namespace}.decoder.generate"
            ).client()
            await decode_worker_client.wait_for_instances()
            logger.info("Connected to decode worker for disaggregated mode")

        # Choose handler based on worker type
        if config.multimodal_decode_worker:
            handler = MultimodalDecodeWorkerHandler(
                runtime,
                engine_client,
                config,
                shutdown_event,
                generate_endpoint=generate_endpoint,
            )
        else:
            handler = MultimodalPDWorkerHandler(
                runtime,
                engine_client,
                config,
                encode_worker_client,
                decode_worker_client,
                shutdown_event,
                generate_endpoint=generate_endpoint,
            )
        handler.add_temp_dir(prometheus_temp_dir)

        await handler.async_init(runtime)

        # Set up KV event publisher for prefix caching if enabled
        kv_publisher = self.setup_kv_event_publisher(
            config, generate_endpoint, vllm_config
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
            serve_tasks = [
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=metrics_labels,
                ),
                clear_endpoint.serve_endpoint(
                    handler.clear_kv_blocks,
                    metrics_labels=metrics_labels,
                ),
            ]

            if lora_enabled:
                serve_tasks.extend(
                    [
                        load_lora_endpoint.serve_endpoint(
                            handler.load_lora,
                            metrics_labels=metrics_labels,
                        ),
                        unload_lora_endpoint.serve_endpoint(
                            handler.unload_lora,
                            metrics_labels=metrics_labels,
                        ),
                        list_loras_endpoint.serve_endpoint(
                            handler.list_loras,
                            metrics_labels=metrics_labels,
                        ),
                    ]
                )

            await asyncio.gather(*serve_tasks)
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
        shutdown_endpoints: list,  # mutated in place
    ) -> None:
        """Initialize standalone multimodal encode worker."""
        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        shutdown_endpoints[:] = [generate_endpoint]

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
