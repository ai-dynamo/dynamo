# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization factory for vLLM workers."""

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo import prometheus_names
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import LLMBackendMetrics
from dynamo.llm import ModelInput, ModelType
from dynamo.runtime import DistributedRuntime

from .args import Config
from .constants import DisaggregationMode
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .health_check import VllmHealthCheckPayload, VllmPrefillHealthCheckPayload
from .multimodal_handlers import EncodeWorkerHandler
from .publisher import StatLoggerFactory

logger = logging.getLogger(__name__)

# (engine_client, vllm_config, default_sampling_params, prometheus_temp_dir, component_gauges)
EngineSetupResult = tuple[AsyncLLM, VllmConfig, Any, Any, LLMBackendMetrics]

SetupVllmEngineFn = Callable[..., EngineSetupResult]
SetupKvEventPublisherFn = Callable[..., Optional[Any]]
RegisterVllmModelFn = Callable[..., Awaitable[None]]
SetupFpmRelayFn = Callable[..., Optional[list]]
SetupMetricsCollectionFn = Callable[..., None]


class WorkerFactory:
    """Factory for creating and initializing multimodal vLLM workers."""

    def __init__(
        self,
        setup_vllm_engine_fn: SetupVllmEngineFn,
        setup_kv_event_publisher_fn: SetupKvEventPublisherFn,
        register_vllm_model_fn: RegisterVllmModelFn,
        setup_fpm_relay_fn: SetupFpmRelayFn,
        setup_metrics_collection_fn: SetupMetricsCollectionFn,
    ):
        self.setup_vllm_engine = setup_vllm_engine_fn
        self.setup_kv_event_publisher = setup_kv_event_publisher_fn
        self.register_vllm_model = register_vllm_model_fn
        self.setup_fpm_relay = setup_fpm_relay_fn
        self.setup_metrics_collection = setup_metrics_collection_fn

    async def create(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,
        snapshot_engine: Optional[EngineSetupResult] = None,
    ) -> None:
        """Create the appropriate multimodal worker based on config flags."""

        if config.disaggregation_mode == DisaggregationMode.ENCODE:
            await self._create_multimodal_encode_worker(
                runtime, config, shutdown_event, shutdown_endpoints
            )
        elif config.disaggregation_mode == DisaggregationMode.PREFILL:
            await self._create_prefill_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                snapshot_engine=snapshot_engine,
            )
        else:
            # AGGREGATED or DECODE
            await self._create_decode_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                snapshot_engine=snapshot_engine,
            )
        return

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

        handler = EncodeWorkerHandler(
            config.engine_args, config.embedding_transfer_mode  # type: ignore[arg-type]
        )
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

    async def _create_decode_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
        snapshot_engine: Optional[EngineSetupResult] = None,
    ) -> None:
        """
        Instantiate and serve
        """

        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        clear_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.clear_kv_blocks"
        )

        shutdown_endpoints[:] = [
            generate_endpoint,
            clear_endpoint,
        ]

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
                [
                    load_lora_endpoint,
                    unload_lora_endpoint,
                    list_loras_endpoint,
                ]
            )

        # Use pre-created engine if provided (checkpoint mode), otherwise create new
        fpm_worker_id = str(generate_endpoint.connection_id())
        if snapshot_engine is not None:
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                component_gauges,
            ) = snapshot_engine
            vllm_config.additional_config["fpm_worker_id"] = fpm_worker_id
            # Factory is created after unpack so component_gauges is available
            factory = StatLoggerFactory(
                endpoint=generate_endpoint,
                component_gauges=component_gauges,
            )
        else:
            # Factory is created without component_gauges; setup_vllm_engine() will
            # create the gauges after setup_multiprocess_prometheus() and set them
            # on the factory before vLLM calls create_stat_logger().
            factory = StatLoggerFactory(
                endpoint=generate_endpoint,
            )
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                component_gauges,
            ) = self.setup_vllm_engine(config, factory, fpm_worker_id=fpm_worker_id)

        # TODO Hack to get data, move this to registering in TBD
        factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
        factory.init_publish()

        # Currently routing to worker is still controlled by the worker
        # as the worker has logic to determine whether remote encode should be
        # performed
        encode_worker_client = await self._maybe_get_encode_worker_client(
            runtime, config
        )

        handler = DecodeWorkerHandler(
            runtime,
            config,
            engine_client,
            default_sampling_params,
            getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
            enable_multimodal=config.enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=config.use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=config.frontend_decoding,
            encode_worker_client=encode_worker_client,
        )
        handler.add_temp_dir(prometheus_temp_dir)

        # Check if kv event consolidator is enabled (port was allocated in setup_vllm_engine)
        consolidator_enabled = False
        consolidator_port = None

        _consolidator_eps = vllm_config.additional_config.get("consolidator_endpoints")
        if _consolidator_eps:
            # Extract connect endpoint (third element) for clients to subscribe
            # consolidator_endpoints = (vllm_endpoint, bind_endpoint, connect_endpoint)
            consolidator_output_endpoint = _consolidator_eps[2]
            consolidator_port = int(consolidator_output_endpoint.split(":")[-1])
            consolidator_enabled = True

        # Set up KV event publisher for prefix caching if enabled
        # If kv event consolidator is enabled, publisher will subscribe to kv event consolidator's output
        kv_publishers = self.setup_kv_event_publisher(
            config,
            generate_endpoint,
            vllm_config,
            consolidator_enabled=consolidator_enabled,
            consolidator_port=consolidator_port,
        )
        if kv_publishers:
            handler.kv_publishers = kv_publishers

        # Set up forward pass metrics relay (child ZMQ -> event plane).
        # In checkpoint mode the engine was created before the runtime, so
        # ForwardPassMetrics.worker_id will be empty (relay still works).
        fpm_relays = self.setup_fpm_relay(generate_endpoint, vllm_config)
        if fpm_relays:
            handler.fpm_relays = fpm_relays

        self.setup_metrics_collection(config, generate_endpoint, logger)
        # Register sleep/wake_up engine routes
        runtime.register_engine_route("sleep", handler.sleep)
        runtime.register_engine_route("wake_up", handler.wake_up)
        runtime.register_engine_route("scale_elastic_ep", handler.scale_elastic_ep)
        logger.info(
            "Registered engine routes: /engine/sleep, /engine/wake_up, /engine/scale_elastic_ep"
        )

        # Parse endpoint types from --endpoint-types flag
        model_type = parse_endpoint_types(config.endpoint_types)
        logger.info(f"Registering model with endpoint types: {config.endpoint_types}")

        model_input = (
            ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens
        )

        # Warn if custom template provided but chat endpoint not enabled
        if config.custom_jinja_template and "chat" not in config.endpoint_types:
            logger.warning(
                "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
                "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
            )

        if config.gms_shadow_mode:
            # Shadow mode: lock-driven activation.
            # Flow: sleep → startup probe passes → block on lock → wake → register.
            await handler._quiesce_controller.quiesce(1)

            runtime.set_health_status(True)
            logger.info(
                "[Shadow] Engine sleeping, startup probe now passing, waiting for lock"
            )

            from gpu_memory_service.failover_lock.flock import FlockFailoverLock

            lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
            engine_id = os.environ.get("ENGINE_ID", "0")
            lock = FlockFailoverLock(lock_path)
            await lock.acquire(engine_id=f"engine-{engine_id}")
            logger.info("[Shadow] Lock acquired, waking engine")

            await handler._quiesce_controller.resume()
            handler._quiesce_controller.mark_resumed()
            logger.info("[Shadow] Engine awake, registering with discovery")

        await self.register_vllm_model(
            model_input,
            model_type,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
        )

        health_check_payload = VllmHealthCheckPayload(
            engine_client, use_text_input=config.use_vllm_tokenizer
        ).to_dict()

        try:
            logger.debug("Starting serve_endpoint for decode worker")

            model_metrics_labels = [
                (
                    prometheus_names.labels.MODEL,
                    config.served_model_name or config.model,
                ),
                (
                    prometheus_names.labels.MODEL_NAME,
                    config.served_model_name or config.model,
                ),
            ]

            serve_tasks = [
                # for decode, we want to transfer the in-flight requests to other decode engines,
                # because waiting them to finish can take a long time for long OSLs
                generate_endpoint.serve_endpoint(
                    handler.generate,  # type: ignore
                    graceful_shutdown=True,
                    metrics_labels=model_metrics_labels,
                    health_check_payload=health_check_payload,
                ),
                clear_endpoint.serve_endpoint(
                    handler.clear_kv_blocks,
                    metrics_labels=model_metrics_labels,
                ),
            ]

            if lora_enabled:
                serve_tasks.extend(
                    [
                        load_lora_endpoint.serve_endpoint(
                            handler.load_lora,
                            metrics_labels=model_metrics_labels,
                        ),
                        unload_lora_endpoint.serve_endpoint(
                            handler.unload_lora,
                            metrics_labels=model_metrics_labels,
                        ),
                        list_loras_endpoint.serve_endpoint(
                            handler.list_loras,
                            metrics_labels=model_metrics_labels,
                        ),
                    ]
                )

            await asyncio.gather(*serve_tasks)
            logger.debug("serve_endpoint completed for decode worker")
        except Exception as e:
            logger.error(f"Failed to serve endpoints: {e}")
            raise
        finally:
            logger.debug("Cleaning up decode worker")
            # Cleanup background tasks
            handler.cleanup()

    async def _create_prefill_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
        snapshot_engine: Optional[EngineSetupResult] = None,
    ) -> None:
        """
        Instantiate and serve
        """
        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        clear_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.clear_kv_blocks"
        )

        # Use pre-created engine if provided (checkpoint mode), otherwise create new
        fpm_worker_id = str(generate_endpoint.connection_id())
        if snapshot_engine is not None:
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = snapshot_engine
            # TODO: The scheduler in the child process still has worker_id=""
            # because the engine was forked before the runtime existed.
            # Propagating the new ID to the child requires shared memory or
            # a restart of the EngineCore process.
            vllm_config.additional_config["fpm_worker_id"] = fpm_worker_id
        else:
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = self.setup_vllm_engine(config, fpm_worker_id=fpm_worker_id)

        encode_worker_client = await self._maybe_get_encode_worker_client(
            runtime, config
        )

        handler = PrefillWorkerHandler(
            runtime,
            config,
            engine_client,
            default_sampling_params,
            getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
            enable_multimodal=config.enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=config.use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=config.frontend_decoding,
            encode_worker_client=encode_worker_client,
        )
        handler.add_temp_dir(prometheus_temp_dir)

        # Check if kv event consolidator is enabled (port was allocated in setup_vllm_engine)
        consolidator_enabled = False
        consolidator_port = None

        _consolidator_eps = vllm_config.additional_config.get("consolidator_endpoints")
        if _consolidator_eps:
            # Extract connect endpoint (third element) for clients to subscribe
            # consolidator_endpoints = (vllm_endpoint, bind_endpoint, connect_endpoint)
            consolidator_output_endpoint = _consolidator_eps[2]
            consolidator_port = int(consolidator_output_endpoint.split(":")[-1])
            consolidator_enabled = True

        # Set up KV event publishers for prefix caching if enabled (one per dp_rank)
        # If kv event consolidator is enabled, publisher will subscribe to kv event consolidator's output
        kv_publishers = self.setup_kv_event_publisher(
            config,
            generate_endpoint,
            vllm_config,
            consolidator_enabled=consolidator_enabled,
            consolidator_port=consolidator_port,
        )
        if kv_publishers:
            handler.kv_publishers = kv_publishers

        # Set up forward pass metrics relay (child ZMQ -> event plane).
        # In checkpoint mode the engine was created before the runtime, so
        # ForwardPassMetrics.worker_id will be empty (relay still works).
        fpm_relays = self.setup_fpm_relay(generate_endpoint, vllm_config)
        if fpm_relays:
            handler.fpm_relays = fpm_relays

        self.setup_metrics_collection(config, generate_endpoint, logger)
        # Register sleep/wake_up engine routes
        runtime.register_engine_route("sleep", handler.sleep)
        runtime.register_engine_route("wake_up", handler.wake_up)
        runtime.register_engine_route("scale_elastic_ep", handler.scale_elastic_ep)
        logger.info(
            "Registered engine routes: /engine/sleep, /engine/wake_up, /engine/scale_elastic_ep"
        )

        shutdown_endpoints[:] = [generate_endpoint, clear_endpoint]

        # Register prefill model with ModelType.Prefill
        model_input = (
            ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens
        )
        await self.register_vllm_model(
            model_input,
            ModelType.Prefill,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
        )

        health_check_payload = VllmPrefillHealthCheckPayload(
            engine_client, use_text_input=config.use_vllm_tokenizer
        ).to_dict()

        try:
            logger.debug("Starting serve_endpoint for prefill worker")
            await asyncio.gather(
                # for prefill, we want to shutdown the engine after all prefill requests are finished because
                #     (temp reason): we don't support re-routing prefill requests
                #     (long-term reason): prefill engine should pull from a global queue so there is
                #                         only a few in-flight requests that can be quickly finished
                generate_endpoint.serve_endpoint(
                    handler.generate,  # type: ignore
                    graceful_shutdown=True,
                    # In practice config.served_model_name is always set, but mypy needs the "or" here.
                    metrics_labels=[
                        (
                            prometheus_names.labels.MODEL,
                            config.served_model_name or config.model,
                        ),
                        (
                            prometheus_names.labels.MODEL_NAME,
                            config.served_model_name or config.model,
                        ),
                    ],
                    health_check_payload=health_check_payload,
                ),
                clear_endpoint.serve_endpoint(
                    handler.clear_kv_blocks,  # type: ignore
                    metrics_labels=[
                        (
                            prometheus_names.labels.MODEL,
                            config.served_model_name or config.model,
                        ),
                        (
                            prometheus_names.labels.MODEL_NAME,
                            config.served_model_name or config.model,
                        ),
                    ],
                ),
            )
            logger.debug("serve_endpoint completed for prefill worker")
        except Exception as e:
            logger.error(f"Failed to serve endpoints: {e}")
            raise
        finally:
            logger.debug("Cleaning up prefill worker")
            handler.cleanup()

    async def _maybe_get_encode_worker_client(
        self, runtime: DistributedRuntime, config: Config
    ) -> Optional[Any]:
        """Helper function to get encode worker client if routing to encoder is enabled."""
        if config.route_to_encoder:
            # [gluo NOTE] hardcoded component name
            encode_worker_client = await runtime.endpoint(
                f"{config.namespace}.encode.generate"
            ).client()
            logger.info("Waiting for Encoder Worker Instances ...")
            await encode_worker_client.wait_for_instances()
            logger.info("Connected to encode workers")
            return encode_worker_client
        return None
