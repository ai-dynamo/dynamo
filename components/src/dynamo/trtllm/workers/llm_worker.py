# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM worker initialization for TensorRT-LLM backend.

This module handles the initialization and lifecycle of text and multimodal
LLM workers using TensorRT-LLM.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Optional

from prometheus_client import REGISTRY
from tensorrt_llm.llmapi import (
    CapacitySchedulerPolicy,
    DynamicBatchConfig,
    KvCacheConfig,
    SchedulerConfig,
)
from tensorrt_llm.llmapi.llm import SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.llmapi.tokenizer import tokenizer_factory
from tensorrt_llm.metrics import MetricsCollector
from torch.cuda import device_count
from transformers import AutoConfig

import dynamo.nixl_connect as nixl_connect
from dynamo import prometheus_names
from dynamo.common.config_dump import dump_config
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import (
    LLMBackendMetrics,
    register_engine_metrics_callback,
)
from dynamo.common.utils.runtime import parse_endpoint
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    register_model,
)
from dynamo.runtime import DistributedRuntime
from dynamo.trtllm.args import Config
from dynamo.trtllm.constants import DisaggregationMode, Modality
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine, get_llm_engine
from dynamo.trtllm.health_check import TrtllmHealthCheckPayload
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import DYNAMO_COMPONENT_REGISTRY, get_publisher
from dynamo.trtllm.request_handlers.handlers import (
    RequestHandlerConfig,
    RequestHandlerFactory,
)
from dynamo.trtllm.utils.trtllm_utils import deep_update

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024


async def get_engine_runtime_config(
    engine: TensorRTLLMEngine, config: Config
) -> ModelRuntimeConfig:
    """Retrieve runtime configuration from TensorRT-LLM engine."""
    runtime_config = ModelRuntimeConfig()

    try:
        # Extract total_kv_blocks from engine stats
        stats = engine.llm.get_stats_async(timeout=5)
        stat = await anext(stats)
        runtime_config.total_kv_blocks = stat["kvCacheStats"]["maxNumBlocks"]
        logging.info(
            f"Set runtime config total_kv_blocks: {runtime_config.total_kv_blocks}"
        )

        # Extract max number of sequences
        runtime_config.max_num_seqs = config.max_batch_size
        logging.info(f"Set runtime config max_num_seqs: {runtime_config.max_num_seqs}")

        # Get max_num_batched_tokens from config
        runtime_config.max_num_batched_tokens = config.max_num_tokens
        logging.info(
            f"Set runtime config max_num_batched_tokens: {runtime_config.max_num_batched_tokens}"
        )
    except Exception as e:
        logging.error(f"Failed to get runtime config from TensorRT-LLM engine: {e}")
        # Keep default/None values if retrieval fails

    return runtime_config


def build_kv_connector_config(config: Config):
    if config.connector:
        if config.connector[0] == "kvbm":
            return KvCacheConnectorConfig(
                connector_module="kvbm.trtllm_integration.connector",
                connector_scheduler_class="DynamoKVBMConnectorLeader",
                connector_worker_class="DynamoKVBMConnectorWorker",
            )
        elif config.connector[0] == "none":
            return None
        else:
            logging.error(f"Invalid connector: {config.connector[0]}")
            sys.exit(1)
    return None


async def init_llm_worker(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    shutdown_endpoints: Optional[list] = None,
) -> None:
    """Initialize and run the LLM worker.

    This function handles text and multimodal LLM modalities using TensorRT-LLM.

    Args:
        runtime: The Dynamo distributed runtime.
        config: Configuration parsed from command line.
        shutdown_event: Event to signal shutdown.
        shutdown_endpoints: Optional list to populate with endpoints for graceful shutdown.
    """

    encode_client = None
    if config.encode_endpoint:
        logging.info(
            f"Initializing encode worker client for endpoint: {config.encode_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.encode_endpoint
        )
        encode_client = await runtime.endpoint(
            f"{parsed_namespace}.{parsed_component_name}.{parsed_endpoint_name}"
        ).client()

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model)

    if config.gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")
    else:
        gpus_per_node = config.gpus_per_node

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=config.free_gpu_memory_fraction
    )

    if config.has_connector("kvbm"):
        kv_cache_config.enable_partial_reuse = False

    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128,
    )
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        dynamic_batch_config=dynamic_batch_config,
    )
    kv_connector_config = build_kv_connector_config(config)

    arg_map = {
        "model": model_path,
        "scheduler_config": scheduler_config,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "moe_expert_parallel_size": config.expert_parallel_size,
        "enable_attention_dp": config.enable_attention_dp,
        "backend": Backend.PYTORCH,
        "kv_cache_config": kv_cache_config,
        "gpus_per_node": gpus_per_node,
        "max_num_tokens": config.max_num_tokens,
        "max_seq_len": config.max_seq_len,
        "max_beam_width": config.max_beam_width,
        "max_batch_size": config.max_batch_size,
        "return_perf_metrics": config.publish_events_and_metrics,
        # enable_iter_perf_stats is required for PyTorch backend to compute iteration-level
        # stats (KV cache utilization, hit rate). TensorRT backend always has this enabled.
        # See TRT-LLM PR #11243: MetricsCollector.log_iteration_stats() needs these stats.
        "enable_iter_perf_stats": config.publish_events_and_metrics,
        "kv_connector_config": kv_connector_config,
    }

    if config.extra_engine_args != "":
        # TODO: Support extra engine args from json file as well.
        arg_map = update_llm_args_with_extra_options(arg_map, config.extra_engine_args)

    # Apply override_engine_args if provided
    if config.override_engine_args != "":
        try:
            overrides = json.loads(config.override_engine_args)
            logging.info(f"Applying engine arg overrides: {overrides}")

            deep_update(arg_map, overrides)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse override_engine_args as JSON: {e}")
            sys.exit(1)

    if config.publish_events_and_metrics:
        # 'event_buffer_max_size' is required to enable TRTLLM to publish kv cache events.
        # Add it to kv_cache_config while preserving all settings from YAML
        current_kv_config = arg_map["kv_cache_config"]
        if isinstance(current_kv_config, KvCacheConfig):
            # Convert KvCacheConfig object to dict, preserving ALL existing settings
            # This ensures YAML overrides are not lost when adding event_buffer_max_size
            kv_config_dict = current_kv_config.model_dump(exclude_none=True)
            kv_config_dict["event_buffer_max_size"] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE
            arg_map["kv_cache_config"] = kv_config_dict
        elif isinstance(current_kv_config, dict):
            # Add event_buffer_max_size while preserving cache_transceiver_config and other YAML settings
            current_kv_config[
                "event_buffer_max_size"
            ] = DEFAULT_KV_EVENT_BUFFER_MAX_SIZE

        # Only pytorch backend is supported for now to publish events and metrics.
        if "backend" not in arg_map:
            arg_map["backend"] = Backend.PYTORCH
        elif arg_map["backend"] not in Backend:
            logging.error(
                "Only %s supported for now to publish events and metrics. Got: %s",
                [b.value for b in Backend],
                arg_map["backend"],
            )
            sys.exit(1)

    trtllm_zmq_bind_endpoint = None  # Endpoint for TensorRT-LLM to bind and publish
    consolidator_output_endpoint = (
        None  # Endpoint where consolidator publishes (workers subscribe to this)
    )

    try:
        from kvbm.trtllm_integration.consolidator_config import (
            get_consolidator_endpoints,
            should_enable_consolidator,
        )

        if should_enable_consolidator(arg_map):
            # get_consolidator_endpoints returns (trtllm_bind_endpoint, output_bind_endpoint, output_connect_endpoint)
            consolidator_endpoints = get_consolidator_endpoints()
            trtllm_zmq_bind_endpoint = consolidator_endpoints[0]  # TRTLLM bind endpoint
            consolidator_output_endpoint = consolidator_endpoints[
                1
            ]  # Consolidator output bind endpoint (for KVBM connector)
            consolidator_output_connect_endpoint = consolidator_endpoints[
                2
            ]  # Consolidator output connect endpoint (for worker publisher)
    except ImportError:
        # kvbm package is not installed
        logging.info(
            "kvbm package not installed - skipping KV event consolidator setup."
        )
    except Exception as e:
        logging.error(
            f"Failed to set up consolidator endpoints: {e}. "
            "Continuing without KV event consolidation.",
            exc_info=True,
        )

    logging.info(f"TensorRT-LLM engine args: {arg_map}")
    engine_args = arg_map

    # Populate default sampling params from the model
    tokenizer = tokenizer_factory(arg_map["model"])
    default_sampling_params = SamplingParams()

    # Enable perf metrics so prompt_tokens_details can be returned
    if hasattr(default_sampling_params, "return_perf_metrics"):
        default_sampling_params.return_perf_metrics = True
    model_input = ModelInput.Tokens

    # Set model type based on disaggregation mode for unified frontend support
    if config.disaggregation_mode == DisaggregationMode.PREFILL:
        model_type = ModelType.Prefill
    else:
        model_type = parse_endpoint_types(config.endpoint_types)
        logging.info(f"Registering model with endpoint types: {config.endpoint_types}")

        # Warn if custom template provided but chat endpoint not enabled
        if config.custom_jinja_template and "chat" not in config.endpoint_types:
            logging.warning(
                "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --endpoint-types. "
                "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
            )

    multimodal_processor = None

    if os.getenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR") == "1":
        # We need to initialize the tokenizer for the test logits processor
        # But detokenizing still happens in the rust engine, so we do _not_ want
        # to set default_sampling_params.detokenize to True.
        # This overrides the skip_tokenizer_init=True set earlier
        engine_args["skip_tokenizer_init"] = False

    if config.modality == Modality.MULTIMODAL:
        engine_args["skip_tokenizer_init"] = False
        model_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
        multimodal_processor = MultimodalRequestProcessor(
            model_type=model_config.model_type,
            model_dir=config.model,
            max_file_size_mb=config.max_file_size_mb,
            tokenizer=tokenizer,
            allowed_local_media_path=config.allowed_local_media_path,
        )

    else:
        # We already detokenize inside HandlerBase. No need to also do it in TRTLLM.
        default_sampling_params.detokenize = False

    connector = None
    logging.info("Initializing NIXL Connect.")
    connector = nixl_connect.Connector()

    dump_config(
        config.dump_config_to, {"engine_args": engine_args, "dynamo_args": config}
    )

    # Prepare model name for metrics
    model_name_for_metrics = config.served_model_name or config.model

    # Construct Prometheus gauges directly; passed through to the engine and publisher
    # via explicit parameters (no module-level global).
    component_gauges = LLMBackendMetrics(
        registry=DYNAMO_COMPONENT_REGISTRY,
        model_name=model_name_for_metrics,
        component_name=config.component,
    )

    async with get_llm_engine(
        engine_args,
        config.disaggregation_mode,
        component_gauges=component_gauges,
    ) as engine:
        endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )

        if shutdown_endpoints is not None:
            shutdown_endpoints[:] = [endpoint]

        # should ideally call get_engine_runtime_config
        # this is because we don't have a good way to
        # get total_kv_blocks from the engine yet without calling get_stats_async
        # This causes an issue because get_stats_async doesn't work when no requests are sent to the engine
        # So for now, we just set the parsers from the config
        # TODO: fix this once we have a better way to get total_kv_blocks
        runtime_config = ModelRuntimeConfig()

        # Set values from config that are available immediately
        # Note: We populate max_num_seqs and max_num_batched_tokens from config
        # to ensure Prometheus metrics are available even without engine stats

        # Naming clarification:
        # - In vLLM: max_num_seqs = maximum concurrent requests (this is an unusual name due to vLLM's historic reasons)
        # - In TensorRT-LLM: max_batch_size = maximum concurrent requests (clearer name)
        # Both parameters control the same thing: how many requests can be processed simultaneously
        runtime_config.max_num_seqs = config.max_batch_size
        runtime_config.max_num_batched_tokens = config.max_num_tokens
        runtime_config.reasoning_parser = config.dyn_reasoning_parser
        runtime_config.tool_call_parser = config.dyn_tool_call_parser
        # Decode workers don't create the WorkerKvQuery endpoint, so don't advertise local indexer
        runtime_config.enable_local_indexer = (
            config.enable_local_indexer
            and config.disaggregation_mode != DisaggregationMode.DECODE
        )
        # Set data_parallel_size for attention DP mode
        # This enables the router's scheduler to correctly iterate over all dp_ranks
        # Need to name ADP as `data_parallel_size` for parity with other frameworks
        attention_dp_size = engine.get_attention_dp_size()
        runtime_config.data_parallel_size = attention_dp_size

        logging.info(f"Set runtime config max_num_seqs: {runtime_config.max_num_seqs}")
        logging.info(
            f"Set runtime config max_num_batched_tokens: {runtime_config.max_num_batched_tokens}"
        )
        logging.info(f"Set runtime config data_parallel_size: {attention_dp_size}")

        # The get_engine_runtime_config function exists but is not called here due to:
        # 1. get_stats_async requires active requests to work properly
        # 2. We need runtime config during registration, before any requests are made
        # 3. total_kv_blocks would ideally come from engine stats but is not critical for basic operation

        # Initialize TensorRT-LLM MetricsCollector and register with global REGISTRY
        # This enables exposing TRT-LLM's native Prometheus metrics (request latency, TTFT, TPOT, etc.)
        metrics_collector = None
        if config.publish_events_and_metrics:
            try:
                model_name_for_metrics = config.served_model_name or config.model
                metrics_collector = MetricsCollector(
                    {"model_name": model_name_for_metrics, "engine_type": "trtllm"}
                )
                logging.info("TensorRT-LLM MetricsCollector initialized")

                # Register TRT-LLM metrics (TRT-LLM natively outputs trtllm_* metrics after traffic)
                # Auto-label injection: hierarchy labels are added automatically
                register_engine_metrics_callback(
                    endpoint=endpoint,
                    registry=REGISTRY,
                    metric_prefix_filters=["trtllm_"],
                    namespace_name=config.namespace,
                    component_name=config.component,
                    endpoint_name="generate",
                    model_name=model_name_for_metrics,
                )
                logging.info("TensorRT-LLM Prometheus metrics registered")
            except Exception as e:
                logging.warning(
                    f"Failed to initialize TensorRT-LLM Prometheus metrics: {e}"
                )

        # Register callback for Dynamo component metrics using dedicated registry
        register_engine_metrics_callback(
            endpoint=endpoint,
            registry=DYNAMO_COMPONENT_REGISTRY,
        )
        logging.debug("DYNAMO_COMPONENT_REGISTRY callback registered successfully")

        # publisher will be set later if publishing is enabled.
        handler_config = RequestHandlerConfig(
            engine=engine,
            default_sampling_params=default_sampling_params,
            publisher=None,
            disaggregation_mode=config.disaggregation_mode,
            encode_client=encode_client,
            multimodal_processor=multimodal_processor,
            connector=connector,
            runtime=runtime,  # Pass runtime for graceful shutdown
            metrics_collector=metrics_collector,
            kv_block_size=config.kv_block_size,
            shutdown_event=shutdown_event,
            encoder_cache_capacity_gb=config.multimodal_embedding_cache_capacity_gb,
            disable_request_abort=config.disable_request_abort,
        )

        # Register the model with runtime config
        # Encode workers do NOT register - they're internal workers only
        # Prefill and decode workers register - frontend detects their role via ModelType
        if config.disaggregation_mode != DisaggregationMode.ENCODE:
            await register_model(
                model_input,
                model_type,
                endpoint,
                config.model,
                config.served_model_name,
                kv_cache_block_size=config.kv_block_size,
                runtime_config=runtime_config,
                custom_template_path=config.custom_jinja_template,
            )

        # Get health check payload (checks env var and falls back to TensorRT-LLM default)
        health_check_payload = TrtllmHealthCheckPayload(tokenizer=tokenizer).to_dict()

        if config.publish_events_and_metrics:
            # Initialize and pass in the publisher to the request handler to
            # publish events and metrics.
            # Use model as fallback if served_model_name is not provided
            model_name_for_metrics = config.served_model_name or config.model
            metrics_labels = [
                (
                    prometheus_names.labels.MODEL,
                    model_name_for_metrics,
                ),  # OpenAI standard
                (
                    prometheus_names.labels.MODEL_NAME,
                    model_name_for_metrics,
                ),  # Native engine compatibility
            ]

            # Create worker-side publisher for consolidated events if consolidator is enabled
            # This subscribes to consolidator's ZMQ output and publishes to NATS with worker_id
            consolidator_publisher = None
            if consolidator_output_endpoint:
                # Use the connect endpoint directly (already provided by get_consolidator_endpoints)
                consolidator_publisher = KvEventPublisher(
                    endpoint=endpoint,
                    kv_block_size=config.kv_block_size,
                    zmq_endpoint=consolidator_output_connect_endpoint,
                    zmq_topic="",
                )
                logging.info(
                    f"Created worker-side publisher for consolidated events: "
                    f"subscribing to {consolidator_output_connect_endpoint}, worker_id={endpoint.connection_id()}"
                )

            async with get_publisher(
                endpoint,
                engine,
                int(endpoint.connection_id()),
                config.kv_block_size,
                metrics_labels,
                component_gauges=component_gauges,
                zmq_endpoint=trtllm_zmq_bind_endpoint,
                enable_local_indexer=config.enable_local_indexer,
                metrics_collector=metrics_collector,
            ) as publisher:
                handler_config.publisher = publisher
                handler = RequestHandlerFactory().get_request_handler(handler_config)
                await endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=metrics_labels,
                    health_check_payload=health_check_payload,
                )

            # Shutdown consolidator publisher if it was created
            if consolidator_publisher:
                consolidator_publisher.shutdown()
        else:
            handler = RequestHandlerFactory().get_request_handler(handler_config)
            await endpoint.serve_endpoint(
                handler.generate, health_check_payload=health_check_payload
            )
