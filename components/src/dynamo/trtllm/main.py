# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import os
import signal
import sys

# Configure TLLM_LOG_LEVEL before importing tensorrt_llm
# This must happen before any tensorrt_llm imports
if "TLLM_LOG_LEVEL" not in os.environ and os.getenv(
    "DYN_SKIP_TRTLLM_LOG_FORMATTING"
) not in ("1", "true", "TRUE"):
    # This import is safe because it doesn't trigger tensorrt_llm imports
    from dynamo.runtime.logging import map_dyn_log_to_tllm_level

    dyn_log = os.environ.get("DYN_LOG", "info")
    tllm_level = map_dyn_log_to_tllm_level(dyn_log)
    os.environ["TLLM_LOG_LEVEL"] = tllm_level
import uvloop
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
from dynamo.common.config_dump import dump_config
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    register_llm,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.trtllm.engine import Backend, TensorRTLLMEngine, get_llm_engine
from dynamo.trtllm.health_check import TrtllmHealthCheckPayload
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.publisher import get_publisher
from dynamo.trtllm.request_handlers.handler_base import DisaggregationMode
from dynamo.trtllm.request_handlers.handlers import (
    RequestHandlerConfig,
    RequestHandlerFactory,
)
from dynamo.trtllm.utils.trtllm_utils import (
    Config,
    cmd_line_args,
    deep_update,
    parse_endpoint,
)

# Default buffer size for kv cache events.
DEFAULT_KV_EVENT_BUFFER_MAX_SIZE = 1024

configure_dynamo_logging()


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


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
    if config.connector is not None:
        if config.connector == "kvbm":
            return KvCacheConnectorConfig(
                connector_module="kvbm.trtllm_integration.connector",
                connector_scheduler_class="DynamoKVBMConnectorLeader",
                connector_worker_class="DynamoKVBMConnectorWorker",
            )
        elif config.connector == "none":
            return None
        else:
            logging.error(f"Invalid connector: {config.connector}")
            sys.exit(1)
    return None


async def worker():
    config = cmd_line_args()

    loop = asyncio.get_running_loop()
    # Enable NATS based on use_kv_events flag (derived from publish_events_and_metrics)
    runtime = DistributedRuntime(
        loop, config.store_kv, config.request_plane, config.use_kv_events
    )

    # Set up signal handler for graceful shutdown
    def signal_handler():
        # Schedule the shutdown coroutine instead of calling it directly
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.info("Signal handlers set up for graceful shutdown")

    await init(runtime, config)


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    logging.info(f"Initializing the worker with config: {config}")

    encode_client = None
    if config.encode_endpoint:
        logging.info(
            f"Initializing encode worker client for endpoint: {config.encode_endpoint}"
        )
        parsed_namespace, parsed_component_name, parsed_endpoint_name = parse_endpoint(
            config.encode_endpoint
        )
        encode_client = (
            await runtime.namespace(parsed_namespace)
            .component(parsed_component_name)
            .endpoint(parsed_endpoint_name)
            .client()
        )

    component = runtime.namespace(config.namespace).component(config.component)

    # Convert model path to Path object if it's a local path, otherwise keep as string
    model_path = str(config.model_path)

    if config.gpus_per_node is None:
        gpus_per_node = device_count()
        if gpus_per_node == 0:
            raise ValueError("No GPU devices found on the node")
    else:
        gpus_per_node = config.gpus_per_node

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=config.free_gpu_memory_fraction
    )

    if config.connector is not None and "kvbm" in config.connector:
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

    modality = getattr(config, "modality", None) or "text"
    arg_map = {
        "model": model_path,
        "scheduler_config": scheduler_config,
        "tensor_parallel_size": config.tensor_parallel_size,
        "pipeline_parallel_size": config.pipeline_parallel_size,
        "moe_expert_parallel_size": config.expert_parallel_size,
        "backend": Backend.PYTORCH,
        "kv_cache_config": kv_cache_config,
        "gpus_per_node": gpus_per_node,
        "max_num_tokens": config.max_num_tokens,
        "max_seq_len": config.max_seq_len,
        "max_beam_width": config.max_beam_width,
        "max_batch_size": config.max_batch_size,
        "return_perf_metrics": config.publish_events_and_metrics,
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
        model_type = parse_endpoint_types(config.dyn_endpoint_types)
        logging.info(
            f"Registering model with endpoint types: {config.dyn_endpoint_types}"
        )

        # Warn if custom template provided but chat endpoint not enabled
        if config.custom_jinja_template and "chat" not in config.dyn_endpoint_types:
            logging.warning(
                "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
                "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
            )

    multimodal_processor = None

    if os.getenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR") == "1":
        # We need to initialize the tokenizer for the test logits processor
        # But detokenizing still happens in the rust engine, so we do _not_ want
        # to set default_sampling_params.detokenize to True.
        # This overrides the skip_tokenizer_init=True set earlier
        engine_args["skip_tokenizer_init"] = False

    if modality == "multimodal":
        engine_args["skip_tokenizer_init"] = False
        model_config = AutoConfig.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        multimodal_processor = MultimodalRequestProcessor(
            model_type=model_config.model_type,
            model_dir=config.model_path,
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

    async with get_llm_engine(engine_args, config.disaggregation_mode) as engine:
        endpoint = component.endpoint(config.endpoint)

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
        runtime_config.reasoning_parser = config.reasoning_parser
        runtime_config.tool_call_parser = config.tool_call_parser
        runtime_config.enable_local_indexer = config.enable_local_indexer

        logging.info(f"Set runtime config max_num_seqs: {runtime_config.max_num_seqs}")
        logging.info(
            f"Set runtime config max_num_batched_tokens: {runtime_config.max_num_batched_tokens}"
        )

        # The get_engine_runtime_config function exists but is not called here due to:
        # 1. get_stats_async requires active requests to work properly
        # 2. We need runtime config during registration, before any requests are made
        # 3. total_kv_blocks would ideally come from engine stats but is not critical for basic operation

        # Initialize TensorRT-LLM MetricsCollector and register with global REGISTRY
        # This enables exposing TRT-LLM's native Prometheus metrics (request latency, TTFT, TPOT, etc.)
        metrics_collector = None
        if config.publish_events_and_metrics:
            try:
                model_name_for_metrics = config.served_model_name or config.model_path
                metrics_collector = MetricsCollector(
                    {"model_name": model_name_for_metrics, "engine_type": "trtllm"}
                )
                logging.info("TensorRT-LLM MetricsCollector initialized")

                # Register callback to expose TRT-LLM metrics via Dynamo endpoint
                # Filter out python_/process_ metrics and add trtllm_ prefix to remaining metrics
                register_engine_metrics_callback(
                    endpoint=endpoint,
                    registry=REGISTRY,
                    exclude_prefixes=["python_", "process_"],
                    add_prefix="trtllm_",
                )
                logging.info("TensorRT-LLM Prometheus metrics registered")
            except Exception as e:
                logging.warning(
                    f"Failed to initialize TensorRT-LLM Prometheus metrics: {e}"
                )

        # publisher will be set later if publishing is enabled.
        handler_config = RequestHandlerConfig(
            component=component,
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
        )

        # Register the model with runtime config
        # Encode workers do NOT register - they're internal workers only
        # Prefill and decode workers register - frontend detects their role via ModelType
        if config.disaggregation_mode != DisaggregationMode.ENCODE:
            await register_llm(
                model_input,
                model_type,
                endpoint,
                config.model_path,
                config.served_model_name,
                kv_cache_block_size=config.kv_block_size,
                migration_limit=config.migration_limit,
                runtime_config=runtime_config,
                custom_template_path=config.custom_jinja_template,
            )

        # Get health check payload (checks env var and falls back to TensorRT-LLM default)
        health_check_payload = TrtllmHealthCheckPayload(tokenizer=tokenizer).to_dict()

        if config.publish_events_and_metrics:
            # Initialize and pass in the publisher to the request handler to
            # publish events and metrics.
            kv_listener = runtime.namespace(config.namespace).component(
                config.component
            )
            # Use model_path as fallback if served_model_name is not provided
            model_name_for_metrics = config.served_model_name or config.model_path
            metrics_labels = [("model", model_name_for_metrics)]

            # Create worker-side publisher for consolidated events if consolidator is enabled
            # This subscribes to consolidator's ZMQ output and publishes to NATS with worker_id
            consolidator_publisher = None
            if consolidator_output_endpoint:
                # Use the connect endpoint directly (already provided by get_consolidator_endpoints)
                consolidator_config = ZmqKvEventPublisherConfig(
                    worker_id=int(endpoint.connection_id()),
                    kv_block_size=config.kv_block_size,
                    zmq_endpoint=consolidator_output_connect_endpoint,
                    zmq_topic="",  # Empty topic = all topics
                )
                consolidator_publisher = ZmqKvEventPublisher(
                    component, consolidator_config
                )
                logging.info(
                    f"Created worker-side publisher for consolidated events: "
                    f"subscribing to {consolidator_output_connect_endpoint}, worker_id={endpoint.connection_id()}"
                )

            async with get_publisher(
                component,
                engine,
                kv_listener,
                int(endpoint.connection_id()),
                config.kv_block_size,
                metrics_labels,
                zmq_endpoint=trtllm_zmq_bind_endpoint,
                enable_local_indexer=config.enable_local_indexer,
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


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
