# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import signal
import tempfile
from typing import Optional

import uvloop
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    ZmqKvEventPublisher,
    ZmqKvEventPublisherConfig,
    fetch_llm,
    register_llm,
)
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.multimodal_handlers import (
    ECProcessorHandler,
    EncodeWorkerHandler,
    MultimodalDecodeWorkerHandler,
    MultimodalPDWorkerHandler,
    PreprocessedHandler,
    VLLMEncodeWorkerHandler,
)
from dynamo.vllm.multimodal_utils.encode_utils import create_ec_transfer_config

from .args import Config, overwrite_args, parse_args
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .health_check import VllmHealthCheckPayload, VllmPrefillHealthCheckPayload
from .publisher import StatLoggerFactory

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def graceful_shutdown(runtime):
    """
    Shutdown dynamo distributed runtime.
    The endpoints will be immediately invalidated so no new requests will be accepted.
    For endpoints served with graceful_shutdown=True, the serving function will wait until all in-flight requests are finished.
    For endpoints served with graceful_shutdown=False, the serving function will return immediately.
    """
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


async def worker():
    config = parse_args()

    loop = asyncio.get_running_loop()
    overwrite_args(config)

    # Enable NATS based on use_kv_events flag (derived from kv_events_config)
    runtime = DistributedRuntime(
        loop, config.store_kv, config.request_plane, config.use_kv_events
    )

    # Set up signal handler for graceful shutdown
    def signal_handler():
        asyncio.create_task(graceful_shutdown(runtime))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logging.debug("Signal handlers set up for graceful shutdown")

    dump_config(config.dump_config_to, config)

    # Name the model. Use either the full path (vllm and sglang do the same),
    # or the HF name (e.g. "Qwen/Qwen3-0.6B"), depending on cmd line params.
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    # Download the model if necessary using modelexpress.
    # We want it on disk before we start vllm to avoid downloading from HuggingFace.
    #
    # We don't set `config.engine_args.model` to the local path fetch_llm returns
    # because vllm will send that name to its Ray pipeline-parallel workers, which
    # may not have the local path.
    # vllm will attempt to download the model again, but find it in the HF cache.
    # For non-HF models use a path instead of an HF name, and ensure all workers have
    # that path (ideally via a shared folder).
    if not os.path.exists(config.model):
        await fetch_llm(config.model)

    # Route to appropriate initialization based on config flags
    if config.vllm_native_encoder_worker:
        await init_vllm_native_encoder(runtime, config)
        logger.debug("init_vllm_native_encoder completed")
    elif config.ec_processor:
        await init_ec_processor(runtime, config)
        logger.debug("init_ec_processor completed")
    elif config.multimodal_processor:
        await init_multimodal_processor(runtime, config)
        logger.debug("init_multimodal_processor completed")
    elif config.multimodal_encode_worker:
        await init_multimodal_encode_worker(runtime, config)
        logger.debug("init_multimodal_encode_worker completed")
    elif (
        config.multimodal_worker
        or config.multimodal_decode_worker
        or config.multimodal_encode_prefill_worker
    ):
        await init_multimodal_worker(runtime, config)
        logger.debug("init_multimodal_worker completed")
    elif config.is_prefill_worker:
        await init_prefill(runtime, config)
        logger.debug("init_prefill completed")
    else:
        await init(runtime, config)
        logger.debug("init completed")

    logger.debug("Worker function completed, exiting...")


def setup_metrics_collection(config: Config, generate_endpoint, logger):
    """Set up metrics collection for vLLM and LMCache metrics.

    In multiprocess mode (PROMETHEUS_MULTIPROC_DIR set), metrics are stored:
      1. In-memory: Metric objects in global REGISTRY
      2. On-disk: Metric values in .db files (PROMETHEUS_MULTIPROC_DIR)

    MultiProcessCollector reads from .db files but adding it to REGISTRY can fail
    with "Duplicated timeseries" if PROMETHEUS_MULTIPROC_DIR was set before process
    started (K8s deployments) because metrics are already in REGISTRY.

    Solution: Try adding MultiProcessCollector to REGISTRY. If that fails, use
    separate registry for multiprocess collection and register callbacks to both
    registries to ensure all metrics (vllm, lmcache, dynamo_component) are collected.
    """
    if config.engine_args.disable_log_stats is False:
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            try:
                # MultiProcessCollector reads metrics from .db files in PROMETHEUS_MULTIPROC_DIR
                # Adding it to REGISTRY allows collecting both in-memory and .db file metrics
                multiprocess.MultiProcessCollector(REGISTRY)
                logger.debug("Added MultiProcessCollector to global REGISTRY")
                register_engine_metrics_callback(
                    endpoint=generate_endpoint,
                    registry=REGISTRY,
                    metric_prefix_filters=["vllm:", "lmcache:"],
                )
            except ValueError as e:
                # Conflict: metrics already in REGISTRY, MultiProcessCollector tries to add same metrics from .db files
                # Solution: Use separate registry that ONLY reads from .db files (no in-memory conflicts)
                logger.debug(
                    f"Could not add MultiProcessCollector to REGISTRY ({e}), using separate registry"
                )
                multiproc_registry = CollectorRegistry()
                multiprocess.MultiProcessCollector(multiproc_registry)

                # Register both registries to collect all metrics
                # Global REGISTRY has in-memory metrics (vllm, dynamo_component)
                register_engine_metrics_callback(
                    endpoint=generate_endpoint,
                    registry=REGISTRY,
                    metric_prefix_filters=["vllm:", "dynamo_component:"],
                )
                # Multiproc registry has .db file metrics (lmcache, possibly vllm duplicates)
                register_engine_metrics_callback(
                    endpoint=generate_endpoint,
                    registry=multiproc_registry,
                    metric_prefix_filters=["vllm:", "lmcache:"],
                )
        else:
            # No multiprocess mode
            register_engine_metrics_callback(
                endpoint=generate_endpoint,
                registry=REGISTRY,
                metric_prefix_filters=["vllm:", "lmcache:"],
            )


def setup_kv_event_publisher(
    config: Config,
    component,
    generate_endpoint,
    vllm_config,
    consolidator_enabled: bool = False,
    consolidator_port: Optional[int] = 5558,
) -> Optional[ZmqKvEventPublisher]:
    """
    Set up KV event publishers for prefix caching if enabled.
    Creates one publisher per dp_rank since each dp_rank publishes to a different port.
    Args:
        config: Worker configuration
        component: Component for runtime integration
        generate_endpoint: Endpoint for worker ID
        vllm_config: vLLM configuration
        consolidator_enabled: If True, subscribe to kv eventconsolidator's ZMQ endpoint
        consolidator_port: Port where kv event consolidator publishes (default: 5558)

    Returns:
        List of ZmqKvEventPublisher instances (one per dp_rank) if prefix caching is enabled, None otherwise.
    """
    if not config.engine_args.enable_prefix_caching:
        return None

    # Skip KV event publishing for decode workers
    if config.is_decode_worker:
        logger.info("Skipping KV event publisher setup for decode worker")
        return None

    if config.engine_args.kv_events_config is None:
        return None

    # Check if kv_cache_events are explicitly disabled
    if not config.engine_args.kv_events_config.enable_kv_cache_events:
        logger.info(
            "KV event publishing skipped: enable_kv_cache_events=False in kv_events_config"
        )
        return None

    # Get data_parallel_size to create publishers for all dp_ranks
    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    kv_publishers = []

    for dp_rank in range(data_parallel_size):
        if consolidator_enabled:
            # TODO: Use different port for each dp_rank once KVBM supports DP
            zmq_endpoint = f"tcp://127.0.0.1:{consolidator_port}"
            logger.info(
                f"KV event publisher for dp_rank={dp_rank} subscribing to consolidator at {zmq_endpoint}"
            )
        else:
            # Each dp_rank publishes to a different port
            zmq_endpoint = ZmqEventPublisher.offset_endpoint_port(
                config.engine_args.kv_events_config.endpoint,
                data_parallel_rank=dp_rank,
            ).replace("*", "127.0.0.1")
            logger.info(
                f"KV event publisher for dp_rank={dp_rank} subscribing to vLLM at {zmq_endpoint}"
            )

        zmq_config = ZmqKvEventPublisherConfig(
            worker_id=generate_endpoint.connection_id(),
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
            enable_local_indexer=config.enable_local_indexer,
        )
        kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)
        kv_publishers.append(kv_publisher)

        logger.info(
            f"Worker reading KV events for dp_rank={dp_rank} from {zmq_endpoint}"
        )

    return kv_publishers if kv_publishers else None


def setup_vllm_engine(config, stat_logger=None):
    # vLLM v0.11.0 bug: vllm/v1.metrics/prometheus.py:79 passes TemporaryDirectory object
    # instead of .name string, causing false error on exit. Set PROMETHEUS_MULTIPROC_DIR
    # ourselves to avoid this and handle cleanup properly.
    prometheus_temp_dir = None
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        prometheus_temp_dir = tempfile.TemporaryDirectory(prefix="vllm_prometheus_")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_temp_dir.name
        logger.debug(
            f"Created PROMETHEUS_MULTIPROC_DIR at: {os.environ['PROMETHEUS_MULTIPROC_DIR']}"
        )

    setup_multiprocess_prometheus()  # call vLLM's library's function to setup multiprocess prometheus
    logger.debug(
        f"Prometheus multiproc dir set to: {os.environ.get('PROMETHEUS_MULTIPROC_DIR')}"
    )

    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    engine_args = config.engine_args

    if engine_args.enable_lora:
        if "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in os.environ:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        if "VLLM_LORA_MODULES_LOADING_TIMEOUT" not in os.environ:
            os.environ["VLLM_LORA_MODULES_LOADING_TIMEOUT"] = "600"
    # Load default sampling params from `generation_config.json`
    default_sampling_params = (
        engine_args.create_model_config().get_diff_sampling_param()
    )

    # Taken from build_async_engine_client_from_engine_args()
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # Set up consolidator endpoints if KVBM is enabled
    consolidator_endpoints = None
    if config.has_connector("kvbm"):
        try:
            from kvbm.vllm_integration.consolidator_config import (
                get_consolidator_endpoints,
            )

            consolidator_endpoints = get_consolidator_endpoints(vllm_config)
        except Exception as e:
            logger.warning(
                f"KVBM connector is enabled but failed to get consolidator endpoints: {e}. "
                "Continuing without KV event consolidation. "
                "Ensure 'kvbm' package is installed if this feature is needed."
            )
    vllm_config.consolidator_endpoints = consolidator_endpoints

    factory = []
    if stat_logger:
        factory.append(stat_logger)

    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        stat_loggers=factory,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )

    logger.info(f"VllmWorker for {config.served_model_name} has been initialized")

    return engine_client, vllm_config, default_sampling_params, prometheus_temp_dir


async def register_vllm_model(
    model_input: ModelInput,
    model_type: ModelType,
    generate_endpoint,
    config: Config,
    engine_client: AsyncLLM,
    vllm_config,
    migration_limit: int,
):
    """
    Helper function to register a vLLM model with runtime configuration.

    Args:
        model_input: Input type for the model (e.g., ModelInput.Tokens)
        model_type: Type of model (e.g., ModelType.Chat, ModelType.Prefill)
        generate_endpoint: Endpoint to register
        config: Configuration object
        engine_client: vLLM engine client
        vllm_config: vLLM configuration
        migration_limit: Migration limit for the model
    """
    runtime_config = ModelRuntimeConfig()

    # Get runtime configuration from vLLM engine
    logging.info(
        f"Getting engine runtime configuration metadata from vLLM engine for {model_type}..."
    )
    runtime_values = get_engine_cache_info(engine_client)
    runtime_config.total_kv_blocks = runtime_values["num_gpu_blocks"]
    runtime_config.max_num_seqs = runtime_values["max_num_seqs"]
    runtime_config.max_num_batched_tokens = runtime_values["max_num_batched_tokens"]
    runtime_config.enable_local_indexer = config.enable_local_indexer

    # Add tool/reasoning parsers for decode models
    if model_type != ModelType.Prefill:
        runtime_config.tool_call_parser = config.tool_call_parser
        runtime_config.reasoning_parser = config.reasoning_parser

    # Get data_parallel_size from vllm_config (defaults to 1)
    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    runtime_config.data_parallel_size = data_parallel_size

    await register_llm(
        model_input,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
        migration_limit=migration_limit,
        runtime_config=runtime_config,
        custom_template_path=config.custom_jinja_template,
    )


async def init_prefill(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """
    component = runtime.namespace(config.namespace).component(config.component)

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    (
        engine_client,
        vllm_config,
        default_sampling_params,
        prometheus_temp_dir,
    ) = setup_vllm_engine(config)

    handler = PrefillWorkerHandler(
        runtime,
        component,
        engine_client,
        default_sampling_params,
        getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
        enable_multimodal=config.enable_multimodal,
        generate_endpoint=generate_endpoint,
        config=config,
        use_vllm_tokenizer=config.use_vllm_tokenizer,
    )
    handler.add_temp_dir(prometheus_temp_dir)

    # Check if kv event consolidator is enabled (port was allocated in setup_vllm_engine)
    consolidator_enabled = False
    consolidator_port = None

    if (
        hasattr(vllm_config, "consolidator_endpoints")
        and vllm_config.consolidator_endpoints
    ):
        # Extract connect endpoint (third element) for clients to subscribe
        # consolidator_endpoints = (vllm_endpoint, bind_endpoint, connect_endpoint)
        consolidator_output_endpoint = vllm_config.consolidator_endpoints[2]
        consolidator_port = int(consolidator_output_endpoint.split(":")[-1])
        consolidator_enabled = True

    # Set up KV event publishers for prefix caching if enabled (one per dp_rank)
    # If kv event consolidator is enabled, publisher will subscribe to kv event consolidator's output
    kv_publishers = setup_kv_event_publisher(
        config,
        component,
        generate_endpoint,
        vllm_config,
        consolidator_enabled=consolidator_enabled,
        consolidator_port=consolidator_port,
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    setup_metrics_collection(config, generate_endpoint, logger)

    # Register prefill model with ModelType.Prefill
    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        model_input = (
            ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens
        )
        await register_vllm_model(
            model_input,
            ModelType.Prefill,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            migration_limit=0,  # Prefill doesn't support migration
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
                handler.generate,
                graceful_shutdown=True,
                # In practice config.served_model_name is always set, but mypy needs the "or" here.
                metrics_labels=[("model", config.served_model_name or config.model)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", config.served_model_name)],
            ),
        )
        logger.debug("serve_endpoint completed for prefill worker")
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.debug("Cleaning up prefill worker")
        handler.cleanup()


async def init(runtime: DistributedRuntime, config: Config):
    """
    Instantiate and serve
    """

    component = runtime.namespace(config.namespace).component(config.component)

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")
    load_lora_endpoint = component.endpoint("load_lora")
    unload_lora_endpoint = component.endpoint("unload_lora")
    list_loras_endpoint = component.endpoint("list_loras")

    factory = StatLoggerFactory(
        component,
        config.engine_args.data_parallel_rank or 0,
        metrics_labels=[("model", config.served_model_name or config.model)],
    )
    (
        engine_client,
        vllm_config,
        default_sampling_params,
        prometheus_temp_dir,
    ) = setup_vllm_engine(config, factory)

    # TODO Hack to get data, move this to registering in TBD
    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.set_request_total_slots_all(vllm_config.scheduler_config.max_num_seqs)
    factory.init_publish()

    handler = DecodeWorkerHandler(
        runtime,
        component,
        engine_client,
        default_sampling_params,
        getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
        enable_multimodal=config.enable_multimodal,
        generate_endpoint=generate_endpoint,
        config=config,
        use_vllm_tokenizer=config.use_vllm_tokenizer,
    )
    handler.add_temp_dir(prometheus_temp_dir)

    # Check if kv event consolidator is enabled (port was allocated in setup_vllm_engine)
    consolidator_enabled = False
    consolidator_port = None

    if (
        hasattr(vllm_config, "consolidator_endpoints")
        and vllm_config.consolidator_endpoints
    ):
        # Extract connect endpoint (third element) for clients to subscribe
        # consolidator_endpoints = (vllm_endpoint, bind_endpoint, connect_endpoint)
        consolidator_output_endpoint = vllm_config.consolidator_endpoints[2]
        consolidator_port = int(consolidator_output_endpoint.split(":")[-1])
        consolidator_enabled = True

    # Set up KV event publisher for prefix caching if enabled
    # If kv event consolidator is enabled, publisher will subscribe to kv event consolidator's output
    kv_publishers = setup_kv_event_publisher(
        config,
        component,
        generate_endpoint,
        vllm_config,
        consolidator_enabled=consolidator_enabled,
        consolidator_port=consolidator_port,
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    setup_metrics_collection(config, generate_endpoint, logger)

    if not config.engine_args.data_parallel_rank:  # if rank is 0 or None then register
        # Parse endpoint types from --dyn-endpoint-types flag
        model_type = parse_endpoint_types(config.dyn_endpoint_types)
        logger.info(
            f"Registering model with endpoint types: {config.dyn_endpoint_types}"
        )

        model_input = (
            ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens
        )

        # Warn if custom template provided but chat endpoint not enabled
        if config.custom_jinja_template and "chat" not in config.dyn_endpoint_types:
            logger.warning(
                "Custom Jinja template provided (--custom-jinja-template) but 'chat' not in --dyn-endpoint-types. "
                "The chat template will be loaded but the /v1/chat/completions endpoint will not be available."
            )

        await register_vllm_model(
            model_input,
            model_type,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            migration_limit=config.migration_limit,
        )

    health_check_payload = VllmHealthCheckPayload(
        engine_client, use_text_input=config.use_vllm_tokenizer
    ).to_dict()

    try:
        logger.debug("Starting serve_endpoint for decode worker")
        await asyncio.gather(
            # for decode, we want to transfer the in-flight requests to other decode engines,
            # because waiting them to finish can take a long time for long OSLs
            generate_endpoint.serve_endpoint(
                handler.generate,
                graceful_shutdown=config.migration_limit <= 0,
                metrics_labels=[("model", config.served_model_name or config.model)],
                health_check_payload=health_check_payload,
            ),
            clear_endpoint.serve_endpoint(
                handler.clear_kv_blocks,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            load_lora_endpoint.serve_endpoint(
                handler.load_lora,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            unload_lora_endpoint.serve_endpoint(
                handler.unload_lora,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
            list_loras_endpoint.serve_endpoint(
                handler.list_loras,
                metrics_labels=[("model", config.served_model_name or config.model)],
            ),
        )
        logger.debug("serve_endpoint completed for decode worker")
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        logger.debug("Cleaning up decode worker")
        # Cleanup background tasks
        handler.cleanup()


def get_engine_cache_info(engine: AsyncLLM):
    """Retrieve cache configuration information from [`AsyncLLM`] engine."""

    try:
        # Get values directly from vllm_config instead of collective_rpc
        cache_values = {
            "num_gpu_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
        }

        scheduler_values = {
            "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
        }

        logging.info(f"Cache config values: {cache_values}")
        logging.info(f"Scheduler config values: {scheduler_values}")
        return {
            "num_gpu_blocks": cache_values["num_gpu_blocks"],
            "max_num_seqs": scheduler_values["max_num_seqs"],
            "max_num_batched_tokens": scheduler_values["max_num_batched_tokens"],
        }
    except Exception as e:
        logging.error(f"Failed to get configuration values from vLLM config: {e}")
        raise


async def init_multimodal_processor(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal processor component"""
    component = runtime.namespace(config.namespace).component(config.component)

    generate_endpoint = component.endpoint(config.endpoint)

    # Get encode worker client
    encode_worker_client = (
        await runtime.namespace(config.namespace)
        .component("encoder")
        .endpoint("generate")
        .client()
    )

    pd_worker_client = (
        await runtime.namespace(config.namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    handler = PreprocessedHandler(
        config.engine_args,
        encode_worker_client,
        pd_worker_client,
    )

    logger.info("Waiting for Encoder Worker Instances ...")
    await encode_worker_client.wait_for_instances()

    # Register the endpoint as entrypoint to a model
    await register_llm(
        ModelInput.Tokens,
        ModelType.Chat,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
    )

    logger.info("Starting to serve the processor endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve endpoints: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_encode_worker(runtime: DistributedRuntime, config: Config):
    """Initialize multimodal encode worker component"""
    component = runtime.namespace(config.namespace).component(config.component)

    generate_endpoint = component.endpoint(config.endpoint)

    # Get PD worker client
    # In multimodal mode, the PD worker always registers as "backend"
    # (even in disaggregated mode with prefill/decode split, we still connect to "backend")
    pd_worker_client = (
        await runtime.namespace(config.namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    handler = EncodeWorkerHandler(
        config.engine_args,
        pd_worker_client,
    )
    await handler.async_init(runtime)
    logger.info("Waiting for PD Worker Instances ...")
    await pd_worker_client.wait_for_instances()
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


async def init_vllm_native_encoder(runtime: DistributedRuntime, config: Config):
    """
    Initialize vLLM-native encoder worker component (ECConnector mode).
    In this mode, vLLM handles encoder execution, caching, and storage automatically.
    """
    # Create component and endpoint
    component = runtime.namespace(config.namespace).component(config.component)
    generate_endpoint = component.endpoint(config.endpoint)

    # Configure ECTransferConfig for producer role
    instance_id = 0
    engine_id = f"{config.namespace}.{config.component}.encoder.{instance_id}"

    # Configure encoder with producer role, it will be responsible for creating embeddings and storing them in the shared storage
    ec_transfer_config = create_ec_transfer_config(
        engine_id=engine_id,
        ec_role="ec_producer",
        ec_connector_backend=config.ec_connector_backend,
        ec_storage_path=config.ec_storage_path,
        ec_extra_config=config.ec_extra_config,
    )

    # Set ECTransferConfig on engine args
    config.engine_args.ec_transfer_config = ec_transfer_config

    # Setup vLLM engine
    (
        engine_client,
        vllm_config,
        default_sampling_params,
        prometheus_temp_dir,
    ) = setup_vllm_engine(config)

    # Initialize vLLM Native Encoder Worker Handler
    handler = VLLMEncodeWorkerHandler(
        runtime,
        component,
        engine_client,
        config,
    )
    handler.add_temp_dir(prometheus_temp_dir)

    # 5. No async init needed - vLLM handles everything
    # await handler.async_init(runtime)  # Not needed for ECConnector mode

    logger.info("Starting to serve vLLM-native encoder endpoint...")

    # 6. Serve endpoint
    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve vLLM-native encoder endpoint: {e}")
        raise
    finally:
        handler.cleanup()


async def init_ec_processor(runtime: DistributedRuntime, config: Config):
    """
    Initialize ECConnector processor component.

    Simple processor that routes multimodal requests using ECConnector pattern:
    1. Preprocess request (same as regular processor)
    2. Send multimodal items to encoder workers (stores to shared storage)
    3. Forward preprocessed request to PD worker (loads from shared storage)
    4. Stream response back to client
    """
    # Create component and endpoint
    component = runtime.namespace(config.namespace).component(config.component)
    generate_endpoint = component.endpoint(config.endpoint)

    # Get encoder worker client
    encoder_client = (
        await runtime.namespace(config.namespace)
        .component("encoder")
        .endpoint("generate")
        .client()
    )

    # Get PD worker client
    pd_client = (
        await runtime.namespace(config.namespace)
        .component("backend")
        .endpoint("generate")
        .client()
    )

    # Get prompt template from args (must be passed via environment or command line)
    mm_prompt_template = config.mm_prompt_template

    # Create EC processor handler (with preprocessing like regular processor)
    handler = ECProcessorHandler(
        config.engine_args,
        encoder_worker_client=encoder_client,
        pd_worker_client=pd_client,
        prompt_template=mm_prompt_template,
    )

    logger.info("Waiting for encoder and PD worker instances...")
    await encoder_client.wait_for_instances()
    await pd_client.wait_for_instances()

    # Register the endpoint as entrypoint to a model (same as preprocessed_handler)
    await register_llm(
        ModelInput.Tokens,  # Use Rust tokenization for better performance and multi-image support
        ModelType.Chat,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
    )

    logger.info("Starting to serve EC processor endpoint...")

    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(
                handler.generate, metrics_labels=[("model", config.model)]
            ),
        )
    except Exception as e:
        logger.error(f"Failed to serve EC processor endpoint: {e}")
        raise
    finally:
        handler.cleanup()


async def init_multimodal_worker(runtime: DistributedRuntime, config: Config):
    """
    Initialize multimodal worker component.

    Supports two modes:
    1. --multimodal-worker: Receives embeddings from separate encoder
    2. --multimodal-encode-prefill-worker: Handles inline encoding (e.g., Llama 4)

    Both can operate in aggregated (P+D) or disaggregated (P→D) mode.

    When --ec-consumer-mode is enabled, configures as ECConnector consumer
    to load encoder embeddings from shared storage.
    """
    component = runtime.namespace(config.namespace).component(config.component)

    generate_endpoint = component.endpoint(config.endpoint)
    clear_endpoint = component.endpoint("clear_kv_blocks")

    # Configure ECConnector consumer mode if enabled
    if config.ec_consumer_mode:
        logger.info("Configuring as ECConnector consumer for encoder embeddings")
        instance_id = 0
        engine_id = f"{config.namespace}.{config.component}.backend.{instance_id}"

        # The PD Worker just load the embeddings from the shared storage, so it is a consumer
        ec_transfer_config = create_ec_transfer_config(
            engine_id=engine_id,
            ec_role="ec_consumer",
            ec_connector_backend=config.ec_connector_backend,
            ec_storage_path=config.ec_storage_path,
            ec_extra_config=config.ec_extra_config,
        )

        # Set ECTransferConfig on engine args
        config.engine_args.ec_transfer_config = ec_transfer_config
        logger.info(f"Configured as ECConnector consumer with engine_id={engine_id}")

    (
        engine_client,
        vllm_config,
        default_sampling_params,
        prometheus_temp_dir,
    ) = setup_vllm_engine(config)

    # Set up decode worker client for disaggregated mode
    decode_worker_client = None
    if config.is_prefill_worker:
        # Prefill worker needs to connect to decode worker
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
            runtime, component, engine_client, config
        )
    else:
        handler = MultimodalPDWorkerHandler(
            runtime, component, engine_client, config, decode_worker_client
        )
    handler.add_temp_dir(prometheus_temp_dir)

    await handler.async_init(runtime)

    # Set up KV event publisher for prefix caching if enabled
    kv_publisher = setup_kv_event_publisher(
        config, component, generate_endpoint, vllm_config
    )
    if kv_publisher:
        handler.kv_publisher = kv_publisher

    metrics_labels = [("model", config.model)]
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


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
