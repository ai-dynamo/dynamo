# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os
import tempfile
import time
from typing import Optional

import uvloop
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess
from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.entrypoints.cli.serve import run_headless
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

from dynamo import prometheus_names
from dynamo.common.config_dump import dump_config
from dynamo.common.storage import get_fs
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.output_modalities import get_output_modalities
from dynamo.common.utils.prometheus import (
    LLMBackendMetrics,
    register_engine_metrics_callback,
)
from dynamo.common.utils.runtime import create_runtime
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    fetch_model,
    register_model,
)

# Optional imports for frontend decoding support
try:
    from dynamo.llm import MediaDecoder, MediaFetcher

    MEDIA_DECODER_AVAILABLE = True
except ImportError:
    MediaDecoder = None
    MediaFetcher = None
    MEDIA_DECODER_AVAILABLE = False

from dynamo.runtime import DistributedRuntime, Endpoint
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.worker_factory import WorkerFactory

from .args import Config, parse_args
from .checkpoint_restore import get_checkpoint_config
from .constants import DisaggregationMode
from .handlers import DecodeWorkerHandler, PrefillWorkerHandler
from .health_check import (
    VllmHealthCheckPayload,
    VllmOmniHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
)
from .publisher import DYNAMO_COMPONENT_REGISTRY, StatLoggerFactory

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []
CHECKPOINT_SLEEP_MODE_LEVEL = 1


async def _handle_non_leader_node(dp_rank: int) -> None:
    """
    Handle non-leader node (data_parallel_rank >= 1) in multi-node deployments.
    Non-leader nodes run vLLM workers but don't serve Dynamo endpoints.
    """
    logger.info(
        f"Non-leader node detected (data_parallel_rank={dp_rank}). "
        "Skipping endpoint serving."
    )
    # Wait indefinitely - process terminated via signal handlers
    await asyncio.Event().wait()


def build_headless_namespace(config: Config) -> argparse.Namespace:
    """Build an argparse Namespace from engine_args for vLLM's run_headless().

    run_headless() expects the raw CLI namespace. We reconstruct it from
    the already-parsed AsyncEngineArgs so parse_args() doesn't need to
    leak transport details.
    """
    ns = argparse.Namespace(**vars(config.engine_args))
    # run_headless() reads api_server_count; default to 0 (no API server)
    if not hasattr(ns, "api_server_count"):
        ns.api_server_count = 0
    return ns


def run_dynamo_headless(config: Config) -> None:
    """Run in headless mode for multi-node TP/PP.

    Secondary nodes spawn vLLM workers only — no engine core, no scheduler,
    no Dynamo endpoints. Bypasses DistributedRuntime entirely (no NATS/etcd).
    """
    args = build_headless_namespace(config)
    run_headless(args)


async def worker():
    config = parse_args()

    dump_config(config.dump_config_to, config)

    # Name the model. Use either the full path (vllm and sglang do the same),
    # or the HF name (e.g. "Qwen/Qwen3-0.6B"), depending on cmd line params.
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    # Check checkpoint mode and validate env vars EARLY (fail fast if misconfigured)
    early_exit, checkpoint_cfg = get_checkpoint_config()
    if early_exit:
        return

    # Download the model if necessary using modelexpress.
    # We want it on disk before we start vllm to avoid downloading from HuggingFace.
    #
    # We don't set `config.engine_args.model` to the local path fetch_model returns
    # because vllm will send that name to its Ray pipeline-parallel workers, which
    # may not have the local path.
    # vllm will attempt to download the model again, but find it in the HF cache.
    # For non-HF models use a path instead of an HF name, and ensure all workers have
    # that path (ideally via a shared folder).
    if not os.path.exists(config.model):
        await fetch_model(config.model)

    # HEADLESS MODE: bypass DistributedRuntime entirely.
    # Workers run vLLM only (no NATS, etcd, or dynamo endpoints).
    if config.headless:
        if checkpoint_cfg is not None:
            raise ValueError(
                "--headless is incompatible with checkpoint mode "
                "(DYN_CHECKPOINT_SIGNAL_FILE is set). "
                "Remove --headless or unset DYN_CHECKPOINT_SIGNAL_FILE."
            )
        run_dynamo_headless(config)
        return

    # CHECKPOINT MODE: Load engine BEFORE runtime creation
    # This allows checkpointing GPU state before runtime connections are established
    pre_created_engine = None
    if checkpoint_cfg is not None:
        logger.info("Checkpoint mode enabled (watcher-driven signals)")

        # Checkpoint mode requires sleep mode — enable before engine init
        config.engine_args.enable_sleep_mode = True

        pre_created_engine = setup_vllm_engine(config)
        engine_client = pre_created_engine[0]

        if not await checkpoint_cfg.run_lifecycle(
            engine_client, CHECKPOINT_SLEEP_MODE_LEVEL
        ):
            return

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        use_kv_events=config.use_kv_events,
    )

    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    # Route to appropriate initialization based on config flags
    if WorkerFactory.handles(config):
        # Create worker factory with setup functions
        factory = WorkerFactory(
            setup_vllm_engine_fn=setup_vllm_engine,
            setup_kv_event_publisher_fn=setup_kv_event_publisher,
            register_vllm_model_fn=register_vllm_model,
        )
        await factory.create(
            runtime,
            config,
            shutdown_event,
            shutdown_endpoints,
            pre_created_engine=pre_created_engine,
        )
        logger.debug("multimodal worker completed")
    elif config.omni:
        await init_omni(runtime, config, shutdown_event)
        logger.debug("init_omni completed")
    elif config.disaggregation_mode == DisaggregationMode.PREFILL:
        await init_prefill(
            runtime, config, shutdown_event, pre_created_engine=pre_created_engine
        )
        logger.debug("init_prefill completed")
    else:
        await init(
            runtime, config, shutdown_event, pre_created_engine=pre_created_engine
        )
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

    Auto-label injection:
        Hierarchy labels (dynamo_namespace, dynamo_component, dynamo_endpoint) are automatically
        injected into engine metrics to align Python metrics with Rust auto-labels.
        Additional labels can be provided via inject_labels parameter.
    """
    if config.engine_args.disable_log_stats is False:
        # Register the dedicated dynamo_component registry callback
        # IMPORTANT: We do NOT use MultiProcessCollector for DYNAMO_COMPONENT_REGISTRY
        # because our gauges use in-memory values which work fine for single-process
        # and multi-process (each process has its own gauge with dp_rank label).
        # Using MultiProcessCollector would read from .db files which causes stale
        # values to accumulate across test runs.
        register_engine_metrics_callback(
            endpoint=generate_endpoint,
            registry=DYNAMO_COMPONENT_REGISTRY,
        )

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
                    namespace_name=config.namespace,
                    component_name=config.component,
                    endpoint_name=config.endpoint,
                    model_name=config.model,
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
                # Global REGISTRY has in-memory metrics (vllm)
                register_engine_metrics_callback(
                    endpoint=generate_endpoint,
                    registry=REGISTRY,
                    metric_prefix_filters=["vllm:"],
                    namespace_name=config.namespace,
                    component_name=config.component,
                    endpoint_name=config.endpoint,
                    model_name=config.model,
                )
                # Multiproc registry has .db file metrics (lmcache, possibly vllm duplicates)
                register_engine_metrics_callback(
                    endpoint=generate_endpoint,
                    registry=multiproc_registry,
                    metric_prefix_filters=["vllm:", "lmcache:"],
                    namespace_name=config.namespace,
                    component_name=config.component,
                    endpoint_name=config.endpoint,
                    model_name=config.model,
                )
        else:
            # No multiprocess mode
            register_engine_metrics_callback(
                endpoint=generate_endpoint,
                registry=REGISTRY,
                metric_prefix_filters=["vllm:", "lmcache:"],
                namespace_name=config.namespace,
                component_name=config.component,
                endpoint_name=config.endpoint,
                model_name=config.model,
            )


def setup_kv_event_publisher(
    config: Config,
    generate_endpoint: Endpoint,
    vllm_config: VllmConfig,
    consolidator_enabled: bool = False,
    consolidator_port: Optional[int] = 5558,
) -> Optional[KvEventPublisher]:
    """
    Set up KV event publishers for prefix caching if enabled.
    Creates one publisher per dp_rank since each dp_rank publishes to a different port.
    Args:
        config: Worker configuration
        generate_endpoint: Endpoint for worker ID
        vllm_config: vLLM configuration
        consolidator_enabled: If True, subscribe to kv eventconsolidator's ZMQ endpoint
        consolidator_port: Port where kv event consolidator publishes (default: 5558)

    Returns:
        List of KvEventPublisher instances (one per dp_rank) if prefix caching is enabled, None otherwise.
    """
    if not config.engine_args.enable_prefix_caching:
        return None

    # Skip KV event publishing for decode workers
    if config.disaggregation_mode == DisaggregationMode.DECODE:
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

        kv_publisher = KvEventPublisher(
            endpoint=generate_endpoint,
            kv_block_size=vllm_config.cache_config.block_size,
            zmq_endpoint=zmq_endpoint,
            zmq_topic="",
            enable_local_indexer=config.enable_local_indexer,
            dp_rank=dp_rank,
        )
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

    # Construct Prometheus gauges AFTER setup_multiprocess_prometheus() so Gauge objects
    # see the correct ValueClass (multiprocess vs in-memory).
    component_gauges = LLMBackendMetrics(
        registry=DYNAMO_COMPONENT_REGISTRY,
        model_name=config.served_model_name or "",
        component_name=config.component or "",
    )

    # If a StatLoggerFactory was provided, give it the gauges so the loggers
    # it creates can publish Prometheus metrics.
    if stat_logger is not None:
        stat_logger.component_gauges = component_gauges

    os.environ["VLLM_NO_USAGE_STATS"] = "1"  # Avoid internal HTTP requests
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    engine_args = config.engine_args

    if engine_args.enable_lora:
        if "VLLM_ALLOW_RUNTIME_LORA_UPDATING" not in os.environ:
            os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        if "VLLM_LORA_MODULES_LOADING_TIMEOUT" not in os.environ:
            os.environ["VLLM_LORA_MODULES_LOADING_TIMEOUT"] = "600"

    if engine_args.load_format == "gms":
        engine_args.worker_cls = "gpu_memory_service.integrations.vllm.worker.GMSWorker"

    if engine_args.load_format in ("mx-source", "mx-target"):
        try:
            from modelexpress import register_modelexpress_loaders

            # Ensure the ModelExpress server URL env var is set for the model loader
            if config.model_express_url:
                os.environ["MODEL_EXPRESS_URL"] = config.model_express_url
            register_modelexpress_loaders()
            # Use wrapper worker to ensure loaders are registered in spawned worker processes
            engine_args.worker_cls = "modelexpress.vllm_worker.ModelExpressWorker"
        except ImportError as e:
            raise ImportError(
                f"ModelExpress package required for --load-format={engine_args.load_format}. "
                "Install with: pip install modelexpress"
            ) from e

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

    # Time engine initialization
    start_time = time.time()
    engine_client = AsyncLLM.from_vllm_config(
        vllm_config=vllm_config,
        usage_context=usage_context,
        stat_loggers=factory,
        enable_log_requests=engine_args.enable_log_requests,
        disable_log_stats=engine_args.disable_log_stats,
    )
    load_time = time.time() - start_time

    # Record model load time
    component_gauges.set_model_load_time(load_time)

    logger.info(f"VllmWorker for {config.served_model_name} has been initialized")

    return (
        engine_client,
        vllm_config,
        default_sampling_params,
        prometheus_temp_dir,
        component_gauges,
    )


async def register_vllm_model(
    model_input: ModelInput,
    model_type: ModelType,
    generate_endpoint,
    config: Config,
    engine_client: AsyncLLM,
    vllm_config,
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
    # Decode workers don't create the WorkerKvQuery endpoint, so don't advertise local indexer
    runtime_config.enable_local_indexer = (
        config.enable_local_indexer
        and config.disaggregation_mode != DisaggregationMode.DECODE
    )

    # Add tool/reasoning parsers for decode models
    if model_type != ModelType.Prefill:
        runtime_config.tool_call_parser = config.dyn_tool_call_parser
        runtime_config.reasoning_parser = config.dyn_reasoning_parser

    # Get data_parallel_size from vllm_config (defaults to 1)
    data_parallel_size = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    runtime_config.data_parallel_size = data_parallel_size

    # Configure media decoder for frontend image decoding when enabled
    # This enables frontend to decode images and transfer via NIXL RDMA
    media_decoder = None
    media_fetcher = None
    if config.frontend_decoding:
        if not MEDIA_DECODER_AVAILABLE:
            raise RuntimeError(
                "--frontend-decoding requires MediaDecoder support. "
                "Ensure dynamo.llm module includes MediaDecoder and MediaFetcher."
            )
        media_decoder = MediaDecoder()
        media_decoder.enable_image({"limits": {"max_alloc": 128 * 1024 * 1024}})
        # media_decoder.enable_video({})

        media_fetcher = MediaFetcher()
        media_fetcher.timeout_ms(30000)
        media_fetcher.allow_direct_port(True)

    await register_model(
        model_input,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=runtime_values["block_size"],
        runtime_config=runtime_config,
        custom_template_path=config.custom_jinja_template,
        media_decoder=media_decoder,
        media_fetcher=media_fetcher,
    )


async def init_prefill(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    pre_created_engine=None,
):
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
    if pre_created_engine is not None:
        (
            engine_client,
            vllm_config,
            default_sampling_params,
            prometheus_temp_dir,
            _component_gauges,
        ) = pre_created_engine
    else:
        (
            engine_client,
            vllm_config,
            default_sampling_params,
            prometheus_temp_dir,
            _component_gauges,
        ) = setup_vllm_engine(config)

    handler = PrefillWorkerHandler(
        runtime,
        engine_client,
        default_sampling_params,
        getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
        enable_multimodal=config.enable_multimodal,
        generate_endpoint=generate_endpoint,
        config=config,
        use_vllm_tokenizer=config.use_vllm_tokenizer,
        shutdown_event=shutdown_event,
        enable_frontend_decoding=config.frontend_decoding,
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
        generate_endpoint,
        vllm_config,
        consolidator_enabled=consolidator_enabled,
        consolidator_port=consolidator_port,
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    setup_metrics_collection(config, generate_endpoint, logger)

    # Register sleep/wake_up engine routes
    runtime.register_engine_route("sleep", handler.sleep)
    runtime.register_engine_route("wake_up", handler.wake_up)
    logger.info("Registered engine routes: /engine/sleep, /engine/wake_up")

    # Handle non-leader nodes - don't serve endpoints
    if config.engine_args.data_parallel_rank:
        await _handle_non_leader_node(config.engine_args.data_parallel_rank)
        return
    shutdown_endpoints[:] = [generate_endpoint, clear_endpoint]

    # Register prefill model with ModelType.Prefill
    model_input = ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens
    await register_vllm_model(
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
                handler.generate,
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
                handler.clear_kv_blocks,
                metrics_labels=[
                    (prometheus_names.labels.MODEL, config.served_model_name),
                    (prometheus_names.labels.MODEL_NAME, config.served_model_name),
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


async def init(
    runtime: DistributedRuntime,
    config: Config,
    shutdown_event: asyncio.Event,
    pre_created_engine=None,
):
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
    if pre_created_engine is not None:
        (
            engine_client,
            vllm_config,
            default_sampling_params,
            prometheus_temp_dir,
            component_gauges,
        ) = pre_created_engine
        # Factory is created after unpack so component_gauges is available
        factory = StatLoggerFactory(
            endpoint=generate_endpoint,
            component_gauges=component_gauges,
            dp_rank=config.engine_args.data_parallel_rank or 0,
        )
    else:
        # Factory is created without component_gauges; setup_vllm_engine() will
        # create the gauges after setup_multiprocess_prometheus() and set them
        # on the factory before vLLM calls create_stat_logger().
        factory = StatLoggerFactory(
            endpoint=generate_endpoint,
            dp_rank=config.engine_args.data_parallel_rank or 0,
        )
        (
            engine_client,
            vllm_config,
            default_sampling_params,
            prometheus_temp_dir,
            component_gauges,
        ) = setup_vllm_engine(config, factory)

    # TODO Hack to get data, move this to registering in TBD
    factory.set_num_gpu_blocks_all(vllm_config.cache_config.num_gpu_blocks)
    factory.init_publish()

    handler = DecodeWorkerHandler(
        runtime,
        engine_client,
        default_sampling_params,
        getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
        enable_multimodal=config.enable_multimodal,
        generate_endpoint=generate_endpoint,
        config=config,
        use_vllm_tokenizer=config.use_vllm_tokenizer,
        shutdown_event=shutdown_event,
        enable_frontend_decoding=config.frontend_decoding,
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
        generate_endpoint,
        vllm_config,
        consolidator_enabled=consolidator_enabled,
        consolidator_port=consolidator_port,
    )
    if kv_publishers:
        handler.kv_publishers = kv_publishers

    setup_metrics_collection(config, generate_endpoint, logger)

    # Register sleep/wake_up engine routes
    runtime.register_engine_route("sleep", handler.sleep)
    runtime.register_engine_route("wake_up", handler.wake_up)
    logger.info("Registered engine routes: /engine/sleep, /engine/wake_up")

    # Handle non-leader nodes - don't serve endpoints
    if config.engine_args.data_parallel_rank:
        await _handle_non_leader_node(config.engine_args.data_parallel_rank)
        return

    # Parse endpoint types from --endpoint-types flag
    model_type = parse_endpoint_types(config.endpoint_types)
    logger.info(f"Registering model with endpoint types: {config.endpoint_types}")

    model_input = ModelInput.Text if config.use_vllm_tokenizer else ModelInput.Tokens

    # Warn if custom template provided but chat endpoint not enabled
    if config.custom_jinja_template and "chat" not in config.endpoint_types:
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
                handler.generate,
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


def get_engine_cache_info(engine: AsyncLLM):
    """Retrieve cache configuration information from [`AsyncLLM`] engine."""

    try:
        # Get values directly from vllm_config instead of collective_rpc
        cache_values = {
            "num_gpu_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
            "block_size": engine.vllm_config.cache_config.block_size,
        }

        scheduler_values = {
            "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
        }

        logging.info(f"Cache config values: {cache_values}")
        logging.info(f"Scheduler config values: {scheduler_values}")
        return {
            "num_gpu_blocks": cache_values["num_gpu_blocks"],
            "block_size": cache_values["block_size"],
            "max_num_seqs": scheduler_values["max_num_seqs"],
            "max_num_batched_tokens": scheduler_values["max_num_batched_tokens"],
        }
    except Exception as e:
        logging.error(f"Failed to get configuration values from vLLM config: {e}")
        raise


async def init_omni(
    runtime: DistributedRuntime, config: Config, shutdown_event: asyncio.Event
):
    """Initialize Omni worker for multi-stage pipeline generation using vLLM-Omni.

    Supports text-to-text, text-to-image, and text-to-video generation
    through a single unified OmniHandler.
    """
    from dynamo.vllm.omni import OmniHandler

    generate_endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )

    shutdown_endpoints[:] = [generate_endpoint]

    # Initialize media filesystem for storing generated images/videos
    media_fs = (
        get_fs(config.media_output_fs_url) if config.media_output_fs_url else None
    )

    # Initialize unified OmniHandler
    handler = OmniHandler(
        runtime=runtime,
        config=config,
        default_sampling_params={},
        shutdown_event=shutdown_event,
        media_output_fs=media_fs,
        media_output_http_url=config.media_output_http_url,
    )

    logger.info(f"Omni worker initialized for model: {config.model}")

    # Set up metrics collection for vLLM and LMCache metrics
    setup_metrics_collection(config, generate_endpoint, logger)

    # Handle non-leader nodes - don't serve endpoints
    if config.engine_args.data_parallel_rank:
        await _handle_non_leader_node(config.engine_args.data_parallel_rank)
        return

    # TODO: extend for multi-stage pipelines
    model_type = get_output_modalities(config.output_modalities, config.model)
    if model_type is None:
        # Default to Images
        model_type = ModelType.Images
    await register_model(
        ModelInput.Text,
        model_type,
        generate_endpoint,
        config.model,
        config.served_model_name,
        kv_cache_block_size=config.engine_args.block_size,
    )

    logger.info("Starting to serve Omni worker endpoint...")

    health_check_payload = (
        await VllmOmniHealthCheckPayload.create(handler.engine_client)
    ).to_dict()

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
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
        )
    except Exception as e:
        logger.error(f"Failed to serve Omni endpoint: {e}")
        raise
    finally:
        logger.debug("Cleaning up Omni worker")
        handler.cleanup()


def main():
    uvloop.run(worker())


if __name__ == "__main__":
    main()
