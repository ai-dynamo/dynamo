# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Optional

import uvloop
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess
from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
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
from dynamo.runtime import Endpoint
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.omni.args import OmniConfig
from dynamo.vllm.worker_factory import WorkerFactory

from . import envs
from .args import Config, _uses_dynamo_connector, parse_args
from .constants import DisaggregationMode
from .handlers import get_dp_range_for_worker
from .publisher import DYNAMO_COMPONENT_REGISTRY, StatLoggerFactory
from .snapshot import prepare_snapshot_engine

# Optional imports for frontend decoding support
MediaDecoder: type | None = None
MediaFetcher: type | None = None
try:
    from dynamo.llm import MediaDecoder, MediaFetcher

    MEDIA_DECODER_AVAILABLE = True
except ImportError:
    MediaDecoder = None
    MediaFetcher = None
    MEDIA_DECODER_AVAILABLE = False

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []


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
    # Propagate worker_cls for custom load formats so headless workers use
    # the same model loader and patches as the leader node.
    if config.engine_args.load_format == "gms":
        config.engine_args.worker_cls = (
            "gpu_memory_service.integrations.vllm.worker.GMSWorker"
        )

        if config.gms_shadow_mode:
            from gpu_memory_service.integrations.vllm.utils import (
                configure_gms_lock_mode,
                validate_cudagraph_mode,
            )

            os.environ["DYN_GMS_SHADOW_MODE"] = "1"
            configure_gms_lock_mode(config.engine_args)
            validate_cudagraph_mode(config.engine_args)

    elif config.engine_args.load_format in ("mx-source", "mx-target"):
        config.engine_args.worker_cls = "modelexpress.vllm_worker.ModelExpressWorker"

    # Keep the upstream CLI import local so tests that only exercise
    # build_headless_namespace() do not pull in vLLM's full CLI import graph.
    from vllm.entrypoints.cli.serve import run_headless

    args = build_headless_namespace(config)
    run_headless(args)


async def worker() -> None:
    config = parse_args()

    dump_config(config.dump_config_to, config)

    # Name the model. Use either the full path (vllm and sglang do the same),
    # or the HF name (e.g. "Qwen/Qwen3-0.6B"), depending on cmd line params.
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

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

    # CHECKPOINT MODE: Load engine BEFORE runtime creation
    # This allows checkpointing GPU state before runtime connections are established
    snapshot_controller = await prepare_snapshot_engine(
        config,
        setup_vllm_engine,
    )

    snapshot_engine = None
    if snapshot_controller is not None:
        snapshot_engine = snapshot_controller.engine
        (
            config.namespace,
            config.discovery_backend,
        ) = snapshot_controller.reload_restore_identity()

    # HEADLESS MODE: bypass DistributedRuntime entirely.
    # Workers run vLLM only (no NATS, etcd, or dynamo endpoints).
    if config.headless:
        run_dynamo_headless(config)
        return

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        use_kv_events=config.use_kv_events,
    )

    # [gluo FIXME] should be after init() below? 'shutdown_endpoints' are populated
    # there
    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    # Use WorkerFactory to appropriate initialize worker based on config flags
    factory = WorkerFactory(
        setup_vllm_engine_fn=setup_vllm_engine,
        setup_kv_event_publisher_fn=setup_kv_event_publisher,
        register_vllm_model_fn=register_vllm_model,
        setup_fpm_relay_fn=setup_fpm_relay,
        setup_metrics_collection_fn=setup_metrics_collection,
    )
    await factory.create(
        runtime,
        config,
        shutdown_event,
        shutdown_endpoints,
        snapshot_engine=snapshot_engine,
    )

    logger.debug("Worker function completed, exiting...")


def setup_metrics_collection(
    config: Config | OmniConfig, generate_endpoint: Endpoint, logger: logging.Logger
) -> None:
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

        multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        # After CRIU restore to another node, env still has the checkpoint pod's path
        # but that directory exists only on the checkpoint node; create it here if missing.
        if multiproc_dir and not os.path.isdir(multiproc_dir):
            try:
                os.makedirs(multiproc_dir, exist_ok=True)
            except OSError:
                pass
        if multiproc_dir and os.path.isdir(multiproc_dir):
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
            if multiproc_dir:
                logger.warning(
                    f"PROMETHEUS_MULTIPROC_DIR={multiproc_dir} is not a valid directory, "
                    "falling back to single-process metrics"
                )
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
) -> Optional[list[KvEventPublisher]]:
    """
    list[KvEventPublisher] | None
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

    # Get DP rank range managed by this worker to create publishers for corresponding dp_ranks,
    # all served workers should cover all ranks.
    dp_start, dp_size = get_dp_range_for_worker(vllm_config)
    kv_publishers = []

    for dp_rank in range(dp_start, dp_start + dp_size):
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


def setup_fpm_relay(
    generate_endpoint: Endpoint,
    vllm_config: VllmConfig,
) -> Optional[list]:
    """
    Set up forward pass metrics relays for the event plane.

    Creates one FpmEventRelay per dp_rank. Each relay subscribes to the
    local raw ZMQ PUB from InstrumentedScheduler (in the EngineCore child
    process) and re-publishes to the Dynamo event plane with automatic
    discovery registration.

    Returns:
        List of FpmEventRelay instances, or None if FPM is not enabled.
    """
    if not envs.is_set("DYN_FORWARDPASS_METRIC_PORT"):
        return None

    try:
        from dynamo.llm import FpmEventRelay
    except ImportError:
        logger.warning(
            "FpmEventRelay not available (Rust bindings not built with FPM support). "
            "Forward pass metrics will not be relayed to the event plane."
        )
        return None

    dp_start, dp_size = get_dp_range_for_worker(vllm_config)
    relays = []

    for dp_rank in range(dp_start, dp_start + dp_size):
        base_port = envs.DYN_FORWARDPASS_METRIC_PORT
        zmq_endpoint = f"tcp://127.0.0.1:{base_port + dp_rank}"

        relay = FpmEventRelay(
            endpoint=generate_endpoint,
            zmq_endpoint=zmq_endpoint,
        )
        relays.append(relay)

        logger.info(f"FPM relay for dp_rank={dp_rank} subscribing to {zmq_endpoint}")

    return relays if relays else None


def setup_vllm_engine(
    config: Config,
    stat_logger: Optional[StatLoggerFactory] = None,
    fpm_worker_id: Optional[str] = None,
) -> tuple[AsyncLLM, VllmConfig, Any, Any, LLMBackendMetrics]:
    # vLLM v0.11.0 bug: vllm/v1.metrics/prometheus.py:79 passes TemporaryDirectory object
    # instead of .name string, causing false error on exit. Set PROMETHEUS_MULTIPROC_DIR
    # ourselves to avoid this and handle cleanup properly.
    prometheus_temp_dir = None
    existing_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if existing_dir and not os.path.isdir(existing_dir):
        logger.warning(
            f"PROMETHEUS_MULTIPROC_DIR={existing_dir} does not exist, recreating"
        )
        os.makedirs(existing_dir, exist_ok=True)
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

        if config.gms_shadow_mode:
            from gpu_memory_service.integrations.vllm.utils import (
                configure_gms_lock_mode,
                validate_cudagraph_mode,
            )

            os.environ["DYN_GMS_SHADOW_MODE"] = "1"
            logger.info(
                "[Shadow] Enabled shadow mode: will skip KV cache allocation at startup"
            )
            # ENGINE_ID=0 writes weights, all others import (RO).
            # Prevents deadlock during TP>1 failover.
            configure_gms_lock_mode(engine_args)
            validate_cudagraph_mode(engine_args)

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

    # Configure ec_both mode with DynamoMultimodalEmbeddingCacheConnector.
    # Must happen BEFORE engine setup so vLLM sees ec_transfer_config.
    if (
        not config.route_to_encoder
        and config.multimodal_embedding_cache_capacity_gb > 0
    ):
        from vllm.config import ECTransferConfig

        logger.info(
            "Configuring ec_both mode with DynamoMultimodalEmbeddingCacheConnector "
            "(capacity=%.2f GB)",
            config.multimodal_embedding_cache_capacity_gb,
        )
        instance_id = 0
        engine_id = f"{config.namespace}.{config.component}.backend.{instance_id}"
        engine_args.ec_transfer_config = ECTransferConfig(
            engine_id=engine_id,
            ec_role="ec_both",
            ec_connector="DynamoMultimodalEmbeddingCacheConnector",
            ec_connector_module_path="dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector",
            ec_connector_extra_config={
                "multimodal_embedding_cache_capacity_gb": config.multimodal_embedding_cache_capacity_gb,
            },
        )
        logger.info("Configured ec_both with engine_id=%s", engine_id)

    # Taken from build_async_engine_client_from_engine_args()
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    # Set up consolidator endpoints if KVBM (DynamoConnector) is enabled
    consolidator_endpoints = None
    if _uses_dynamo_connector(config.engine_args):
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
    # Store consolidator endpoints in additional_config (vLLM 0.16+ uses strict
    # dataclass fields; monkey-patching attributes onto VllmConfig is no longer safe).
    vllm_config.additional_config["consolidator_endpoints"] = consolidator_endpoints

    # Pass worker identity to InstrumentedScheduler via additional_config.
    if fpm_worker_id is not None:
        vllm_config.additional_config["fpm_worker_id"] = fpm_worker_id

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
    generate_endpoint: Endpoint,
    config: Config,
    engine_client: AsyncLLM,
    vllm_config: VllmConfig,
) -> None:
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
    num_gpu_blocks = runtime_values["num_gpu_blocks"]
    if num_gpu_blocks is None:
        # TODO(upstream-vllm): remove this workaround once vLLM propagates
        # num_gpu_blocks from Ray DP workers back to the main-process vllm_config.
        # With Ray-based data-parallel backend, num_gpu_blocks is computed inside
        # Ray worker processes and is never written back to the main-process
        # vllm_config.  Use 0 as a sentinel so the Rust runtime can still register
        # the model; KV-cache capacity metrics will be unavailable in this mode.
        logging.warning(
            "num_gpu_blocks is None (expected when using --data-parallel-backend ray). "
            "Setting total_kv_blocks=0 for model registration."
        )
        num_gpu_blocks = 0
    runtime_config.total_kv_blocks = num_gpu_blocks
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
    runtime_config.exclude_tools_when_tool_choice_none = (
        config.exclude_tools_when_tool_choice_none
    )

    # Get data_parallel_size from vllm_config (defaults to 1)
    dp_range = get_dp_range_for_worker(vllm_config)
    runtime_config.data_parallel_start_rank = dp_range[0]
    runtime_config.data_parallel_size = dp_range[1]

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
        assert MediaDecoder is not None and MediaFetcher is not None
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
        context_length=vllm_config.model_config.max_model_len,
        kv_cache_block_size=runtime_values["block_size"],
        runtime_config=runtime_config,
        custom_template_path=config.custom_jinja_template,
        media_decoder=media_decoder,
        media_fetcher=media_fetcher,
    )


def get_engine_cache_info(engine: AsyncLLM) -> dict[str, Any]:
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


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
