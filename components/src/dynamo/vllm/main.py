# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from dynamo.vllm.omni.args import OmniConfig


def _early_truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _maybe_isolate_jit_cache_dirs_by_container() -> dict[str, tuple[str | None, str]]:
    """Keep runtime JIT/cache locks pod-local and engine-container-local.

    Bulwark primary and shadow containers can compile or materialize FlashInfer
    artifacts at the same time. Some Kubernetes-mounted filesystems do not
    reliably support fcntl locks under that contention, so keep these caches in
    a container-specific tmpfs namespace before vLLM imports FlashInfer.
    """

    if not _early_truthy_env(
        "DYN_VLLM_ISOLATE_JIT_CACHE_BY_CONTAINER",
        default=_early_truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE"),
    ):
        return {}

    container = os.environ.get("CONTAINER_NAME") or os.environ.get("ENGINE_ID")
    if not container:
        return {}

    base = os.environ.get("DYN_VLLM_JIT_CACHE_BASE", "/dev/shm/dynamo-jit")
    container_base = os.path.join(base, container)
    mappings = {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "xdg-cache",
        "VLLM_CACHE_ROOT": "vllm-cache",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor-cache",
        "TRITON_CACHE_DIR": "triton-cache",
        "CUDA_CACHE_PATH": "cuda-cache",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer-workspace",
        "FLASHINFER_CUBIN_DIR": "flashinfer-cubins",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "vllm-flashinfer-autotune",
        "TILELANG_CACHE_DIR": "tilelang-cache",
        "TILELANG_TMP_DIR": "tilelang-tmp",
    }

    changed: dict[str, tuple[str | None, str]] = {}
    force = _early_truthy_env("DYN_VLLM_FORCE_CONTAINER_JIT_CACHE", default=True)
    for name, suffix in mappings.items():
        previous = os.environ.get(name)
        value = os.path.join(container_base, suffix)
        if previous == value:
            continue
        if previous and not force and not previous.startswith(base):
            continue
        os.environ[name] = value
        changed[name] = (previous, value)
    return changed


_JIT_CACHE_DIR_ENV_CHANGES = _maybe_isolate_jit_cache_dirs_by_container()

import uvloop
from gpu_memory_service.integrations.vllm.kv_identity import (
    private_bootstrap_scratch_warmup_enabled,
)
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess
from vllm.config import VllmConfig
from vllm.distributed.kv_events import ZmqEventPublisher
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

from dynamo.common.config_dump import dump_config
from dynamo.common.model_fetch import fetch_model
from dynamo.common.snapshot.restore_context import (
    parse_snapshot_restore_runtime_config,
    refresh_snapshot_restore_config,
)
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.prometheus import (
    LLMBackendMetrics,
    register_engine_metrics_callback,
)
from dynamo.common.utils.runtime import create_runtime
from dynamo.common.utils.topology import apply_topology_config
from dynamo.gms_router_policy import resolve_vllm_gms_daemon_socket
from dynamo.llm import (
    KvEventPublisher,
    MediaDecoder,
    MediaFetcher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import Endpoint
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.worker_factory import WorkerFactory

from . import envs
from .args import Config, _uses_dynamo_connector, configure_rl_logprobs_mode, parse_args
from .cache_info import get_configured_kv_event_block_size
from .capacity import (
    get_metrics_model_name,
    get_spec_decode_runtime_data,
    per_rank_kv_blocks,
)
from .constants import DisaggregationMode
from .handlers import get_dp_range_for_worker
from .instrumented_scheduler import ENV_FPM_BENCHMARK_OUTPUT_PATH, ENV_FPM_WORKER_ID
from .publisher import DYNAMO_COMPONENT_REGISTRY, StatLoggerFactory
from .snapshot import prepare_snapshot_engine

configure_dynamo_logging()
logger = logging.getLogger(__name__)
if _JIT_CACHE_DIR_ENV_CHANGES:
    logger.info(
        "[GMS] Isolated vLLM runtime JIT/cache dirs for container=%s: %s",
        os.environ.get("CONTAINER_NAME") or os.environ.get("ENGINE_ID"),
        {name: value for name, (_, value) in _JIT_CACHE_DIR_ENV_CHANGES.items()},
    )
GMS_VLLM_WORKER_CLS = "gpu_memory_service.integrations.vllm.worker.GMSWorker"
shutdown_endpoints: list = []
SPEC_DECODE_RUNTIME_KEY = "spec_decode"
MX_LOAD_FORMATS = {"modelexpress", "mx"}


def uses_modelexpress_load_format(config: Config) -> bool:
    return getattr(config.engine_args, "load_format", None) in MX_LOAD_FORMATS


def should_prefetch_model(config: Config) -> bool:
    if os.path.exists(config.model):
        return False
    return not uses_modelexpress_load_format(config)


def should_register_model_ignore_weights(config: Config) -> bool:
    return uses_modelexpress_load_format(config)


def _register_model_source_path(config: Config, vllm_config: VllmConfig) -> str:
    """Pick the path passed to `register_model` for MDC construction.

    When `--model` is an object-storage URI (`s3://...`, `gs://...`, `az://...`),
    vLLM's `maybe_pull_model_tokenizer_for_runai` (vllm/config/model.py) pulls
    metadata files to a local temp dir and rewrites `vllm_config.model_config`:

      - `.model_weights = <original URI>`  (used by runai-streamer / mx plugin)
      - `.model = <local temp dir>`        (contains config.json, tokenizer, …)

    Dynamo's `register_model` would otherwise try to resolve the raw URI via
    `hub.rs` → ModelExpress, which has no S3 provider and 404s. Returning the
    local dir lets `register_model` take its `fs::exists` shortcut.

    Temporary vLLM-only workaround until `hub.rs` learns object-storage routing.
    Falls back to `config.model` whenever vLLM did not pull (HF id, local path,
    or older vLLM without `model_weights`).
    """
    if getattr(vllm_config.model_config, "model_weights", ""):
        return vllm_config.model_config.model
    return config.model


def _truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _gms_failover_shadow_member() -> bool:
    if not (
        _truthy_env("DYN_GMS_FAILOVER_SHADOW_MODE")
        or _truthy_env("DYN_VLLM_GMS_SHADOW_MODE")
    ):
        return False
    if _truthy_env("DYN_VLLM_GMS_ACTIVE_LOCK_HELD"):
        return False
    if _truthy_env("DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV"):
        return True
    engine_id = os.environ.get("ENGINE_ID", "0")
    primary_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    return engine_id != primary_id


def _gms_private_bootstrap_shadow_member(config: Config) -> bool:
    engine_args = getattr(config, "engine_args", None)
    if getattr(engine_args, "load_format", None) != "gms":
        return False
    if not getattr(config, "gms_shadow_mode", False):
        return False
    if not _truthy_env(
        "DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV",
        default=_truthy_env("GMS_VLLM_PRIVATE_BOOTSTRAP_KV"),
    ):
        return False
    return _gms_failover_shadow_member()


def _maybe_disable_gms_shadow_graphs_before_vllm_config(config: Config) -> bool:
    if not _gms_private_bootstrap_shadow_member(config):
        return False
    if _truthy_env("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_CUDAGRAPH"):
        return False
    if private_bootstrap_scratch_warmup_enabled():
        os.environ["GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED"] = "1"
        logger.info(
            "[GMS] Allowing vLLM graph warmup for scratch-backed "
            "private-bootstrap shadow KV"
        )
        return False

    previous = os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH")
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    logger.info(
        "[GMS] Disabled vLLM breakable cudagraph for private-bootstrap "
        "shadow before vLLM config creation (previous=%r)",
        previous,
    )
    return True


def _apply_gms_shadow_graph_config(vllm_config: VllmConfig, *, disabled: bool) -> None:
    if not disabled:
        return

    from vllm.config import CompilationMode, CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    previous_mode = getattr(compilation_config, "cudagraph_mode", None)
    previous_compile_mode = getattr(compilation_config, "mode", None)
    compilation_config.mode = CompilationMode.NONE
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    logger.info(
        "[GMS] Forced private-bootstrap shadow vLLM graph mode to eager: "
        "compile_mode %s -> %s, cudagraph_mode %s -> %s",
        previous_compile_mode,
        compilation_config.mode,
        previous_mode,
        compilation_config.cudagraph_mode,
    )


def _is_gms_load_format(engine_args: Any) -> bool:
    return str(getattr(engine_args, "load_format", "")) == "gms"


def _configure_gms_vllm_worker(engine_args: Any) -> None:
    """Force GMS worker setup early enough for spawned vLLM workers."""

    current = getattr(engine_args, "worker_cls", None)
    if current not in (None, "auto", GMS_VLLM_WORKER_CLS):
        logger.warning(
            "[GMS] Overriding user-provided vLLM worker_cls=%s with %s "
            "because --load-format=gms requires the GMS worker integration",
            current,
            GMS_VLLM_WORKER_CLS,
        )

    engine_args.worker_cls = GMS_VLLM_WORKER_CLS

    # Import eagerly so model-loader/KV patches fail before vLLM starts worker
    # processes. Worker subprocesses still resolve the class by string.
    import gpu_memory_service.integrations.vllm.worker  # noqa: F401

    logger.info("[GMS] vLLM worker_cls configured as %s", GMS_VLLM_WORKER_CLS)


def _verify_gms_vllm_worker_config(vllm_config: VllmConfig) -> None:
    worker_cls = getattr(vllm_config.parallel_config, "worker_cls", None)
    if worker_cls != GMS_VLLM_WORKER_CLS:
        raise RuntimeError(
            "GMS load format requires vLLM worker_cls="
            f"{GMS_VLLM_WORKER_CLS}, got {worker_cls!r}"
        )
    logger.info("[GMS] Final vLLM parallel_config.worker_cls=%s", worker_cls)


def _gms_shadow_init_geometry_wait_ms() -> int:
    names = (
        "DYN_VLLM_GMS_SHADOW_INIT_GEOMETRY_WAIT_MS",
        "GMS_VLLM_KV_GEOMETRY_WAIT_MS",
        "GMS_KV_LEASE_GEOMETRY_WAIT_MS",
    )
    wait_ms = 300_000
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            wait_ms = int(value)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, value)
            continue
        break
    return max(wait_ms, 0)


def _maybe_wait_for_gms_primary_kv_before_init(config: Config) -> None:
    if getattr(config.engine_args, "load_format", None) != "gms":
        return
    if not config.gms_shadow_mode:
        return
    if not _gms_failover_shadow_member():
        return
    if not _truthy_env("DYN_VLLM_GMS_WAIT_FOR_PRIMARY_KV_BEFORE_INIT", default=True):
        return

    from gpu_memory_service.integrations.vllm.install_vmm_ipc_kv import (
        _existing_shared_kv_blocks,
    )

    wait_ms = _gms_shadow_init_geometry_wait_ms()
    logger.info(
        "[GMS] Shadow engine waiting up to %d ms for primary KV geometry "
        "before vLLM engine initialization",
        wait_ms,
    )
    blocks = _existing_shared_kv_blocks(wait_ms=wait_ms)
    if blocks is None:
        raise RuntimeError(
            "Timed out waiting for primary GMS KV geometry before shadow "
            "vLLM engine initialization"
        )
    logger.info(
        "[GMS] Shadow engine observed primary KV geometry before init: blocks=%d",
        blocks,
    )


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
    if _is_gms_load_format(config.engine_args):
        _configure_gms_vllm_worker(config.engine_args)

        if config.gms_shadow_mode:
            from gpu_memory_service.integrations.vllm.utils import (
                configure_gms_lock_mode,
                configure_mx_ports,
            )

            configure_gms_lock_mode(config.engine_args)
            configure_mx_ports(config.engine_args)

    # ModelExpress uses vLLM's plugin path with --load-format=modelexpress.
    # Dynamo does not set a custom worker class here.

    # Cross-node liveness: heartbeat the leader so a crash of this worker node is
    # detected in ~one heartbeat-timeout instead of via the NCCL collective timeout.
    _maybe_start_vllm_rank_liveness_client(config)

    # Keep the upstream CLI import local so tests that only exercise
    # build_headless_namespace() do not pull in vLLM's full CLI import graph.
    from vllm.entrypoints.cli.serve import run_headless

    args = build_headless_namespace(config)
    run_headless(args)


def _maybe_start_vllm_rank_liveness_client(config: Config) -> None:
    """Start the worker side of the ZMQ rank-liveness channel (headless nodes)."""
    from dynamo.common import rank_liveness as rl

    if not rl.liveness_enabled() or not config.gms_shadow_mode:
        return
    ea = config.engine_args
    node_rank = int(getattr(ea, "node_rank", 0) or 0)
    leader_host = getattr(ea, "master_addr", None)
    nnodes = int(getattr(ea, "nnodes", 1) or 1)
    if nnodes <= 1 or node_rank < 1 or not leader_host:
        return
    rl.RankLivenessClient(leader_host, node_rank).start()


async def worker(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    config = parse_args(argv)

    dump_config(config.dump_config_to, config)

    # Name the model. Use either the full path (vllm and sglang do the same),
    # or the HF name (e.g. "Qwen/Qwen3-0.6B"), depending on cmd line params.
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model

    configure_rl_logprobs_mode(config)

    # Download the model if necessary using Dynamo's generic model fetch path.
    # We want it on disk before we start vllm to avoid downloading from HuggingFace.
    # When vLLM uses the ModelExpress plugin, the plugin owns acquisition through
    # P2P, ModelStreamer, GDS, or vLLM's native fallback.
    #
    # We don't set `config.engine_args.model` to the local path fetch_model returns
    # because vllm will send that name to its Ray pipeline-parallel workers, which
    # may not have the local path.
    # vllm will attempt to download the model again, but find it in the HF cache.
    # For non-HF models use a path instead of an HF name, and ensure all workers have
    # that path (ideally via a shared folder).
    if should_prefetch_model(config):
        await fetch_model(config.model)

    # Snapshot mode: load engine before runtime creation so there are no
    # runtime connections when CRIU captures GPU state.
    snapshot_controller = await prepare_snapshot_engine(
        config,
        setup_vllm_engine,
    )

    snapshot_engine = None
    if snapshot_controller is not None:
        snapshot_engine = snapshot_controller.engine
        config = await refresh_snapshot_restore_config(
            config,
            lambda: parse_snapshot_restore_runtime_config(argv),
        )

    # HEADLESS MODE: bypass DistributedRuntime entirely.
    # Workers run vLLM only (no NATS, etcd, or dynamo endpoints).
    if config.headless:
        run_dynamo_headless(config)
        return

    shutdown_event = asyncio.Event()
    engine_holder: list = []
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
    )

    async def cleanup_engine() -> None:
        if not engine_holder:
            logger.info("vLLM engine not initialized; skipping cleanup")
            return

        engine = engine_holder[0]
        try:
            engine.shutdown()
            logger.info("vLLM engine shutdown complete")
        except Exception as exc:
            logger.warning("vLLM engine shutdown failed: %s", exc)

    # [gluo FIXME] should be after init() below? 'shutdown_endpoints' are populated
    # there
    install_signal_handlers(
        loop,
        runtime,
        shutdown_endpoints,
        shutdown_event,
        cleanup_callback=cleanup_engine,
    )

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
        engine_holder=engine_holder,
    )

    logger.debug("Worker function completed, exiting...")


def setup_metrics_collection(
    config: "Config | OmniConfig", generate_endpoint: Endpoint, logger: logging.Logger
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
    metrics_model_name = get_metrics_model_name(config)
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
        # After CRIU restore to another node, env still has the snapshot pod's path
        # but that directory exists only on that node; create it here if missing.
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
                    model_name=metrics_model_name,
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
                    model_name=metrics_model_name,
                )
                # Multiproc registry has .db file metrics (lmcache, possibly vllm duplicates)
                register_engine_metrics_callback(
                    endpoint=generate_endpoint,
                    registry=multiproc_registry,
                    metric_prefix_filters=["vllm:", "lmcache:"],
                    namespace_name=config.namespace,
                    component_name=config.component,
                    endpoint_name=config.endpoint,
                    model_name=metrics_model_name,
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
                model_name=metrics_model_name,
            )


def _resolve_image_token_id(config: Config, vllm_config: VllmConfig) -> Optional[int]:
    """Routing-side image-placeholder token id for the served model.

    Resolved via the SAME Rust logic the frontend uses
    (`dynamo._core.resolve_routing_image_token_id` ->
    `lightseek_mm::resolve_routing_tokens`), returning `chat_placeholder_token_id`
    so the KV-event normalizer keys on the identical token the frontend
    substitutes `pad_value` over — no per-family drift between the two.

    Returns None when the bindings lack the `mm-routing` feature or the model
    isn't in the MM-routing registry — in both cases the frontend also skips MM
    routing, so a worker-side no-op is consistent (events pass through).
    """
    try:
        from dynamo._core import resolve_routing_image_token_id
    except ImportError:
        return None

    # vLLM has already resolved the model to a local dir (config.json +
    # tokenizer.json on disk) during engine init; read from there.
    return resolve_routing_image_token_id(config.model, vllm_config.model_config.model)


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
    kv_event_block_size = get_configured_kv_event_block_size(vllm_config)
    # The image-placeholder token id the frontend substitutes pad_value over.
    # Passed to the KV publisher so the router-side normalizer rewrites those
    # runs in vLLM BlockStored events to the same canonical pad_value scheme.
    # None (no mm-routing, model not in registry, text-only) leaves events
    # unchanged — consistent with the frontend also skipping MM routing.
    image_token_id = _resolve_image_token_id(config, vllm_config)

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
            kv_block_size=kv_event_block_size,
            zmq_endpoint=zmq_endpoint,
            zmq_topic="",
            enable_local_indexer=config.enable_local_indexer,
            dp_rank=dp_rank,
            image_token_id=image_token_id,
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
) -> tuple[AsyncLLM, VllmConfig, Any, Any, Optional[LLMBackendMetrics]]:
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
    #
    # Embedding workers (pooling engines) have no KV cache, no scheduler
    # gauges, and no model_load_time hook -- registering the chat-shaped
    # LLMBackendMetrics on them publishes zeros forever. Skip the
    # construction entirely on that path so /metrics stays clean.
    embedding_worker = stat_logger is not None and stat_logger.embedding_worker
    component_gauges: Optional[LLMBackendMetrics] = None
    if not embedding_worker:
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

    if _is_gms_load_format(engine_args):
        _configure_gms_vllm_worker(engine_args)

        if config.gms_shadow_mode:
            from gpu_memory_service.integrations.vllm.utils import (
                configure_gms_lock_mode,
                configure_mx_ports,
            )

            # ENGINE_ID=0 writes weights, all others import (RO).
            # Prevents deadlock during TP>1 failover.
            configure_gms_lock_mode(engine_args)
            configure_mx_ports(engine_args)

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

    gms_shadow_graphs_disabled = _maybe_disable_gms_shadow_graphs_before_vllm_config(
        config
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
    default_sampling_params = vllm_config.model_config.get_diff_sampling_param()
    if _is_gms_load_format(engine_args):
        _verify_gms_vllm_worker_config(vllm_config)
    _apply_gms_shadow_graph_config(vllm_config, disabled=gms_shadow_graphs_disabled)

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

    # Pass runtime-only worker identity to InstrumentedScheduler via the
    # environment so it does not perturb vLLM's config hash.
    if fpm_worker_id is not None:
        os.environ[ENV_FPM_WORKER_ID] = fpm_worker_id

    # Pass benchmark config to InstrumentedScheduler via additional_config.
    if hasattr(config, "_benchmark_additional_config"):
        bench = config._benchmark_additional_config
        if fpm_worker_id and bench["output_path"] == "/tmp/benchmark_results.json":
            short_id = fpm_worker_id[-8:]
            os.environ[
                ENV_FPM_BENCHMARK_OUTPUT_PATH
            ] = f"/tmp/benchmark_results_{short_id}.json"
        vllm_config.additional_config["benchmark"] = bench
        logger.info("Benchmark config injected into additional_config")

    factory = []
    if stat_logger:
        factory.append(stat_logger)

    _maybe_wait_for_gms_primary_kv_before_init(config)

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

    # Record model load time. ``component_gauges`` is None on the
    # embedding-worker path -- pooling engines have no chat-shaped gauges
    # registered, so model_load_time has no collector to publish to.
    # Skip rather than fabricating a zero-valued sample.
    if component_gauges is not None:
        component_gauges.set_model_load_time(load_time)

    logger.info(f"VllmWorker for {config.served_model_name} has been initialized")

    # update block_size in vllm_config based on final engine cache info for later use
    runtime_values = get_engine_cache_info(engine_client)
    vllm_config.cache_config.block_size = runtime_values["block_size"]

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
    worker_type: WorkerType,
    needs: list[list[WorkerType]] | None = None,
) -> None:
    """
    Helper function to register a vLLM model with runtime configuration.

    Args:
        model_input: Input type for the model (e.g., ModelInput.Tokens)
        model_type: OpenAI surface this card exposes (e.g., ModelType.Chat).
            Prefill workers have no OpenAI surface — their role is carried by
            `worker_type=WorkerType.Prefill` — but pass the legacy
            `ModelType.Prefill` marker bit (not a surface) so an old frontend
            still detects them during the cross-version rollout.
        generate_endpoint: Endpoint to register
        config: Configuration object
        engine_client: vLLM engine client
        vllm_config: vLLM configuration
        worker_type: The disaggregation role this worker plays
            (Prefill / Decode / Encode / Aggregated). Required by the
            frontend's model-serving-readiness check.
        needs: Peer worker types required to serve traffic, in DNF form
            (list of alternative AND-sets).
    """
    runtime_config = ModelRuntimeConfig()
    runtime_config.context_length = vllm_config.model_config.max_model_len

    # Get runtime configuration from vLLM engine
    logging.info(
        f"Getting engine runtime configuration metadata from vLLM engine for {model_type}..."
    )
    runtime_values = get_engine_cache_info(engine_client)
    num_gpu_blocks = runtime_values["num_gpu_blocks"]
    # Get data_parallel_size from vllm_config (defaults to 1)
    dp_range = get_dp_range_for_worker(vllm_config)
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
    runtime_config.total_kv_blocks = per_rank_kv_blocks(num_gpu_blocks, dp_range[1])
    runtime_config.max_num_seqs = runtime_values["max_num_seqs"]
    runtime_config.max_num_batched_tokens = runtime_values["max_num_batched_tokens"]
    # Decode workers don't create the WorkerKvQuery endpoint, so don't advertise local indexer
    runtime_config.enable_local_indexer = (
        config.enable_local_indexer
        and config.disaggregation_mode != DisaggregationMode.DECODE
    )

    # Add tool/reasoning parsers for decode/aggregated workers. Prefill
    # workers have no OpenAI surface and don't run a parser — key off
    # `worker_type` to skip them.
    if worker_type != WorkerType.Prefill:
        runtime_config.tool_call_parser = config.dyn_tool_call_parser
        runtime_config.reasoning_parser = config.dyn_reasoning_parser
    runtime_config.exclude_tools_when_tool_choice_none = (
        config.exclude_tools_when_tool_choice_none
    )
    runtime_config.set_structural_tag_mode(
        "on" if config.dyn_enable_structural_tag else "off"
    )
    runtime_config.set_structural_tag_scope(config.dyn_structural_tag_scope)
    runtime_config.set_structural_tag_schema(config.dyn_structural_tag_schema)

    # Propagate stream_interval so the frontend can respect --stream-interval.
    # set_engine_specific requires a JSON-encoded string (the Rust binding
    # parses it with serde_json::from_str); str(int) happens to be valid JSON.
    stream_interval = getattr(config.engine_args, "stream_interval", None)
    if stream_interval is not None:
        runtime_config.set_engine_specific("stream_interval", str(stream_interval))

    spec_decode = get_spec_decode_runtime_data(config, vllm_config)
    if spec_decode is not None:
        runtime_config.set_engine_specific(
            SPEC_DECODE_RUNTIME_KEY, json.dumps(spec_decode)
        )
        logging.info("Published vLLM spec decode runtime metadata: %s", spec_decode)

    if resolve_vllm_gms_daemon_socket(vllm_config):
        runtime_config.set_gms_placement_enabled()
        logging.info("Publishing GMS placement capability to discovery")

    runtime_config.data_parallel_start_rank = dp_range[0]
    runtime_config.data_parallel_size = dp_range[1]

    # Set topology and KV transfer policy for topology-aware routing
    apply_topology_config(runtime_config)

    # Configure media decoder for frontend image decoding when enabled
    # This enables frontend to decode images and transfer via NIXL RDMA
    media_decoder = None
    media_fetcher = None
    if config.frontend_decoding:
        media_decoder = MediaDecoder()
        media_decoder.enable_image({"limits": {"max_alloc": 128 * 1024 * 1024}})
        # media_decoder.enable_video({})

        media_fetcher = MediaFetcher()
        media_fetcher.timeout_ms(30000)
        allow_internal = os.getenv("DYN_MM_ALLOW_INTERNAL", "0") == "1"
        media_fetcher.allow_direct_ip(allow_internal)
        media_fetcher.allow_direct_port(allow_internal)

    await register_model(
        model_input,
        model_type,
        generate_endpoint,
        _register_model_source_path(config, vllm_config),
        config.served_model_name,
        kv_cache_block_size=runtime_values["kv_event_block_size"],
        runtime_config=runtime_config,
        custom_template_path=config.custom_jinja_template,
        media_decoder=media_decoder,
        media_fetcher=media_fetcher,
        worker_type=worker_type,
        needs=needs,
        ignore_weights=should_register_model_ignore_weights(config),
    )


def get_engine_cache_info(engine: AsyncLLM) -> dict[str, Any]:
    """Return vLLM cache and scheduler limits used for model registration."""

    try:
        # Get values directly from vllm_config instead of collective_rpc
        kv_event_block_size = get_configured_kv_event_block_size(engine.vllm_config)
        cache_values = {
            "num_gpu_blocks": engine.vllm_config.cache_config.num_gpu_blocks,
            "block_size": engine.vllm_config.cache_config.block_size,
            "kv_event_block_size": kv_event_block_size,
        }

        scheduler_values = {
            "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": engine.vllm_config.scheduler_config.max_num_batched_tokens,
        }

        logging.debug(f"Cache config values: {cache_values}")
        logging.debug(f"Scheduler config values: {scheduler_values}")
        return {
            "num_gpu_blocks": cache_values["num_gpu_blocks"],
            "block_size": cache_values["block_size"],
            "kv_event_block_size": cache_values["kv_event_block_size"],
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
