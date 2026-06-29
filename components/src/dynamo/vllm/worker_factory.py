# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization factory for vLLM workers."""

import asyncio
import json
import logging
import os
import time as _time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo import prometheus_names
from dynamo.common.gms_failover import (
    release_attached_gms_failover_lock,
    run_gms_failover_post_lock_fence,
    run_gms_failover_promotion_warmup,
)
from dynamo.common.rl import first_endpoint_response, register_rl_routes
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import (
    LLMBackendMetrics,
    register_embedding_cache_metrics,
)
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime

from .args import Config
from .cache_info import configure_kv_event_block_size
from .capacity import per_rank_kv_blocks
from .constants import DisaggregationMode
from .handlers import (
    BaseWorkerHandler,
    DecodeWorkerHandler,
    EmbeddingWorkerHandler,
    PrefillWorkerHandler,
    get_dp_range_for_worker,
)
from .health_check import (
    VllmEmbeddingHealthCheckPayload,
    VllmHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
)
from .instrumented_scheduler import ENV_FPM_BENCHMARK_OUTPUT_PATH, ENV_FPM_WORKER_ID
from .multimodal_handlers import EncodeWorkerHandler
from .publisher import StatLoggerFactory

logger = logging.getLogger(__name__)

# (engine_client, vllm_config, default_sampling_params, prometheus_temp_dir, component_gauges)
# component_gauges is None on the embedding-worker path: pooling engines
# have no KV cache / scheduler gauges, so setup_vllm_engine() skips the
# LLMBackendMetrics registration there.
EngineSetupResult = tuple[AsyncLLM, VllmConfig, Any, Any, Optional[LLMBackendMetrics]]


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r", name, value)
        return default


def _truthy_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _vllm_prelock_shadow_warmup_enabled() -> bool:
    return _truthy_env("DYN_VLLM_GMS_PREWARM_SHADOW_BEFORE_LOCK")


def _vllm_scratch_private_bootstrap_enabled() -> bool:
    return _truthy_env(
        "DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_SCRATCH_WARMUP",
        default=_truthy_env("GMS_VLLM_PRIVATE_BOOTSTRAP_SCRATCH_WARMUP"),
    )


def _vllm_failover_shape_warmup_payload(base_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a small real-request canary that covers post-failover serving JIT."""

    payload = dict(base_payload)
    payload.pop("_HEALTH_CHECK", None)
    max_tokens = max(
        1,
        _int_env(
            "DYN_VLLM_GMS_FAILOVER_SHAPE_WARMUP_MAX_TOKENS",
            _int_env("DYN_GMS_FAILOVER_SHAPE_WARMUP_MAX_TOKENS", 16),
        ),
    )
    token_count = max(
        1,
        _int_env(
            "DYN_VLLM_GMS_FAILOVER_SHAPE_WARMUP_INPUT_TOKENS",
            _int_env("DYN_GMS_FAILOVER_SHAPE_WARMUP_INPUT_TOKENS", 25),
        ),
    )

    if "token_ids" in payload:
        token_ids = payload.get("token_ids") or [1]
        token_id = int(token_ids[0])
        payload["token_ids"] = [token_id] * token_count
        sampling_options = dict(payload.get("sampling_options") or {})
        sampling_options["temperature"] = 0.0
        payload["sampling_options"] = sampling_options
        stop_conditions = dict(payload.get("stop_conditions") or {})
        stop_conditions["max_tokens"] = max_tokens
        payload["stop_conditions"] = stop_conditions
        return payload

    payload["prompt"] = os.environ.get(
        "DYN_VLLM_GMS_FAILOVER_SHAPE_WARMUP_PROMPT",
        "Return one concise deterministic sentence about GMS failover validation.",
    )
    payload["temperature"] = 0.0
    payload["max_tokens"] = max_tokens
    return payload


async def _wait_and_load_benchmark(bench_cfg: dict, vllm_config: VllmConfig) -> dict:
    """Wait for benchmark result files and aggregate across DP ranks."""
    base_path = Path(
        os.environ.get(ENV_FPM_BENCHMARK_OUTPUT_PATH, bench_cfg["output_path"])
    )
    timeout = int(bench_cfg.get("timeout", 300))

    try:
        dp_start, dp_size = get_dp_range_for_worker(vllm_config)
    except Exception:
        logger.warning(
            "Could not determine DP range, assuming single rank",
            exc_info=True,
        )
        dp_start, dp_size = 0, 1

    rank_paths = []
    for dp_rank in range(dp_start, dp_start + dp_size):
        if dp_rank == 0:
            rank_paths.append(base_path)
        else:
            stem, ext = os.path.splitext(str(base_path))
            rank_paths.append(Path(f"{stem}_dp{dp_rank}{ext}"))

    logger.info(
        "Waiting for benchmark to complete (files: %s, timeout: %ds)...",
        rank_paths,
        timeout,
    )

    deadline = _time.monotonic() + timeout
    for p in rank_paths:
        while not p.exists():
            if _time.monotonic() > deadline:
                raise TimeoutError(
                    f"Benchmark did not complete within {timeout}s. Missing: {p}"
                )
            await asyncio.sleep(0.1)

    merged: dict = {}
    for i, p in enumerate(rank_paths):
        with open(p) as f:
            data = json.load(f)
        if i == 0:
            merged = data
            for r in merged.get("results", []):
                r["point"]["dp_rank"] = dp_start
        else:
            dp_rank = dp_start + i
            for r in data.get("results", []):
                r["point"]["dp_rank"] = dp_rank
            merged.setdefault("results", []).extend(data.get("results", []))

    logger.info(
        "Benchmark complete, %d points across %d rank(s)",
        len(merged.get("results", [])),
        len(rank_paths),
    )
    return merged


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
        engine_holder: Optional[list] = None,
    ) -> None:
        """Create the appropriate multimodal worker based on config flags."""

        # Embedding worker is selected first because it crosses worker shapes
        # (pooling AsyncLLM, ModelType.Embedding) rather than being a variant
        # of decode. Aggregated-only — exclusivity with disagg modes is
        # enforced earlier in DynamoVllmConfig._validate_embedding_worker_exclusivity.
        if config.embedding_worker:
            await self._create_embedding_worker(
                runtime, config, shutdown_event, shutdown_endpoints
            )
            return

        # NOTE: --benchmark-mode is only supported for prefill/decode workers.
        # The encode worker path does not wire benchmark waiting or
        # the get_perf_metrics endpoint.
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
                engine_holder=engine_holder,
            )
        else:
            # AGGREGATED or DECODE
            await self._create_decode_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                snapshot_engine=snapshot_engine,
                engine_holder=engine_holder,
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
            config.engine_args,
            config.embedding_transfer_mode,  # type: ignore[arg-type]
        )
        await handler.async_init(runtime)

        # Encode workers register a model card so the frontend's
        # serving-readiness gate can count them. The card carries no OpenAI
        # surface (`ModelType.Empty`) — the encode endpoint isn't routed by
        # the OpenAI dispatch. `needs` is the DNF for an encode worker:
        # either a P+D pair or a single Aggregated peer.
        await register_model(
            ModelInput.Tokens,
            ModelType.Empty,
            generate_endpoint,
            config.model,
            model_name=config.served_model_name or config.model,
            worker_type=WorkerType.Encode,
            needs=[
                [WorkerType.Prefill, WorkerType.Decode],
                [WorkerType.Aggregated],
            ],
        )
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

    async def _create_embedding_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
    ) -> None:
        """Initialize an aggregated text-embedding worker.

        Pooling models have no KV cache, no decode phase, and no streamed
        output, so several pieces of the decode-worker setup are intentionally
        skipped here:

        - KV-events publisher: no KV cache → nothing to publish.
        - Forward-pass-metrics relay: relays decode-phase ZMQ metrics; no
          decode here.
        - StatLoggerFactory wiring: built around per-batch sampling/decoding
          stats which the pooling engine does not emit.
        - InstrumentedScheduler: hard-codes ``pooling_params=None`` (see
          components/src/dynamo/vllm/instrumented_scheduler.py), which would
          silently disable the pooling pass. ``setup_vllm_engine`` only
          installs it when ``--benchmark-mode`` is set, which is rejected
          for embedding workers via config validation.

          We are deliberately not extending ``--benchmark-mode`` with an
          ``embed`` choice. That flag exists primarily to expose a worker's
          capability curve (RPS / p99 vs. concurrency, throughput knee) at
          startup for capacity planning, engine-arg tuning, and as input to
          the Dynamo planner's auto-scaling decisions. Decode workloads
          benefit because they have many interacting knobs (max-num-seqs,
          chunked prefill, prefill/decode mix). Embedding workloads are
          essentially ``(batch_size × ISL → latency)`` -- a clean two-axis
          function -- so the value of in-process self-profiling is much
          lower than external HTTP load testing, which is what every other
          embedding-serving stack uses anyway. The single remaining wedge
          is planner integration: if/when the Dynamo planner needs
          in-process embedding capability curves to auto-scale embedding
          fleets, add ``--benchmark-mode embed`` at that point together
          with the planner's embedding-capability model.

        The engine itself is the standard ``AsyncLLM`` constructed by
        ``setup_vllm_engine``; pooling vs. generation is selected by the
        user's ``--runner pooling`` argument flowing through ``engine_args``.
        """
        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        shutdown_endpoints[:] = [generate_endpoint]

        fpm_worker_id = str(generate_endpoint.connection_id())
        # Embedding workers run on pooling engines: no KV cache, no
        # scheduler stats, no decode loop. The factory still has to exist
        # because vLLM unconditionally invokes it during AsyncLLM init,
        # but it returns a no-op stat logger and setup_vllm_engine() skips
        # the chat-shaped LLMBackendMetrics registration.
        factory = StatLoggerFactory(
            endpoint=generate_endpoint,
            embedding_worker=True,
        )
        (
            engine_client,
            vllm_config,
            _default_sampling_params,
            _prometheus_temp_dir,
            _component_gauges,
        ) = self.setup_vllm_engine(config, factory, fpm_worker_id=fpm_worker_id)

        handler = EmbeddingWorkerHandler(
            runtime=runtime,
            engine=engine_client,
            config=config,
            shutdown_event=shutdown_event,
        )

        embedding_health_check_payload = VllmEmbeddingHealthCheckPayload(
            model_name=config.served_model_name or config.model
        ).to_dict()

        logger.info("Starting to serve the embedding worker endpoint...")
        try:
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=[("model", config.model)],
                    health_check_payload=embedding_health_check_payload,
                ),
                self.register_vllm_model(
                    ModelInput.Text,
                    ModelType.Embedding,
                    generate_endpoint,
                    config,
                    engine_client,
                    vllm_config,
                    # Embedding workers have no prefill/decode split — they
                    # always serve a single pooling pass, so they advertise
                    # as Aggregated with no peer dependencies.
                    worker_type=WorkerType.Aggregated,
                    needs=[],
                ),
            )
        except Exception as e:
            logger.error(f"Failed to serve embedding worker endpoint: {e}")
            raise
        finally:
            handler.cleanup()

    @staticmethod
    def _truthy_env(name: str, *, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() not in ("", "0", "false", "no", "off")

    @classmethod
    def _private_bootstrap_requested(cls) -> bool:
        return cls._truthy_env(
            "DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV",
            default=cls._truthy_env(
                "GMS_VLLM_PRIVATE_BOOTSTRAP_KV",
                default=cls._truthy_env("DYN_GMS_FAILOVER_PRIVATE_BOOTSTRAP_KV"),
            ),
        )

    async def _acquire_failover_lock(self, *, timeout: float | None = None):
        from gpu_memory_service.failover_lock.flock import FlockFailoverLock

        lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
        engine_id = os.environ.get("ENGINE_ID", "0")
        lock = FlockFailoverLock(lock_path)
        await lock.acquire(engine_id=f"engine-{engine_id}", timeout=timeout)
        return lock

    def _maybe_start_rank_liveness_monitor(self, handler, config: Config):
        """Leader side of the cross-node ZMQ rank-liveness channel.

        Only the rank-0 leader runs worker_factory (headless workers bypass it),
        so reaching here means we are the active leader. On a worker node going
        silent (process death), release the failover lock so the warm shadow
        promotes in ~one heartbeat-timeout, then signal a graceful shutdown of the
        now-broken leader — instead of waiting out the NCCL collective timeout.
        """
        import signal

        from dynamo.common import rank_liveness as rl

        if not rl.liveness_enabled() or not getattr(config, "gms_shadow_mode", False):
            return None
        nnodes = int(getattr(config.engine_args, "nnodes", 1) or 1)
        if nnodes <= 1:
            return None

        loop = asyncio.get_running_loop()

        def on_rank_lost(rank: int, reason: str) -> None:
            logger.warning(
                "[GMS liveness] vLLM worker rank %d lost (%s); releasing lock + "
                "shutting down broken leader for fast shadow promotion",
                rank,
                reason,
            )
            try:
                asyncio.run_coroutine_threadsafe(
                    release_attached_gms_failover_lock(handler, backend_name="vllm"),
                    loop,
                )
            except Exception:
                logger.debug("[GMS liveness] lock release on rank loss failed", exc_info=True)
            # The cohort is broken (a TP rank is gone); bring the leader down so it
            # stops holding the GPU/KV and the shadow (now lock holder) serves.
            os.kill(os.getpid(), signal.SIGTERM)

        monitor = rl.RankLivenessMonitor(
            on_rank_lost, expected_ranks=range(1, nnodes)
        )
        setattr(handler, "_gms_rank_liveness_monitor", monitor)
        monitor.start()
        logger.info("[GMS liveness] started vLLM leader rank-liveness monitor")
        return monitor

    async def _configure_gms_preinit_failover_role(
        self,
        config: Config,
    ) -> tuple[Any | None, bool]:
        """Choose shared vs private-bootstrap KV before vLLM initializes.

        vLLM sizes/profiles memory before the worker handler exists. In a
        Bulwark pair that uses private-bootstrap shadows, the process that owns
        the failover lock may initialize against the stable shared KV namespace;
        all other processes must initialize against member-scoped private KV.
        This is dynamic: after failover, a restarted ENGINE_ID=0 container is a
        standby while ENGINE_ID=1 owns the lock.
        """

        os.environ.pop("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", None)
        os.environ.pop("DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV", None)

        if not config.gms_shadow_mode:
            return None, False
        if getattr(config.engine_args, "load_format", None) != "gms":
            return None, False
        if not self._private_bootstrap_requested():
            return None, False
        if not self._truthy_env("DYN_VLLM_GMS_DYNAMIC_PREINIT_ROLE", default=True):
            return None, False

        engine_id = os.environ.get("ENGINE_ID", "0")
        primary_engine_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
        is_static_primary = engine_id == primary_engine_id

        if not is_static_primary:
            os.environ["DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV"] = "1"
            os.environ["DYN_VLLM_GMS_ACTIVE_LOCK_HELD"] = "0"
            logger.info(
                "[GMS] Static shadow engine-%s initializing with private-bootstrap KV",
                engine_id,
            )
            return None, False

        logger.info(
            "[GMS] Static primary engine-%s attempting active failover lock "
            "before vLLM engine init",
            engine_id,
        )
        try:
            lock = await self._acquire_failover_lock(timeout=0.0)
        except Exception as exc:  # noqa: BLE001 - lock contention is expected.
            try:
                from gpu_memory_service.failover_lock.interface import FailoverLockError
            except ImportError:  # pragma: no cover
                FailoverLockError = RuntimeError  # type: ignore[assignment]

            if not isinstance(exc, FailoverLockError):
                raise
            os.environ["DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV"] = "1"
            os.environ["DYN_VLLM_GMS_ACTIVE_LOCK_HELD"] = "0"
            logger.info(
                "[GMS] Active failover lock is held; static primary engine-%s "
                "will initialize as private-bootstrap standby",
                engine_id,
            )
            return None, False

        os.environ["DYN_VLLM_GMS_ACTIVE_LOCK_HELD"] = "1"
        os.environ["DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV"] = "0"
        await run_gms_failover_post_lock_fence(
            backend_name="vllm",
            role=f"engine-{engine_id}-pre-init",
        )
        logger.info(
            "[GMS] Static primary engine-%s owns active lock before vLLM init",
            engine_id,
        )
        return lock, True

    async def _maybe_wait_for_failover_lock(
        self,
        handler,
        runtime: DistributedRuntime,
        config: Config,
        *,
        lock_already_acquired: bool = False,
        promotion_warmup: Callable[[], Awaitable[None]] | None = None,
        prelock_warmup: Callable[[], Awaitable[None]] | None = None,
        post_lock_fence_already_run: bool = False,
    ) -> None:
        # Shadow mode: lock-driven activation.
        # Default safe flow for GMS shared KV is lock-before-init, because vLLM
        # warmup writes KV before this handler can quiesce the engine. The legacy
        # warm-standby path remains behind DYN_VLLM_GMS_LOCK_BEFORE_INIT=0.
        if not config.gms_shadow_mode:
            return

        if lock_already_acquired:
            if not post_lock_fence_already_run:
                await run_gms_failover_post_lock_fence(
                    backend_name="vllm",
                    role="pre-init",
                )
            if promotion_warmup is not None:
                logger.info(
                    "[GMS failover] Skipping promotion warmup for pre-init "
                    "active lock; warmup is shadow-only"
                )
            logger.info(
                "[Shadow] Failover lock already acquired before engine init; "
                "registering with discovery"
            )
            return

        engine_id = os.environ.get("ENGINE_ID", "0")
        primary_engine_id = os.environ.get("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
        forced_private_bootstrap = self._truthy_env(
            "DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV"
        )
        is_shadow = forced_private_bootstrap or engine_id != primary_engine_id
        private_bootstrap = self._private_bootstrap_requested() and is_shadow

        if not is_shadow:
            logger.info(
                "[Primary] Engine initialized; acquiring active failover lock "
                "before discovery registration"
            )
            lock = await self._acquire_failover_lock()
            setattr(handler, "_gms_failover_lock", lock)
            await run_gms_failover_post_lock_fence(
                backend_name="vllm",
                role="active",
            )
            if promotion_warmup is not None:
                logger.info(
                    "[GMS failover] Skipping promotion warmup for active primary; "
                    "warmup is shadow-only"
                )
            self._maybe_start_rank_liveness_monitor(handler, config)
            logger.info("[Primary] Active lock acquired, registering with discovery")
            return

        skip_startup_sleep = os.environ.get(
            "DYN_VLLM_GMS_SHADOW_SKIP_STARTUP_SLEEP",
            "1" if private_bootstrap else "0",
        ).lower() not in {"0", "false", "no", "off"}
        if skip_startup_sleep:
            prelock_warmup_ran = False
            if (
                private_bootstrap
                and prelock_warmup is not None
                and _vllm_prelock_shadow_warmup_enabled()
            ):
                if _vllm_scratch_private_bootstrap_enabled():
                    logger.info(
                        "[Shadow] Running pre-lock scratch-backed warmup while "
                        "undiscovered"
                    )
                    await prelock_warmup()
                    prelock_warmup_ran = True
                    logger.info(
                        "[Shadow] Pre-lock scratch-backed warmup complete; "
                        "waiting for active lock"
                    )
                else:
                    logger.warning(
                        "[Shadow] Skipping pre-lock warmup because scratch-backed "
                        "private-bootstrap KV is not enabled"
                    )

            prepromoted_private_bootstrap = False
            if private_bootstrap and self._truthy_env(
                "DYN_VLLM_GMS_PREPROMOTE_SHADOW_KV"
            ):
                logger.info(
                    "[Shadow] Pre-promoting private-bootstrap KV while "
                    "undiscovered; lock still gates serving"
                )
                await handler.engine_client.collective_rpc(
                    "wake_up",
                    kwargs={"tags": ["kv_pool"]},
                )
                prepromoted_private_bootstrap = True
                logger.info(
                    "[Shadow] Private-bootstrap KV pre-promotion complete; "
                    "waiting for active lock"
                )

            runtime.set_health_status(True)
            logger.info(
                "[Shadow] Engine initialized and kept awake but undiscovered; "
                "startup probe now passing, waiting for lock"
            )

            lock = await self._acquire_failover_lock()
            setattr(handler, "_gms_failover_lock", lock)
            await run_gms_failover_post_lock_fence(
                backend_name="vllm",
                role="private-bootstrap-shadow",
            )
            if prepromoted_private_bootstrap:
                logger.info(
                    "[Shadow] Private-bootstrap KV already promoted before "
                    "lock; skipping promotion before discovery registration"
                )
            else:
                logger.info(
                    "[Shadow] Promoting private-bootstrap KV before discovery "
                    "registration"
                )
                await handler.engine_client.collective_rpc(
                    "wake_up",
                    kwargs={"tags": ["kv_pool"]},
                )
            if promotion_warmup is not None and not prelock_warmup_ran:
                await promotion_warmup()
            elif prelock_warmup_ran:
                logger.info(
                    "[Shadow] Pre-lock warmup already ran; skipping post-lock "
                    "promotion warmup"
                )
            logger.info("[Shadow] Lock acquired, registering with discovery")
            return

        await handler._pause_controller.pause(1, clear_cache=False)

        runtime.set_health_status(True)
        logger.info(
            "[Shadow] Engine sleeping, startup probe now passing, waiting for lock"
        )

        lock = await self._acquire_failover_lock()
        setattr(handler, "_gms_failover_lock", lock)
        await run_gms_failover_post_lock_fence(
            backend_name="vllm",
            role="legacy-shadow",
        )
        logger.info("[Shadow] Lock acquired, waking engine")

        await handler._pause_controller.resume()
        handler._pause_controller.mark_resumed()
        if promotion_warmup is not None:
            await promotion_warmup()
        logger.info("[Shadow] Engine awake, registering with discovery")

    async def _create_decode_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
        snapshot_engine: Optional[EngineSetupResult] = None,
        engine_holder: Optional[list] = None,
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
        rl_endpoint = (
            runtime.endpoint(f"{config.namespace}.{config.component}.rl")
            if config.enable_rl
            else None
        )

        shutdown_endpoints[:] = [
            generate_endpoint,
            clear_endpoint,
        ]
        if rl_endpoint is not None:
            shutdown_endpoints.append(rl_endpoint)

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

        early_failover_lock = None
        early_failover_fence_run = False
        (
            early_failover_lock,
            early_failover_fence_run,
        ) = await self._configure_gms_preinit_failover_role(config)
        lock_before_init = os.environ.get("DYN_VLLM_GMS_LOCK_BEFORE_INIT", "1").lower()
        if (
            early_failover_lock is None
            and config.gms_shadow_mode
            and lock_before_init not in {"0", "false", "no", "off"}
        ):
            logger.info(
                "[Shadow] Waiting for failover lock before vLLM engine init "
                "to protect shared GMS KV warmup"
            )
            early_failover_lock = await self._acquire_failover_lock()
            await run_gms_failover_post_lock_fence(
                backend_name="vllm",
                role=f"engine-{os.environ.get('ENGINE_ID', '0')}-pre-init",
            )
            early_failover_fence_run = True
            os.environ["DYN_VLLM_GMS_ACTIVE_LOCK_HELD"] = "1"
            os.environ["DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV"] = "0"
            logger.info("[Shadow] Failover lock acquired before vLLM engine init")

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
            os.environ[ENV_FPM_WORKER_ID] = fpm_worker_id
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
        await configure_kv_event_block_size(engine_client, vllm_config)
        if engine_holder is not None:
            engine_holder.append(engine_client)

        # TODO Hack to get data, move this to registering in TBD
        _, dp_size = get_dp_range_for_worker(vllm_config)
        per_rank_num_gpu_blocks = per_rank_kv_blocks(
            vllm_config.cache_config.num_gpu_blocks,
            dp_size,
        )
        factory.set_num_gpu_blocks_all(per_rank_num_gpu_blocks or 0)
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
            model_config=getattr(vllm_config, "model_config", None),
            enable_multimodal=config.enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=config.use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=config.frontend_decoding,
            encode_worker_client=encode_worker_client,
        )
        handler.add_temp_dir(prometheus_temp_dir)
        if early_failover_lock is not None:
            setattr(handler, "_gms_failover_lock", early_failover_lock)

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

        embedding_cache = getattr(handler, "embedding_cache_manager", None)
        if embedding_cache is not None:
            register_embedding_cache_metrics(
                endpoint=generate_endpoint,
                cache=embedding_cache,
                model_name=config.served_model_name or config.model,
                component_name=config.component,
            )

        # Register engine routes
        self.register_engine_routes(runtime, handler, lora_enabled=lora_enabled)

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

        health_check_payload = VllmHealthCheckPayload(
            engine_client, use_text_input=config.use_vllm_tokenizer
        ).to_dict()

        async def promotion_warmup() -> None:
            await run_gms_failover_promotion_warmup(
                handler.generate, health_check_payload, backend_name="vllm"
            )

        shape_warmup_payload = _vllm_failover_shape_warmup_payload(health_check_payload)

        async def prelock_warmup() -> None:
            await run_gms_failover_promotion_warmup(
                handler.generate, shape_warmup_payload, backend_name="vllm"
            )

        await self._maybe_wait_for_failover_lock(
            handler,
            runtime,
            config,
            lock_already_acquired=early_failover_lock is not None,
            promotion_warmup=promotion_warmup,
            prelock_warmup=prelock_warmup,
            post_lock_fence_already_run=early_failover_fence_run,
        )

        # Wait for self-benchmark to complete before registering.
        bench_cfg = vllm_config.additional_config.get("benchmark")
        if bench_cfg:
            handler._benchmark_results = await _wait_and_load_benchmark(
                bench_cfg, vllm_config
            )

        # Model-serving-readiness role.
        # _create_decode_worker handles both DECODE and AGGREGATED disaggregation modes.
        # `--route-to-encoder` adds Encode to the AND-set of required peers
        # (encode workers register their own card in
        # `_create_multimodal_encode_worker`).
        if config.disaggregation_mode == DisaggregationMode.DECODE:
            worker_type = WorkerType.Decode
            needs_set: list[WorkerType] = [WorkerType.Prefill]
        else:
            # AGGREGATED
            worker_type = WorkerType.Aggregated
            needs_set = []
        if config.route_to_encoder:
            needs_set.append(WorkerType.Encode)
        needs: list[list[WorkerType]] = [needs_set] if needs_set else []

        await self.register_vllm_model(
            model_input,
            model_type,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            worker_type=worker_type,
            needs=needs,
        )

        perf_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.get_perf_metrics"
        )
        shutdown_endpoints.append(perf_endpoint)

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
                perf_endpoint.serve_endpoint(
                    handler.get_perf_metrics,
                    metrics_labels=model_metrics_labels,
                ),
            ]

            if rl_endpoint is not None:
                serve_tasks.append(
                    rl_endpoint.serve_endpoint(
                        handler.rl_dispatch,
                        metrics_labels=model_metrics_labels,
                    )
                )

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
        engine_holder: Optional[list] = None,
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
        rl_endpoint = (
            runtime.endpoint(f"{config.namespace}.{config.component}.rl")
            if config.enable_rl
            else None
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
            os.environ[ENV_FPM_WORKER_ID] = fpm_worker_id
        else:
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = self.setup_vllm_engine(config, fpm_worker_id=fpm_worker_id)
        await configure_kv_event_block_size(engine_client, vllm_config)
        if engine_holder is not None:
            engine_holder.append(engine_client)

        encode_worker_client = await self._maybe_get_encode_worker_client(
            runtime, config
        )

        handler = PrefillWorkerHandler(
            runtime,
            config,
            engine_client,
            default_sampling_params,
            getattr(getattr(vllm_config, "model_config", None), "max_model_len", None),
            model_config=getattr(vllm_config, "model_config", None),
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

        embedding_cache = getattr(handler, "embedding_cache_manager", None)
        if embedding_cache is not None:
            register_embedding_cache_metrics(
                endpoint=generate_endpoint,
                cache=embedding_cache,
                model_name=config.served_model_name or config.model,
                component_name=config.component,
            )

        # Register engine routes
        self.register_engine_routes(
            runtime, handler, lora_enabled=config.engine_args.enable_lora
        )

        health_check_payload = VllmPrefillHealthCheckPayload(
            engine_client, use_text_input=config.use_vllm_tokenizer
        ).to_dict()

        async def promotion_warmup() -> None:
            await run_gms_failover_promotion_warmup(
                handler.generate, health_check_payload, backend_name="vllm"
            )

        await self._maybe_wait_for_failover_lock(
            handler, runtime, config, promotion_warmup=promotion_warmup
        )

        # Wait for self-benchmark to complete before registering.
        bench_cfg = vllm_config.additional_config.get("benchmark")
        if bench_cfg:
            handler._benchmark_results = await _wait_and_load_benchmark(
                bench_cfg, vllm_config
            )

        perf_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.get_perf_metrics"
        )
        shutdown_endpoints[:] = [generate_endpoint, clear_endpoint, perf_endpoint]
        if rl_endpoint is not None:
            shutdown_endpoints.append(rl_endpoint)

        # Prefill workers expose no OpenAI surface — the role is carried by
        # `worker_type=Prefill`. We register the legacy `ModelType.Prefill`
        # marker bit (not a surface) so an OLD frontend, which detects prefill
        # via that bit, still routes disaggregated traffic to this worker
        # during the cross-version rollout. A new frontend ignores the bit and
        # dispatches off `worker_type`. When
        # --route-to-encoder is set, Encode joins the AND-set of needs.
        # ModelInput here is the inter-worker contract, not an engine-local
        # tokenization preference: prefill only ever receives token IDs from
        # its decode peer, so this is Tokens regardless of
        # config.use_vllm_tokenizer (which only swaps the frontend↔decode
        # boundary and the engine-local health-check payload below).
        prefill_needs_set: list[WorkerType] = [WorkerType.Decode]
        if config.route_to_encoder:
            prefill_needs_set.append(WorkerType.Encode)
        await self.register_vllm_model(
            ModelInput.Tokens,
            ModelType.Prefill,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            worker_type=WorkerType.Prefill,
            needs=[prefill_needs_set],
        )

        prefill_metrics_labels = [
            (
                prometheus_names.labels.MODEL,
                config.served_model_name or config.model,
            ),
            (
                prometheus_names.labels.MODEL_NAME,
                config.served_model_name or config.model,
            ),
        ]

        try:
            logger.debug("Starting serve_endpoint for prefill worker")
            serve_tasks = [
                generate_endpoint.serve_endpoint(
                    handler.generate,  # type: ignore
                    graceful_shutdown=True,
                    metrics_labels=prefill_metrics_labels,
                    health_check_payload=health_check_payload,
                ),
                clear_endpoint.serve_endpoint(
                    handler.clear_kv_blocks,  # type: ignore
                    metrics_labels=prefill_metrics_labels,
                ),
                perf_endpoint.serve_endpoint(
                    handler.get_perf_metrics,
                    metrics_labels=prefill_metrics_labels,
                ),
            ]
            if rl_endpoint is not None:
                serve_tasks.append(
                    rl_endpoint.serve_endpoint(
                        handler.rl_dispatch,
                        metrics_labels=prefill_metrics_labels,
                    )
                )
            await asyncio.gather(*serve_tasks)
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

    def register_engine_routes(
        self,
        runtime: DistributedRuntime,
        handler: BaseWorkerHandler,
        lora_enabled: bool = False,
    ) -> None:
        """Register all engine routes for this handler.

        Args:
            runtime: The DistributedRuntime instance to register routes on.
        """
        runtime.register_engine_route("control/start_profile", handler.start_profile)
        runtime.register_engine_route("control/stop_profile", handler.stop_profile)
        runtime.register_engine_route("control/sleep", handler.sleep)
        runtime.register_engine_route("control/wake_up", handler.wake_up)
        runtime.register_engine_route(
            "control/scale_elastic_ep", handler.scale_elastic_ep
        )

        rl_routes: dict = {
            "liveness_probe": handler.liveness_probe,
            "pause_generation": handler.pause_generation,
            "resume_generation": handler.resume_generation,
            "flush_cache": handler.flush_cache,
            "abort_request": handler.abort_request,
            "update_weights_from_disk": handler.update_weights_from_disk,
            "update_weights_from_distributed": handler.update_weights_from_distributed,
            "update_weights_from_tensor": handler.update_weights_from_tensor,
            "init_weights_update_group": handler.init_weights_update_group,
            "destroy_weights_update_group": handler.destroy_weights_update_group,
            "get_weight_version": handler.get_weight_version,
        }

        if lora_enabled:

            async def load_lora(body: dict) -> dict:
                return await first_endpoint_response(handler.load_lora, body)

            async def unload_lora(body: dict) -> dict:
                return await first_endpoint_response(handler.unload_lora, body)

            rl_routes["load_lora"] = load_lora
            rl_routes["unload_lora"] = unload_lora

        register_rl_routes(
            runtime,
            handler.rl_route_registry,
            rl_routes,
            enable_dispatch=handler.config.enable_rl,
        )

        logger.info(
            "Registered engine routes: control/sleep, control/wake_up, "
            "control/scale_elastic_ep, control/start_profile, control/stop_profile, "
            "and RL admin routes: %s%s",
            ", ".join(sorted(rl_routes)),
            " (LoRA routes: load_lora, unload_lora)" if lora_enabled else "",
        )
