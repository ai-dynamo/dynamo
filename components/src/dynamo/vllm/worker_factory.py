# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker initialization factory for vLLM workers."""

import asyncio
import copy
import json
import logging
import math
import os
import time as _time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo import prometheus_names
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


def _benchmark_rank_path(base_path: Path, dp_rank: int) -> Path:
    if dp_rank == 0:
        return base_path
    stem, ext = os.path.splitext(str(base_path))
    return Path(f"{stem}_dp{dp_rank}{ext}")


def _benchmark_merged_path(base_path: Path, dp_start: int) -> Path:
    stem, ext = os.path.splitext(str(base_path))
    rank_suffix = "" if dp_start == 0 else f"_dp{dp_start}"
    return Path(f"{stem}{rank_suffix}_merged{ext}")


def _merge_benchmark_rank_results(
    rank_data: list[tuple[int, Path, dict]],
    merged_path: Path,
) -> dict:
    """Validate and flatten globally synchronized benchmark iterations."""
    if not rank_data:
        raise RuntimeError("No self-benchmark rank results were loaded")

    source_ranks = [rank for rank, _, _ in rank_data]
    reference_rank, _, reference = rank_data[0]
    run_id = reference.get("run_id")
    grid_digest = reference.get("grid_digest")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Self-benchmark rank results are missing run_id")
    if not isinstance(grid_digest, str) or not grid_digest:
        raise RuntimeError("Self-benchmark rank results are missing grid_digest")

    global_size = reference.get("dp", {}).get("size")
    if not isinstance(global_size, int) or global_size < 1:
        raise RuntimeError("Self-benchmark results have invalid global DP size")
    global_ranks = list(range(global_size))

    reference_groups = reference.get("iteration_groups")
    if not isinstance(reference_groups, list) or not reference_groups:
        raise RuntimeError(
            "Self-benchmark rank results are missing synchronized iteration groups"
        )

    groups_by_id: dict[int, dict] = {}
    for group in reference_groups:
        benchmark_id = group.get("benchmark_id")
        if not isinstance(benchmark_id, int) or benchmark_id < 1:
            raise RuntimeError(
                f"Self-benchmark rank {reference_rank} has invalid benchmark_id "
                f"{benchmark_id}"
            )
        if benchmark_id in groups_by_id:
            raise RuntimeError(
                f"Self-benchmark rank {reference_rank} has duplicate "
                f"benchmark_id={benchmark_id}"
            )
        if group.get("point", {}).get("benchmark_id") != benchmark_id:
            raise RuntimeError(
                "Self-benchmark iteration group point id mismatch for "
                f"benchmark_id={benchmark_id}"
            )
        if group.get("expected_dp_ranks") != global_ranks or not group.get("complete"):
            raise RuntimeError(
                "Self-benchmark iteration group is incomplete for "
                f"benchmark_id={benchmark_id}: "
                f"expected={global_ranks} "
                f"actual={group.get('expected_dp_ranks')}"
            )

        rank_results = group.get("rank_results")
        if not isinstance(rank_results, list):
            raise RuntimeError(
                f"Self-benchmark iteration group {benchmark_id} has no rank results"
            )
        actual_ranks = [result.get("dp_rank") for result in rank_results]
        if actual_ranks != global_ranks:
            raise RuntimeError(
                "Self-benchmark iteration group rank mismatch for "
                f"benchmark_id={benchmark_id}: "
                f"expected={global_ranks} actual={actual_ranks}"
            )

        wall_times: list[float] = []
        for rank_result in rank_results:
            dp_rank = rank_result["dp_rank"]
            fpms = rank_result.get("fpms")
            if not isinstance(fpms, list) or len(fpms) != 1:
                raise RuntimeError(
                    "Each self-benchmark rank-point must contain exactly one FPM: "
                    f"rank={dp_rank} benchmark_id={benchmark_id}"
                )
            fpm = fpms[0]
            if fpm.get("counter_id") != benchmark_id:
                raise RuntimeError(
                    "Self-benchmark FPM counter mismatch: "
                    f"rank={dp_rank} benchmark_id={benchmark_id} "
                    f"counter_id={fpm.get('counter_id')}"
                )
            if fpm.get("dp_rank") != dp_rank:
                raise RuntimeError(
                    "Self-benchmark FPM rank mismatch: "
                    f"result_rank={dp_rank} fpm_rank={fpm.get('dp_rank')}"
                )
            wall_times.append(float(fpm.get("wall_time", 0.0)))
        expected_wall_time = max(wall_times, default=0.0)
        if group.get("wall_time") != expected_wall_time:
            raise RuntimeError(
                "Self-benchmark iteration wall time mismatch for "
                f"benchmark_id={benchmark_id}: "
                f"expected={expected_wall_time} actual={group.get('wall_time')}"
            )
        groups_by_id[benchmark_id] = group

    expected_ids = sorted(groups_by_id)
    if expected_ids != list(range(1, len(expected_ids) + 1)):
        raise RuntimeError(
            f"Self-benchmark ids are not a contiguous 1-based sequence: {expected_ids}"
        )

    measured_iteration_seconds = sum(
        float(group.get("wall_time", 0.0)) for group in reference_groups
    )
    rank_timings: list[tuple[int, dict]] = []

    for dp_rank, path, data in rank_data:
        if data.get("run_id") != run_id:
            raise RuntimeError(
                f"Self-benchmark run_id mismatch at {path}: "
                f"expected={run_id} actual={data.get('run_id')}"
            )
        if data.get("grid_digest") != grid_digest:
            raise RuntimeError(
                f"Self-benchmark grid mismatch at {path}: "
                f"expected={grid_digest} actual={data.get('grid_digest')}"
            )
        recorded_rank = data.get("dp", {}).get("rank")
        if recorded_rank != dp_rank:
            raise RuntimeError(
                f"Self-benchmark rank metadata mismatch at {path}: "
                f"expected={dp_rank} actual={recorded_rank}"
            )
        if data.get("dp", {}).get("size") != global_size:
            raise RuntimeError(
                f"Self-benchmark global DP size mismatch at {path}: "
                f"expected={global_size} actual={data.get('dp', {}).get('size')}"
            )
        if data.get("iteration_groups") != reference_groups:
            raise RuntimeError(
                f"Self-benchmark synchronized iteration groups differ at {path}"
            )

        timing = data.get("timing")
        if not isinstance(timing, dict):
            raise RuntimeError(f"Self-benchmark rank {dp_rank} is missing timing")
        started_at = timing.get("started_at")
        completed_at = timing.get("completed_at")
        elapsed_seconds = timing.get("benchmark_elapsed_seconds")
        measured_seconds = timing.get("measured_iteration_seconds")
        if not isinstance(started_at, str) or not isinstance(completed_at, str):
            raise RuntimeError(
                f"Self-benchmark rank {dp_rank} has invalid timing timestamps"
            )
        if (
            not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
        ):
            raise RuntimeError(
                f"Self-benchmark rank {dp_rank} has invalid elapsed timing"
            )
        if (
            not isinstance(measured_seconds, (int, float))
            or not math.isfinite(measured_seconds)
            or not math.isclose(
                measured_seconds,
                measured_iteration_seconds,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"Self-benchmark rank {dp_rank} measured timing does not match "
                "the synchronized iteration groups"
            )
        if measured_seconds > elapsed_seconds + 1e-12:
            raise RuntimeError(
                f"Self-benchmark rank {dp_rank} measured timing exceeds elapsed timing"
            )
        rank_timings.append((dp_rank, timing))

        results_by_id: dict[int, dict] = {}
        for result in data.get("results", []):
            point = result.get("point", {})
            benchmark_id = point.get("benchmark_id")
            if benchmark_id in results_by_id:
                raise RuntimeError(
                    f"Self-benchmark rank {dp_rank} has duplicate "
                    f"benchmark_id={benchmark_id}"
                )
            results_by_id[benchmark_id] = result
        if sorted(results_by_id) != expected_ids:
            raise RuntimeError(
                f"Self-benchmark rank {dp_rank} has a different id set: "
                f"expected={expected_ids} actual={sorted(results_by_id)}"
            )

        for benchmark_id in expected_ids:
            result = results_by_id[benchmark_id]
            group = groups_by_id[benchmark_id]
            if result.get("point") != group.get("point"):
                raise RuntimeError(
                    "Self-benchmark point mismatch for "
                    f"benchmark_id={benchmark_id} on rank {dp_rank}"
                )
            fpms = result.get("fpms") or []
            if len(fpms) != 1:
                raise RuntimeError(
                    f"Self-benchmark rank {dp_rank} must have exactly one FPM for "
                    f"benchmark_id={benchmark_id}"
                )
            fpm = fpms[0]
            if fpm.get("counter_id") != benchmark_id:
                raise RuntimeError(
                    "Self-benchmark FPM counter mismatch: "
                    f"rank={dp_rank} benchmark_id={benchmark_id} "
                    f"counter_id={fpm.get('counter_id')}"
                )
            if fpm.get("dp_rank") != dp_rank:
                raise RuntimeError(
                    "Self-benchmark FPM rank mismatch: "
                    f"file_rank={dp_rank} fpm_rank={fpm.get('dp_rank')}"
                )
            group_fpms = group["rank_results"][dp_rank]["fpms"]
            if fpms != group_fpms:
                raise RuntimeError(
                    "Self-benchmark local result differs from synchronized group: "
                    f"rank={dp_rank} benchmark_id={benchmark_id}"
                )

    flattened_results: list[dict] = []
    for benchmark_id in expected_ids:
        group = groups_by_id[benchmark_id]
        canonical_point = group["point"]
        for rank_result in group["rank_results"]:
            dp_rank = rank_result["dp_rank"]
            fpms = copy.deepcopy(rank_result["fpms"])
            point = copy.deepcopy(canonical_point)
            point["dp_rank"] = dp_rank
            flattened_results.append({"point": point, "fpms": fpms})

    merged = copy.deepcopy(reference)
    merged["artifact_type"] = "merged"
    merged["dp"] = {
        "ranks": global_ranks,
        "source_ranks": source_ranks,
        "managed_size": len(source_ranks),
        "global_size": global_size,
    }
    merged["rank_files"] = [str(path) for _, path, _ in rank_data]
    merged["merged_output_path"] = str(merged_path)
    merged["results"] = flattened_results
    merged["iteration_groups"] = copy.deepcopy(reference_groups)
    _, slowest_timing = max(
        rank_timings,
        key=lambda item: float(item[1]["benchmark_elapsed_seconds"]),
    )
    merged["timing"] = copy.deepcopy(slowest_timing)
    merged["timing"]["benchmark_elapsed_seconds"] = max(
        float(timing["benchmark_elapsed_seconds"]) for _, timing in rank_timings
    )
    merged["timing"]["measured_iteration_seconds"] = measured_iteration_seconds
    merged["timing"]["rank_benchmark_elapsed_seconds"] = {
        str(rank): float(timing["benchmark_elapsed_seconds"])
        for rank, timing in rank_timings
    }
    rank_point_count = len(expected_ids) * global_size
    merged["coverage"] = {
        "expected_points": rank_point_count,
        "completed_points": rank_point_count,
        "skipped_points": 0,
    }
    merged["skipped_points"] = []
    return merged


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


async def _wait_and_load_benchmark(bench_cfg: dict, vllm_config: VllmConfig) -> dict:
    """Wait for benchmark result files and aggregate across DP ranks."""
    base_path = Path(
        os.environ.get(ENV_FPM_BENCHMARK_OUTPUT_PATH, bench_cfg["output_path"])
    )
    timeout = int(bench_cfg.get("timeout", 900))

    dp_start, dp_size = get_dp_range_for_worker(vllm_config)

    dp_ranks = list(range(dp_start, dp_start + dp_size))
    rank_paths = [_benchmark_rank_path(base_path, dp_rank) for dp_rank in dp_ranks]
    merged_path = _benchmark_merged_path(base_path, dp_start)
    try:
        merged_path.unlink()
    except FileNotFoundError:
        pass

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

    rank_data: list[tuple[int, Path, dict]] = []
    for dp_rank, p in zip(dp_ranks, rank_paths):
        with open(p) as f:
            data = json.load(f)
        if data.get("valid") is False:
            raise RuntimeError(
                f"Self-benchmark produced incomplete results at {p}: "
                f"coverage={data.get('coverage')} "
                f"skipped_points={data.get('skipped_points')} "
                f"missing_phases={data.get('missing_phases')}"
            )
        rank_data.append((dp_rank, p, data))

    merged = _merge_benchmark_rank_results(rank_data, merged_path)
    _write_json_atomic(merged_path, merged)

    logger.info(
        "Benchmark complete, %d rank-points across %d rank(s); merged results: %s",
        len(merged.get("results", [])),
        len(rank_paths),
        merged_path,
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

    async def _maybe_wait_for_failover_lock(
        self,
        handler,
        runtime: DistributedRuntime,
        config: Config,
    ) -> None:
        # Shadow mode: lock-driven activation.
        # Flow: sleep → startup probe passes → block on lock → wake → register.
        if not config.gms_shadow_mode:
            return

        await handler._pause_controller.pause(1)

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

        await handler._pause_controller.resume()
        handler._pause_controller.mark_resumed()
        logger.info("[Shadow] Engine awake, registering with discovery")

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
        fpm_relays = self.setup_fpm_relay(config, generate_endpoint, vllm_config)
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

        await self._maybe_wait_for_failover_lock(handler, runtime, config)

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

        health_check_payload = VllmHealthCheckPayload(
            engine_client, use_text_input=config.use_vllm_tokenizer
        ).to_dict()

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
        fpm_relays = self.setup_fpm_relay(config, generate_endpoint, vllm_config)
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

        await self._maybe_wait_for_failover_lock(handler, runtime, config)

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

        health_check_payload = VllmPrefillHealthCheckPayload(
            engine_client, use_text_input=config.use_vllm_tokenizer
        ).to_dict()

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
