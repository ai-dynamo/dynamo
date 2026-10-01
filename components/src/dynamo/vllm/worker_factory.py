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
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

from dynamo import prometheus_names
from dynamo.common.model_taints import register_model_taint_route
from dynamo.common.rl import first_endpoint_response, register_rl_routes
from dynamo.common.snapshot.lifecycle import elect_and_wake
from dynamo.common.utils.endpoint_types import parse_endpoint_types
from dynamo.common.utils.prometheus import (
    LLMBackendMetrics,
    register_embedding_cache_metrics,
)
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, Endpoint

from .args import Config
from .cache_info import configure_kv_event_block_size
from .capacity import per_rank_kv_blocks
from .constants import DisaggregationMode
from .dp_topology import get_dp_range_for_worker
from .handlers import (
    BaseWorkerHandler,
    DecodeWorkerHandler,
    EmbeddingWorkerHandler,
    PrefillWorkerHandler,
)
from .health_check import (
    VllmEmbeddingHealthCheckPayload,
    VllmHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
)
from .instrumented_scheduler import (
    ENV_FPM_BENCHMARK_OUTPUT_PATH,
    ENV_FPM_WORKER_ID,
    benchmark_content_point_key,
)
from .multimodal_handlers import EncodeWorkerHandler
from .pooling_handlers import ClassifyWorkerHandler
from .publisher import StatLoggerFactory
from .realtime import RealtimeHandler, RealtimeTranscriptionHandler
from .state_agent import StateAgentLifecycle, state_agent_settings

logger = logging.getLogger(__name__)

# The active point has an 8s FPM deadline. ADP schedule/result, decision/commit,
# boundary, and final cleanup phases each have a 10s bound. Their worst-case
# healthy stop path is about 70s, with the remainder reserved for JSON writing
# and scheduler-loop slack before failing closed.
BENCHMARK_SOFT_TIMEOUT_GRACE_SECONDS = 90

# Bound for the post-benchmark worker GC stop RPC. Restoring GC in a healthy
# worker is sub-second; a longer wait means the engine is gone, and neither
# serving nor error propagation may hang on it.
WORKER_GC_STOP_TIMEOUT_SECONDS = 30.0

# Bound for the post-benchmark engine provenance probe. It reads attributes
# off already-built layers, so a healthy worker answers immediately; a longer
# wait means the engine is gone and provenance must not hold up teardown.
ENGINE_PROBE_TIMEOUT_SECONDS = 30.0

# Schema name of the sidecar the post-benchmark probe writes next to the
# merged artifact; the rank and merged artifacts keep their startup engine block.
WORKER_PROBE_SIDECAR_SCHEMA = "dynamo.fpm.benchmark_worker_probe"

# (engine_client, vllm_config, default_sampling_params, cleanup_resource, component_gauges)
# component_gauges is None on the embedding-worker path: pooling engines
# have no KV cache / scheduler gauges, so setup_vllm_engine() skips the
# LLMBackendMetrics registration there.
EngineSetupResult = tuple[AsyncLLM, VllmConfig, Any, Any, Optional[LLMBackendMetrics]]
SnapshotEngineSetupResult = tuple[EngineSetupResult, StatLoggerFactory]


def _benchmark_rank_path(base_path: Path, dp_rank: int) -> Path:
    if dp_rank == 0:
        return base_path
    stem, ext = os.path.splitext(str(base_path))
    return Path(f"{stem}_dp{dp_rank}{ext}")


def _benchmark_merged_path(base_path: Path, dp_start: int) -> Path:
    stem, ext = os.path.splitext(str(base_path))
    rank_suffix = "" if dp_start == 0 else f"_dp{dp_start}"
    return Path(f"{stem}{rank_suffix}_merged{ext}")


def _benchmark_worker_probe_path(merged_path: Path) -> Path:
    """Sidecar next to the merged artifact that holds the post-run worker probe."""
    stem, ext = os.path.splitext(str(merged_path))
    return Path(f"{stem}_worker_probe{ext}")


def _validate_benchmark_rank_payload(data: dict, path: Path) -> str:
    """Validate status/coverage invariants before accepting a rank artifact."""
    status = data.get("status", "complete" if data.get("valid") is True else "failed")

    def invalid(reason: str) -> RuntimeError:
        return RuntimeError(
            f"Self-benchmark produced incomplete results at {path}: {reason}; "
            f"coverage={data.get('coverage')} "
            f"skipped_points={data.get('skipped_points')} "
            f"missing_phases={data.get('missing_phases')}"
        )

    if status not in {"complete", "partial"}:
        raise invalid(f"status={status!r}")

    coverage = data.get("coverage")
    if not isinstance(coverage, dict):
        raise invalid("missing coverage")
    expected = coverage.get("expected_points")
    completed = coverage.get("completed_points")
    skipped = coverage.get("skipped_points")
    if (
        not isinstance(expected, int)
        or not isinstance(completed, int)
        or not isinstance(skipped, int)
        or min(expected, completed, skipped) < 0
        or completed + skipped > expected
    ):
        raise invalid("invalid coverage arithmetic")

    skipped_points = data.get("skipped_points", [])
    missing_phases = data.get("missing_phases", [])
    results = data.get("results")
    iteration_groups = data.get("iteration_groups")
    if not isinstance(skipped_points, list) or len(skipped_points) != skipped:
        raise invalid("skipped point count mismatch")
    if not isinstance(missing_phases, list):
        raise invalid("invalid missing phases")
    if not isinstance(results, list) or len(results) != completed:
        raise invalid("result count does not match completed coverage")
    if not isinstance(iteration_groups, list) or len(iteration_groups) != completed:
        raise invalid("iteration group count does not match completed coverage")

    if status == "complete":
        if (
            data.get("valid") is not True
            or ("usable" in data and data.get("usable") is not True)
            or data.get("stop_reason") is not None
            or completed != expected
            or skipped != 0
            or missing_phases
            or data.get("error") is not None
        ):
            raise invalid("inconsistent complete status")
    elif (
        data.get("valid") is not False
        or data.get("usable") is not True
        or data.get("stop_reason") != "timeout"
        or completed >= expected
        or skipped != 0
        or missing_phases
        or data.get("error") is not None
    ):
        raise invalid("inconsistent partial status")
    return status


def _benchmark_engine_identity(data: dict) -> Optional[dict]:
    """The part of a rank artifact's ``engine`` block that every DP rank of a
    run must agree on.

    ``parallel.data_parallel_rank`` legitimately differs per rank, and
    Worker observations are written after collection, so they do not
    participate in the startup identity comparison.
    """
    engine = data.get("engine")
    if not isinstance(engine, dict):
        return None
    identity = copy.deepcopy(engine)
    identity.pop("resolved", None)
    identity.pop("resolution", None)
    identity.pop("resolved_scope", None)
    identity.pop("worker_probe", None)
    attention = identity.get("attention")
    if isinstance(attention, dict):
        for field in ("backend_resolved", "mla_prefill_backend_resolved", "resolution"):
            attention.pop(field, None)
    parallel = identity.get("parallel")
    if isinstance(parallel, dict):
        parallel.pop("data_parallel_rank", None)
    return identity


def _engine_degraded(engine_block: Any) -> bool:
    """True when a rank's ``engine`` block cannot be trusted for an identity
    comparison: no block at all, or the capture itself failed partway
    through (``_bench_capture_engine`` still emits whatever sub-blocks it
    built before the exception, alongside ``capture_error``).
    """
    return not isinstance(engine_block, dict) or "capture_error" in engine_block


def _validate_execution_measurement(
    fpm: dict, measurement: dict, previous_forward_index: int
) -> int:
    """Check raw observations against their declared point/rank and timing."""
    rank = measurement["dp_rank"]
    benchmark_id = fpm["counter_id"]

    def invalid(reason: str) -> RuntimeError:
        return RuntimeError(
            "Self-benchmark execution evidence is invalid: "
            f"rank={rank} benchmark_id={benchmark_id}: {reason}"
        )

    raw = measurement.get("raw_fpms")
    expected = measurement.get("expected_internal_samples")
    if (
        not isinstance(raw, list)
        or not raw
        or type(expected) is not int
        or expected < len(raw)
        or "benchmark_sample" in fpm
    ):
        raise invalid("missing raw samples or contradictory retained observation")
    for index, sample_fpm in enumerate(raw):
        if not isinstance(sample_fpm, dict) or (
            type(sample_fpm.get("counter_id")) is not int
            or sample_fpm["counter_id"] != benchmark_id
            or type(sample_fpm.get("dp_rank")) is not int
            or sample_fpm["dp_rank"] != rank
            or sample_fpm.get("worker_id") != fpm.get("worker_id")
        ):
            raise invalid(f"raw sample {index} has a different point/rank identity")
        sample = sample_fpm.get("benchmark_sample")
        if (
            not isinstance(sample, dict)
            or type(sample.get("sample_index")) is not int
            or sample["sample_index"] != index
            or type(sample.get("forward_index")) is not int
            or sample["forward_index"] <= previous_forward_index
        ):
            raise invalid(f"raw sample {index} has an invalid sample/forward index")
        previous_forward_index = sample["forward_index"]
        timing = sample.get("timing")
        if not isinstance(timing, dict) or timing.get("basis") not in {
            "schedule_to_output",
            "inter_output",
        }:
            raise invalid(f"raw sample {index} has no timing interval")
        start, end = timing.get("start_monotonic"), timing.get("end_monotonic")
        wall = sample_fpm.get("wall_time")
        if (
            end is None
            or type(end) not in (int, float)
            or not math.isfinite(end)
            or wall is None
            or type(wall) not in (int, float)
            or not math.isfinite(wall)
            or wall < 0
            or (
                start is not None
                and (
                    type(start) not in (int, float)
                    or not math.isfinite(start)
                    or start > end
                    or not math.isclose(end - start, wall, rel_tol=1e-9, abs_tol=1e-9)
                )
            )
        ):
            raise invalid(f"raw sample {index} contradicts its timing interval")
        graph = sample.get("cudagraph")
        if not isinstance(graph, dict):
            raise invalid(f"raw sample {index} has no graph observation status")
        mode = graph.get("runtime_mode")
        unpadded, padded = (
            graph.get("num_unpadded_tokens"),
            graph.get("num_padded_tokens"),
        )
        paddings = graph.get("num_paddings")
        if graph.get("status") == "observed":
            if (
                not isinstance(mode, str)
                or not mode
                or type(unpadded) is not int
                or type(padded) is not int
                or unpadded < 0
                or padded < unpadded
                or type(paddings) is not int
                or paddings != padded - unpadded
            ):
                raise invalid(f"raw sample {index} has invalid observed graph data")
        elif graph.get("status") != "unavailable" or any(
            value is not None for value in (mode, unpadded, padded, paddings)
        ):
            raise invalid(
                f"raw sample {index} contradicts its graph observation status"
            )
    estimate = measurement.get("estimate")
    if not isinstance(estimate, dict):
        raise invalid("missing reduction inputs")
    indices = estimate.get("raw_sample_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or not 0 <= index < len(raw) for index in indices)
        or indices != sorted(set(indices))
    ):
        raise invalid("invalid reduction sample indices")
    method = estimate.get("method")
    if (
        (method == "single_step" and (len(raw) != 1 or indices != [0]))
        or (method == "last_step" and indices != [len(raw) - 1])
        or (method == "adjacent_upper_median" and indices != list(range(1, len(raw))))
        or method not in {"single_step", "last_step", "adjacent_upper_median"}
    ):
        raise invalid("reduction inputs contradict the reduction method")
    walls = sorted(raw[index]["wall_time"] for index in indices)
    if not math.isclose(
        fpm["wall_time"], walls[len(walls) // 2], rel_tol=1e-9, abs_tol=1e-12
    ):
        raise invalid("retained wall time contradicts the raw reduction inputs")
    return previous_forward_index


def _validate_warmup_evidence(data: dict, rank: int) -> dict:
    """Require a ledger status; an unavailable history must remain unknown."""
    ledger = data.get("warmup_evidence")
    if (
        not isinstance(ledger, dict)
        or ledger.get("status") not in {"recorded", "unavailable"}
        or not isinstance(ledger.get("records"), list)
        or (ledger["status"] == "unavailable" and ledger["records"])
    ):
        raise RuntimeError(f"Self-benchmark warmup evidence is invalid: rank={rank}")
    for record in ledger["records"]:
        if (
            not isinstance(record, dict)
            or record.get("status") not in {"running", "completed", "failed"}
            or not isinstance(record.get("validation"), dict)
            or record["validation"].get("status")
            not in {"passed", "failed", "not_performed"}
            or (
                record["validation"]["status"] == "passed"
                and record["status"] != "completed"
            )
            or (
                record["validation"]["status"] == "failed"
                and record["status"] != "failed"
            )
        ):
            raise RuntimeError(
                f"Self-benchmark warmup evidence is invalid: rank={rank}"
            )
        start, end = record.get("forward_index_start"), record.get("forward_index_end")
        if (
            (start is not None and (type(start) is not int or start < 0))
            or (end is not None and (type(end) is not int or end < 0))
            or (start is not None and end is not None and end < start)
            or (record["status"] == "completed" and end is None)
        ):
            raise RuntimeError(
                f"Self-benchmark warmup forward range is invalid: rank={rank}"
            )
        count = record.get("observed_forward_count")
        if count is not None and (
            type(count) is not int
            or count < 0
            or (start is not None and end is not None and count != end - start)
        ):
            raise RuntimeError(
                f"Self-benchmark warmup forward count is invalid: rank={rank}"
            )
    return ledger


def _merge_benchmark_rank_results(
    rank_data: list[tuple[int, Path, dict]],
    merged_path: Path,
) -> dict:
    """Validate and flatten globally synchronized benchmark iterations."""
    if not rank_data:
        raise RuntimeError("No self-benchmark rank results were loaded")

    source_ranks = [rank for rank, _, _ in rank_data]
    # Row-level KV seed provenance travels from each rank artifact into the
    # merged artifact unchanged (per rank, per benchmark id).
    regimes: dict[tuple[int, int], object] = {}
    reference_rank, reference_path, reference = rank_data[0]
    run_id = reference.get("run_id")
    grid_digest = reference.get("grid_digest")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("Self-benchmark rank results are missing run_id")
    if not isinstance(grid_digest, str) or not grid_digest:
        raise RuntimeError("Self-benchmark rank results are missing grid_digest")
    has_measurement_protocol = "measurement_protocol" in reference
    measurement_protocol = reference.get("measurement_protocol")
    if has_measurement_protocol and (
        not isinstance(measurement_protocol, dict)
        or measurement_protocol.get("schema_version") != 1
    ):
        raise RuntimeError("Self-benchmark results have invalid measurement protocol")
    for _, path, data in rank_data:
        if ("measurement_protocol" in data) != has_measurement_protocol or data.get(
            "measurement_protocol"
        ) != measurement_protocol:
            raise RuntimeError(
                f"Self-benchmark measurement protocol mismatch at {path}: "
                "every contributing rank must record the same protocol"
            )
    execution_evidence = (
        measurement_protocol.get("execution_evidence")
        if isinstance(measurement_protocol, dict)
        else None
    )
    has_execution_evidence = (
        isinstance(measurement_protocol, dict)
        and "execution_evidence" in measurement_protocol
    )
    if has_execution_evidence and (
        not isinstance(execution_evidence, dict)
        or execution_evidence.get("schema_version") != 1
        or execution_evidence.get("sample_field") != "benchmark_sample"
        or execution_evidence.get("warmup_field") != "warmup_evidence"
    ):
        raise RuntimeError(
            "Self-benchmark results have invalid execution evidence contract"
        )
    warmup_ledgers = {}
    for rank, _, data in rank_data:
        if has_execution_evidence:
            warmup_ledgers[rank] = _validate_warmup_evidence(data, rank)
        elif "warmup_evidence" in data:
            raise RuntimeError(
                "Self-benchmark execution evidence has no declared contract"
            )
    # A rank is "degraded" when its engine capture failed or is missing while
    # at least one other rank has one: it cannot be trusted for an identity
    # comparison, so it is excluded below and carried into the merged
    # document's engine.capture_errors instead. A run where NO rank ever
    # captured an engine block (e.g. a pre-Task-1 artifact) is not degraded
    # -- it simply carries no engine provenance, so the merge stays
    # byte-for-byte backward compatible (no engine key, no warning).
    any_engine_present = any(isinstance(d.get("engine"), dict) for _, _, d in rank_data)
    engine_capture_errors: dict[str, str] = {}
    reference_engine: Optional[dict] = None
    reference_engine_block: Optional[dict] = None
    reference_engine_rank: Optional[int] = None
    if any_engine_present:
        for rank, _, data in rank_data:
            engine_block = data.get("engine")
            if _engine_degraded(engine_block):
                engine_capture_errors[str(rank)] = (
                    str(engine_block["capture_error"])
                    if isinstance(engine_block, dict)
                    else "missing engine block"
                )
            elif reference_engine is None:
                # The first rank whose capture succeeded is the identity
                # reference; a rank that comes first in rank_data but was
                # itself degraded is not eligible to be the reference.
                reference_engine = _benchmark_engine_identity(data)
                reference_engine_block = engine_block
                reference_engine_rank = rank

    reference_status = _validate_benchmark_rank_payload(reference, reference_path)

    reference_coverage = reference.get("coverage")
    if not isinstance(reference_coverage, dict):
        raise RuntimeError("Self-benchmark rank results are missing coverage")
    expected_points_per_rank = reference_coverage.get("expected_points")
    completed_points_per_rank = reference_coverage.get("completed_points")
    skipped_points_per_rank = reference_coverage.get("skipped_points")
    if (
        not isinstance(expected_points_per_rank, int)
        or not isinstance(completed_points_per_rank, int)
        or not isinstance(skipped_points_per_rank, int)
        or expected_points_per_rank < completed_points_per_rank
        or min(completed_points_per_rank, skipped_points_per_rank) < 0
    ):
        raise RuntimeError("Self-benchmark rank results have invalid coverage")

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
    previous_forward_indices: dict[int, int] = {}
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
        measurement_point_key = (
            benchmark_content_point_key(group["point"])
            if has_measurement_protocol
            else None
        )
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
            measurement = fpm.get("benchmark_measurement")
            if has_measurement_protocol:
                if (
                    not isinstance(measurement, dict)
                    or measurement.get("schema_version") != 1
                    or measurement.get("dp_rank") != dp_rank
                ):
                    raise RuntimeError(
                        "Self-benchmark measurement evidence missing or invalid: "
                        f"rank={dp_rank} benchmark_id={benchmark_id}"
                    )
                point_key = measurement.get("point_key")
                preparation = measurement.get("preparation")
                if (
                    not isinstance(point_key, str)
                    or not point_key
                    or not isinstance(preparation, dict)
                    or preparation.get("grid_digest") != grid_digest
                    or point_key != measurement_point_key
                ):
                    raise RuntimeError(
                        "Self-benchmark measurement identity mismatch: "
                        f"rank={dp_rank} benchmark_id={benchmark_id}"
                    )
                if has_execution_evidence:
                    previous_forward_indices[dp_rank] = _validate_execution_measurement(
                        fpm, measurement, previous_forward_indices.get(dp_rank, -1)
                    )
                    warmups_before = preparation.get("warmup_records_before")
                    ledger = warmup_ledgers.get(dp_rank)
                    if (
                        (
                            warmups_before is not None
                            and (type(warmups_before) is not int or warmups_before < 0)
                        )
                        or (
                            ledger is not None
                            and ledger["status"] == "recorded"
                            and (
                                warmups_before is None
                                or warmups_before > len(ledger["records"])
                            )
                        )
                        or (
                            ledger is not None
                            and ledger["status"] == "unavailable"
                            and warmups_before is not None
                        )
                    ):
                        raise RuntimeError(
                            f"Self-benchmark warmup reference is invalid: rank={dp_rank} benchmark_id={benchmark_id}"
                        )
                    if ledger is not None and ledger["status"] == "recorded":
                        first_forward = measurement["raw_fpms"][0]["benchmark_sample"][
                            "forward_index"
                        ]
                        if any(
                            record["status"] == "running"
                            or (
                                record.get("forward_index_end") is not None
                                and record["forward_index_end"] > first_forward
                            )
                            for record in ledger["records"][:warmups_before]
                        ):
                            raise RuntimeError(
                                f"Self-benchmark warmup reference overlaps measurement: rank={dp_rank} benchmark_id={benchmark_id}"
                            )
                elif any(
                    "benchmark_sample" in raw for raw in measurement.get("raw_fpms", [])
                ):
                    raise RuntimeError(
                        "Self-benchmark execution evidence has no declared contract"
                    )
            elif "benchmark_measurement" in fpm:
                raise RuntimeError(
                    "Self-benchmark measurement evidence has no shared protocol: "
                    f"rank={dp_rank} benchmark_id={benchmark_id}"
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
    if completed_points_per_rank != len(expected_ids):
        raise RuntimeError(
            "Self-benchmark completed coverage does not match iteration groups: "
            f"coverage={completed_points_per_rank} groups={len(expected_ids)}"
        )

    measured_iteration_seconds = sum(
        float(group.get("wall_time", 0.0)) for group in reference_groups
    )
    rank_timings: list[tuple[int, dict]] = []

    for dp_rank, path, data in rank_data:
        data_status = _validate_benchmark_rank_payload(data, path)
        if data_status != reference_status:
            raise RuntimeError(
                f"Self-benchmark status mismatch at {path}: "
                f"expected={reference_status} actual={data_status}"
            )
        if data.get("coverage") != reference_coverage:
            raise RuntimeError(
                f"Self-benchmark coverage mismatch at {path}: "
                f"expected={reference_coverage} actual={data.get('coverage')}"
            )
        if data.get("stop_reason") != reference.get("stop_reason"):
            raise RuntimeError(f"Self-benchmark stop reason mismatch at {path}")
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
        if reference_engine is not None and not _engine_degraded(data.get("engine")):
            data_engine_identity = _benchmark_engine_identity(data) or {}
            if data_engine_identity != reference_engine:
                mismatched = sorted(
                    key
                    for key in set(data_engine_identity) | set(reference_engine)
                    if data_engine_identity.get(key) != reference_engine.get(key)
                )
                # The two dicts can differ (`!=`) even when every key's
                # .get() agrees, e.g. a top-level key present-with-None on
                # one rank and absent on the other: .get() returns None
                # either way, so no single key explains it. Name the whole
                # block instead of indexing into a possibly empty list.
                field = mismatched[0] if mismatched else "<top-level key set>"
                raise RuntimeError(
                    f"Self-benchmark engine provenance mismatch at {path}: "
                    f"field={field} "
                    f"reference_rank={reference_engine_rank}: the ranks of "
                    "one run must share an engine configuration"
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
            if "kv_seed_regime" in result:
                regimes[(dp_rank, benchmark_id)] = result["kv_seed_regime"]
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
            entry: dict = {"point": point, "fpms": fpms}
            if (dp_rank, benchmark_id) in regimes:
                entry["kv_seed_regime"] = regimes[(dp_rank, benchmark_id)]
            flattened_results.append(entry)

    merged = copy.deepcopy(reference)
    if any_engine_present:
        if (
            _engine_degraded(reference.get("engine"))
            and reference_engine_block is not None
        ):
            # The reference rank's own capture failed or was absent, but
            # another rank's did not: reseed the merged block from that rank
            # instead of losing the provenance the run did capture.
            merged["engine"] = copy.deepcopy(reference_engine_block)
        merged_engine = merged.get("engine")
        if isinstance(merged_engine, dict) and isinstance(
            merged_engine.get("parallel"), dict
        ):
            # The merged document describes every rank, so a single rank's
            # own DP rank would be a lie here.
            merged_engine["parallel"]["data_parallel_rank"] = None
        if engine_capture_errors:
            logger.warning(
                "Self-benchmark engine provenance capture failed or was "
                "absent on rank(s) %s; the merged artifact's engine "
                "provenance is unverified across ranks",
                ", ".join(sorted(engine_capture_errors, key=int)),
            )
            if not isinstance(merged_engine, dict):
                merged_engine = {}
                merged["engine"] = merged_engine
            merged_engine["capture_errors"] = engine_capture_errors
    merged["artifact_type"] = "merged"
    merged["dp"] = {
        "ranks": global_ranks,
        "source_ranks": source_ranks,
        "managed_size": len(source_ranks),
        "global_size": global_size,
    }
    merged["rank_files"] = [str(path) for _, path, _ in rank_data]
    merged["merged_output_path"] = str(merged_path)
    if isinstance(merged.get("engine"), dict):
        # The post-run worker probe (_attach_engine_resolved) writes its replies
        # to a sidecar instead of rewriting the artifacts. Its path sits next to
        # merged_output_path, outside the engine block, which stays free of
        # per-run paths.
        merged["worker_probe_path"] = str(_benchmark_worker_probe_path(merged_path))
    merged["results"] = flattened_results
    merged["iteration_groups"] = copy.deepcopy(reference_groups)
    if has_execution_evidence:
        merged.pop("warmup_evidence", None)
        merged["rank_warmup_evidence"] = {
            str(rank): copy.deepcopy(
                warmup_ledgers.get(
                    rank,
                    {
                        "status": "unavailable",
                        "records": [],
                        "reason": "rank_artifact_not_loaded",
                    },
                )
            )
            for rank in global_ranks
        }
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
    merged["coverage"] = {
        "expected_points": expected_points_per_rank * global_size,
        "completed_points": completed_points_per_rank * global_size,
        "skipped_points": skipped_points_per_rank * global_size,
    }
    merged["skipped_points"] = copy.deepcopy(reference.get("skipped_points", []))
    return merged


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp_path = Path(f"{path}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _make_engine_probe() -> Callable[[Any], dict]:
    """Build the worker-side engine probe.

    The probe is defined *inside* this function so cloudpickle serialises it
    by value: the model worker then runs it without importing
    ``dynamo.vllm.worker_factory``, which would pull the whole launcher stack
    into the model process. For the same reason its body may reference only
    builtins and attributes of the worker object it is handed.

    vLLM chooses the attention backend inside the worker
    (``Attention.__init__`` -> ``get_attn_backend``) and never writes the
    result back into ``VllmConfig``, so the layer registry in
    ``compilation_config.static_forward_context`` is the only place the
    resolved names exist. Same attribute names in vLLM 0.28.0 and 0.29.0.
    """

    def probe(self):
        vllm_config = getattr(self, "vllm_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        context = getattr(compilation_config, "static_forward_context", None) or {}
        backends = {}
        prefill_backends = []
        capture_errors = {}
        for name, layer in context.items():
            get_backend = getattr(layer, "get_attn_backend", None)
            if not callable(get_backend):
                continue
            try:
                backends[str(name)] = get_backend().get_name()
            except Exception as error:
                # Provenance is best effort; retain the failed layer as
                # evidence instead of claiming a complete observation.
                capture_errors[str(name)] = f"{type(error).__name__}: {error}"
                continue
            prefill = getattr(layer, "prefill_backend", None)
            if prefill is not None:
                # MLA layers hold a backend instance; tolerate a bare class.
                prefill_backends.append(
                    getattr(prefill, "__name__", None) or type(prefill).__name__
                )
        unique_prefill = sorted(set(prefill_backends))
        worker_rank = getattr(self, "rank", None)
        # vLLM zeroes parallel_config.data_parallel_rank on every engine for a
        # dense (non-MoE) model under external DP, keeping the true rank only
        # in data_parallel_index -- the same trap
        # InstrumentedScheduler._resolve_dp_rank already routes around, so
        # engine.resolved.dp_rank does not silently disagree with
        # engine.parallel.data_parallel_rank / dp.rank in the same artifact.
        dp_rank = getattr(parallel_config, "data_parallel_index", None)
        if dp_rank is None:
            dp_rank = getattr(parallel_config, "data_parallel_rank", None)
        # No live process group is guaranteed wherever this callable runs.
        # vLLM orders ranks as DP x PP x PCP x TP, so derive the model-parallel
        # ranks from the worker rank while keeping the explicit DP identity.
        tensor_parallel_size = getattr(parallel_config, "tensor_parallel_size", None)
        pipeline_parallel_size = getattr(
            parallel_config, "pipeline_parallel_size", None
        )
        prefill_context_parallel_size = getattr(
            parallel_config, "prefill_context_parallel_size", None
        )
        tp_rank = None
        pp_rank = None
        pcp_rank = None
        if (
            type(worker_rank) is int
            and worker_rank >= 0
            and type(tensor_parallel_size) is int
            and tensor_parallel_size > 0
        ):
            tp_rank = worker_rank % tensor_parallel_size
            if (
                type(prefill_context_parallel_size) is int
                and prefill_context_parallel_size > 0
            ):
                pcp_rank = (
                    worker_rank // tensor_parallel_size
                ) % prefill_context_parallel_size
                if type(pipeline_parallel_size) is int and pipeline_parallel_size > 0:
                    pp_rank = (
                        worker_rank
                        // (tensor_parallel_size * prefill_context_parallel_size)
                    ) % pipeline_parallel_size
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        return {
            "tp_rank": tp_rank,
            "pp_rank": pp_rank,
            "pcp_rank": pcp_rank,
            "worker_rank": worker_rank,
            "dp_rank": dp_rank,
            "attention_backends": backends,
            "mla_prefill_backend": (
                unique_prefill[0] if len(unique_prefill) == 1 else None
            ),
            "cudagraph_mode_resolved": getattr(cudagraph_mode, "name", None),
            "cudagraph_capture_sizes_resolved": [
                int(size)
                for size in (
                    getattr(compilation_config, "cudagraph_capture_sizes", None) or []
                )
            ],
            "capture_errors": capture_errors,
        }

    return probe


def _worker_probe_responses(results: Any) -> list[dict]:
    """Keep every reply, including malformed or duplicate worker evidence."""
    responses = []
    identities: dict[tuple[int, int], list[int]] = {}
    # A malformed top-level result is itself evidence; it is not an empty RPC.
    for index, result in enumerate(results if isinstance(results, list) else [results]):
        issues = []
        try:
            response = json.loads(json.dumps(result, allow_nan=False))
        except (TypeError, ValueError) as error:
            response = {
                "unserializable_type": type(result).__name__,
                "repr": repr(result),
            }
            issues.append(f"non-JSON response: {type(error).__name__}")
        if not isinstance(result, dict):
            issues.append("response is not an object")
        else:
            for field in ("dp_rank", "worker_rank", "tp_rank"):
                if type(result.get(field)) is not int or result[field] < 0:
                    issues.append(f"missing or invalid {field}")
            for field in ("pp_rank", "pcp_rank"):
                local_rank = result.get(field)
                if local_rank is not None and (
                    type(local_rank) is not int or local_rank < 0
                ):
                    issues.append(f"invalid {field}")
            backends = result.get("attention_backends")
            if not isinstance(backends, dict) or any(
                not isinstance(layer, str) or not isinstance(name, str) or not name
                for layer, name in backends.items()
            ):
                issues.append("invalid attention_backends")
            for field in ("mla_prefill_backend", "cudagraph_mode_resolved"):
                if result.get(field) is not None and not isinstance(result[field], str):
                    issues.append(f"invalid {field}")
            sizes = result.get("cudagraph_capture_sizes_resolved")
            if not isinstance(sizes, list) or any(
                type(size) is not int or size < 1 for size in sizes
            ):
                issues.append("invalid cudagraph_capture_sizes_resolved")
            if not issues:
                identity = (result["dp_rank"], result["worker_rank"])
                identities.setdefault(identity, []).append(index)
        responses.append({"rpc_index": index, "response": response, "issues": issues})
    for indices in identities.values():
        if len(indices) > 1:
            for index in indices:
                responses[index]["issues"].append("duplicate (dp_rank, worker_rank)")
    return responses


def _apply_engine_resolved(
    document: dict, responses: list[dict], failure: str | None
) -> None:
    """Attach labelled post-run evidence without extending the RPC's scope."""
    engine = document.get("engine")
    if not isinstance(engine, dict):
        return
    parallel = engine.get("parallel") or {}
    dp = document.get("dp") or {}
    rank = dp.get("rank", parallel.get("data_parallel_rank"))
    expected_ranks = [rank] if type(rank) is int else dp.get("ranks")
    if not isinstance(expected_ranks, list):
        expected_ranks = None
    tp_size = parallel.get("tensor_parallel_size")
    pp_size = parallel.get("pipeline_parallel_size")
    pcp_size = parallel.get("prefill_context_parallel_size")
    topology: tuple[int, int, int] | None = None
    if (
        type(tp_size) is int
        and tp_size > 0
        and type(pp_size) is int
        and pp_size > 0
        and type(pcp_size) is int
        and pcp_size > 0
    ):
        topology = (tp_size, pcp_size, pp_size)
    expected_workers_per_dp = math.prod(topology) if topology is not None else None
    scoped_responses = copy.deepcopy(responses)
    workers = []
    slots: dict[tuple[int, int, int, int], list[dict]] = {}
    for entry in scoped_responses:
        response = entry["response"]
        if entry["issues"]:
            continue
        if expected_ranks is not None and response["dp_rank"] not in expected_ranks:
            continue
        if topology is not None:
            tp_size, pcp_size, pp_size = topology
            worker_rank = response["worker_rank"]
            expected_tp = worker_rank % tp_size
            expected_pcp = (worker_rank // tp_size) % pcp_size
            expected_pp = (worker_rank // (tp_size * pcp_size)) % pp_size
            if (
                response["tp_rank"] != expected_tp
                or response.get("pp_rank") != expected_pp
                or response.get("pcp_rank") != expected_pcp
            ):
                entry["issues"].append("worker ranks disagree with TP/PCP/PP topology")
                continue
            slot = (response["dp_rank"], expected_tp, expected_pcp, expected_pp)
            slots.setdefault(slot, []).append(entry)
        workers.append(entry)
    for entries in slots.values():
        if len(entries) > 1:
            for entry in entries:
                entry["issues"].append(
                    "duplicate (dp_rank, tp_rank, pcp_rank, pp_rank)"
                )
    workers = [entry for entry in workers if not entry["issues"]]
    observed = [entry["response"] for entry in workers]
    observed_ranks = sorted({response["dp_rank"] for response in observed})
    complete = (
        expected_ranks is not None
        and expected_workers_per_dp is not None
        and len(observed) == len(expected_ranks) * expected_workers_per_dp
        and not any(entry["issues"] for entry in scoped_responses)
        and not any(response.get("capture_errors") for response in observed)
    )
    disagreements = []
    # PP stages own different layers. Only compare backend choices for a
    # layer that more than one worker actually observed.
    layer_backends: dict[str, set[str]] = {}
    for response in observed:
        for layer, name in response["attention_backends"].items():
            layer_backends.setdefault(layer, set()).add(name)
    if any(len(names) > 1 for names in layer_backends.values()):
        disagreements.append("attention_backends")
    for field in (
        "mla_prefill_backend",
        "cudagraph_mode_resolved",
        "cudagraph_capture_sizes_resolved",
    ):
        values = [
            response[field] for response in observed if response.get(field) is not None
        ]
        if values and any(value != values[0] for value in values[1:]):
            disagreements.append(field)
    if failure:
        resolution = failure
    elif not observed:
        resolution = "worker_probe_unobserved"
    elif disagreements:
        resolution = "worker_probe_mixed"
    else:
        resolution = "worker_probe" if complete else "worker_probe_partial"
    engine["worker_probe"] = {
        "schema_version": 1,
        "scope": "post_collection_collective_rpc",
        "responses": scoped_responses,
        "coverage": {
            "scope": "artifact_workers",
            "expected_dp_ranks": expected_ranks,
            "expected_workers_per_dp": expected_workers_per_dp,
            "observed_workers": [
                {
                    key: response.get(key)
                    for key in (
                        "dp_rank",
                        "worker_rank",
                        "tp_rank",
                        "pp_rank",
                        "pcp_rank",
                    )
                }
                for response in observed
            ],
            "missing_dp_ranks": (
                sorted(set(expected_ranks) - set(observed_ranks))
                if expected_ranks is not None
                else None
            ),
            "complete": complete,
        },
        "disagreements": disagreements,
        "error": failure,
    }
    # Keep the legacy field as a representative, explicitly labelled with
    # its actual worker identity. It never substitutes for an unobserved DP rank.
    engine["resolved"] = copy.deepcopy(observed[0]) if observed else None
    engine["resolved_scope"] = "representative_worker" if observed else None
    engine["resolution"] = resolution
    attention = engine.get("attention")
    if not isinstance(attention, dict):
        return
    names = sorted(
        {
            name
            for response in observed
            for name in response["attention_backends"].values()
        }
    )
    attention["backend_resolved"] = names[0] if len(names) == 1 else None
    prefill_names = {response.get("mla_prefill_backend") for response in observed}
    attention["mla_prefill_backend_resolved"] = (
        next(iter(prefill_names)) if len(prefill_names) == 1 else None
    )
    attention["resolution"] = "worker_probe_mixed" if len(names) > 1 else resolution


def _engine_probe_view(engine: dict) -> dict:
    """The fields ``_apply_engine_resolved`` sets on one ``engine`` block."""
    view = {
        field: engine.get(field)
        for field in ("resolved", "resolved_scope", "resolution", "worker_probe")
    }
    attention = engine.get("attention")
    if isinstance(attention, dict):
        view["attention"] = {
            field: attention.get(field)
            for field in (
                "backend_resolved",
                "mla_prefill_backend_resolved",
                "resolution",
            )
        }
    return view


def _worker_probe_sidecar(
    merged: dict, responses: list[dict], failure: str | None
) -> dict:
    """Probe evidence for the merged artifact and for each contributing rank.

    A rank's view is what ``_apply_engine_resolved`` would record for that
    rank, built without reading its artifact. The merge verified that the
    non-degraded ranks' startup ``engine`` blocks match except for the DP
    rank, so the merged block with that rank's DP rank stands in for each of
    them. A rank listed in ``engine.capture_errors`` uses the merged block's
    TP/PP/PCP sizes, which every DP rank of one launcher shares; the
    per-reply topology check rejects replies that contradict them. A rank's
    view never claims another DP rank's observation.
    """
    engine = merged["engine"]
    parallel = engine.get("parallel")
    source_ranks = (merged.get("dp") or {}).get("source_ranks") or []
    ranks: dict[str, dict] = {}
    for dp_rank, rank_file in zip(source_ranks, merged.get("rank_files") or []):
        rank_engine: dict = {
            "parallel": {
                **(parallel if isinstance(parallel, dict) else {}),
                "data_parallel_rank": dp_rank,
            }
        }
        if isinstance(engine.get("attention"), dict):
            rank_engine["attention"] = {}
        # "ranks" keeps the DP filter on this rank even if dp_rank is not an int.
        _apply_engine_resolved(
            {"engine": rank_engine, "dp": {"rank": dp_rank, "ranks": [dp_rank]}},
            responses,
            failure,
        )
        ranks[str(dp_rank)] = {
            "rank_file": rank_file,
            **_engine_probe_view(rank_engine),
        }
    return {
        "schema": WORKER_PROBE_SIDECAR_SCHEMA,
        "schema_version": 1,
        "run_id": merged.get("run_id"),
        "merged_output_path": merged.get("merged_output_path"),
        "rank_files": list(merged.get("rank_files") or []),
        "merged": _engine_probe_view(engine),
        "ranks": ranks,
    }


def _warn_provenance_write_failed(path: object) -> None:
    logger.warning("Could not record engine provenance in %s", path, exc_info=True)


async def _attach_engine_resolved(merged: dict, engine_client: AsyncLLM) -> None:
    """Keep all answers to one bounded, post-collection worker probe.

    The RPC may cover only one DP engine. Missing workers are unobserved,
    never inferred from another worker's configuration. The answers update
    the in-memory merged document (served by ``get_perf_metrics``) and are
    written to one sidecar next to the merged artifact; the rank and merged
    artifacts on disk are neither read nor rewritten. Probe, sidecar, and
    recording errors are logged, never raised: they must not discard the
    valid measurements already on disk or keep the launcher from restoring
    the workers.
    """
    if not isinstance(merged.get("engine"), dict):
        return

    def record(responses: list[dict], failure: str | None) -> None:
        _apply_engine_resolved(merged, responses, failure)
        merged_output_path = merged.get("merged_output_path")
        if not merged_output_path:
            return
        sidecar_path: Path | None = None
        try:
            sidecar_path = _benchmark_worker_probe_path(Path(merged_output_path))
            _write_json_atomic(
                sidecar_path, _worker_probe_sidecar(merged, responses, failure)
            )
        except Exception:
            # Name the sidecar once its path is known, else the value that broke it.
            _warn_provenance_write_failed(sidecar_path or merged_output_path)

    try:
        results = await asyncio.wait_for(
            engine_client.collective_rpc(
                _make_engine_probe(), timeout=ENGINE_PROBE_TIMEOUT_SECONDS
            ),
            timeout=ENGINE_PROBE_TIMEOUT_SECONDS,
        )
        responses = _worker_probe_responses(results)
        failure = None
        if not responses or all(entry["issues"] for entry in responses):
            failure = "probe_failed: no valid worker probe responses"
        record(responses, failure)
    except Exception as error:
        # A bare TimeoutError's str() is empty, and it is the single most
        # likely production failure (a dead or wedged engine) -- the
        # exception type name keeps the recorded reason from being useless.
        resolution = f"probe_failed: {type(error).__name__}: {error}"
        logger.warning("Engine provenance probe failed: %s", resolution, exc_info=True)
        try:
            record([], resolution)
        except Exception:
            # Best effort: the launcher must still get to restore the workers.
            logger.warning(
                "Could not record the engine provenance probe failure", exc_info=True
            )


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
    # The merged artifact written below names the probe sidecar's path, so a
    # sidecar left by an earlier run must not survive to pass for this run's.
    for stale_path in (merged_path, _benchmark_worker_probe_path(merged_path)):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass

    logger.info(
        "Waiting for benchmark to complete (files: %s, timeout: %ds)...",
        rank_paths,
        timeout,
    )

    deadline = _time.monotonic() + timeout
    hard_deadline = deadline + BENCHMARK_SOFT_TIMEOUT_GRACE_SECONDS
    timeout_warning_emitted = False
    while True:
        missing_paths = [path for path in rank_paths if not path.exists()]
        if not missing_paths:
            break
        now = _time.monotonic()
        if now > deadline:
            if not timeout_warning_emitted:
                logger.warning(
                    "Self-benchmark exceeded the %ds soft timeout; waiting up to "
                    "%ds for the current profiling iteration, rank cleanup, and "
                    "partial result write. Missing: %s",
                    timeout,
                    BENCHMARK_SOFT_TIMEOUT_GRACE_SECONDS,
                    missing_paths,
                )
                timeout_warning_emitted = True
            if now > hard_deadline:
                raise TimeoutError(
                    "Self-benchmark did not publish results within the soft "
                    f"timeout plus {BENCHMARK_SOFT_TIMEOUT_GRACE_SECONDS}s cleanup "
                    f"grace. Missing: {missing_paths}"
                )
        await asyncio.sleep(0.1)

    rank_data: list[tuple[int, Path, dict]] = []
    for dp_rank, p in zip(dp_ranks, rank_paths):
        with open(p) as f:
            data = json.load(f)
        _validate_benchmark_rank_payload(data, p)
        rank_data.append((dp_rank, p, data))

    merged = _merge_benchmark_rank_results(rank_data, merged_path)
    _write_json_atomic(merged_path, merged)

    if merged.get("status") == "partial":
        logger.warning(
            "Self-benchmark soft timeout returned partial results: coverage=%s. "
            "Engine startup will continue.",
            merged.get("coverage"),
        )

    logger.info(
        "Benchmark complete, %d rank-points across %d rank(s); merged results: %s",
        len(merged.get("results", [])),
        len(rank_paths),
        merged_path,
    )
    return merged


async def _stop_worker_gc_policy(engine_client: AsyncLLM) -> None:
    """Restore worker-process GC once the self-benchmark has finished.

    Model workers auto-start the FPM freeze policy when
    ``worker_extension_cls`` resolves (importing ``dynamo.vllm.gc_policy``
    starts it), while ``InstrumentedScheduler`` only restores the
    engine-core process. Without this symmetric stop the workers would keep
    serving real traffic with automatic gen2 collection disabled and the
    freeze daemon alive, so cyclic garbage would never be reclaimed.
    Awaiting the RPC holds serving until every worker has restored its
    thresholds and collected the previously frozen heap; both normal
    completion and benchmark abort funnel through the same benchmark-wait
    call sites. A failure here must propagate: serving on a worker that is
    not GC-equivalent to a never-benchmarked one is worse than failing
    startup.
    """
    if os.environ.get("DYN_FPM_GC_POLICY", "").strip().lower() != "freeze":
        return
    await engine_client.collective_rpc("fpm_gc_stop")
    logger.info("FPM GC policy stopped in all model workers")


def _disable_engine_client_cudagraph_metrics(engine_client: Any) -> int:
    """Turn cudagraph_metrics off in the engine client's process.

    Two independent steps, each logging its own WARNING when it fails, so a
    failure in one never skips the other. First drops the graph-dispatch
    logging vLLM's stat loggers built at startup, which is what stops the
    collection. Then clears the flag on the client's own config so a stat
    logger vLLM builds later (it rebuilds them on an elastic-EP scale-up)
    builds none. Returns the number of stat loggers whose graph logging was
    dropped.
    """
    cleared = 0
    try:
        logger_manager = getattr(engine_client, "logger_manager", None)
        stat_loggers = list(getattr(logger_manager, "stat_loggers", None) or [])
        for stat_logger in list(stat_loggers):
            # A per-engine adapter holds one LoggingStatLogger per DP engine.
            per_engine = getattr(stat_logger, "per_engine_stat_loggers", None)
            if isinstance(per_engine, dict):
                stat_loggers.extend(per_engine.values())
        for stat_logger in stat_loggers:
            if getattr(stat_logger, "cudagraph_logging", None) is not None:
                stat_logger.cudagraph_logging = None
                cleared += 1
    except Exception:
        logger.warning(
            "Could not turn vLLM cudagraph_metrics off in the stat loggers of "
            "the engine client after the self-benchmark; they may keep "
            "collecting CUDA graph dispatch statistics while serving",
            exc_info=True,
        )
    try:
        vllm_config = getattr(engine_client, "vllm_config", None)
        observability_config = getattr(vllm_config, "observability_config", None)
        if observability_config is not None and hasattr(
            observability_config, "cudagraph_metrics"
        ):
            observability_config.cudagraph_metrics = False
    except Exception:
        logger.warning(
            "Could not turn vLLM cudagraph_metrics off in the config of the "
            "engine client after the self-benchmark; stat loggers vLLM builds "
            "later may collect CUDA graph dispatch statistics",
            exc_info=True,
        )
    return cleared


def _restore_cudagraph_metrics(bench_cfg: dict, engine_client: AsyncLLM) -> None:
    """Turn cudagraph_metrics back off before serving if Dynamo turned it on.

    Benchmark mode enables the option so each benchmark sample records its
    CUDA graph dispatch (``update_engine_config_with_dynamo``). Left on,
    vLLM's default stat logger keeps collecting the per-step dispatch
    statistics, without bound when periodic stat logging is off, and logs
    them. An option the user enabled is left alone. Fail-soft: serving does
    not depend on this.
    """
    if not bench_cfg.get("cudagraph_metrics_auto_enabled", False):
        return
    try:
        cleared = _disable_engine_client_cudagraph_metrics(engine_client)
    except Exception:
        logger.warning(
            "Could not turn vLLM cudagraph_metrics off in the engine client "
            "after the self-benchmark; its stat loggers may keep collecting "
            "CUDA graph dispatch statistics while serving",
            exc_info=True,
        )
        return
    logger.info(
        "Turned vLLM cudagraph_metrics off in the engine client after the "
        "self-benchmark (stat loggers cleared: %d)",
        cleared,
    )


async def _restore_benchmark_workers(bench_cfg: dict, engine_client: AsyncLLM) -> None:
    try:
        if bench_cfg.get("randomize_kda_state", False):
            await engine_client.collective_rpc("finish_benchmark_kda_state")
    finally:
        try:
            await _stop_worker_gc_policy(engine_client)
        finally:
            # Last, so the GC stop never waits behind it; it never raises.
            _restore_cudagraph_metrics(bench_cfg, engine_client)


async def _await_benchmark_then_restore_workers(
    bench_cfg: dict, vllm_config: VllmConfig, engine_client: AsyncLLM
) -> dict:
    """Wait for the self-benchmark and restore worker state and GC on every exit path.

    The worker stop must not depend on the wait succeeding: ``_bench_abort``
    publishes ``status="failed"`` artifacts, so an aborted benchmark makes
    ``_wait_and_load_benchmark`` raise during validation, and the workers
    would otherwise keep automatic gen2 collection disabled through teardown
    or, if a caller survives the error, into serving. On the success path a
    stop failure fails closed; on the failure path it is logged and the
    original error propagates unchanged.
    """
    try:
        results = await _wait_and_load_benchmark(bench_cfg, vllm_config)
    except BaseException:
        # The failure may be the engine dying; an unbounded RPC would then
        # hang the launcher on the very path that is supposed to surface the
        # error, so the cleanup stop is time-boxed and the original error
        # always wins.
        try:
            await asyncio.wait_for(
                _restore_benchmark_workers(bench_cfg, engine_client),
                timeout=WORKER_GC_STOP_TIMEOUT_SECONDS,
            )
        except BaseException:
            logger.exception(
                "Failed to restore model workers while "
                "handling a self-benchmark failure"
            )
        raise
    await _attach_engine_resolved(results, engine_client)
    await asyncio.wait_for(
        _restore_benchmark_workers(bench_cfg, engine_client),
        timeout=WORKER_GC_STOP_TIMEOUT_SECONDS,
    )
    return results


SetupVllmEngineFn = Callable[..., EngineSetupResult]
SetupKvEventPublisherFn = Callable[..., Optional[Any]]
SetupKvStateAttachmentOwnerFn = Callable[..., Awaitable[Optional[Any]]]
RegisterVllmModelFn = Callable[..., Awaitable[None]]
SetupFpmRelayFn = Callable[..., Optional[list]]
SetupMetricsCollectionFn = Callable[..., None]


@dataclass
class _DecodeWorkerLifecycle:
    engine_client: Optional[AsyncLLM] = None
    vllm_config: Optional[VllmConfig] = None
    handler: Optional[BaseWorkerHandler] = None
    shutdown_event: asyncio.Event | None = None

    def __enter__(self) -> "_DecodeWorkerLifecycle":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        original_error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        try:
            self.cleanup()
        except Exception:
            if original_error is None:
                raise
            logger.exception(
                "Failed to clean up decode worker after an earlier failure"
            )

    def cleanup(self) -> None:
        """Release resources in reverse construction order."""
        logger.debug("Cleaning up decode worker")
        if self.shutdown_event is not None:
            self.shutdown_event.set()
        try:
            if self.handler is not None:
                self.handler.cleanup()
        finally:
            if self.engine_client is not None and self.vllm_config is not None:
                self.engine_client.shutdown(timeout=self.vllm_config.shutdown_timeout)


class WorkerFactory:
    """Factory for creating and initializing multimodal vLLM workers."""

    def __init__(
        self,
        setup_vllm_engine_fn: SetupVllmEngineFn,
        setup_kv_event_publisher_fn: SetupKvEventPublisherFn,
        register_vllm_model_fn: RegisterVllmModelFn,
        setup_fpm_relay_fn: SetupFpmRelayFn,
        setup_metrics_collection_fn: SetupMetricsCollectionFn,
        setup_kv_state_attachment_owner_fn: SetupKvStateAttachmentOwnerFn | None = None,
        state_agent_lifecycle: StateAgentLifecycle | None = None,
    ):
        self.setup_vllm_engine = setup_vllm_engine_fn
        self.setup_kv_event_publisher = setup_kv_event_publisher_fn
        self.setup_kv_state_attachment_owner = setup_kv_state_attachment_owner_fn
        self.register_vllm_model = register_vllm_model_fn
        self.setup_fpm_relay = setup_fpm_relay_fn
        self.setup_metrics_collection = setup_metrics_collection_fn
        self.state_agent_lifecycle = state_agent_lifecycle or StateAgentLifecycle()

    async def _setup_kv_routing(
        self,
        config: Config,
        generate_endpoint: Endpoint,
        vllm_config: VllmConfig,
        *,
        consolidator_enabled: bool,
        consolidator_port: int | None,
    ) -> Optional[Any]:
        if state_agent_settings(config) is None:
            return self.setup_kv_event_publisher(
                config,
                generate_endpoint,
                vllm_config,
                consolidator_enabled=consolidator_enabled,
                consolidator_port=consolidator_port,
            )

        # NOTE: Source mode is immutable for one worker lifecycle. An opted-in
        # worker never falls back to, or concurrently starts, the ordinary KV
        # publisher; setup failure disables KV routing while serving continues.
        try:
            if self.setup_kv_state_attachment_owner is None:
                raise RuntimeError("KV state-agent attachment setup is unavailable")
            owner = await self.setup_kv_state_attachment_owner(
                config, generate_endpoint, vllm_config
            )
            if owner is not None:
                await self.state_agent_lifecycle.install(owner)
        except Exception:
            logger.exception(
                "KV state-agent attachment setup failed; KV routing remains disabled"
            )
        return None

    async def create(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,
        snapshot_engine: Optional[SnapshotEngineSetupResult] = None,
    ) -> None:
        """Create the appropriate multimodal worker based on config flags."""

        if config.realtime:
            await self._create_realtime_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                snapshot_engine=snapshot_engine,
            )
            return

        # Embedding worker is selected first because it crosses worker shapes
        # (pooling AsyncLLM, ModelType.Embedding) rather than being a variant
        # of decode. Aggregated-only — exclusivity with disagg modes is
        # enforced earlier in DynamoVllmConfig._validate_embedding_worker_exclusivity.
        if config.embedding_worker:
            await self._create_embedding_worker(
                runtime, config, shutdown_event, shutdown_endpoints
            )
            return

        if config.classify_worker:
            await self._create_classify_worker(
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

    async def _create_realtime_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,
        snapshot_engine: Optional[SnapshotEngineSetupResult] = None,
    ) -> None:
        """Initialize an aggregated vLLM realtime worker."""
        del shutdown_event  # Connection cancellation is carried by Dynamo Context.

        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        shutdown_endpoints[:] = [generate_endpoint]

        fpm_worker_id = str(generate_endpoint.connection_id())
        if snapshot_engine is not None:
            engine_setup, factory = snapshot_engine
            (
                engine_client,
                vllm_config,
                _default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = engine_setup
            os.environ[ENV_FPM_WORKER_ID] = fpm_worker_id
            factory.bind_endpoint(generate_endpoint)
        else:
            factory = StatLoggerFactory(endpoint=generate_endpoint)
            (
                engine_client,
                vllm_config,
                _default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = self.setup_vllm_engine(
                config,
                factory,
                fpm_worker_id=fpm_worker_id,
            )
        await configure_kv_event_block_size(engine_client, vllm_config)
        _, dp_size = get_dp_range_for_worker(vllm_config)
        num_gpu_blocks = per_rank_kv_blocks(
            vllm_config.cache_config.num_gpu_blocks,
            dp_size,
        )
        factory.set_num_gpu_blocks_all(num_gpu_blocks or 0)
        factory.init_publish()

        model_name = config.served_model_name or config.model
        handler = RealtimeHandler(
            {
                "transcription": RealtimeTranscriptionHandler.from_engine(
                    engine_client=engine_client,
                    model_name=model_name,
                    model_path=config.model,
                )
            }
        )
        self.setup_metrics_collection(config, generate_endpoint, logger)

        await self.register_vllm_model(
            ModelInput.Text,
            ModelType.Realtime,
            generate_endpoint,
            config,
            engine_client,
            vllm_config,
            worker_type=WorkerType.Aggregated,
            needs=[],
        )
        register_model_taint_route(runtime, generate_endpoint)

        metrics_labels = [
            (prometheus_names.labels.MODEL, model_name),
            (prometheus_names.labels.MODEL_NAME, model_name),
        ]
        logger.info(
            "Starting realtime worker endpoint for model: %s",
            model_name,
        )
        try:
            await generate_endpoint.serve_bidirectional_endpoint(
                handler.generate,
                graceful_shutdown=True,
                metrics_labels=metrics_labels,
            )
        except Exception as exc:
            logger.error("Realtime worker failed: %s", exc)
            raise
        finally:
            if prometheus_temp_dir is not None:
                prometheus_temp_dir.cleanup()

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
            enable_frontend_decoding=config.frontend_decoding,
            embedding_cache_capacity_gb=config.multimodal_embedding_cache_capacity_gb,
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
        register_model_taint_route(runtime, generate_endpoint)
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
            engine_cleanup_resource,
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

        register_model_taint_route(runtime, generate_endpoint)
        logger.info("Starting to serve the embedding worker endpoint...")
        try:
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=[("model", config.model)],
                    health_check_payload=embedding_health_check_payload,
                ),
                self.register_vllm_model(
                    (
                        ModelInput.Tokens
                        if config.embedding_frontend_tokenization
                        else ModelInput.Text
                    ),
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
            # Attached multi-client AsyncLLMs do not own EngineCore. Close all
            # clients first, then let the parent cleanup resource terminate
            # child endpoints and finally the shared EngineCore.
            try:
                engine_client.shutdown()
            except Exception:
                logger.exception("Failed to shut down embedding AsyncLLM client")
            if engine_cleanup_resource is not None:
                try:
                    engine_cleanup_resource.cleanup()
                except Exception:
                    logger.exception("Failed to clean up embedding engine resources")

    async def _create_classify_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
    ) -> None:
        """Initialize an aggregated sequence-classification worker.

        Like the embeddings worker, this uses a pooling ``AsyncLLM`` and skips
        the generation-only KV-cache and scheduler machinery. The combined
        model type advertises both pooling-family endpoints.
        """
        generate_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.{config.endpoint}"
        )
        shutdown_endpoints[:] = [generate_endpoint]

        fpm_worker_id = str(generate_endpoint.connection_id())
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

        handler = ClassifyWorkerHandler(
            runtime=runtime,
            engine=engine_client,
            config=config,
            model_config=getattr(vllm_config, "model_config", None),
            shutdown_event=shutdown_event,
        )

        classify_health_check_payload = VllmEmbeddingHealthCheckPayload(
            model_name=config.served_model_name or config.model
        ).to_dict()

        logger.info("Starting to serve the classify worker endpoint...")
        try:
            await asyncio.gather(
                generate_endpoint.serve_endpoint(
                    handler.generate,
                    metrics_labels=[("model", config.model)],
                    health_check_payload=classify_health_check_payload,
                ),
                self.register_vllm_model(
                    ModelInput.Text,
                    ModelType.Classify | ModelType.Pooling,
                    generate_endpoint,
                    config,
                    engine_client,
                    vllm_config,
                    worker_type=WorkerType.Aggregated,
                    needs=[],
                ),
            )
        except Exception as e:
            logger.error(f"Failed to serve classify worker endpoint: {e}")
            raise
        finally:
            handler.cleanup()

    def _maybe_create_failover_metrics(self, config: Config, generate_endpoint):
        """Create + register per-engine failover metrics (shadow mode only).

        Called before the model loads so ``init`` spans the load and a restarted
        engine re-exposes its persisted switch counters within seconds. Uses a
        dedicated registry surfaced on ``generate_endpoint``'s system /metrics.
        """
        if config.gms_shadow_mode is not True:
            return None
        from gpu_memory_service.failover_lock.failover_metrics import (
            create_failover_metrics,
        )

        persist_dir = os.path.dirname(
            os.path.abspath(
                os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
            )
        )
        failover_metrics = create_failover_metrics(
            endpoint=generate_endpoint,
            engine_id=os.environ.get("ENGINE_ID", "0"),
            model_name=config.served_model_name or config.model,
            component_name=config.component,
            persist_dir=persist_dir,
        )
        failover_metrics.set_state("init")
        return failover_metrics

    async def _maybe_wait_for_failover_lock(
        self,
        handler,
        runtime: DistributedRuntime,
        config: Config,
        failover_metrics=None,
    ) -> bool:
        # Shadow mode: sleep → probe → block on lock → wake. True only for a real
        # (contended) failover, not the initial bootup. The election itself is
        # shared with the snapshot restore path, which arrives already paused;
        # a cold-start engine is awake, so it sleeps here first.
        if config.gms_shadow_mode is not True:
            return False

        await handler._pause_controller.pause(1)
        lock = await elect_and_wake(
            handler._pause_controller,
            runtime,
            lock_path=os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock"),
            failover_metrics=failover_metrics,
        )
        return lock is not None and lock.was_contended

    async def _create_decode_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
        snapshot_engine: Optional[SnapshotEngineSetupResult] = None,
    ) -> None:
        """
        Instantiate and serve
        """
        with _DecodeWorkerLifecycle(shutdown_event=shutdown_event) as lifecycle:
            try:
                await self._run_decode_worker(
                    runtime,
                    config,
                    shutdown_event,
                    shutdown_endpoints,
                    snapshot_engine=snapshot_engine,
                    lifecycle=lifecycle,
                )
            finally:
                await self.state_agent_lifecycle.close()

    async def _run_decode_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
        snapshot_engine: Optional[SnapshotEngineSetupResult],
        lifecycle: _DecodeWorkerLifecycle,
    ) -> None:
        """Initialize and serve a decode worker."""

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

        # Shadow mode: create metrics + enter 'init' before load, so 'init' spans it.
        failover_metrics = self._maybe_create_failover_metrics(
            config, generate_endpoint
        )

        # Use pre-created engine if provided (checkpoint mode), otherwise create new
        fpm_worker_id = str(generate_endpoint.connection_id())
        if snapshot_engine is not None:
            engine_setup, factory = snapshot_engine
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = engine_setup
            os.environ[ENV_FPM_WORKER_ID] = fpm_worker_id
            factory.bind_endpoint(generate_endpoint)
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
                _component_gauges,
            ) = self.setup_vllm_engine(config, factory, fpm_worker_id=fpm_worker_id)
        lifecycle.engine_client = engine_client
        lifecycle.vllm_config = vllm_config
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
        lifecycle.handler = handler
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
        kv_publishers = await self._setup_kv_routing(
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
            runtime,
            generate_endpoint,
            handler,
            lora_enabled=lora_enabled,
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

        was_failover = False
        if snapshot_engine is None:
            was_failover = await self._maybe_wait_for_failover_lock(
                handler, runtime, config, failover_metrics
            )

        # Wait for self-benchmark to complete before registering.
        bench_cfg = vllm_config.additional_config.get("benchmark")
        if bench_cfg:
            handler._benchmark_results = await _await_benchmark_then_restore_workers(
                bench_cfg, vllm_config, handler.engine_client
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

        handler._first_token_source = await generate_endpoint.first_token_source(
            worker_type
        )

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
        # Serving now: a failover that got here succeeded. Gated on was_failover
        # (same as the attempt) so bootup isn't counted and success pairs with attempt.
        if failover_metrics is not None:
            failover_metrics.set_state("active")
            if was_failover:
                failover_metrics.record_switch_success()

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

    async def _create_prefill_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,
        snapshot_engine: Optional[SnapshotEngineSetupResult] = None,
    ) -> None:
        try:
            await self._run_prefill_worker(
                runtime,
                config,
                shutdown_event,
                shutdown_endpoints,
                snapshot_engine,
            )
        except BaseException:
            await self.state_agent_lifecycle.close()
            raise

    async def _run_prefill_worker(
        self,
        runtime: DistributedRuntime,
        config: Config,
        shutdown_event: asyncio.Event,
        shutdown_endpoints: list,  # mutated in place
        snapshot_engine: Optional[SnapshotEngineSetupResult] = None,
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

        # Shadow mode: create metrics + enter 'init' before load, so 'init' spans it.
        failover_metrics = self._maybe_create_failover_metrics(
            config, generate_endpoint
        )

        # Use pre-created engine if provided (checkpoint mode), otherwise create new
        fpm_worker_id = str(generate_endpoint.connection_id())
        snapshot_factory: Optional[StatLoggerFactory] = None
        if snapshot_engine is not None:
            engine_setup, snapshot_factory = snapshot_engine
            (
                engine_client,
                vllm_config,
                default_sampling_params,
                prometheus_temp_dir,
                _component_gauges,
            ) = engine_setup
            snapshot_factory.bind_endpoint(generate_endpoint)
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

        if snapshot_factory is not None:
            _, dp_size = get_dp_range_for_worker(vllm_config)
            per_rank_num_gpu_blocks = per_rank_kv_blocks(
                vllm_config.cache_config.num_gpu_blocks,
                dp_size,
            )
            snapshot_factory.set_num_gpu_blocks_all(per_rank_num_gpu_blocks or 0)
            snapshot_factory.init_publish()

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
        kv_publishers = await self._setup_kv_routing(
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
            runtime,
            generate_endpoint,
            handler,
            lora_enabled=config.engine_args.enable_lora,
        )

        was_failover = False
        if snapshot_engine is None:
            was_failover = await self._maybe_wait_for_failover_lock(
                handler, runtime, config, failover_metrics
            )

        # Wait for self-benchmark to complete before registering.
        bench_cfg = vllm_config.additional_config.get("benchmark")
        if bench_cfg:
            handler._benchmark_results = await _await_benchmark_then_restore_workers(
                bench_cfg, vllm_config, handler.engine_client
            )

        perf_endpoint = runtime.endpoint(
            f"{config.namespace}.{config.component}.get_perf_metrics"
        )
        shutdown_endpoints[:] = [generate_endpoint, clear_endpoint, perf_endpoint]
        if rl_endpoint is not None:
            shutdown_endpoints.append(rl_endpoint)
        if lora_enabled:
            shutdown_endpoints.extend(
                [load_lora_endpoint, unload_lora_endpoint, list_loras_endpoint]
            )

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
        # Serving now: a failover that got here succeeded. Gated on was_failover
        # (same as the attempt) so bootup isn't counted and success pairs with attempt.
        if failover_metrics is not None:
            failover_metrics.set_state("active")
            if was_failover:
                failover_metrics.record_switch_success()

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
            if lora_enabled:
                serve_tasks.extend(
                    [
                        load_lora_endpoint.serve_endpoint(
                            handler.load_lora,
                            metrics_labels=prefill_metrics_labels,
                        ),
                        unload_lora_endpoint.serve_endpoint(
                            handler.unload_lora,
                            metrics_labels=prefill_metrics_labels,
                        ),
                        list_loras_endpoint.serve_endpoint(
                            handler.list_loras,
                            metrics_labels=prefill_metrics_labels,
                        ),
                    ]
                )
            await asyncio.gather(*serve_tasks)
            logger.debug("serve_endpoint completed for prefill worker")
        except Exception as e:
            logger.error(f"Failed to serve endpoints: {e}")
            raise
        finally:
            logger.debug("Cleaning up prefill worker")
            await self.state_agent_lifecycle.close()
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
        generate_endpoint: Endpoint,
        handler: BaseWorkerHandler,
        lora_enabled: bool = False,
    ) -> None:
        """Register all engine routes for this handler.

        Args:
            runtime: The DistributedRuntime instance to register routes on.
            generate_endpoint: Worker endpoint whose model taints can be updated.
        """
        register_model_taint_route(runtime, generate_endpoint)
        runtime.register_engine_route("control/start_profile", handler.start_profile)
        runtime.register_engine_route("control/stop_profile", handler.stop_profile)
        runtime.register_engine_route("control/sleep", handler.sleep)
        runtime.register_engine_route("control/wake_up", handler.wake_up)
        runtime.register_engine_route(
            "control/scale_elastic_ep", handler.scale_elastic_ep
        )
        runtime.register_engine_route("control/ep_capacity", handler.get_ep_capacity)

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
            "set_weight_version": handler.set_weight_version,
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
            "control/scale_elastic_ep, control/ep_capacity, "
            "control/start_profile, control/stop_profile, "
            "and RL admin routes: %s%s",
            ", ".join(sorted(rl_routes)),
            " (LoRA routes: load_lora, unload_lora)" if lora_enabled else "",
        )
