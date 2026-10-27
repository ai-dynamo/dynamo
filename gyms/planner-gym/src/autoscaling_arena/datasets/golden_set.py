# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build reusable Golden workload sets from externally supplied base traces.

The recipe deliberately contains no source path.  A caller supplies one or more
homogeneous trace collections at runtime, and this module writes replay JSONL
plus a path-redacted manifest.  Source request/session identifiers are used only
for deduplication and whole-session selection; they are never emitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import stat
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoscaling_arena.datasets.dynamo_trace import (
    TRACE_SCHEMA,
    discover_trace_shards,
    iter_trace_lines,
    parse_trace_event,
)
from autoscaling_arena.workloads.generator import validate_mooncake_trace

GOLDEN_SET_SCHEMA_VERSION = 1
SOURCE_FORMATS = {"auto", "dynamo_request_trace_v1", "replay_jsonl"}
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


@dataclass(frozen=True)
class BaseRequest:
    """One normalized source request.  ``source_key`` is never serialized."""

    source_key: str
    source_timestamp_ms: int | float
    input_length: int
    output_length: int
    hash_ids: tuple[int, ...]
    block_size: int
    session_id: str | None = None


@dataclass
class SourceStats:
    files: int = 0
    rows: int = 0
    request_rows: int = 0
    non_request_rows: int = 0
    duplicate_requests: int = 0
    filtered_requests: int = 0
    filtered_session_requests: int = 0
    session_tagged_requests: int = 0
    complete_sessions: int = 0
    loaded_hash_references: int = 0
    block_sizes: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class GoldenSetRecipe:
    """Validated Golden Set recipe."""

    data: Mapping[str, Any]
    fingerprint_sha256: str


@dataclass(frozen=True)
class GoldenSetResult:
    output_dir: Path
    manifest_path: Path
    match_config_fragment_path: Path
    trace_paths: tuple[Path, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class _RequestGroup:
    key: str
    requests: tuple[BaseRequest, ...]


@dataclass(frozen=True)
class _PhasePlan:
    name: str
    pool: str
    start_ms: int
    end_ms: int
    timestamps_ms: tuple[int, ...]
    schedule: Mapping[str, Any]


def load_golden_set_recipe(path: Path) -> GoldenSetRecipe:
    """Load and strictly validate a Golden Set YAML recipe."""

    # Keep package imports usable in lightweight environments. PyYAML is a
    # required runtime dependency, but only Golden Set YAML I/O needs it.
    import yaml

    try:
        value = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read recipe: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid recipe YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("recipe must be a YAML mapping")
    data = _validate_recipe(dict(value))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return GoldenSetRecipe(
        data=data, fingerprint_sha256=hashlib.sha256(canonical).hexdigest()
    )


def build_golden_set(
    recipe: GoldenSetRecipe,
    sources: Sequence[Path],
    output_dir: Path,
    *,
    source_format: str = "auto",
    source_block_size: int | None = None,
    reference_rps: float,
    seed: int = 0,
    max_source_requests: int | None = 100_000,
    max_source_hashes: int | None = 5_000_000,
) -> GoldenSetResult:
    """Construct replay traces from external base traffic.

    Each workload receives an independently derived random stream, so adding or
    reordering another workload cannot perturb existing outputs.  Selection is
    without replacement within one workload and may reuse source requests across
    different workloads.
    """

    # Match fragment serialization is the only YAML dependency on this path.
    import yaml

    if source_format not in SOURCE_FORMATS:
        raise ValueError(f"unsupported source format {source_format!r}")
    if not math.isfinite(reference_rps) or reference_rps <= 0:
        raise ValueError("reference_rps must be a finite positive number")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if source_block_size is not None and source_block_size <= 0:
        raise ValueError("source_block_size must be positive")
    if max_source_requests is not None and max_source_requests <= 0:
        raise ValueError("max_source_requests must be positive or null")
    if max_source_hashes is not None and max_source_hashes <= 0:
        raise ValueError("max_source_hashes must be positive or null")
    if not sources:
        raise ValueError("at least one source path is required")
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new directory"
        )

    requests, stats, resolved_format, source_fingerprint = _load_sources(
        sources,
        source_format=source_format,
        source_block_size=source_block_size,
        max_source_requests=max_source_requests,
        max_source_hashes=max_source_hashes,
        selection=recipe.data["selection"],
    )
    if not requests:
        raise ValueError("source contains no replayable requests after filtering")
    if len(stats.block_sizes) != 1:
        raise ValueError(
            "source contains mixed trace block sizes "
            f"{sorted(stats.block_sizes)}; build each block-size collection separately"
        )
    block_size = next(iter(stats.block_sizes))

    groups = _group_requests(
        requests,
        preserve_sessions=recipe.data["selection"]["session_policy"] == "preserve",
    )
    pools = _materialize_pools(groups, recipe.data["pools"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir, publish_mode = _create_staging_directory(output_dir)
    trace_paths: list[Path] = []
    workload_manifests: list[dict[str, Any]] = []
    committed = False
    try:
        for workload in recipe.data["workloads"]:
            trace_path, workload_manifest = _build_workload(
                workload,
                groups=groups,
                pools=pools,
                output_dir=build_dir,
                block_size=block_size,
                reference_rps=reference_rps,
                seed=seed,
                arrival_process=recipe.data["arrival_process"]["type"],
                max_count_error_fraction=recipe.data["selection"][
                    "max_count_error_fraction"
                ],
            )
            trace_paths.append(trace_path)
            workload_manifests.append(workload_manifest)

        source_summary = {
            "format": resolved_format,
            "fingerprint_sha256": source_fingerprint,
            "files": stats.files,
            "rows": stats.rows,
            "request_rows": stats.request_rows,
            "unique_request_rows": stats.request_rows - stats.duplicate_requests,
            "non_request_rows": stats.non_request_rows,
            "duplicate_requests": stats.duplicate_requests,
            "filtered_requests": stats.filtered_requests,
            "filtered_session_requests": stats.filtered_session_requests,
            "replayable_requests": len(requests),
            "loaded_hash_references": stats.loaded_hash_references,
            "block_size": block_size,
            "session_tagged_requests": stats.session_tagged_requests,
            "session_coverage_fraction": round(
                stats.session_tagged_requests
                / max(1, stats.request_rows - stats.duplicate_requests),
                8,
            ),
            "complete_sessions_within_input_scope": stats.complete_sessions,
        }
        manifest: dict[str, Any] = {
            "schema": "autoscaling_arena.golden_set_manifest.v1",
            "recipe_schema_version": GOLDEN_SET_SCHEMA_VERSION,
            "recipe_fingerprint_sha256": recipe.fingerprint_sha256,
            "seed": seed,
            "reference_rps": reference_rps,
            "arrival_process": recipe.data["arrival_process"]["type"],
            "selection": {
                "replacement": False,
                "session_policy": recipe.data["selection"]["session_policy"],
                "missing_session": "singleton",
                "whole_session_selection": (
                    recipe.data["selection"]["session_policy"] == "preserve"
                ),
                "within_session_source_order_preserved": (
                    recipe.data["selection"]["session_policy"] == "preserve"
                ),
                "intra_session_gaps_preserved": False,
                "session_completeness": "within_input_scope",
                "replay_session_ids_emitted": False,
                "replay_arrival_semantics": "open_loop",
            },
            "source": source_summary,
            "workloads": workload_manifests,
        }
        manifest_path = build_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        fragment = {
            "evaluations": {
                "traces": [
                    {
                        "name": item["name"],
                        "path": item["file"],
                        "block_size": block_size,
                        "presorted": True,
                    }
                    for item in workload_manifests
                ],
                "defaults": {
                    "seed": 0,
                    "max_requests": None,
                    "arrival_speedup": 1.0,
                },
            }
        }
        fragment_path = build_dir / "match-config.fragment.yaml"
        fragment_path.write_text(yaml.safe_dump(fragment, sort_keys=False))

        # Recheck immediately before the atomic publish. Another process may
        # have created the destination while this build was in progress.
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"output directory already exists: {output_dir}; choose a new directory"
            )
        build_dir.chmod(publish_mode)
        build_dir.rename(output_dir)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(build_dir, ignore_errors=True)

    return GoldenSetResult(
        output_dir=output_dir,
        manifest_path=output_dir / "manifest.json",
        match_config_fragment_path=output_dir / "match-config.fragment.yaml",
        trace_paths=tuple(output_dir / path.name for path in trace_paths),
        manifest=manifest,
    )


def _create_staging_directory(output_dir: Path) -> tuple[Path, int]:
    """Create a private sibling directory and remember normal publish mode."""

    for _ in range(100):
        candidate = output_dir.parent / (
            f".{output_dir.name}.tmp-{secrets.token_hex(8)}"
        )
        try:
            # Path.mkdir's default 0777 mode honors the caller's umask. Keep
            # that mode for publication, but hide incomplete contents while
            # construction is in progress.
            candidate.mkdir()
        except FileExistsError:
            continue
        publish_mode = stat.S_IMODE(candidate.stat().st_mode)
        candidate.chmod(0o700)
        return candidate, publish_mode
    raise FileExistsError(f"could not allocate staging directory for {output_dir}")


def _validate_recipe(data: dict[str, Any]) -> dict[str, Any]:
    _only_keys(
        data,
        {"schema_version", "selection", "arrival_process", "pools", "workloads"},
        "recipe",
    )
    if data.get("schema_version") != GOLDEN_SET_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {GOLDEN_SET_SCHEMA_VERSION}")

    selection = _mapping(data.get("selection", {}), "selection")
    _only_keys(
        selection,
        {
            "replacement",
            "session_policy",
            "missing_session",
            "max_count_error_fraction",
            "filters",
        },
        "selection",
    )
    replacement = selection.get("replacement", False)
    if replacement is not False:
        raise ValueError("selection.replacement: only false is currently supported")
    session_policy = selection.get("session_policy", "preserve")
    if session_policy not in {"preserve", "ignore"}:
        raise ValueError("selection.session_policy: expected preserve or ignore")
    if selection.get("missing_session", "singleton") != "singleton":
        raise ValueError("selection.missing_session: only singleton is supported")
    max_error = _number(
        selection.get("max_count_error_fraction", 0.02),
        "selection.max_count_error_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    filters = _mapping(selection.get("filters", {}), "selection.filters")
    filter_keys = {
        "min_input_length",
        "min_output_length",
        "max_input_length",
        "max_output_length",
        "max_total_length",
    }
    _only_keys(filters, filter_keys, "selection.filters")
    normalized_filters = {
        "min_input_length": _integer(
            filters.get("min_input_length", 1),
            "selection.filters.min_input_length",
            minimum=1,
        ),
        "min_output_length": _integer(
            filters.get("min_output_length", 1),
            "selection.filters.min_output_length",
            minimum=1,
        ),
        "max_input_length": _optional_integer(
            filters.get("max_input_length"),
            "selection.filters.max_input_length",
            minimum=1,
        ),
        "max_output_length": _optional_integer(
            filters.get("max_output_length"),
            "selection.filters.max_output_length",
            minimum=1,
        ),
        "max_total_length": _optional_integer(
            filters.get("max_total_length"),
            "selection.filters.max_total_length",
            minimum=1,
        ),
    }
    normalized_selection = {
        "replacement": False,
        "session_policy": session_policy,
        "missing_session": "singleton",
        "max_count_error_fraction": max_error,
        "filters": normalized_filters,
    }

    arrival = _mapping(data.get("arrival_process", {}), "arrival_process")
    _only_keys(arrival, {"type"}, "arrival_process")
    arrival_type = arrival.get("type", "poisson")
    if arrival_type not in {"poisson", "deterministic"}:
        raise ValueError("arrival_process.type: expected poisson or deterministic")

    raw_pools = _mapping(data.get("pools", {"all": {"type": "all"}}), "pools")
    if not raw_pools:
        raise ValueError("pools must not be empty")
    pools: dict[str, dict[str, Any]] = {}
    for name, raw_pool in raw_pools.items():
        _validate_name(name, f"pools.{name}")
        pool = _mapping(raw_pool, f"pools.{name}")
        _only_keys(
            pool,
            {"type", "field", "lower_quantile", "upper_quantile"},
            f"pools.{name}",
        )
        pool_type = pool.get("type")
        if pool_type == "all":
            if set(pool) != {"type"}:
                raise ValueError(f"pools.{name}: an all pool accepts only type")
            pools[name] = {"type": "all"}
            continue
        if pool_type != "quantile":
            raise ValueError(f"pools.{name}.type: expected all or quantile")
        field_name = pool.get("field")
        if field_name not in {
            "input_length",
            "output_length",
            "total_length",
            "input_to_output_ratio",
            "output_to_input_ratio",
        }:
            raise ValueError(
                f"pools.{name}.field: expected input_length, output_length, "
                "total_length, input_to_output_ratio, or output_to_input_ratio"
            )
        lower = _number(
            pool.get("lower_quantile", 0.0),
            f"pools.{name}.lower_quantile",
            minimum=0.0,
            maximum=1.0,
        )
        upper = _number(
            pool.get("upper_quantile", 1.0),
            f"pools.{name}.upper_quantile",
            minimum=0.0,
            maximum=1.0,
        )
        if lower > upper:
            raise ValueError(f"pools.{name}: lower_quantile exceeds upper_quantile")
        pools[name] = {
            "type": "quantile",
            "field": field_name,
            "lower_quantile": lower,
            "upper_quantile": upper,
        }

    raw_workloads = data.get("workloads")
    if not isinstance(raw_workloads, list) or not raw_workloads:
        raise ValueError("workloads must be a non-empty list")
    # Imported lazily to avoid coupling source-reader import time to the
    # built-in workload registry.
    from autoscaling_arena.workloads.registry import WORKLOADS

    reserved_evaluation_names = set(WORKLOADS).union({"synthetic", "recorded", "all"})
    workloads: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_workload in enumerate(raw_workloads):
        path = f"workloads[{index}]"
        workload = _mapping(raw_workload, path)
        name = workload.get("name")
        _validate_name(name, f"{path}.name")
        if name in seen_names:
            raise ValueError(f"{path}.name: duplicate workload {name!r}")
        # Generated names become external evaluation names in the emitted Match
        # Config fragment, where built-in workload and suite names are reserved.
        if name in reserved_evaluation_names:
            raise ValueError(
                f"{path}.name: {name!r} conflicts with a built-in workload or suite"
            )
        seen_names.add(name)
        description = workload.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"{path}.description: expected a string")
        is_burst = "burst_overlay" in workload or "baseline_load_factor" in workload
        if is_burst:
            normalized = _validate_burst_workload(workload, path, pools)
        else:
            normalized = _validate_phase_workload(workload, path, pools)
        normalized["name"] = name
        normalized["description"] = description
        workloads.append(normalized)

    return {
        "schema_version": GOLDEN_SET_SCHEMA_VERSION,
        "selection": normalized_selection,
        "arrival_process": {"type": arrival_type},
        "pools": pools,
        "workloads": workloads,
    }


def _validate_phase_workload(
    workload: Mapping[str, Any], path: str, pools: Mapping[str, Any]
) -> dict[str, Any]:
    _only_keys(workload, {"name", "description", "repeat", "phases"}, path)
    repeat = _integer(workload.get("repeat", 1), f"{path}.repeat", minimum=1)
    raw_phases = workload.get("phases")
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError(f"{path}.phases: expected a non-empty list")
    phases = []
    for index, raw_phase in enumerate(raw_phases):
        phase_path = f"{path}.phases[{index}]"
        phase = _mapping(raw_phase, phase_path)
        _only_keys(
            phase, {"duration_s", "load_factor", "interpolation", "pool"}, phase_path
        )
        duration = _number(
            phase.get("duration_s"), f"{phase_path}.duration_s", minimum=0.001
        )
        load_factor = _load_factor(
            phase.get("load_factor"), f"{phase_path}.load_factor"
        )
        interpolation = phase.get("interpolation")
        if isinstance(load_factor, list):
            if interpolation != "linear":
                raise ValueError(
                    f"{phase_path}.interpolation: a load-factor range requires linear"
                )
        elif interpolation is not None:
            raise ValueError(
                f"{phase_path}.interpolation: only valid with a load-factor range"
            )
        pool = phase.get("pool", "all")
        if pool not in pools:
            raise ValueError(f"{phase_path}.pool: unknown pool {pool!r}")
        phases.append(
            {
                "duration_s": duration,
                "load_factor": load_factor,
                "interpolation": interpolation,
                "pool": pool,
            }
        )
    return {"kind": "phases", "repeat": repeat, "phases": phases}


def _validate_burst_workload(
    workload: Mapping[str, Any], path: str, pools: Mapping[str, Any]
) -> dict[str, Any]:
    _only_keys(
        workload,
        {
            "name",
            "description",
            "duration_s",
            "baseline_load_factor",
            "pool",
            "burst_overlay",
        },
        path,
    )
    duration = _number(workload.get("duration_s"), f"{path}.duration_s", minimum=0.001)
    baseline = _number(
        workload.get("baseline_load_factor"),
        f"{path}.baseline_load_factor",
        minimum=0.0,
    )
    pool = workload.get("pool", "all")
    if pool not in pools:
        raise ValueError(f"{path}.pool: unknown pool {pool!r}")
    burst = _mapping(workload.get("burst_overlay"), f"{path}.burst_overlay")
    _only_keys(
        burst,
        {"count", "width_s", "load_factor", "minimum_gap_s"},
        f"{path}.burst_overlay",
    )
    count = _integer(burst.get("count"), f"{path}.burst_overlay.count", minimum=1)
    widths = _range(
        burst.get("width_s"), f"{path}.burst_overlay.width_s", minimum=0.001
    )
    factors = _range(
        burst.get("load_factor"), f"{path}.burst_overlay.load_factor", minimum=0.0
    )
    gap = _number(
        burst.get("minimum_gap_s", 0.0),
        f"{path}.burst_overlay.minimum_gap_s",
        minimum=0.0,
    )
    maximum_required = count * widths[1] + max(0, count - 1) * gap
    if maximum_required > duration:
        raise ValueError(f"{path}.burst_overlay: bursts cannot fit within duration")
    return {
        "kind": "bursts",
        "duration_s": duration,
        "baseline_load_factor": baseline,
        "pool": pool,
        "burst_overlay": {
            "count": count,
            "width_s": widths,
            "load_factor": factors,
            "minimum_gap_s": gap,
        },
    }


def _load_sources(
    sources: Sequence[Path],
    *,
    source_format: str,
    source_block_size: int | None,
    max_source_requests: int | None,
    max_source_hashes: int | None,
    selection: Mapping[str, Any],
) -> tuple[list[BaseRequest], SourceStats, str, str]:
    shards: list[Path] = []
    seen_paths: set[Path] = set()
    for source in sources:
        for shard in discover_trace_shards(source.expanduser().resolve()):
            resolved = shard.resolve()
            if resolved not in seen_paths:
                shards.append(resolved)
                seen_paths.add(resolved)
    shards.sort()
    if not shards:
        raise ValueError("no .jsonl or .jsonl.gz source shards found")

    detected = {_detect_shard_format(shard) for shard in shards}
    detected.discard("empty")
    if not detected:
        raise ValueError("source contains no recognizable trace rows")
    if len(detected) != 1:
        raise ValueError(f"source contains mixed trace formats: {sorted(detected)}")
    resolved_format = next(iter(detected))
    if source_format != "auto" and source_format != resolved_format:
        raise ValueError(
            f"source format is {resolved_format}, not requested {source_format}"
        )
    if resolved_format == "replay_jsonl" and source_block_size is None:
        raise ValueError("--source-block-size is required for replay_jsonl input")

    stats = SourceStats(files=len(shards))
    by_key: dict[str, BaseRequest] = {}
    invalid_sessions: set[str] = set()
    unkeyed_counter = 0
    filters = selection["filters"]

    for shard in shards:
        for line_number, line in iter_trace_lines(shard):
            stats.rows += 1
            location = f"{shard.name}:{line_number}"
            if resolved_format == "dynamo_request_trace_v1":
                event = parse_trace_event(line, location)
                if event is None or event.request is None:
                    stats.non_request_rows += 1
                    continue
                source = event.request
                record = BaseRequest(
                    source_key=source.request_id,
                    source_timestamp_ms=source.received_ms,
                    input_length=source.input_length,
                    output_length=source.output_length,
                    hash_ids=source.input_sequence_hashes,
                    block_size=source.trace_block_size,
                    session_id=source.session_id,
                )
                dedupe_key = f"request:{source.request_id}"
            else:
                unkeyed_counter += 1
                record, request_id = _parse_replay_row(
                    line,
                    location,
                    block_size=source_block_size or 0,
                    ordinal=unkeyed_counter,
                )
                dedupe_key = (
                    f"request:{request_id}"
                    if request_id is not None
                    else f"row:{unkeyed_counter}"
                )

            stats.request_rows += 1
            stats.block_sizes.add(record.block_size)
            if (
                max_source_requests is not None
                and stats.request_rows > max_source_requests
            ):
                raise ValueError(
                    f"source exceeds max_source_requests={max_source_requests}; "
                    "narrow the source collection or raise the explicit request limit"
                )
            if source_block_size is not None and record.block_size != source_block_size:
                raise ValueError(
                    f"trace block size {record.block_size} does not match requested "
                    f"source_block_size {source_block_size} at {location}"
                )
            existing = by_key.get(dedupe_key)
            if existing is not None:
                if existing != record:
                    raise ValueError(
                        f"conflicting duplicate request identifier at {location}"
                    )
                stats.duplicate_requests += 1
                continue
            loaded_hashes = stats.loaded_hash_references + len(record.hash_ids)
            if max_source_hashes is not None and loaded_hashes > max_source_hashes:
                raise ValueError(
                    f"source exceeds max_source_hashes={max_source_hashes}; "
                    "narrow the source collection or raise the explicit hash limit"
                )
            stats.loaded_hash_references = loaded_hashes
            by_key[dedupe_key] = record
            if record.session_id is not None:
                stats.session_tagged_requests += 1
            if not _passes_filters(record, filters):
                stats.filtered_requests += 1
                if selection["session_policy"] == "preserve" and record.session_id:
                    invalid_sessions.add(record.session_id)

    filtered: list[BaseRequest] = []
    for record in by_key.values():
        if not _passes_filters(record, filters):
            continue
        if record.session_id is not None and record.session_id in invalid_sessions:
            stats.filtered_session_requests += 1
            continue
        filtered.append(record)
    filtered.sort(key=lambda item: (item.source_timestamp_ms, item.source_key))
    stats.complete_sessions = len(
        {record.session_id for record in filtered if record.session_id is not None}
    )

    fingerprint = hashlib.sha256()
    for record in filtered:
        fingerprint.update(
            json.dumps(
                {
                    "key": record.source_key,
                    "timestamp": record.source_timestamp_ms,
                    "input_length": record.input_length,
                    "output_length": record.output_length,
                    "hash_ids": record.hash_ids,
                    "block_size": record.block_size,
                    "session_id": record.session_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        fingerprint.update(b"\n")
    return filtered, stats, resolved_format, fingerprint.hexdigest()


def _detect_shard_format(path: Path) -> str:
    for line_number, line in iter_trace_lines(path):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"invalid JSON at {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"trace row is not an object at {path.name}:{line_number}")
        if set(value) == {"verification"}:
            return "dynamo_request_trace_v1"
        # Converted replay rows may retain producer provenance such as
        # ``schema: dynamo.request.trace.v1``.  The complete replay field set is
        # authoritative; raw request traces carry those fields under
        # event.request/replay instead of at the row root.
        if {"timestamp", "input_length", "output_length", "hash_ids"}.issubset(value):
            return "replay_jsonl"
        event = value.get("event", value)
        if isinstance(event, dict) and event.get("schema") == TRACE_SCHEMA:
            return "dynamo_request_trace_v1"
        raise ValueError(f"unrecognized trace row at {path.name}:{line_number}")
    return "empty"


def _parse_replay_row(
    line: bytes, location: str, *, block_size: int, ordinal: int
) -> tuple[BaseRequest, str | None]:
    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid JSON at {location}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"trace row is not an object at {location}")
    timestamp = value.get("timestamp")
    input_length = value.get("input_length")
    output_length = value.get("output_length")
    hashes = value.get("hash_ids")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or timestamp < 0
    ):
        raise ValueError(f"invalid timestamp at {location}")
    if (
        isinstance(input_length, bool)
        or not isinstance(input_length, int)
        or input_length < 0
    ):
        raise ValueError(f"invalid input_length at {location}")
    if (
        isinstance(output_length, bool)
        or not isinstance(output_length, int)
        or output_length < 0
    ):
        raise ValueError(f"invalid output_length at {location}")
    if not isinstance(hashes, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in hashes
    ):
        raise ValueError(f"invalid hash_ids at {location}")
    expected = math.ceil(input_length / block_size)
    if len(hashes) != expected:
        raise ValueError(
            f"input_length {input_length} with block size {block_size} requires "
            f"{expected} hashes, got {len(hashes)} at {location}"
        )
    request_id = value.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise ValueError(f"invalid request_id at {location}")
    session_id = value.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise ValueError(f"invalid session_id at {location}")
    source_key = request_id or f"row-{ordinal}"
    return (
        BaseRequest(
            source_key=source_key,
            source_timestamp_ms=timestamp,
            input_length=input_length,
            output_length=output_length,
            hash_ids=tuple(hashes),
            block_size=block_size,
            session_id=session_id,
        ),
        request_id,
    )


def _passes_filters(record: BaseRequest, filters: Mapping[str, int | None]) -> bool:
    if record.input_length < (filters["min_input_length"] or 0):
        return False
    if record.output_length < (filters["min_output_length"] or 0):
        return False
    if (
        filters["max_input_length"] is not None
        and record.input_length > filters["max_input_length"]
    ):
        return False
    if (
        filters["max_output_length"] is not None
        and record.output_length > filters["max_output_length"]
    ):
        return False
    return (
        filters["max_total_length"] is None
        or record.input_length + record.output_length <= filters["max_total_length"]
    )


def _group_requests(
    requests: Sequence[BaseRequest], *, preserve_sessions: bool
) -> tuple[_RequestGroup, ...]:
    grouped: dict[str, list[BaseRequest]] = {}
    for index, request in enumerate(requests):
        key = (
            f"session:{request.session_id}"
            if preserve_sessions and request.session_id is not None
            else f"singleton:{index}:{request.source_key}"
        )
        grouped.setdefault(key, []).append(request)
    return tuple(
        _RequestGroup(
            key=key,
            requests=tuple(
                sorted(
                    items, key=lambda item: (item.source_timestamp_ms, item.source_key)
                )
            ),
        )
        for key, items in sorted(grouped.items())
    )


def _materialize_pools(
    groups: Sequence[_RequestGroup], pool_specs: Mapping[str, Mapping[str, Any]]
) -> dict[str, frozenset[str]]:
    pools: dict[str, frozenset[str]] = {}
    for name, spec in pool_specs.items():
        if spec["type"] == "all":
            pools[name] = frozenset(group.key for group in groups)
            continue
        field_name = spec["field"]
        scored = [(_group_score(group, field_name), group.key) for group in groups]
        values = sorted(score for score, _ in scored)
        lower = _quantile(values, spec["lower_quantile"])
        upper = _quantile(values, spec["upper_quantile"])
        pools[name] = frozenset(key for score, key in scored if lower <= score <= upper)
        if not pools[name]:
            raise ValueError(f"pool {name!r} selects no source request groups")
    return pools


def _group_score(group: _RequestGroup, field_name: str) -> float:
    def value(request: BaseRequest) -> int | float:
        if field_name == "total_length":
            return request.input_length + request.output_length
        if field_name == "input_to_output_ratio":
            return request.input_length / request.output_length
        if field_name == "output_to_input_ratio":
            return request.output_length / request.input_length
        return int(getattr(request, field_name))

    return float(statistics.median(value(request) for request in group.requests))


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile over an empty source")
    position = quantile * (len(values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] * (1.0 - fraction) + values[upper_index] * fraction


def _build_workload(
    workload: Mapping[str, Any],
    *,
    groups: Sequence[_RequestGroup],
    pools: Mapping[str, frozenset[str]],
    output_dir: Path,
    block_size: int,
    reference_rps: float,
    seed: int,
    arrival_process: str,
    max_count_error_fraction: float,
) -> tuple[Path, dict[str, Any]]:
    name = workload["name"]
    schedule_rng = _derived_rng(seed, name, "schedule")
    phases = _plan_phases(
        workload,
        reference_rps=reference_rps,
        arrival_process=arrival_process,
        rng=schedule_rng,
    )
    used_pools = sorted({phase.pool for phase in phases})
    for index, left in enumerate(used_pools):
        for right in used_pools[index + 1 :]:
            if pools[left] == pools[right]:
                raise ValueError(
                    f"workload {name!r} uses pools {left!r} and {right!r}, but "
                    "they select identical request groups for this source"
                )
    by_key = {group.key: group for group in groups}
    available = set(by_key)
    selected_rows: list[tuple[int, BaseRequest, str]] = []
    phase_summaries: list[dict[str, Any]] = []

    for phase_index, phase in enumerate(phases):
        target_count = len(phase.timestamps_ms)
        # Sort before shuffling: set iteration is randomized independently of
        # the recipe seed by PYTHONHASHSEED.
        eligible = sorted(pools[phase.pool].intersection(available))
        selection_rng = _derived_rng(seed, name, f"phase-{phase_index}", "selection")
        selection_rng.shuffle(eligible)
        selected_keys, actual_count = _select_group_keys(
            eligible, by_key=by_key, target_count=target_count
        )
        error_fraction = abs(target_count - actual_count) / max(1, target_count)
        if error_fraction > max_count_error_fraction:
            available_requests = sum(len(by_key[key].requests) for key in eligible)
            raise ValueError(
                f"workload {name!r} phase {phase.name!r} needs about {target_count} "
                f"requests from pool {phase.pool!r}, selected {actual_count} from "
                f"{available_requests} available without splitting sessions; "
                "supply a larger source, lower reference_rps, or relax "
                "max_count_error_fraction"
            )
        for key in selected_keys:
            available.remove(key)
        chosen = sorted(
            (request for key in selected_keys for request in by_key[key].requests),
            key=lambda item: (item.source_timestamp_ms, item.source_key),
        )
        timestamps = _spread_subsample(phase.timestamps_ms, len(chosen))
        for timestamp, request in zip(timestamps, chosen):
            selected_rows.append((timestamp, request, phase.name))
        phase_summaries.append(
            {
                "name": phase.name,
                "pool": phase.pool,
                "start_ms": phase.start_ms,
                "end_ms": phase.end_ms,
                "target_requests": target_count,
                "actual_requests": actual_count,
                "count_error_fraction": round(error_fraction, 8),
                "schedule": dict(phase.schedule),
            }
        )

    selected_rows.sort(
        key=lambda item: (item[0], item[1].source_timestamp_ms, item[1].source_key)
    )
    hash_map: dict[int, int] = {}
    phase_values: dict[str, list[tuple[int, int]]] = {}
    output_hash = hashlib.sha256()
    request_count = 0
    last_timestamp_ms = 0
    path = output_dir / f"{name}.jsonl"
    with path.open("wb") as handle:
        for timestamp, request, phase_name in selected_rows:
            remapped_hashes = []
            for source_hash in request.hash_ids:
                if source_hash not in hash_map:
                    hash_map[source_hash] = len(hash_map)
                remapped_hashes.append(hash_map[source_hash])
            record = {
                "timestamp": timestamp,
                "input_length": request.input_length,
                "output_length": request.output_length,
                "hash_ids": remapped_hashes,
            }
            line = (json.dumps(record, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            output_hash.update(line)
            request_count += 1
            last_timestamp_ms = timestamp
            phase_values.setdefault(phase_name, []).append(
                (request.input_length, request.output_length)
            )

    validate_mooncake_trace(path, block_size=block_size, presorted=True)
    for phase in phase_summaries:
        values = phase_values.get(phase["name"], [])
        phase["median_input_length"] = (
            statistics.median(value[0] for value in values) if values else None
        )
        phase["median_output_length"] = (
            statistics.median(value[1] for value in values) if values else None
        )
        phase["median_input_to_output_ratio"] = (
            round(statistics.median(value[0] / value[1] for value in values), 8)
            if values
            else None
        )
    return path, {
        "name": name,
        "description": workload["description"],
        "file": path.name,
        "sha256": output_hash.hexdigest(),
        "request_count": request_count,
        "duration_ms": max((phase.end_ms for phase in phases), default=0),
        "last_timestamp_ms": last_timestamp_ms,
        "distinct_compacted_hashes": len(hash_map),
        "phases": phase_summaries,
    }


def _select_group_keys(
    eligible: Sequence[str],
    *,
    by_key: Mapping[str, _RequestGroup],
    target_count: int,
) -> tuple[list[str], int]:
    """Choose a maximum-cardinality atomic subset no larger than the target.

    This is a bounded subset-sum solved with a Python integer bitset.  Across
    all candidates, at most ``target_count`` predecessor entries are retained,
    while the caller-provided (seeded) candidate order determines which exact
    solution wins.  It avoids a greedy failure such as choosing an 8-turn
    session for a target of 10 when 6+4 is feasible.
    """

    if target_count <= 0:
        return [], 0
    reachable = 1  # Bit N means a subset totaling N requests is reachable.
    mask = (1 << (target_count + 1)) - 1
    parents: dict[int, tuple[int, str]] = {}
    for key in eligible:
        size = len(by_key[key].requests)
        if size > target_count:
            continue
        shifted = (reachable << size) & mask
        new_sums = shifted & ~reachable
        bits = new_sums
        while bits:
            lowest = bits & -bits
            total = lowest.bit_length() - 1
            parents[total] = (total - size, key)
            bits ^= lowest
        reachable |= shifted
        if reachable & (1 << target_count):
            break

    actual_count = min(target_count, reachable.bit_length() - 1)
    selected: list[str] = []
    cursor = actual_count
    while cursor:
        previous, key = parents[cursor]
        selected.append(key)
        cursor = previous
    selected.reverse()
    return selected, actual_count


def _plan_phases(
    workload: Mapping[str, Any],
    *,
    reference_rps: float,
    arrival_process: str,
    rng: random.Random,
) -> tuple[_PhasePlan, ...]:
    if workload["kind"] == "bursts":
        duration = workload["duration_s"]
        intervals = _place_bursts(duration, workload["burst_overlay"], rng)

        def rate_factor(t: float) -> float:
            for start, end, factor in intervals:
                if start <= t < end:
                    return factor
            return workload["baseline_load_factor"]

        max_factor = max(
            [workload["baseline_load_factor"]] + [factor for _, _, factor in intervals]
        )
        relative = _sample_rate_function(
            duration,
            rate_factor,
            max_rate=reference_rps * max_factor,
            reference_rps=reference_rps,
            arrival_process=arrival_process,
            rng=rng,
        )
        return (
            _PhasePlan(
                name="bursts",
                pool=workload["pool"],
                start_ms=0,
                end_ms=int(duration * 1000),
                timestamps_ms=tuple(relative),
                schedule={
                    "kind": "random_bursts",
                    "baseline_load_factor": workload["baseline_load_factor"],
                    "bursts": [
                        {
                            "start_ms": int(start * 1000),
                            "end_ms": int(end * 1000),
                            "load_factor": round(factor, 8),
                        }
                        for start, end, factor in intervals
                    ],
                },
            ),
        )

    plans: list[_PhasePlan] = []
    offset_s = 0.0
    for repeat_index in range(workload["repeat"]):
        for phase_index, phase in enumerate(workload["phases"]):
            duration = phase["duration_s"]
            load = phase["load_factor"]
            start_factor, end_factor = (
                (load[0], load[1]) if isinstance(load, list) else (load, load)
            )

            def rate_factor(
                t: float, start=start_factor, end=end_factor, span=duration
            ) -> float:
                if start == end:
                    return start
                return start + (end - start) * (t / span)

            relative = _sample_rate_function(
                duration,
                rate_factor,
                max_rate=reference_rps * max(start_factor, end_factor),
                reference_rps=reference_rps,
                arrival_process=arrival_process,
                rng=rng,
            )
            start_ms = round(offset_s * 1000)
            timestamps = tuple(start_ms + timestamp for timestamp in relative)
            plans.append(
                _PhasePlan(
                    name=f"r{repeat_index + 1}-p{phase_index + 1}",
                    pool=phase["pool"],
                    start_ms=start_ms,
                    end_ms=round((offset_s + duration) * 1000),
                    timestamps_ms=timestamps,
                    schedule={
                        "kind": "linear" if start_factor != end_factor else "constant",
                        "load_factor_start": start_factor,
                        "load_factor_end": end_factor,
                    },
                )
            )
            offset_s += duration
    return tuple(plans)


def _sample_rate_function(
    duration_s: float,
    rate_factor,
    *,
    max_rate: float,
    reference_rps: float,
    arrival_process: str,
    rng: random.Random,
) -> list[int]:
    if max_rate <= 0:
        return []
    if arrival_process == "poisson":
        out: list[int] = []
        t = 0.0
        while True:
            t += rng.expovariate(max_rate)
            if t >= duration_s:
                break
            rate = max(0.0, reference_rps * rate_factor(t))
            if rng.random() <= rate / max_rate:
                out.append(min(int(t * 1000), max(0, int(duration_s * 1000) - 1)))
        return out

    # Deterministic arrivals use numerical integration over small bins, then
    # invert the cumulative intensity.  This is deterministic for arbitrary
    # burst overlays as well as linear phase ramps.
    bins = max(1, math.ceil(duration_s * 10))
    step = duration_s / bins
    cumulative = [0.0]
    for index in range(bins):
        midpoint = (index + 0.5) * step
        cumulative.append(
            cumulative[-1] + max(0.0, reference_rps * rate_factor(midpoint)) * step
        )
    count = round(cumulative[-1])
    out = []
    cursor = 0
    for index in range(count):
        target = (index + 0.5) * cumulative[-1] / count
        while cursor + 1 < len(cumulative) and cumulative[cursor + 1] < target:
            cursor += 1
        span = cumulative[cursor + 1] - cumulative[cursor]
        fraction = 0.0 if span == 0 else (target - cumulative[cursor]) / span
        t = (cursor + fraction) * step
        out.append(min(int(t * 1000), max(0, int(duration_s * 1000) - 1)))
    return out


def _place_bursts(
    duration_s: float, spec: Mapping[str, Any], rng: random.Random
) -> list[tuple[float, float, float]]:
    count = spec["count"]
    widths = [rng.uniform(*spec["width_s"]) for _ in range(count)]
    gap = spec["minimum_gap_s"]
    required = sum(widths) + max(0, count - 1) * gap
    if required > duration_s:
        raise ValueError(
            "sampled burst widths do not fit; reduce width_s/count/minimum_gap_s"
        )
    slack = duration_s - required
    weights = [rng.expovariate(1.0) for _ in range(count + 1)]
    total_weight = sum(weights)
    gaps = [slack * weight / total_weight for weight in weights]
    intervals = []
    cursor = gaps[0]
    for index, width in enumerate(widths):
        factor = rng.uniform(*spec["load_factor"])
        intervals.append((cursor, cursor + width, factor))
        cursor += width
        if index + 1 < count:
            cursor += gap + gaps[index + 1]
    return intervals


def _spread_subsample(values: Sequence[int], count: int) -> tuple[int, ...]:
    if count == 0:
        return ()
    if count > len(values):
        raise ValueError("cannot assign more requests than target timestamps")
    if count == len(values):
        return tuple(values)
    return tuple(
        values[min(len(values) - 1, math.floor((index + 0.5) * len(values) / count))]
        for index in range(count)
    )


def _derived_rng(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256((str(seed) + "\0" + "\0".join(parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _load_factor(value: Any, path: str) -> float | list[float]:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"{path}: expected one number or a two-value range")
        return [
            _number(item, f"{path}[{index}]", minimum=0.0)
            for index, item in enumerate(value)
        ]
    return _number(value, path, minimum=0.0)


def _range(value: Any, path: str, *, minimum: float) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path}: expected a two-value range")
    out = [
        _number(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    ]
    if out[0] > out[1]:
        raise ValueError(f"{path}: lower value exceeds upper value")
    return out


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return dict(value)


def _only_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"{path}: unexpected keys: {unexpected}")


def _validate_name(value: Any, path: str) -> None:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(f"{path}: expected a lowercase kebab-case name")


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path}: expected a finite number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{path}: must be at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{path}: must be at most {maximum}")
    return number


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    if value < minimum:
        raise ValueError(f"{path}: must be at least {minimum}")
    return value


def _optional_integer(value: Any, path: str, *, minimum: int) -> int | None:
    if value is None:
        return None
    return _integer(value, path, minimum=minimum)
