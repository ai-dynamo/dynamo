# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute and publish fully-expanded :mod:`match_config` matrices."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from autoscaling_arena import __version__
from autoscaling_arena.match_config import (
    EngineConfig,
    MatchConfig,
    MatchRun,
    RealAutoscalerConfig,
    RealBackendConfig,
    SimAutoscalerConfig,
    SimBackendConfig,
    SLAProfileConfig,
)
from autoscaling_arena.scorecard import SLOProfile
from autoscaling_arena.workloads import Workload, get_workload

ProgressCallback = Callable[[str, MatchRun, Optional[dict[str, Any]]], None]

_LOWER_IS_BETTER = {
    "gpu_hours",
    "duration_s",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "p99_itl_ms",
    "mean_e2e_ms",
    "p99_e2e_ms",
    "p95_e2e_latency_ms",
    "oscillation_count",
    "scale_events",
}
_SENSITIVE_ARGS = {
    "--access-token",
    "--api-key",
    "--apikey",
    "--authorization",
    "--auth-token",
    "--aws-secret-access-key",
    "--bearer-token",
    "--client-secret",
    "--password",
    "--refresh-token",
    "--secret-key",
    "--session-token",
    "--token",
}
_HEADER_ARGS = {"--header", "-h"}
_SENSITIVE_CONFIG_KEYS = {
    "access_token",
    "aeg_sas_key",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "bearer_token",
    "client_secret",
    "connection_string",
    "credential",
    "credentials",
    "ocp_apim_subscription_key",
    "password",
    "private_key",
    "proxy_authorization",
    "refresh_token",
    "sas_token",
    "secret",
    "secret_access_key",
    "secret_key",
    "session_token",
    "token",
    "trace_path",
    "trace_paths",
    "x_amz_security_token",
    "x_api_key",
    "x_functions_key",
    "x_goog_api_key",
}


class MatchPublishError(RuntimeError):
    """A configured result destination could not be published safely."""


@dataclass
class _ExecutionContext:
    session_id: str
    session_root: Path
    external_trace_root: Optional[Path] = None
    trace_paths: dict[tuple[Any, ...], Path] = field(default_factory=dict)
    trace_fingerprints: dict[Path, dict[str, Any]] = field(default_factory=dict)
    arrival_series: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)


def execute_match_config(
    config: MatchConfig,
    *,
    on_progress: Optional[ProgressCallback] = None,
    run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Execute selected expanded matrix cells and return a normalized report.

    Failures are recorded per cell.  ``execution.fail_fast`` stops scheduling
    additional cells after the first failed result, but still returns a partial
    report that can be published.  ``run_ids`` is intended for reproducing one
    or more exact matrix cells from a saved report; ``None`` runs the full
    matrix.
    """

    matrix = list(config.iter_runs())
    if run_ids is not None:
        available = {item.run_id for item in matrix}
        unknown = sorted(run_ids - available)
        if unknown:
            raise ValueError("unknown Match Config run id(s): " + ", ".join(unknown))
        matrix = [item for item in matrix if item.run_id in run_ids]
        if not matrix:
            raise ValueError("run_ids must select at least one matrix cell")

    started_wall = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    config_hash = match_config_sha256(config)
    session_prefix = started_wall.strftime("%Y%m%dT%H%M%S%fZ") + "-" + config_hash[:10]
    try:
        config.publish.artifact_root.mkdir(parents=True, exist_ok=True)
        session_root = Path(
            tempfile.mkdtemp(
                prefix=session_prefix + "-",
                dir=config.publish.artifact_root,
            )
        )
    except OSError as exc:
        raise MatchPublishError(
            f"cannot create artifact session under "
            f"{config.publish.artifact_root}: {exc}"
        ) from exc
    session_id = session_root.name
    with tempfile.TemporaryDirectory(
        prefix="autoscaling-arena-external-traces-"
    ) as external_trace_root:
        context = _ExecutionContext(
            session_id=session_id,
            session_root=session_root,
            external_trace_root=Path(external_trace_root).resolve(),
        )
        return _execute_match_matrix(
            config,
            matrix=matrix,
            context=context,
            config_hash=config_hash,
            started_wall=started_wall,
            started_monotonic=started_monotonic,
            on_progress=on_progress,
        )


def _execute_match_matrix(
    config: MatchConfig,
    *,
    matrix: list[MatchRun],
    context: _ExecutionContext,
    config_hash: str,
    started_wall: datetime,
    started_monotonic: float,
    on_progress: Optional[ProgressCallback],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in matrix:
        if on_progress:
            on_progress("started", item, None)
        try:
            if config.backend.type == "sim":
                result = _run_sim_item(config, item, context)
            else:
                result = _run_real_item(config, item, context)
        except Exception as exc:
            result = _failure_result(
                item,
                artifact_dir=context.session_root / "runs" / item.run_id,
                error_type=type(exc).__name__,
                message=_redact_exception_message(config, str(exc)),
            )
        results.append(_sanitize_json(result))
        if on_progress:
            on_progress("finished", item, result)
        if result["status"] != "ok" and config.execution.fail_fast:
            break

    finished_wall = datetime.now(timezone.utc)
    failed = sum(result["status"] != "ok" for result in results)
    succeeded = len(results) - failed
    skipped = len(matrix) - len(results)
    report = {
        "match": {
            "name": config.name,
            "description": config.description,
            "labels": dict(config.labels),
            "backend": config.backend.type,
        },
        "summary": {
            "status": "ok" if failed == 0 and skipped == 0 else "failed",
            "planned_runs": len(matrix),
            "executed_runs": len(results),
            "succeeded_runs": succeeded,
            "failed_runs": failed,
            "skipped_runs": skipped,
            "rank_by": config.metrics.rank_by,
            "metrics": list(config.metrics.include),
        },
        "provenance": {
            "schema_version": config.schema_version,
            "config_path": str(config.source_path),
            "config_file": config.source_path.name,
            "config_sha256": config_hash,
            "replay_config_sha256": match_replay_sha256(config),
            "replay": {
                "kind": "match_config",
                "config_path": _safe_replay_config_path(config),
            },
            "arena_version": __version__,
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "started_at": started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "duration_s": time.monotonic() - started_monotonic,
            "session_id": context.session_id,
            "artifact_root": str(context.session_root),
        },
        "resolved_config": _redact_config(config.to_dict()),
        "matrix": [_matrix_dict(item) for item in matrix],
        "results": results,
    }
    return _sanitize_json(_redact_report_paths(report, config, context))


def publish_match_results(
    config: MatchConfig,
    report: dict[str, Any],
    *,
    force_overwrite: bool = False,
    console: Callable[[str], None] = print,
) -> list[Path]:
    """Publish a normalized report to its configured destinations.

    JSON and standalone HTML writes are staged, committed atomically, and
    refuse to replace existing files unless the destination or caller opts in.
    """

    preflight_publish_destinations(config, force_overwrite=force_overwrite)
    file_destinations = [
        destination
        for destination in config.publish.destinations
        if destination.type in {"json", "html"}
    ]
    payloads: dict[str, str] = {}
    if any(destination.type == "json" for destination in file_destinations):
        try:
            payloads["json"] = (
                json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n"
            )
        except (TypeError, ValueError) as exc:
            raise MatchPublishError(
                f"results are not JSON-serializable: {exc}"
            ) from exc
    if any(destination.type == "html" for destination in file_destinations):
        try:
            from autoscaling_arena.html_report import render_match_report

            payloads["html"] = render_match_report(report)
        except (TypeError, ValueError, OSError) as exc:
            raise MatchPublishError(
                f"cannot render standalone HTML report: {exc}"
            ) from exc

    staged: list[dict[str, Any]] = []
    try:
        for destination in file_destinations:
            assert destination.path is not None
            path = destination.path
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(fd, "w") as handle:
                    handle.write(payloads[destination.type])
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
                raise
            staged.append(
                {
                    "path": path,
                    "temporary": temporary_path,
                    "type": destination.type,
                    "overwrite": force_overwrite or destination.overwrite,
                    "backup": None,
                    "created": False,
                }
            )
    except BaseException as exc:
        for item in staged:
            try:
                item["temporary"].unlink()
            except FileNotFoundError:
                pass
        if isinstance(exc, OSError):
            raise MatchPublishError(f"cannot stage result files: {exc}") from exc
        raise

    committed: list[dict[str, Any]] = []
    try:
        for item in staged:
            path = item["path"]
            temporary_path = item["temporary"]
            if path.is_symlink():
                raise MatchPublishError(
                    f"{item['type'].upper()} destination became a symbolic "
                    f"link: {path}"
                )
            if item["overwrite"]:
                existed = path.exists()
                if existed:
                    backup_fd, backup_name = tempfile.mkstemp(
                        prefix=f".{path.name}.",
                        suffix=".backup",
                        dir=path.parent,
                    )
                    os.close(backup_fd)
                    backup_path = Path(backup_name)
                    backup_path.unlink()
                    os.link(path, backup_path)
                    item["backup"] = backup_path
                os.replace(temporary_path, path)
                item["created"] = not existed
            else:
                # Linking a fully-written same-filesystem temp file is an
                # atomic no-clobber commit.  Unlike check-then-replace, two
                # concurrent publishers cannot both win.
                os.link(temporary_path, path)
                item["created"] = True
                committed.append(item)
                temporary_path.unlink()
                continue
            committed.append(item)
    except BaseException as exc:
        # Roll back every sink already committed by this call.  Existing files
        # have same-filesystem hard-link backups; newly created files are
        # removed.  This makes a multi-destination publish all-or-nothing for
        # ordinary process-level failures.
        for item in reversed(committed):
            path = item["path"]
            backup = item["backup"]
            try:
                if backup is not None:
                    os.replace(backup, path)
                    item["backup"] = None
                elif item["created"]:
                    path.unlink()
            except OSError:
                pass
        for item in staged:
            for key in ("temporary", "backup"):
                temporary = item.get(key)
                if temporary is not None:
                    try:
                        temporary.unlink()
                    except FileNotFoundError:
                        pass
        if isinstance(exc, FileExistsError):
            raise MatchPublishError(
                "a result destination was created concurrently"
            ) from exc
        if isinstance(exc, OSError):
            raise MatchPublishError(f"cannot commit result files: {exc}") from exc
        raise

    written = [item["path"] for item in staged]
    for item in staged:
        backup = item["backup"]
        if backup is not None:
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
        _fsync_directory(item["path"].parent)
    # Console output comes last so a predictable filesystem error cannot leave
    # users with a success-looking table but no configured result artifact.
    for destination in config.publish.destinations:
        if destination.type == "console":
            console(format_match_results(report))
    return written


def preflight_publish_destinations(
    config: MatchConfig, *, force_overwrite: bool = False
) -> None:
    """Check predictable publication failures before an expensive sweep."""

    protected = {config.source_path.resolve(strict=False)}
    protected.update(
        evaluation.trace_path.resolve(strict=False)
        for evaluation in config.evaluations
        if evaluation.trace_path is not None
    )
    protected.update(
        path.resolve(strict=False)
        for evaluation in config.evaluations
        for path in evaluation.trace_paths
    )
    if isinstance(config.backend, RealBackendConfig):
        if config.backend.endpoint_catalog is not None:
            protected.add(config.backend.endpoint_catalog.resolve(strict=False))
    artifact_root = config.publish.artifact_root
    if artifact_root.exists() and not artifact_root.is_dir():
        raise MatchPublishError(
            f"artifact root exists but is not a directory: {artifact_root}"
        )
    _require_directory_ancestor(artifact_root)
    for destination in config.publish.destinations:
        if destination.type not in {"json", "html"}:
            continue
        assert destination.path is not None
        path = destination.path
        if path.resolve(strict=False) in protected:
            raise MatchPublishError(
                f"refusing to overwrite input configuration file: {path}"
            )
        if path.is_symlink():
            raise MatchPublishError(
                f"{destination.type.upper()} destination must not be a "
                f"symbolic link: {path}"
            )
        if path.exists() and path.is_dir():
            raise MatchPublishError(
                f"{destination.type.upper()} destination is a directory: {path}"
            )
        overwrite = force_overwrite or destination.overwrite
        if path.exists() and not overwrite:
            raise MatchPublishError(
                f"{destination.type.upper()} destination already exists: "
                f"{path}; pass --overwrite or set destination.overwrite: true"
            )
        _require_directory_ancestor(path.parent)


def format_match_results(report: dict[str, Any]) -> str:
    """Render a compact, backend-neutral result table."""

    match = report["match"]
    summary = report["summary"]
    lines = [
        "",
        f"Match: {match['name']} ({match['backend']})",
        (
            f"Runs: {summary['succeeded_runs']} succeeded, "
            f"{summary['failed_runs']} failed, {summary['skipped_runs']} skipped "
            f"of {summary['planned_runs']} planned"
        ),
        f"Rank metric: {summary['rank_by']}",
        "",
    ]
    rank_by = summary["rank_by"]
    metrics = summary["metrics"]
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for result in report["results"]:
        key = (result["workload"], result["sla"], result["repetition"])
        groups.setdefault(key, []).append(result)
    for (workload, sla, repetition), group in groups.items():
        lines.append(f"  {workload} | {sla} | repetition {repetition}")
        successful = [result for result in group if result["status"] == "ok"]
        failed = [result for result in group if result["status"] != "ok"]
        reverse = rank_by not in _LOWER_IS_BETTER
        successful.sort(
            key=lambda result: _ranking_value(
                result["metrics"].get(rank_by), reverse=reverse
            ),
            reverse=reverse,
        )
        for result in successful:
            columns = "  ".join(
                f"{metric}={_format_metric(result['metrics'].get(metric))}"
                for metric in metrics
            )
            lines.append(f"    {result['autoscaler']:<20} {columns}")
        for result in failed:
            error = result.get("error", {})
            lines.append(
                f"    {result['autoscaler']:<20} FAILED "
                f"{error.get('type', 'Error')}: {error.get('message', '')}"
            )
    return "\n".join(lines)


def format_match_matrix(config: MatchConfig) -> str:
    """Render the expanded plan without importing either runtime backend."""

    lines = [
        f"Match {config.name!r}: {config.expected_runs} runs "
        f"({config.backend.type})"
    ]
    for item in config.iter_runs():
        lines.append(
            f"  {item.run_id}: autoscaler={item.autoscaler} "
            f"workload={item.workload} sla={item.sla} "
            f"repetition={item.repetition} seed={item.seed}"
        )
    return "\n".join(lines)


def _run_sim_item(
    config: MatchConfig, item: MatchRun, context: _ExecutionContext
) -> dict[str, Any]:
    # Deliberately lazy: validation and real runs must not import Dynamo.
    from autoscaling_arena.runners.sims import run_arena_replay
    from autoscaling_arena.scorecard import scorecard

    backend = config.backend
    assert isinstance(backend, SimBackendConfig)
    autoscaler = _find_sim_autoscaler(backend, item.autoscaler)
    workload = _workload_for_item(item)
    run_dir = context.session_root / "runs" / item.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    if item.trace_format == "dynamo":
        # Native shards are passed straight to Dynamo, in the declared order.
        # Match Config must not sort, merge, cap, or translate exact traces.
        trace_path = None
        trace_paths = tuple(path.resolve() for path in item.trace_paths)
        trace_block_size = item.trace_block_size
        replay_trace_args: dict[str, Any] = {
            "trace_files": [str(path) for path in trace_paths],
            "trace_format": "dynamo",
        }
    else:
        trace_path = _materialize_trace(
            context,
            **_external_trace_options(item),
            workload_name=item.workload,
            seed=item.seed,
            max_requests=item.max_requests,
            arrival_speedup=1.0,
            speedup_is_materialized=False,
        )
        trace_paths = (trace_path,)
        trace_block_size = workload.block_size
        replay_trace_args = {"trace_file": str(trace_path)}

    profile = _to_slo_profile(item.slo_profile)
    planner_config = _sim_planner_config(
        backend, report_filename=run_dir / "planner.html"
    )
    factory_options = {}
    if autoscaler.type == "jev":
        factory_options = {
            "decision_log": run_dir / "jev-decisions.jsonl",
            "jev_context": {
                "slo": asdict(profile),
                "gpu_budget": backend.gpu_budget,
                "engines": {
                    role: {
                        "gpus_per_worker": engine.num_gpus,
                        "cold_start_delay_s": engine.runtime.cold_start_delay_s,
                    }
                    for role in ("prefill", "decode", "aggregate")
                    if (engine := getattr(backend.engines, role)) is not None
                },
            },
        }
    factory = _build_sim_factory(
        autoscaler, topology=backend.topology, **factory_options
    )
    performance_model_metadata = _sim_performance_model_metadata(backend)
    replay_args: dict[str, Any] = {
        **replay_trace_args,
        "autoscaler": factory,
        "substrate_config": planner_config,
        "arrival_speedup_ratio": item.arrival_speedup,
        "trace_block_size": trace_block_size,
        "router_mode": backend.router.mode,
        "replay_concurrency": backend.replay.concurrency,
        "model_name": backend.model.name,
        "sla_ttft_ms": profile.ttft_ms,
        "sla_itl_ms": profile.itl_ms,
        "sla_e2e_ms": profile.e2e_ms,
        "capture_per_request": True,
        "telemetry_sample_interval_s": (backend.replay.telemetry_sample_interval_s),
        "telemetry_jsonl_path": (
            str(run_dir / "telemetry.jsonl")
            if backend.replay.telemetry_sample_interval_s is not None
            else None
        ),
        "ais_bootstrap": backend.replay.ais_bootstrap,
        "performance_model_metadata": performance_model_metadata,
        "report_json": str(run_dir / "trace-report.json"),
    }
    if backend.topology == "disagg":
        assert backend.engines.prefill is not None
        assert backend.engines.decode is not None
        replay_args.update(
            prefill_engine_args=_render_engine_args(
                backend.engines.prefill, backend.model.ais_model_path
            ),
            decode_engine_args=_render_engine_args(
                backend.engines.decode, backend.model.ais_model_path
            ),
            num_prefill_workers=autoscaler.start.prefill,
            num_decode_workers=autoscaler.start.decode,
        )
    else:
        assert backend.engines.aggregate is not None
        replay_args.update(
            extra_engine_args=_render_engine_args(
                backend.engines.aggregate, backend.model.ais_model_path
            ),
            num_workers=autoscaler.start.decode,
        )
    report = run_arena_replay(**replay_args)
    telemetry_artifact = getattr(report, "telemetry_artifact", None) or {}
    raw_scorecard = scorecard(report, profiles=(profile,), sla_profile=profile)
    metrics = _project_sim_metrics(
        raw_scorecard, profile_name=profile.name, include=config.metrics.include
    )
    rank_error = _rank_metric_error(metrics, config.metrics.rank_by)
    result = {
        **_result_identity(item),
        "status": "failed" if rank_error else "ok",
        "autoscaler_type": autoscaler.type,
        "metrics": metrics,
        "raw_metrics": raw_scorecard,
        "timeline": list(getattr(report, "timeline", [])),
        "timeline_semantics": "arriving_and_completed_v2",
        "cache": _prefix_cache_telemetry(report),
        "evaluation": _evaluation_metadata(
            item,
            block_size=trace_block_size,
            trace_path=trace_path,
            trace_paths=trace_paths,
            context=context,
        ),
        "runtime": {
            "substrate_preset": backend.substrate,
            "model": _redact_mapping_secrets(asdict(backend.model)),
            "engines": _sim_engines_metadata(backend),
            "topology": backend.topology,
            "router_mode": backend.router.mode,
            "replay": {
                "telemetry_sample_interval_s": (
                    backend.replay.telemetry_sample_interval_s
                ),
                "telemetry_contract": (
                    "dynamo.replay.telemetry.v1"
                    if backend.replay.telemetry_sample_interval_s is not None
                    else None
                ),
                "telemetry_sample_count": telemetry_artifact.get("sample_count"),
                "telemetry_sha256": telemetry_artifact.get("sha256"),
            },
            "planner_config": _redact_mapping_secrets(planner_config),
            "initial_replicas": asdict(autoscaler.start),
            "autoscaler_config": _redact_mapping_secrets(dict(autoscaler.config)),
        },
        "artifacts": {
            "directory": str(run_dir),
            "trace_report": str(run_dir / "trace-report.json"),
            **(
                {
                    "telemetry": str(run_dir / "telemetry.jsonl"),
                    "telemetry_metadata": {
                        "contract": telemetry_artifact.get(
                            "contract", "dynamo.replay.telemetry.v1"
                        ),
                        "sample_count": telemetry_artifact.get("sample_count"),
                        "sha256": telemetry_artifact.get("sha256"),
                    },
                }
                if backend.replay.telemetry_sample_interval_s is not None
                else {}
            ),
        },
    }
    if rank_error:
        result["error"] = {
            "type": "RankMetricUnavailable",
            "message": rank_error,
        }
    if autoscaler.type == "jev":
        from autoscaling_arena.jev_report import summarize_decisions

        result["artifacts"]["jev_decisions"] = str(run_dir / "jev-decisions.jsonl")
        result["runtime"]["jev"] = summarize_decisions(run_dir / "jev-decisions.jsonl")
    return result


def _run_real_item(
    config: MatchConfig, item: MatchRun, context: _ExecutionContext
) -> dict[str, Any]:
    from autoscaling_arena.runners.real import Endpoint, run_endpoint_match

    backend = config.backend
    assert isinstance(backend, RealBackendConfig)
    autoscaler = _find_real_autoscaler(backend, item.autoscaler)
    endpoint_config = next(
        endpoint
        for endpoint in backend.endpoints
        if endpoint.name == autoscaler.endpoint
    )
    workload = _workload_for_item(item)
    run_dir = context.session_root / "runs" / item.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # AIPerf follows trace timestamps in wall time, so online speedup is baked
    # into the materialized trace.  The sim backend instead passes it to Rust.
    trace_path = _materialize_trace(
        context,
        **_external_trace_options(item),
        workload_name=item.workload,
        seed=item.seed,
        max_requests=item.max_requests,
        arrival_speedup=item.arrival_speedup,
        speedup_is_materialized=True,
    )
    profile = _to_slo_profile(item.slo_profile)
    endpoint = Endpoint(
        name=autoscaler.name,
        url=endpoint_config.url,
        model=endpoint_config.model,
        description=endpoint_config.description,
        endpoint_type=endpoint_config.endpoint_type,
    )
    match_result = run_endpoint_match(
        endpoint,
        trace_path,
        profile,
        artifact_root=run_dir,
        block_size=workload.block_size,
        streaming=backend.aiperf.streaming,
        tokenizer=backend.aiperf.tokenizer,
        aiperf_bin=backend.aiperf.executable,
        timeout_s=backend.aiperf.timeout_s,
        extra_args=backend.aiperf.extra_args,
        workload_name=item.workload,
    )
    raw_metrics = dict(match_result.metrics)
    metrics = _project_real_metrics(raw_metrics, config.metrics.include)
    parse_error = raw_metrics.get("_parse_error")
    missing_summary = not raw_metrics
    rank_error = _rank_metric_error(metrics, config.metrics.rank_by)
    ok = (
        match_result.returncode == 0
        and not parse_error
        and not missing_summary
        and not rank_error
    )
    result: dict[str, Any] = {
        **_result_identity(item),
        "status": "ok" if ok else "failed",
        "autoscaler_type": autoscaler.autoscaler_type,
        "endpoint": autoscaler.endpoint,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "evaluation": _evaluation_metadata(
            item, block_size=workload.block_size, trace_path=trace_path, context=context
        ),
        "runtime": {
            "endpoint": {
                "name": endpoint_config.name,
                "url": _redact_url(endpoint_config.url),
                "model": endpoint_config.model,
                "endpoint_type": endpoint_config.endpoint_type,
                "declared_deployment": _redact_mapping_secrets(
                    endpoint_config.declared_deployment
                ),
            },
            "declared_config": _redact_mapping_secrets(
                dict(autoscaler.declared_config)
            ),
            "returncode": match_result.returncode,
            "stderr_tail": _redact_aiperf_text(
                match_result.stderr_tail, backend.aiperf.extra_args
            ),
        },
        "artifacts": {
            "directory": match_result.artifact_dir,
        },
    }
    if not ok:
        if match_result.returncode != 0:
            message = f"AIPerf exited with return code {match_result.returncode}"
            error_type = "AIPerfProcessError"
        elif parse_error:
            message = str(parse_error)
            error_type = "AIPerfSummaryParseError"
        elif missing_summary:
            message = "AIPerf produced no summary metrics"
            error_type = "AIPerfSummaryMissing"
        else:
            message = rank_error or "configured rank metric is unavailable"
            error_type = "RankMetricUnavailable"
        result["error"] = {"type": error_type, "message": message}
    return result


def _materialize_trace(
    context: _ExecutionContext,
    *,
    workload_name: str,
    seed: int,
    max_requests: Optional[int],
    arrival_speedup: float,
    speedup_is_materialized: bool,
    trace_path: Optional[Path] = None,
    trace_block_size: Optional[int] = None,
    trace_presorted: bool = False,
) -> Path:
    key = (
        workload_name,
        str(trace_path) if trace_path is not None else None,
        trace_block_size,
        trace_presorted,
        seed if trace_path is None else None,
        max_requests,
        arrival_speedup if speedup_is_materialized else 1.0,
    )
    existing = context.trace_paths.get(key)
    if existing is not None:
        return existing
    key_hash = hashlib.sha256(
        json.dumps(key, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    trace_root = context.session_root / "traces"
    if trace_path is not None and context.external_trace_root is not None:
        trace_root = context.external_trace_root
    trace_dir = trace_root / key_hash
    workload = (
        get_workload(workload_name)
        if trace_path is None
        else Workload(
            name="recorded-trace",
            description="User-supplied recorded trace",
            block_size=trace_block_size or 512,
            static_trace=trace_path,
            presorted=trace_presorted,
        )
    )
    path = workload.materialize(
        trace_dir,
        seed=seed,
        max_requests=max_requests,
        arrival_speedup=arrival_speedup if speedup_is_materialized else 1.0,
        force_copy_static=trace_path is not None,
    )
    context.trace_paths[key] = path.resolve()
    return path.resolve()


def _workload_for_item(item: MatchRun) -> Workload:
    if item.trace_format == "dynamo":
        # This wrapper supplies descriptive metadata only. Native traces bypass
        # Workload.materialize() and retain an omitted block size as ``None``
        # when passed to the replay runtime.
        return Workload(
            name=item.workload,
            description="Exact user-supplied native Dynamo trace",
            block_size=item.trace_block_size or 512,
            static_trace=item.trace_paths[0],
            presorted=True,
        )
    if item.trace_path is None:
        return get_workload(item.workload)
    return Workload(
        name=item.workload,
        description="User-supplied recorded trace",
        block_size=item.trace_block_size or 512,
        static_trace=item.trace_path,
        presorted=item.trace_presorted,
    )


def _external_trace_options(item: MatchRun) -> dict[str, Any]:
    if item.trace_path is None:
        return {}
    return {
        "trace_path": item.trace_path,
        "trace_block_size": item.trace_block_size,
        "trace_presorted": item.trace_presorted,
    }


def _trace_fingerprint(path: Path, context: _ExecutionContext) -> dict[str, Any]:
    cached = context.trace_fingerprints.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    stat = path.stat()
    fingerprint = {"sha256": digest.hexdigest(), "size_bytes": stat.st_size}
    context.trace_fingerprints[path] = fingerprint
    return fingerprint


def _evaluation_metadata(
    item: MatchRun,
    *,
    block_size: Optional[int],
    trace_path: Optional[Path],
    trace_paths: tuple[Path, ...] = (),
    context: _ExecutionContext,
) -> dict[str, Any]:
    from autoscaling_arena.html_report import trace_arrival_series

    if item.trace_format == "dynamo":
        resolved_paths = tuple(path.resolve() for path in trace_paths)
        shards = [_trace_fingerprint(path, context) for path in resolved_paths]
        trace_metadata: dict[str, Any] = {
            "format": "dynamo",
            "shard_count": len(shards),
            "shards": shards,
            "size_bytes": sum(item["size_bytes"] for item in shards),
            "staged": False,
            "transformed": False,
        }
        arrival_inputs: Path | tuple[Path, ...] = resolved_paths
    else:
        if trace_path is None:
            raise ValueError("Mooncake evaluation metadata requires trace_path")
        resolved_trace = trace_path.resolve()
        materialized = _trace_fingerprint(resolved_trace, context)
        trace_metadata = dict(materialized)
        if item.trace_path is not None:
            source = _trace_fingerprint(item.trace_path.resolve(), context)
            trace_metadata.update(
                source_sha256=source["sha256"],
                source_size_bytes=source["size_bytes"],
                staged=True,
                transformed=(
                    not item.trace_presorted
                    or item.max_requests is not None
                    or (item.backend == "real" and item.arrival_speedup != 1.0)
                ),
            )
        resolved_paths = (resolved_trace,)
        arrival_inputs = resolved_trace
    speedup = item.arrival_speedup if item.backend == "sim" else 1.0
    arrival_key = (item.trace_format, resolved_paths, speedup)
    arrivals = context.arrival_series.get(arrival_key)
    if arrivals is None:
        if item.trace_format == "dynamo":
            arrivals = trace_arrival_series(
                arrival_inputs,
                trace_format="dynamo",
                speedup=speedup,
            )
        else:
            arrivals = trace_arrival_series(arrival_inputs, speedup=speedup)
        context.arrival_series[arrival_key] = arrivals
    return {
        "workload": item.workload,
        "seed": item.seed,
        "repetition": item.repetition,
        "max_requests": item.max_requests,
        "arrival_speedup": item.arrival_speedup,
        "trace_block_size": block_size,
        "trace": trace_metadata,
        "arrival_series": arrivals,
        "sla": asdict(item.slo_profile),
    }


def _sim_planner_config(
    backend: SimBackendConfig, *, report_filename: Path
) -> dict[str, Any]:
    planner_config: dict[str, Any] = {
        "mode": backend.topology,
        "optimization_target": "sla",
        "ttft_ms": 2000.0,
        "itl_ms": 50.0,
        "enable_load_scaling": True,
        "enable_throughput_scaling": False,
        "pre_deployment_sweeping_mode": "none",
        "load_adjustment_interval_seconds": 5,
        "load_min_observations": 5,
        "load_scaling_down_sensitivity": 80,
        "min_endpoint": 1,
    }
    planner_config.update(backend.planner_config)
    if backend.topology == "disagg":
        assert backend.engines.prefill is not None
        assert backend.engines.decode is not None
        prefill_num_gpus = backend.engines.prefill.num_gpus
        decode_num_gpus = backend.engines.decode.num_gpus
    else:
        assert backend.engines.aggregate is not None
        prefill_num_gpus = backend.engines.aggregate.num_gpus
        decode_num_gpus = backend.engines.aggregate.num_gpus
    planner_config.update(
        {
            "mode": backend.topology,
            "max_gpu_budget": backend.gpu_budget,
            "prefill_engine_num_gpu": prefill_num_gpus,
            "decode_engine_num_gpu": decode_num_gpus,
            "report_filename": str(report_filename),
        }
    )
    return planner_config


def _render_engine_args(engine: EngineConfig, ais_model_path: str) -> str:
    # Typed fields remain authoritative even for programmatically-constructed
    # configs that bypass the YAML parser's reserved-key validation. Replay binds
    # worker_type to the engine slot before Dynamo loads the canonical AIS config.
    args: dict[str, Any] = dict(engine.extra_args)
    perf_config = {
        "model": ais_model_path,
        "system": engine.system,
        "backend": engine.ais_backend,
        "backend_version": engine.ais_backend_version,
        "tp": engine.tp_size,
        "moe_tp_size": engine.moe_tp_size,
        "moe_ep_size": engine.moe_ep_size,
        "attention_dp": engine.attention_dp_size,
        "nextn": args.get("ais_nextn"),
        "estimation_mode": "auto",
        "fallback_policy": "deny",
    }
    args.update(
        engine_type=engine.backend,
        tensor_parallel_size=engine.tp_size,
        dp_size=engine.attention_dp_size or 1,
        ais_perf_config={
            key: value for key, value in perf_config.items() if value is not None
        },
    )
    optional = {
        "startup_time": engine.runtime.cold_start_delay_s,
        "kv_transfer_bandwidth": engine.runtime.kv_transfer_bandwidth_gbps,
        "kv_bytes_per_token": engine.runtime.kv_bytes_per_token,
    }
    args.update({key: value for key, value in optional.items() if value is not None})
    return json.dumps(args, sort_keys=True, allow_nan=False)


def _sim_engines_metadata(backend: SimBackendConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for role in ("prefill", "decode", "aggregate"):
        engine = getattr(backend.engines, role)
        if engine is None:
            continue
        rendered = _render_engine_args(engine, backend.model.ais_model_path)
        metadata[role] = {
            "config": _redact_mapping_secrets(asdict(engine)),
            "rendered_args_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        }
    return metadata


def _sim_performance_model_metadata(
    backend: SimBackendConfig,
) -> dict[str, Any]:
    """Describe Match Config AIS identities in Dynamo's replay contract.

    Optional versions stay omitted here. ``run_arena_replay`` backfills each
    role independently from Dynamo's lowered ``MockEngineArgs`` so an unpinned
    disaggregated deployment never collapses onto one role's AIS identity.
    """

    metadata: dict[str, Any] = {}
    engines = [
        (role, getattr(backend.engines, role))
        for role in ("prefill", "decode", "aggregate")
        if getattr(backend.engines, role) is not None
    ]
    for role, engine in engines:
        config = {
            "backend": engine.ais_backend,
            "backend_version": engine.ais_backend_version,
            "system": engine.system,
            "model_path": backend.model.ais_model_path,
            "tp_size": engine.tp_size,
            "moe_tp_size": engine.moe_tp_size,
            "moe_ep_size": engine.moe_ep_size,
            "attention_dp_size": engine.attention_dp_size,
            "nextn": engine.extra_args.get("ais_nextn"),
        }
        metadata["aggregated" if role == "aggregate" else role] = {
            "provider": "ais",
            "config": {
                key: value for key, value in config.items() if value is not None
            },
        }
    return metadata


def _build_sim_factory(
    autoscaler: SimAutoscalerConfig,
    *,
    topology: str,
    decision_log: Path | None = None,
    jev_context: dict[str, Any] | None = None,
) -> Callable[[Any, Any], Any]:
    # Imports stay inside the sim path so a real-only install needs no Dynamo.
    if autoscaler.type == "planner":
        from autoscaling_arena.adapters import planner_engine_factory

        return planner_engine_factory

    parameters = dict(autoscaler.config)
    if autoscaler.type == "jev":
        from autoscaling_arena.adapters.jev import JevAutoscaler

        def factory(config, capabilities):
            del config
            return JevAutoscaler(
                mode=topology,
                capabilities=capabilities,
                decision_log=decision_log,
                context=jev_context,
                **parameters,
            )

        return factory
    if autoscaler.type == "static":
        from autoscaling_arena.adapters import StaticAutoscaler

        def factory(config, capabilities):
            del config, capabilities
            return StaticAutoscaler(mode=topology, **parameters)

        return factory
    if autoscaler.type == "keda":
        from autoscaling_arena.adapters import KedaAutoscaler

        def factory(config, capabilities):
            del config
            return KedaAutoscaler(
                mode=topology, capabilities=capabilities, **parameters
            )

        return factory
    if autoscaler.type == "reactive":
        from autoscaling_arena.adapters import ReactiveAutoscaler

        def factory(config, capabilities):
            del config
            return ReactiveAutoscaler(
                mode=topology, capabilities=capabilities, **parameters
            )

        return factory
    raise AssertionError(f"unhandled autoscaler type: {autoscaler.type}")


def _project_sim_metrics(
    raw_scorecard: dict[str, Any],
    *,
    profile_name: str,
    include: tuple[str, ...],
) -> dict[str, Any]:
    profile_metrics = raw_scorecard["profiles"][profile_name]
    return {
        metric: (
            profile_metrics.get(metric)
            if metric in profile_metrics
            else raw_scorecard.get(metric)
        )
        for metric in include
    }


def _nice_cache_bucket_width(duration_s: float, max_points: int = 1_500) -> float:
    """Return a human-readable bucket width while bounding report size."""

    raw = max(1.0, duration_s / max_points)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    for step in (1.0, 2.0, 5.0, 10.0):
        if normalized <= step:
            return step * magnitude
    return 10.0 * magnitude


def _prefix_cache_telemetry(report: Any) -> dict[str, Any]:
    """Compact realized prefix reuse before per-request records are discarded.

    The trace report's headline ratio is authoritative.  When Dynamo retained
    per-request records, this additionally builds a completion-attributed,
    token-weighted time series.  It deliberately does not reuse the callback's
    ``avg_kv_hit_rate``: that is a router overlap estimate with different
    semantics and round-robin routing supplies no samples.
    """

    raw_summary = getattr(report, "trace_report", None)
    summary = raw_summary if isinstance(raw_summary, dict) else {}
    ratio = summary.get("prefix_cache_reused_ratio")
    first_ratio = summary.get("first_admission_prefix_cache_reused_ratio")
    total_input_tokens = int(summary.get("total_input_tokens", 0) or 0)
    aggregate_reused_tokens = (
        int(round(float(ratio) * total_input_tokens))
        if ratio is not None and total_input_tokens > 0
        else None
    )
    cache: dict[str, Any] = {
        "prefix_cache_reused_ratio": (float(ratio) if ratio is not None else None),
        "first_admission_prefix_cache_reused_ratio": (
            float(first_ratio) if first_ratio is not None else None
        ),
        "total_input_tokens": total_input_tokens or None,
        "reused_input_tokens": aggregate_reused_tokens,
        "timeline_available": False,
        "timeline_source": None,
        "timeline": [],
    }

    per_request = getattr(report, "per_request", None)
    if not isinstance(per_request, list):
        return cache

    completed: list[tuple[float, int, int]] = []
    for record in per_request:
        if not isinstance(record, dict):
            continue
        status = str(record.get("terminal_status", "")).lower()
        if status != "completed" or record.get("first_admit_ms") is None:
            continue
        try:
            terminal_s = float(record["terminal_time_ms"]) / 1000.0
            input_tokens = max(0, int(record.get("input_length", 0) or 0))
            reused_tokens = max(
                0,
                int(record.get("reused_input_tokens", 0) or 0),
                int(record.get("decode_reused_input_tokens", 0) or 0),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(terminal_s) or terminal_s < 0.0:
            continue
        completed.append((terminal_s, input_tokens, reused_tokens))

    if not completed:
        return cache

    duration_s = max(item[0] for item in completed)
    width_s = _nice_cache_bucket_width(duration_s)
    buckets: dict[int, list[int]] = {}
    for terminal_s, input_tokens, reused_tokens in completed:
        index = int(terminal_s // width_s)
        values = buckets.setdefault(index, [0, 0, 0])
        values[0] += input_tokens
        values[1] += reused_tokens
        values[2] += 1

    last_index = max(buckets)
    timeline: list[dict[str, Any]] = []
    for index in range(last_index + 1):
        input_tokens, reused_tokens, request_count = buckets.get(index, [0, 0, 0])
        timeline.append(
            {
                "window_start_s": index * width_s,
                "time_s": (index + 1) * width_s,
                "input_tokens": input_tokens,
                "reused_input_tokens": reused_tokens,
                "completed_requests": request_count,
                "prefix_cache_reused_ratio": (
                    reused_tokens / input_tokens if input_tokens > 0 else None
                ),
            }
        )

    cache.update(
        timeline_available=True,
        timeline_source="completed_requests",
        timeline=timeline,
        timeline_bucket_width_s=width_s,
    )
    return cache


_REAL_METRIC_FIELDS = {
    "goodput_rps": "goodput_rps",
    "good_rate": "good_request_fraction",
    "good_count": "good_request_count",
    "request_throughput_rps": "request_throughput_rps",
    "completed_requests": "request_count",
    "duration_s": "benchmark_duration_s",
    "mean_ttft_ms": "mean_ttft_ms",
    "p99_ttft_ms": "p99_ttft_ms",
    "mean_itl_ms": "mean_itl_ms",
    "p99_itl_ms": "p99_itl_ms",
    "mean_e2e_ms": "mean_e2e_ms",
    "p99_e2e_ms": "p99_e2e_ms",
}


def _project_real_metrics(
    raw_metrics: dict[str, Any], include: tuple[str, ...]
) -> dict[str, Any]:
    return {metric: raw_metrics.get(_REAL_METRIC_FIELDS[metric]) for metric in include}


def _find_sim_autoscaler(backend: SimBackendConfig, name: str) -> SimAutoscalerConfig:
    return next(item for item in backend.autoscalers if item.name == name)


def _find_real_autoscaler(
    backend: RealBackendConfig, name: str
) -> RealAutoscalerConfig:
    return next(item for item in backend.autoscalers if item.name == name)


def _to_slo_profile(profile: SLAProfileConfig) -> SLOProfile:
    return SLOProfile(
        name=profile.name,
        ttft_ms=profile.ttft_ms,
        itl_ms=profile.itl_ms,
        e2e_ms=profile.e2e_ms,
    )


def _result_identity(item: MatchRun) -> dict[str, Any]:
    return {
        "run_id": item.run_id,
        "backend": item.backend,
        "autoscaler": item.autoscaler,
        "workload": item.workload,
        "sla": item.sla,
        "repetition": item.repetition,
        "seed": item.seed,
    }


def _failure_result(
    item: MatchRun,
    *,
    artifact_dir: Path,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        **_result_identity(item),
        "status": "failed",
        "metrics": {},
        "error": {"type": error_type, "message": message},
        "artifacts": {"directory": str(artifact_dir)},
    }


def _matrix_dict(item: MatchRun) -> dict[str, Any]:
    return {
        **_result_identity(item),
        "max_requests": item.max_requests,
        "arrival_speedup": item.arrival_speedup,
        "sla_target": asdict(item.slo_profile),
    }


def match_config_sha256(config: MatchConfig) -> str:
    """Return the canonical resolved-config digest stored in provenance."""

    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def match_replay_sha256(config: MatchConfig) -> str:
    """Hash execution semantics without checkout- or output-specific paths.

    Source trace locations are guarded independently by replay trace digests.
    Parsed planner and endpoint-catalog contents are already represented in the
    resolved backend, so their source filenames are omitted as well.
    """

    payload = config.to_dict()
    payload.pop("source_path", None)
    payload.pop("publish", None)
    backend = payload.get("backend", {})
    if isinstance(backend, dict):
        backend.pop("planner_config_path", None)
        backend.pop("endpoint_catalog", None)
    evaluations = payload.get("evaluations", [])
    if isinstance(evaluations, list):
        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue
            if evaluation.get("trace_path") is not None:
                evaluation["trace_path"] = "<source-trace>"
            trace_paths = evaluation.get("trace_paths")
            if isinstance(trace_paths, list):
                evaluation["trace_paths"] = [
                    f"<source-trace-{index}>" for index, _ in enumerate(trace_paths)
                ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_replay_config_path(config: MatchConfig) -> Optional[str]:
    """Return a portable replay reference without exposing local paths."""

    repository = Path(__file__).resolve().parents[2]
    source = config.source_path.resolve(strict=False)
    if not source.is_file():
        return None
    try:
        relative = source.relative_to(repository)
    except ValueError:
        return "$MATCH_CONFIG"
    return relative.as_posix()


def _git_commit() -> Optional[str]:
    repository = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else None


def _redact_config(value: Any) -> Any:
    """Redact common credential-bearing AIPerf extra-argument values."""

    if not isinstance(value, dict):
        return value
    redacted = _redact_mapping_secrets(json.loads(json.dumps(value)))
    backend = redacted.get("backend", {})
    endpoints = backend.get("endpoints", [])
    if isinstance(endpoints, list):
        for endpoint in endpoints:
            if isinstance(endpoint, dict) and isinstance(endpoint.get("url"), str):
                endpoint["url"] = _redact_url(endpoint["url"])
    try:
        args = backend["aiperf"]["extra_args"]
    except (KeyError, TypeError):
        return redacted
    backend["aiperf"]["extra_args"] = _redacted_extra_args(
        tuple(str(argument) for argument in args)
    )
    return redacted


def _redact_report_paths(
    value: Any, config: MatchConfig, context: _ExecutionContext
) -> Any:
    """Replace machine-local paths while preserving useful artifact structure."""

    replacements: dict[str, str] = {}

    def protect(path: Optional[Path], replacement: str) -> None:
        if path is not None:
            replacements[str(path.resolve(strict=False))] = replacement

    protect(config.source_path, "<config>")
    protect(config.source_path.parent, "<config-dir>")
    protect(config.publish.artifact_root, "<artifact-root>")
    protect(context.session_root, "<artifact-session>")
    protect(context.external_trace_root, "<external-trace-temp>")
    for evaluation in config.evaluations:
        protect(evaluation.trace_path, "<external-trace>")
        for trace_path in evaluation.trace_paths:
            protect(trace_path, "<external-trace>")
    for destination in config.publish.destinations:
        protect(destination.path, "<publish-destination>")
    if isinstance(config.backend, RealBackendConfig):
        protect(config.backend.endpoint_catalog, "<endpoint-catalog>")
    elif config.backend.planner_config_path is not None:
        protect(config.backend.planner_config_path, "<planner-config>")

    ordered = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)

    def redact(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: redact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, str):
            for source, replacement in ordered:
                item = item.replace(source, replacement)
        return item

    return redact(value)


def _redact_mapping_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            redacted[str(key)] = (
                "<redacted>"
                if normalized in _SENSITIVE_CONFIG_KEYS
                else _redact_mapping_secrets(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_mapping_secrets(item) for item in value]
    return value


def _redact_exception_message(config: MatchConfig, message: str) -> str:
    secrets = sorted(
        _secret_values_from_mapping(config.to_dict()),
        key=len,
        reverse=True,
    )
    for secret in secrets:
        if secret:
            message = message.replace(secret, "<redacted>")
    if isinstance(config.backend, RealBackendConfig):
        message = _redact_aiperf_text(message, config.backend.aiperf.extra_args)
        for endpoint in config.backend.endpoints:
            message = message.replace(endpoint.url, _redact_url(endpoint.url))
    return message


def _secret_values_from_mapping(value: Any) -> list[str]:
    def scalar_values(item: Any) -> list[str]:
        if isinstance(item, dict):
            return [
                secret for child in item.values() for secret in scalar_values(child)
            ]
        if isinstance(item, (list, tuple)):
            return [secret for child in item for secret in scalar_values(child)]
        if isinstance(item, (str, int, float)):
            return [str(item)]
        return []

    secrets: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SENSITIVE_CONFIG_KEYS:
                secrets.extend(scalar_values(item))
            else:
                secrets.extend(_secret_values_from_mapping(item))
    elif isinstance(value, list):
        for item in value:
            secrets.extend(_secret_values_from_mapping(item))
    return secrets


def _redact_aiperf_text(message: str, args: tuple[str, ...]) -> str:
    for secret in _secret_values_from_args(args):
        if secret:
            message = message.replace(secret, "<redacted>")
    return message


def _secret_values_from_args(args: tuple[str, ...]) -> list[str]:
    secrets: list[str] = []
    hide_next = False
    in_header_values = False
    for argument in args:
        rendered = str(argument)
        if hide_next:
            secrets.append(rendered)
            hide_next = False
            continue
        lowered = rendered.lower()
        if lowered in _HEADER_ARGS:
            in_header_values = True
            continue
        if any(lowered.startswith(flag + "=") for flag in _HEADER_ARGS):
            secrets.append(rendered.split("=", 1)[1])
            in_header_values = False
            continue
        if in_header_values:
            if rendered.startswith("-"):
                in_header_values = False
            else:
                secrets.append(rendered)
                continue
        if lowered in _SENSITIVE_ARGS:
            hide_next = True
        elif any(lowered.startswith(key + "=") for key in _SENSITIVE_ARGS):
            secrets.append(rendered.split("=", 1)[1])
    return secrets


def _redacted_extra_args(args: tuple[str, ...]) -> list[str]:
    redacted = list(args)
    hide_next = False
    in_header_values = False
    for index, rendered in enumerate(args):
        lowered = rendered.lower()
        if hide_next:
            redacted[index] = "<redacted>"
            hide_next = False
            continue
        if lowered in _HEADER_ARGS:
            in_header_values = True
            continue
        if any(lowered.startswith(flag + "=") for flag in _HEADER_ARGS):
            flag = rendered.split("=", 1)[0]
            redacted[index] = flag + "=<redacted>"
            in_header_values = False
            continue
        if in_header_values:
            if rendered.startswith("-"):
                in_header_values = False
            else:
                redacted[index] = "<redacted>"
                continue
        if lowered in _SENSITIVE_ARGS:
            hide_next = True
        elif any(lowered.startswith(key + "=") for key in _SENSITIVE_ARGS):
            flag = rendered.split("=", 1)[0]
            redacted[index] = flag + "=<redacted>"
    return redacted


def _redact_url(url: str) -> str:
    # Avoid adding an HTTP parser dependency; cover the two credential-bearing
    # URL forms while preserving the host/path needed for provenance.
    if "://" in url:
        scheme, rest = url.split("://", 1)
        authority, separator, tail = rest.partition("/")
        if "@" in authority:
            authority = authority.rsplit("@", 1)[1]
        url = f"{scheme}://{authority}{separator}{tail}"
    if "?" in url:
        url = url.split("?", 1)[0] + "?<redacted>"
    return url


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float) and not math_is_finite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def math_is_finite(value: float) -> bool:
    # Kept as a tiny seam for tests that need to exercise non-finite metrics.
    return value == value and value not in (float("inf"), float("-inf"))


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "n/a" if value is None else str(value)


def _require_directory_ancestor(path: Path) -> None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.exists() and not candidate.is_dir():
        raise MatchPublishError(f"output parent is not a directory: {candidate}")


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for an atomic rename/link in ``path``."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _ranking_value(value: Any, *, reverse: bool) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("-inf") if reverse else float("inf")


def _rank_metric_error(metrics: dict[str, Any], rank_by: str) -> Optional[str]:
    value = metrics.get(rank_by)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math_is_finite(float(value))
    ):
        return f"configured rank metric {rank_by!r} is not a finite number"
    return None


__all__ = [
    "MatchPublishError",
    "execute_match_config",
    "format_match_matrix",
    "format_match_results",
    "match_config_sha256",
    "match_replay_sha256",
    "preflight_publish_destinations",
    "publish_match_results",
]
