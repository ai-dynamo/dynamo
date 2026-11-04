# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose independent matches with incremental, resumable suite reports."""

from __future__ import annotations

import csv
import fcntl
import html
import io
import json
import os
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoscaling_arena.html_report import render_match_report
from autoscaling_arena.match_config import PublishConfig
from autoscaling_arena.match_runner import (
    MatchPublishError,
    execute_match_config,
    execution_signature,
    match_replay_sha256,
)
from autoscaling_arena.suite_config import (
    SuiteCase,
    SuiteConfig,
    SuiteConfigError,
    suite_config_sha256,
)


def execute_suite_config(
    config: SuiteConfig,
    *,
    output_dir: Path,
    case_ids: set[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run cases sequentially, delegating isolation and limits to each match.

    Each match keeps its engine, policy, warmup, and scoring settings. Only its
    publication settings are replaced: all artifacts stay within this suite's
    output directory. Parallelism is bounded by the current match's execution
    settings; this layer never creates additional concurrent matches.
    """

    cases = config.selected_cases(case_ids)
    root = output_dir.expanduser().resolve()
    manifest = {
        "schema_version": config.schema_version,
        "name": config.name,
        "config_sha256": suite_config_sha256(config),
        "selected_case_ids": [case.case_id for case in cases],
        "cases": {
            case.case_id: {
                "config_file_sha256": case.config_file_sha256,
                "replay_config_sha256": match_replay_sha256(case.config),
                "execution_signature": execution_signature(case.config),
                "labels": case.labels,
                "run_ids": list(case.run_ids) if case.run_ids is not None else None,
            }
            for case in cases
        },
    }
    if resume:
        if not root.is_dir():
            raise SuiteConfigError("resume output directory does not exist")
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".suite.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SuiteConfigError(
                "another process is already executing this suite"
            ) from exc
        try:
            return _execute_cases(cases, root, manifest, resume=resume)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _execute_cases(
    cases: tuple[SuiteCase, ...],
    root: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    index_path = root / "suite.json"
    if resume:
        try:
            previous = json.loads(index_path.read_text())
        except (OSError, ValueError) as exc:
            raise SuiteConfigError(f"cannot read suite checkpoint: {exc}") from exc
        if not isinstance(previous, dict) or _resume_identity(
            previous.get("manifest")
        ) != _resume_identity(manifest):
            raise SuiteConfigError(
                "suite checkpoint does not match this manifest and selection"
            )
        results = previous.get("cases")
        if not isinstance(results, dict) or set(results) - set(
            manifest["selected_case_ids"]
        ):
            raise SuiteConfigError("suite checkpoint has invalid case results")
        if any(
            not isinstance(case, dict)
            or case.get("status") not in {"ok", "failed", "running"}
            for case in results.values()
        ):
            raise SuiteConfigError("suite checkpoint has invalid case status")
        history = previous.get("configuration_history", [])
        if not isinstance(history, list):
            raise SuiteConfigError("suite checkpoint has invalid configuration history")
    else:
        results = {}
        history = []
    history.append(
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": manifest["config_sha256"],
            "cases": {
                case_id: {
                    "config_file_sha256": case["config_file_sha256"],
                    "replay_config_sha256": case["replay_config_sha256"],
                }
                for case_id, case in manifest["cases"].items()
            },
        }
    )
    report = {
        "manifest": manifest,
        "configuration_history": history,
        "summary": {},
        "cases": results,
    }
    _publish_suite(root, report)
    for case in cases:
        case_root = root / "cases" / case.case_id
        case_root.mkdir(parents=True, exist_ok=True)
        execution_dir = case_root / "execution"
        match = replace(
            case.config,
            publish=PublishConfig(artifact_root=case_root, destinations=()),
        )
        started = time.monotonic()
        results[case.case_id] = {
            "status": "running",
            "labels": case.labels,
            "planned_runs": case.expected_runs,
        }
        _publish_suite(root, report)
        # A completed case is deliberately passed through the match checkpoint
        # validator again: it owns input digests and safe result reuse.
        try:
            match_report = execute_match_config(
                match,
                run_ids=set(case.run_ids) if case.run_ids is not None else None,
                resume_dir=execution_dir if resume and execution_dir.exists() else None,
                session_dir=execution_dir if not execution_dir.exists() else None,
            )
        except (MatchPublishError, ValueError) as exc:
            results[case.case_id] = {
                "status": "failed",
                "labels": case.labels,
                "planned_runs": case.expected_runs,
                "wall_time_s": time.monotonic() - started,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            _publish_suite(root, report)
            continue
        _atomic_write(case_root / "results.json", _json_text(match_report))
        _atomic_write(case_root / "index.html", render_match_report(match_report))
        results[case.case_id] = {
            "status": match_report["summary"]["status"],
            "labels": case.labels,
            "planned_runs": case.expected_runs,
            "wall_time_s": time.monotonic() - started,
            "summary": match_report["summary"],
            "provenance": match_report["provenance"],
            "results_path": f"cases/{case.case_id}/results.json",
            "report_path": f"cases/{case.case_id}/index.html",
            # Large time series remain in the linked per-match report.
            "results": [
                _comparison_result(result) for result in match_report["results"]
            ],
        }
        _publish_suite(root, report)
    return report


def _resume_identity(manifest: Any) -> dict[str, Any]:
    """Keep operational budgets out of compatibility, but retain their history."""

    try:
        return {
            "schema_version": manifest["schema_version"],
            "name": manifest["name"],
            "selected_case_ids": manifest["selected_case_ids"],
            "cases": {
                case_id: {
                    "execution_signature": case["execution_signature"],
                    "labels": case["labels"],
                    "run_ids": case["run_ids"],
                }
                for case_id, case in manifest["cases"].items()
            },
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise SuiteConfigError("suite checkpoint has an invalid manifest") from exc


def _publish_suite(root: Path, report: dict[str, Any]) -> None:
    results = report["cases"]
    planned = len(report["manifest"]["selected_case_ids"])
    completed = sum(case["status"] == "ok" for case in results.values())
    failed = sum(case["status"] == "failed" for case in results.values())
    report["summary"] = {
        "status": "ok" if completed == planned else "failed" if failed else "running",
        "planned_cases": planned,
        "succeeded_cases": completed,
        "failed_cases": failed,
        "pending_cases": planned - completed - failed,
    }
    _atomic_write(root / "suite.json", _json_text(report))
    _atomic_write(root / "comparison.csv", _comparison_csv(report))
    _atomic_write(root / "index.html", _suite_html(report))


def _comparison_csv(report: dict[str, Any]) -> str:
    rows = []
    for case_id, case in report["cases"].items():
        for result in case.get("results", []):
            row = {
                "case_id": case_id,
                "run_id": result["run_id"],
                "status": result["status"],
                "labels": json.dumps(case["labels"], sort_keys=True),
                "backend": result["backend"],
                "autoscaler": result["autoscaler"],
                "workload": result["workload"],
                "sla": result["sla"],
                "repetition": result["repetition"],
                "seed": result["seed"],
                **{
                    f"slo.{key}": result.get("sla_target", {}).get(key)
                    for key in ("ttft_ms", "itl_ms", "e2e_ms")
                },
                **{
                    f"metric.{key}": value
                    for key, value in result.get("metrics", {}).items()
                },
            }
            rows.append(row)
    fields = [
        "case_id",
        "run_id",
        "status",
        "labels",
        "backend",
        "autoscaler",
        "workload",
        "sla",
        "repetition",
        "seed",
    ]
    fields.extend(sorted({key for row in rows for key in row if key not in fields}))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _comparison_result(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "run_id",
        "status",
        "backend",
        "autoscaler",
        "workload",
        "sla",
        "sla_target",
        "repetition",
        "seed",
        "metrics",
        "error",
        "execution",
    )
    return {key: result[key] for key in fields if key in result}


def _suite_html(report: dict[str, Any]) -> str:
    rows = []
    for case_id in report["manifest"]["selected_case_ids"]:
        case = report["cases"].get(case_id, {"status": "pending"})
        link = case.get("report_path")
        label = html.escape(case_id)
        if link:
            label = f'<a href="{html.escape(link, quote=True)}">{label}</a>'
        labels = report["manifest"]["cases"][case_id]["labels"]
        summary = case.get("summary", {})
        rows.append(
            "<tr>"
            f"<td>{label}</td><td>{html.escape(case['status'])}</td>"
            f"<td>{html.escape(json.dumps(labels, sort_keys=True))}</td>"
            f"<td>{summary.get('succeeded_runs', '')}</td>"
            f"<td>{summary.get('failed_runs', '')}</td>"
            f"<td>{case.get('wall_time_s', 0):.3f}</td></tr>"
        )
    title = html.escape(report["manifest"]["name"])
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{title}</title><style>body{{font:16px system-ui;margin:2rem}}"
        "table{border-collapse:collapse}th,td{padding:.6rem;border:1px solid #ddd}"
        "</style>"
        f"<h1>{title}</h1><p>Case configurations are evaluated independently. "
        "Metrics from different cases are not combined into an overall ranking.</p>"
        '<p><a href="suite.json">Suite JSON</a> · '
        '<a href="comparison.csv">Comparison CSV</a></p>'
        "<table><thead><tr><th>Case</th><th>Status</th><th>Labels</th>"
        "<th>Successful runs</th><th>Failed runs</th><th>Wall time (s)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></html>\n"
    )


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
