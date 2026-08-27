#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare an RL validation record with the GPU host that will execute it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import check_rl_validation_record

SCHEMA = "dynamo.rl.environment-preflight.v1"
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
GPU_FAMILY = re.compile(r"\b(?:GH|GB|H|A|L|B|V|T)\d{2,4}[A-Z]*\b", re.IGNORECASE)
BACKEND_DISTRIBUTIONS = {
    "vllm": "vllm",
    "sglang": "sglang",
    "tensorrt_llm": "tensorrt-llm",
    "tensorrt-llm": "tensorrt-llm",
}
BASE_REQUIRED_BINARIES = ("git", "nvidia-smi")

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
BinaryLookup = Callable[[str], str | None]
SoftwareProbe = Callable[[str], dict[str, str | None]]


class PreflightError(ValueError):
    """Raised when the preflight input or output cannot be handled safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _safe_run(
    runner: CommandRunner, args: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(args)
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(list(args), 127, "", str(exc))


def _software_probe(backend_name: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "backend_version": None,
        "backend_error": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "torch_error": None,
    }
    distribution = BACKEND_DISTRIBUTIONS.get(backend_name.casefold(), backend_name)
    try:
        result["backend_version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        result["backend_error"] = str(exc)
    try:
        import torch

        result["torch_version"] = str(torch.__version__)
        result["torch_cuda_version"] = (
            str(torch.version.cuda) if torch.version.cuda is not None else None
        )
    except (ImportError, OSError, RuntimeError) as exc:
        result["torch_error"] = str(exc)
    return result


def _check(
    check_id: str,
    passed: bool | None,
    *,
    expected: Any,
    observed: Any,
    detail: str,
) -> dict[str, Any]:
    status = "not_checked" if passed is None else "passed" if passed else "failed"
    return {
        "id": check_id,
        "status": status,
        "expected": expected,
        "observed": observed,
        "detail": detail,
    }


def _git_probe(
    role: str,
    repo: Path,
    expected_commit: str,
    runner: CommandRunner,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head_result = _safe_run(runner, ("git", "-C", str(repo), "rev-parse", "HEAD"))
    status_result = _safe_run(
        runner,
        (
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
        ),
    )
    observed_commit = (
        head_result.stdout.strip() if head_result.returncode == 0 else None
    )
    dirty_entries = (
        [line for line in status_result.stdout.splitlines() if line]
        if status_result.returncode == 0
        else []
    )
    probe = {
        "role": role,
        "path": str(repo.resolve(strict=False)),
        "expected_commit": expected_commit,
        "observed_commit": observed_commit,
        "head_probe_error": head_result.stderr.strip() or None,
        "dirty_entries": dirty_entries,
        "status_probe_error": status_result.stderr.strip() or None,
    }
    checks = [
        _check(
            f"repository.{role}.commit",
            head_result.returncode == 0 and observed_commit == expected_commit,
            expected=expected_commit,
            observed=observed_commit,
            detail="The checked-out commit must match the immutable validation-record pin.",
        ),
        _check(
            f"repository.{role}.clean",
            status_result.returncode == 0 and not dirty_entries,
            expected="clean worktree",
            observed=dirty_entries if status_result.returncode == 0 else None,
            detail="Uncommitted or untracked code can invalidate the recorded pin.",
        ),
    ]
    return probe, checks


def _parse_gpu_rows(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output)):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) != 4:
            raise PreflightError(
                "nvidia-smi GPU query must return index, name, driver version, and memory"
            )
        try:
            index = int(row[0].strip())
            memory_mib = int(row[3].strip())
        except ValueError as exc:
            raise PreflightError(f"cannot parse nvidia-smi GPU row: {row!r}") from exc
        rows.append(
            {
                "index": index,
                "name": row[1].strip(),
                "driver_version": row[2].strip(),
                "memory_total_mib": memory_mib,
            }
        )
    return rows


def _gpu_family(value: str) -> str | None:
    match = GPU_FAMILY.search(value)
    return match.group(0).upper() if match else None


def _gpu_model_matches(expected: str, observed: str) -> bool:
    expected_family = _gpu_family(expected)
    observed_family = _gpu_family(observed)
    if expected_family is not None and observed_family is not None:
        return expected_family == observed_family
    expected_normalized = re.sub(r"[^a-z0-9]", "", expected.casefold())
    observed_normalized = re.sub(r"[^a-z0-9]", "", observed.casefold())
    return expected_normalized == observed_normalized


def _gpu_probe(
    record: dict[str, Any], runner: CommandRunner
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = _safe_run(
        runner,
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
    )
    parse_error: str | None = None
    rows: list[dict[str, Any]] = []
    if query.returncode == 0:
        try:
            rows = _parse_gpu_rows(query.stdout)
        except PreflightError as exc:
            parse_error = str(exc)
    topology = _safe_run(runner, ("nvidia-smi", "topo", "-m"))
    expected = record["hardware"]
    expected_driver = record["environment"]["driver_version"]
    model_match = bool(rows) and all(
        _gpu_model_matches(expected["gpu_model"], row["name"]) for row in rows
    )
    driver_match = bool(rows) and all(
        row["driver_version"] == expected_driver for row in rows
    )
    probe = {
        "query_succeeded": query.returncode == 0 and parse_error is None,
        "query_error": parse_error or query.stderr.strip() or None,
        "gpus": rows,
        "topology_succeeded": topology.returncode == 0,
        "topology": topology.stdout if topology.returncode == 0 else None,
        "topology_error": topology.stderr.strip() or None,
    }
    checks = [
        _check(
            "hardware.gpu_query",
            query.returncode == 0 and parse_error is None and bool(rows),
            expected="at least one queryable NVIDIA GPU",
            observed=len(rows),
            detail="The probe must enumerate only the GPUs visible to the validation job.",
        ),
        _check(
            "hardware.gpu_count",
            len(rows) == expected["gpus_per_node"],
            expected=expected["gpus_per_node"],
            observed=len(rows),
            detail="Visible GPU count must equal the record's per-node allocation.",
        ),
        _check(
            "hardware.gpu_model",
            model_match,
            expected=expected["gpu_model"],
            observed=sorted({row["name"] for row in rows}),
            detail="Every visible GPU must match the recorded accelerator family.",
        ),
        _check(
            "environment.driver_version",
            driver_match,
            expected=expected_driver,
            observed=sorted({row["driver_version"] for row in rows}),
            detail="Every visible GPU must report the recorded driver version.",
        ),
        _check(
            "hardware.topology_capture",
            topology.returncode == 0 and bool(topology.stdout.strip()),
            expected="successful nvidia-smi topo -m capture",
            observed=topology.returncode,
            detail="A reviewer must use the captured topology to verify interconnect and network claims.",
        ),
    ]
    return probe, checks


def build_report(
    record: dict[str, Any],
    *,
    record_path: Path,
    record_sha256: str,
    repositories: dict[str, Path],
    observed_image_digest: str | None,
    required_binaries: Sequence[str],
    runner: CommandRunner = _run_command,
    binary_lookup: BinaryLookup = shutil.which,
    software_probe: SoftwareProbe = _software_probe,
    host_system: str | None = None,
    generated_at: datetime | None = None,
) -> tuple[dict[str, Any], list[str]]:
    checks: list[dict[str, Any]] = []
    structure_findings = check_rl_validation_record.validate_structure(record)
    checks.append(
        _check(
            "validation_record.structure",
            not structure_findings,
            expected=check_rl_validation_record.SCHEMA,
            observed=record.get("schema"),
            detail="; ".join(structure_findings)
            or "Validation-record structure is valid.",
        )
    )
    if structure_findings:
        raise PreflightError(
            "validation record structure is invalid: " + "; ".join(structure_findings)
        )
    expected_commits = {
        "dynamo": record.get("environment", {}).get("dynamo_commit", ""),
        "recipe": record.get("framework", {}).get("recipe_commit", ""),
        "core": record.get("framework", {}).get("core_commit", ""),
    }
    repository_probes: list[dict[str, Any]] = []
    for role in ("dynamo", "recipe", "core"):
        probe, repo_checks = _git_probe(
            role, repositories[role], expected_commits[role], runner
        )
        repository_probes.append(probe)
        checks.extend(repo_checks)

    binaries = sorted(set(BASE_REQUIRED_BINARIES).union(required_binaries))
    binary_probes: list[dict[str, Any]] = []
    for name in binaries:
        location = binary_lookup(name)
        binary_probes.append({"name": name, "path": location})
        checks.append(
            _check(
                f"binary.{name}",
                location is not None,
                expected="available on PATH",
                observed=location,
                detail="Required launch/runtime binary must resolve in the execution environment.",
            )
        )

    gpu_probe, gpu_checks = _gpu_probe(record, runner)
    checks.extend(gpu_checks)
    backend = record.get("environment", {}).get("backend", {})
    software = software_probe(str(backend.get("name", "")))
    checks.extend(
        [
            _check(
                "environment.backend_version",
                software.get("backend_version") == backend.get("version"),
                expected=backend.get("version"),
                observed=software.get("backend_version"),
                detail=software.get("backend_error")
                or "Installed backend distribution must match the validation-record pin.",
            ),
            _check(
                "environment.cuda_version",
                software.get("torch_cuda_version")
                == record.get("environment", {}).get("cuda_version"),
                expected=record.get("environment", {}).get("cuda_version"),
                observed=software.get("torch_cuda_version"),
                detail=software.get("torch_error")
                or "PyTorch's compiled CUDA version must match the recorded runtime value.",
            ),
        ]
    )

    expected_image_digest = record.get("environment", {}).get("container_image_digest")
    digest_well_formed = bool(
        observed_image_digest and IMAGE_DIGEST.fullmatch(observed_image_digest)
    )
    checks.append(
        _check(
            "environment.container_image_digest",
            digest_well_formed and observed_image_digest == expected_image_digest,
            expected=expected_image_digest,
            observed=observed_image_digest,
            detail="Observed digest is an operator-supplied value; preserve independent deployment metadata that proves it.",
        )
    )

    current_system = host_system if host_system is not None else platform.system()
    checks.append(
        _check(
            "host.operating_system",
            current_system == "Linux",
            expected="Linux",
            observed=current_system,
            detail="The documented verl GPU path requires a Linux execution environment.",
        )
    )
    checks.extend(
        [
            _check(
                "hardware.interconnect_review",
                None,
                expected=record.get("hardware", {}).get("interconnect"),
                observed="captured in hardware.topology",
                detail="Human review is required; the probe does not infer interconnect semantics.",
            ),
            _check(
                "hardware.network_review",
                None,
                expected=record.get("hardware", {}).get("network"),
                observed="not automatically detected",
                detail="Preserve scheduler or fabric inventory evidence beside this report.",
            ),
        ]
    )

    failures = [
        f"{item['id']}: {item['detail']}"
        for item in checks
        if item["status"] == "failed"
    ]
    timestamp = generated_at or datetime.now(UTC)
    report = {
        "schema": SCHEMA,
        "generated_at": timestamp.isoformat().replace("+00:00", "Z"),
        "strict_status": "failed" if failures else "passed",
        "validation_record": {
            "path": str(record_path.resolve(strict=False)),
            "sha256": record_sha256,
            "record_id": record.get("record_id", ""),
            "schema": record.get("schema"),
        },
        "host": {
            "system": current_system,
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "hostname_captured": False,
            "environment_variables_captured": False,
        },
        "repositories": repository_probes,
        "container": {
            "expected_image": record.get("environment", {}).get("container_image"),
            "expected_digest": expected_image_digest,
            "observed_digest": observed_image_digest,
            "observed_digest_provenance": "operator_argument",
        },
        "software": {
            **software,
            "required_binaries": binary_probes,
        },
        "hardware": gpu_probe,
        "checks": checks,
        "failed_check_count": len(failures),
        "not_checked_count": sum(item["status"] == "not_checked" for item in checks),
        "claim_boundary": "Host and checkout preflight; not training, correctness, performance, or owner-acceptance evidence.",
    }
    return report, failures


def _output_path(path: Path) -> Path:
    if path.is_symlink():
        raise PreflightError(f"output path must not be a symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--dynamo-repo", type=Path, required=True)
    parser.add_argument("--recipe-repo", type=Path, required=True)
    parser.add_argument("--core-repo", type=Path, required=True)
    parser.add_argument("--observed-image-digest")
    parser.add_argument("--require-binary", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return failure when any machine-checkable preflight check fails",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = check_rl_validation_record.load_record(args.record)
        report, findings = build_report(
            record,
            record_path=args.record,
            record_sha256=_sha256(args.record),
            repositories={
                "dynamo": args.dynamo_repo,
                "recipe": args.recipe_repo,
                "core": args.core_repo,
            },
            observed_image_digest=args.observed_image_digest,
            required_binaries=args.require_binary,
        )
        output = _output_path(args.output_json)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (check_rl_validation_record.RecordError, OSError, PreflightError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.strict and findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        print(f"Wrote failed preflight report to {output}", file=sys.stderr)
        return 1
    print(
        f"RL environment preflight wrote {output} "
        f"({report['strict_status']}; {report['failed_check_count']} failed; "
        f"{report['not_checked_count']} human-review checks)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
