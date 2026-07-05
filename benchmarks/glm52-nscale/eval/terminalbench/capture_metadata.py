#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture secret-free, replayable metadata for a Terminal-Bench Harbor job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_binding import make_wrapper  # noqa: E402
from source_provenance import verify_source_provenance  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], *, timeout: int = 10) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_metadata(repository: Path) -> dict[str, Any]:
    head = command_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
    branch = command_output(["git", "-C", str(repository), "branch", "--show-current"])
    status = command_output(
        ["git", "-C", str(repository), "status", "--short", "--untracked-files=all"]
    )
    status_lines = status.splitlines() if status else []
    return {
        "commit": head,
        "branch": branch or None,
        "clean": not status_lines,
        "changed_path_count": len(status_lines),
    }


def validate_api_base(api_base: str) -> str:
    parsed = urllib.parse.urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--api-base must be an absolute http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "--api-base must not contain credentials, a query string, or a fragment"
        )
    return api_base.rstrip("/")


def endpoint_models(
    api_base: str, served_model: str, serving_context: int
) -> dict[str, Any]:
    url = validate_api_base(api_base) + "/models"
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"GET {url} returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"GET {url} failed: {error.reason}") from error

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"GET {url} did not return JSON") from error

    data = payload.get("data") if isinstance(payload, dict) else None
    models = [item for item in data or [] if isinstance(item, dict)]
    model_ids = sorted(item["id"] for item in models if isinstance(item.get("id"), str))
    matching = [item for item in models if item.get("id") == served_model]
    if len(matching) != 1:
        raise RuntimeError(
            f"GET {url} must advertise {served_model!r} exactly once; found {model_ids}"
        )
    advertised = matching[0]
    if advertised.get("context_window") != serving_context:
        raise RuntimeError(
            f"GET {url} advertised context_window={advertised.get('context_window')!r}; "
            f"expected {serving_context}"
        )

    return {
        "checked_at": utc_now(),
        "url": url,
        "http_status": status,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "model_ids": model_ids,
        "advertised_model": advertised,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }


def runtime_binding(
    path: Path,
    label: str,
    campaign_phase: str,
    api_base: str,
    source_metadata: Path,
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    deployment = json.loads(resolved.read_text())
    preliminary = make_wrapper(
        deployment,
        evaluator=None,
        variant=label,
        campaign_phase=campaign_phase,
        endpoint=validate_api_base(api_base),
    )
    campaign_source = verify_source_provenance(
        source_metadata,
        source_root,
        preliminary["content"]["deployment"]["recipe"]["source_commit"],
    )
    return (
        make_wrapper(
            deployment,
            evaluator={"campaign_source": campaign_source},
            variant=label,
            campaign_phase=campaign_phase,
            endpoint=validate_api_base(api_base),
        ),
        campaign_source,
    )


def harbor_environment(harbor_source: Path) -> dict[str, Any]:
    subprocess.run(
        [
            "uv",
            "sync",
            "--directory",
            str(harbor_source),
            "--frozen",
            "--no-dev",
            "--check",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    python = harbor_source / ".venv/bin/python"
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata as m,json,sys;"
                "p=sorted((d.metadata.get('Name') or '',d.version) for d in m.distributions());"
                "print(json.dumps({'python':sys.version.split()[0],'packages':p},"
                "sort_keys=True,separators=(',',':')))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    environment = json.loads(result.stdout)
    packages = environment.get("packages")
    python_version = environment.get("python")
    if (
        not isinstance(packages, list)
        or not packages
        or not all(
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(value, str) and value for value in item)
            for item in packages
        )
        or not isinstance(python_version, str)
        or not python_version
    ):
        raise RuntimeError("Harbor environment package inventory is invalid")
    normalized_names = [re.sub(r"[-_.]+", "-", item[0]).casefold() for item in packages]
    if len(normalized_names) != len(set(normalized_names)):
        raise RuntimeError("Harbor environment has duplicate normalized package names")
    packages.sort(
        key=lambda item: (re.sub(r"[-_.]+", "-", item[0]).casefold(), item[0], item[1])
    )
    payload = json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
    return {
        "uv_sync_check": "passed",
        "python": python_version,
        "package_count": len(packages),
        "packages_sha256": hashlib.sha256(payload).hexdigest(),
        "packages": packages,
    }


def validate_resume_metadata(
    metadata: dict[str, Any],
    run_spec: dict[str, Any],
    binding: dict[str, Any],
    campaign_source: dict[str, Any],
    harbor_environment_identity: dict[str, Any],
) -> None:
    if (
        isinstance(metadata.get("schema_version"), bool)
        or metadata.get("schema_version") != 2
    ):
        raise RuntimeError("Refusing to resume: run metadata must use schema version 2")
    if metadata.get("run_spec") != run_spec:
        raise RuntimeError("Refusing to resume: run arguments differ from metadata")
    if metadata.get("runtime_binding") != binding:
        raise RuntimeError("Refusing to resume: runtime binding differs from metadata")
    if metadata.get("campaign_source") != campaign_source:
        raise RuntimeError(
            "Refusing to resume: campaign evaluator source differs from metadata"
        )
    if metadata.get("harbor_environment") != harbor_environment_identity:
        raise RuntimeError(
            "Refusing to resume: Harbor environment differs from metadata"
        )


def start(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    harbor_source = args.harbor_source.resolve()
    harbor_head = command_output(["git", "-C", str(harbor_source), "rev-parse", "HEAD"])
    if harbor_head != args.harbor_commit:
        raise RuntimeError(
            f"Harbor commit mismatch: expected {args.harbor_commit}, found {harbor_head}"
        )
    harbor_remote = command_output(
        ["git", "-C", str(harbor_source), "remote", "get-url", "origin"]
    )
    if harbor_remote != args.harbor_repository:
        raise RuntimeError(
            f"Harbor remote mismatch: expected {args.harbor_repository}, "
            f"found {harbor_remote!r}"
        )
    harbor_status = command_output(
        [
            "git",
            "-C",
            str(harbor_source),
            "status",
            "--short",
            "--untracked-files=all",
        ]
    )
    if harbor_status is None:
        raise RuntimeError("Could not inspect the Harbor source checkout")
    if harbor_status:
        raise RuntimeError("Harbor source checkout is not clean")
    harbor_version_output = command_output([str(args.harbor_bin), "--version"])
    if (
        harbor_version_output is None
        or args.harbor_version not in harbor_version_output
    ):
        raise RuntimeError(
            f"Harbor version mismatch: expected {args.harbor_version}, "
            f"found {harbor_version_output!r}"
        )

    resolved_dataset = json.loads(args.dataset_metadata.read_text())
    if resolved_dataset.get("content_hash") != args.dataset_content_hash:
        raise RuntimeError("Resolved dataset metadata has the wrong content hash")
    if resolved_dataset.get("dataset_version_id") != args.dataset_version_id:
        raise RuntimeError("Resolved dataset metadata has the wrong version ID")

    binding, campaign_source = runtime_binding(
        args.runtime_binding,
        args.label,
        args.campaign_phase,
        args.api_base,
        args.source_metadata,
        args.source_root,
    )
    run_spec = {
        "mode": args.mode,
        "label": args.label,
        "campaign_phase": args.campaign_phase,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "dataset_content_hash": args.dataset_content_hash,
        "dataset_version_id": args.dataset_version_id,
        "expected_tasks": args.expected_tasks,
        "attempts_per_task": args.attempts,
        "expected_trials": args.expected_tasks * args.attempts,
        "agent": args.agent,
        "litellm_model": args.model,
        "served_model": args.served_model,
        "api_base": validate_api_base(args.api_base),
        "n_concurrent": args.n_concurrent,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_turns": args.max_turns,
        "max_context_tokens": args.max_context,
        "max_output_tokens": args.max_output,
        "timeout_multiplier": args.timeout_multiplier,
        "job_name": args.job_name,
        "job_dir": str(args.job_dir.resolve()),
        "runtime_deployment_sha256": binding["deployment_sha256"],
    }
    endpoint = endpoint_models(args.api_base, args.served_model, args.serving_context)
    harbor_environment_identity = harbor_environment(harbor_source)
    invocation = {
        "started_at": utc_now(),
        "resume": args.resume,
        "command": args.command,
        "endpoint_preflight": endpoint,
    }

    if output.exists():
        if not args.resume:
            raise FileExistsError(f"Metadata already exists: {output}")
        metadata = json.loads(output.read_text())
        validate_resume_metadata(
            metadata,
            run_spec,
            binding,
            campaign_source,
            harbor_environment_identity,
        )
        metadata.setdefault("invocations", []).append(invocation)
        atomic_write_json(output, metadata)
        return 0

    source = json.loads(args.source_metadata.read_text())
    metadata = {
        "schema_version": 2,
        "created_at": utc_now(),
        "run_spec": run_spec,
        "pins": {
            "harbor_repository": args.harbor_repository,
            "harbor_version": args.harbor_version,
            "harbor_commit": args.harbor_commit,
            "harbor_uv_lock_sha256": sha256_file(harbor_source / "uv.lock"),
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "dataset_content_hash": args.dataset_content_hash,
            "dataset_version_id": args.dataset_version_id,
            "resolved_dataset": resolved_dataset,
        },
        "source": source,
        "campaign_source": campaign_source,
        "runtime_binding": binding,
        "harbor_environment": harbor_environment_identity,
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "docker_version": command_output(
                ["docker", "version", "--format", "{{.Server.Version}}"]
            ),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv,noheader",
                ]
            ),
        },
        "invocations": [invocation],
    }
    atomic_write_json(output, metadata)
    return 0


def finish(args: argparse.Namespace) -> int:
    path = args.metadata.resolve()
    metadata = json.loads(path.read_text())
    if (
        isinstance(metadata.get("schema_version"), bool)
        or metadata.get("schema_version") != 2
    ):
        raise RuntimeError("Run metadata must use schema version 2")
    invocations = metadata.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise RuntimeError(f"No invocation found in {path}")

    job_dir = args.job_dir.resolve()
    trial_results = sorted(
        result
        for result in job_dir.glob("*/result.json")
        if result.parent.name != "summary"
    )
    invocations[-1]["finished_at"] = utc_now()
    invocations[-1]["harbor_exit_code"] = args.exit_code
    invocations[-1]["elapsed_seconds"] = args.elapsed_seconds
    invocations[-1]["observed_trial_result_files"] = len(trial_results)
    invocations[-1]["job_result_sha256"] = sha256_file(job_dir / "result.json")
    invocations[-1]["job_lock_sha256"] = sha256_file(job_dir / "lock.json")
    atomic_write_json(path, metadata)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)

    start_parser = commands.add_parser("start")
    start_parser.set_defaults(function=start)
    start_parser.add_argument("--output", type=Path, required=True)
    start_parser.add_argument("--repo-root", type=Path, required=True)
    start_parser.add_argument("--source-metadata", type=Path, required=True)
    start_parser.add_argument("--source-root", type=Path, required=True)
    start_parser.add_argument("--harbor-source", type=Path, required=True)
    start_parser.add_argument("--harbor-bin", type=Path, required=True)
    start_parser.add_argument(
        "--harbor-repository",
        default="https://github.com/laude-institute/harbor.git",
    )
    start_parser.add_argument("--harbor-version", required=True)
    start_parser.add_argument("--harbor-commit", required=True)
    start_parser.add_argument("--dataset", required=True)
    start_parser.add_argument("--dataset-revision", type=int, required=True)
    start_parser.add_argument("--dataset-content-hash", required=True)
    start_parser.add_argument("--dataset-version-id", required=True)
    start_parser.add_argument("--dataset-metadata", type=Path, required=True)
    start_parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    start_parser.add_argument("--label", required=True)
    start_parser.add_argument(
        "--campaign-phase", choices=("validation", "ab", "ba"), required=True
    )
    start_parser.add_argument("--api-base", required=True)
    start_parser.add_argument("--runtime-binding", type=Path, required=True)
    start_parser.add_argument("--serving-context", type=int, required=True)
    start_parser.add_argument("--served-model", required=True)
    start_parser.add_argument("--model", required=True)
    start_parser.add_argument("--agent", required=True)
    start_parser.add_argument("--expected-tasks", type=int, required=True)
    start_parser.add_argument("--attempts", type=int, required=True)
    start_parser.add_argument("--n-concurrent", type=int, required=True)
    start_parser.add_argument("--temperature", type=float, required=True)
    start_parser.add_argument("--top-p", type=float, required=True)
    start_parser.add_argument("--max-turns", type=int, required=True)
    start_parser.add_argument("--max-context", type=int, required=True)
    start_parser.add_argument("--max-output", type=int, required=True)
    start_parser.add_argument("--timeout-multiplier", type=float, required=True)
    start_parser.add_argument("--job-name", required=True)
    start_parser.add_argument("--job-dir", type=Path, required=True)
    start_parser.add_argument("--resume", action="store_true")
    start_parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)

    finish_parser = commands.add_parser("finish")
    finish_parser.set_defaults(function=finish)
    finish_parser.add_argument("--metadata", type=Path, required=True)
    finish_parser.add_argument("--job-dir", type=Path, required=True)
    finish_parser.add_argument("--exit-code", type=int, required=True)
    finish_parser.add_argument("--elapsed-seconds", type=int, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.function(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"metadata error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
