#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan and run a bounded agent-loadgen campaign with reproducibility metadata."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit


PINNED_LOADGEN_REPOSITORY = "https://github.com/NVIDIA-dev/agent-loadgen"
PINNED_LOADGEN_COMMIT = "9057201e23663baaaf076820f3772d55468dec25"
DEFAULT_PROFILE = (
    Path(__file__).resolve().parent / "profiles" / "codex-causal-smoke.toml"
)
HEADER_NAME_PATTERN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
CAMPAIGN_ID_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,79}$")
DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WRAPPER_SCHEMA_VERSION = 2


class CampaignError(RuntimeError):
    """Report an expected validation or campaign execution failure."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and run a pinned agent-loadgen campaign against a Dynamo recipe endpoint."
    )
    parser.add_argument(
        "--loadgen",
        type=Path,
        required=True,
        help="Path to the pinned agent-loadgen executable.",
    )
    parser.add_argument(
        "--loadgen-source",
        type=Path,
        required=True,
        help="Path to the clean agent-loadgen source checkout used to build the executable.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DYNAMO_BASE_URL"),
        help="Root OpenAI-compatible URL from the deployed recipe.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DYNAMO_MODEL"),
        help="Served model name from the deployed recipe.",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.environ.get("LOADGEN_TOKENIZER"),
        help="Local tokenizer path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Versioned agent-loadgen generator profile.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("agent-loadgen-artifacts"),
        help="Parent directory for new campaign directories.",
    )
    parser.add_argument(
        "--campaign-id",
        help="Optional unique output-directory name; generated when omitted.",
    )
    parser.add_argument(
        "--intent",
        choices=("transport-smoke", "performance-measurement"),
        default="transport-smoke",
        help="Evidence intent. A performance measurement still requires later campaign and telemetry qualification.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Static request header. Repeat for multiple headers.",
    )
    parser.add_argument(
        "--token-path-verified",
        action="store_true",
        help="Declare that supplied token IDs reach the inference engine without re-tokenization.",
    )
    parser.add_argument(
        "--engine-cache-mode",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Declare a verified engine-cache setting. Repeat for multiple settings.",
    )
    parser.add_argument(
        "--max-planned-requests",
        type=int,
        default=4,
        help="Abort before traffic if the planned graph exceeds this request count.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=2,
        help="Maximum simultaneous HTTP requests.",
    )
    parser.add_argument(
        "--warmup-connections",
        type=int,
        default=1,
        help="Connections prepared through /v1/models before traffic.",
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=120, help="Per-request HTTP timeout."
    )
    parser.add_argument(
        "--http-transport",
        choices=("auto", "http1", "http2-prior-knowledge"),
        default="auto",
        help="Transport passed to agent-loadgen.",
    )
    return parser


def _required_text(value: str | None, label: str) -> str:
    if value is None or not value.strip():
        raise CampaignError(f"{label} is required")
    result = value.strip()
    if any(character in result for character in ("\x00", "\r", "\n")):
        raise CampaignError(f"{label} contains a forbidden control character")
    return result


def _normalize_base_url(value: str | None) -> str:
    base_url = _required_text(value, "--base-url or DYNAMO_BASE_URL")
    try:
        parsed = urlsplit(base_url)
        parsed_port = parsed.port
    except ValueError as error:
        raise CampaignError(f"invalid base URL: {error}") from error
    if parsed.scheme not in ("http", "https") or parsed.hostname is None:
        raise CampaignError("base URL must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise CampaignError(
            "base URL must not contain credentials; pass authentication with --header"
        )
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise CampaignError(
            "base URL must be the root service URL, not /v1 or /v1/chat/completions"
        )
    netloc = parsed.hostname
    if ":" in netloc and not netloc.startswith("["):
        netloc = f"[{netloc}]"
    if parsed_port is not None:
        netloc = f"{netloc}:{parsed_port}"
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _parse_key_values(
    values: Sequence[str], label: str, *, redact_values: bool
) -> tuple[list[tuple[str, str]], list[str]]:
    parsed_values: list[tuple[str, str]] = []
    secret_values: list[str] = []
    seen_names: set[str] = set()
    for item in values:
        if "=" not in item:
            raise CampaignError(f"{label} must use NAME=VALUE")
        name, value = item.split("=", 1)
        if not HEADER_NAME_PATTERN.fullmatch(name):
            raise CampaignError(f"{label} has an invalid name: {name!r}")
        if not value or any(character in value for character in ("\x00", "\r", "\n")):
            raise CampaignError(f"{label} {name!r} has an empty or invalid value")
        normalized_name = name.lower()
        if normalized_name in seen_names:
            raise CampaignError(f"{label} repeats {name!r}")
        seen_names.add(normalized_name)
        parsed_values.append((name, value))
        if redact_values:
            secret_values.append(value)
    return parsed_values, secret_values


def _validate_positive(value: int, label: str) -> int:
    if value <= 0:
        raise CampaignError(f"{label} must be greater than zero")
    return value


def _validate_campaign_id(value: str | None) -> str:
    if value is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"
    if value in (".", "..") or CAMPAIGN_ID_PATTERN.fullmatch(value) is None:
        raise CampaignError(
            "--campaign-id must be 1-80 characters using letters, digits, dot, underscore, or hyphen"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(source: Path, arguments: Sequence[str]) -> str:
    command = ["git", "-C", str(source), *arguments]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        raise CampaignError(f"failed to execute git: {error}") from error
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit code {result.returncode}"
        )
        raise CampaignError(f"failed to inspect agent-loadgen source: {detail}")
    return result.stdout.strip()


def _loadgen_source_state(source: Path) -> tuple[str, bool]:
    revision = _run_git(source, ("rev-parse", "HEAD"))
    dirty = bool(
        _run_git(source, ("status", "--porcelain", "--untracked-files=normal"))
    )
    return revision, dirty


def _loadgen_version(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"], check=False, capture_output=True, text=True
        )
    except OSError as error:
        raise CampaignError(
            f"failed to execute agent-loadgen --version: {error}"
        ) from error
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit code {result.returncode}"
        )
        raise CampaignError(f"failed to execute agent-loadgen --version: {detail}")
    version = result.stdout.strip()
    if not version:
        raise CampaignError("agent-loadgen --version returned no version")
    return version


def _redact_text(value: str, secret_values: Sequence[str]) -> str:
    redacted = value
    for secret in sorted(set(secret_values), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _redacted_command(
    command: Sequence[str], secret_values: Sequence[str]
) -> list[str]:
    return [_redact_text(argument, secret_values) for argument in command]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(value, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def _run_stage(
    stage: str, command: Sequence[str], campaign_dir: Path, secret_values: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        redacted_error = _redact_text(str(error), secret_values)
        (campaign_dir / f"{stage}.stdout.log").write_text("", encoding="utf-8")
        (campaign_dir / f"{stage}.stderr.log").write_text(
            f"{redacted_error}\n", encoding="utf-8"
        )
        raise CampaignError(
            f"failed to execute agent-loadgen {stage}: {redacted_error}"
        ) from error
    (campaign_dir / f"{stage}.stdout.log").write_text(
        _redact_text(result.stdout, secret_values), encoding="utf-8"
    )
    (campaign_dir / f"{stage}.stderr.log").write_text(
        _redact_text(result.stderr, secret_values), encoding="utf-8"
    )
    if result.returncode != 0:
        raise CampaignError(
            f"agent-loadgen {stage} failed with exit code {result.returncode}; inspect {stage}.stderr.log"
        )
    return result


def _parse_json_output(value: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise CampaignError(f"{label} was not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return parsed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            parsed = json.load(source)
    except FileNotFoundError as error:
        raise CampaignError(f"{label} was not created at {path}") from error
    except json.JSONDecodeError as error:
        raise CampaignError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return parsed


def _classification(intent: str, run_summary: dict[str, Any]) -> dict[str, Any]:
    transport_passed = run_summary.get("passed") is True
    loadgen_performance_eligible = (
        intent == "performance-measurement"
        and transport_passed
        and run_summary.get("capacity_performance_conclusions_allowed") is True
    )
    return {
        "agent_loadgen_performance_eligible": loadgen_performance_eligible,
        "intent": intent,
        "performance_qualified": False,
        "performance_qualification_blocker": "Complete pinned Router Zoo campaigns, matched no-affinity/affinity/ThunderAgent treatments, and correlated router/engine/GPU telemetry before making performance claims.",
        "transport_passed": transport_passed,
    }


def run_campaign(
    argv: Sequence[str] | None = None, *, expected_commit: str = PINNED_LOADGEN_COMMIT
) -> Path:
    args = _build_parser().parse_args(argv)
    loadgen = args.loadgen.resolve()
    loadgen_source = args.loadgen_source.resolve()
    profile_path = args.profile.resolve()
    output_root = args.output_root.resolve()
    base_url = _normalize_base_url(args.base_url)
    model = _required_text(args.model, "--model or DYNAMO_MODEL")
    tokenizer = _required_text(args.tokenizer, "--tokenizer or LOADGEN_TOKENIZER")
    headers, header_secrets = _parse_key_values(
        args.header, "--header", redact_values=True
    )
    engine_cache_modes, _ = _parse_key_values(
        args.engine_cache_mode, "--engine-cache-mode", redact_values=False
    )
    max_planned_requests = _validate_positive(
        args.max_planned_requests, "--max-planned-requests"
    )
    max_in_flight = _validate_positive(args.max_in_flight, "--max-in-flight")
    warmup_connections = _validate_positive(
        args.warmup_connections, "--warmup-connections"
    )
    timeout_seconds = _validate_positive(args.timeout_seconds, "--timeout-seconds")
    campaign_id = _validate_campaign_id(args.campaign_id)
    campaign_dir = output_root / campaign_id

    if not loadgen.is_file() or not os.access(loadgen, os.X_OK):
        raise CampaignError(
            f"agent-loadgen executable is missing or not executable: {loadgen}"
        )
    if not loadgen_source.is_dir():
        raise CampaignError(
            f"agent-loadgen source directory does not exist: {loadgen_source}"
        )
    if not profile_path.is_file():
        raise CampaignError(f"profile does not exist: {profile_path}")
    if campaign_dir.exists():
        raise CampaignError(
            f"campaign output already exists and will not be overwritten: {campaign_dir}"
        )
    if args.intent == "performance-measurement" and not args.token_path_verified:
        raise CampaignError(
            "--intent performance-measurement requires --token-path-verified"
        )
    if args.intent == "performance-measurement" and not engine_cache_modes:
        raise CampaignError(
            "--intent performance-measurement requires at least one --engine-cache-mode"
        )

    actual_commit, source_dirty = _loadgen_source_state(loadgen_source)
    if actual_commit != expected_commit:
        raise CampaignError(
            f"agent-loadgen source is {actual_commit}, expected pinned commit {expected_commit}"
        )
    if source_dirty:
        raise CampaignError(
            "agent-loadgen source checkout is dirty; build and run from a clean pinned checkout"
        )
    version = _loadgen_version(loadgen)

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        campaign_dir.mkdir()
    except FileExistsError as error:
        raise CampaignError(
            f"campaign output already exists and will not be overwritten: {campaign_dir}"
        ) from error

    plan_dir = campaign_dir / "plan"
    run_dir = campaign_dir / "run"
    metadata_path = campaign_dir / "campaign.json"
    profile_file_sha256 = _sha256(profile_path)
    metadata: dict[str, Any] = {
        "campaign_id": campaign_id,
        "classification": {
            "agent_loadgen_performance_eligible": False,
            "intent": args.intent,
            "performance_qualified": False,
            "transport_passed": False,
        },
        "commands": [],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "profile": {
            "file_sha256": profile_file_sha256,
            "path": str(profile_path),
        },
        "schema_version": WRAPPER_SCHEMA_VERSION,
        "software": {
            "agent_loadgen": {
                "binary_path": str(loadgen),
                "binary_sha256": _sha256(loadgen),
                "expected_commit": expected_commit,
                "repository": PINNED_LOADGEN_REPOSITORY,
                "source_commit": actual_commit,
                "source_dirty": source_dirty,
                "version": version,
            },
            "python": platform.python_version(),
            "wrapper_path": str(Path(__file__).resolve()),
        },
        "status": "running",
        "target": {
            "base_url": base_url,
            "model": model,
            "static_header_names": [name for name, _ in headers],
        },
    }
    _write_json(metadata_path, metadata)

    plan_command = [
        str(loadgen),
        "plan",
        "--config",
        str(profile_path),
        "--output",
        str(plan_dir),
    ]
    generate_command = [
        str(loadgen),
        "generate",
        "--config",
        str(profile_path),
        "--model",
        model,
        "--target",
        base_url,
        "--output",
        str(run_dir),
        "--tokenizer",
        tokenizer,
        "--max-in-flight",
        str(max_in_flight),
        "--warmup-connections",
        str(warmup_connections),
        "--timeout-seconds",
        str(timeout_seconds),
        "--http-transport",
        args.http_transport,
    ]
    if args.token_path_verified:
        generate_command.append("--token-path-verified")
    for name, value in engine_cache_modes:
        generate_command.extend(("--engine-cache-mode", f"{name}={value}"))
    for name, value in headers:
        generate_command.extend(("--header", f"{name}={value}"))
    metadata["commands"] = [
        {"argv": _redacted_command(plan_command, header_secrets), "stage": "plan"},
        {
            "argv": _redacted_command(generate_command, header_secrets),
            "stage": "generate",
        },
    ]
    _write_json(metadata_path, metadata)

    try:
        plan_result = _run_stage("plan", plan_command, campaign_dir, header_secrets)
        plan_summary = _parse_json_output(
            plan_result.stdout, "agent-loadgen plan output"
        )
        planned_requests = plan_summary.get("requests")
        if (
            not isinstance(planned_requests, int)
            or isinstance(planned_requests, bool)
            or planned_requests <= 0
        ):
            raise CampaignError(
                "agent-loadgen plan output has no positive integer request count"
            )
        if planned_requests > max_planned_requests:
            raise CampaignError(
                f"planned request count {planned_requests} exceeds --max-planned-requests {max_planned_requests}"
            )
        profile_digest = plan_summary.get("profile_digest_sha256")
        if (
            not isinstance(profile_digest, str)
            or DIGEST_PATTERN.fullmatch(profile_digest) is None
        ):
            raise CampaignError(
                "agent-loadgen plan output has no valid semantic profile SHA-256 digest"
            )
        scenario_digest = plan_summary.get("scenario_digest_sha256")
        if (
            not isinstance(scenario_digest, str)
            or DIGEST_PATTERN.fullmatch(scenario_digest) is None
        ):
            raise CampaignError(
                "agent-loadgen plan output has no valid scenario SHA-256 digest"
            )
        planned_scenario = _read_json(
            plan_dir / "scenario.json", "agent-loadgen planned scenario"
        )
        if planned_scenario.get("profile_digest_sha256") != profile_digest:
            raise CampaignError(
                "agent-loadgen planned scenario profile digest does not match plan output"
            )
        if planned_scenario.get("scenario_digest_sha256") != scenario_digest:
            raise CampaignError(
                "agent-loadgen planned scenario digest does not match plan output"
            )
        trace_manifest = planned_scenario.get("trace_manifest")
        if (
            not isinstance(trace_manifest, dict)
            or trace_manifest.get("source_digest_sha256") != profile_digest
        ):
            raise CampaignError(
                "agent-loadgen planned trace source digest does not match the semantic profile digest"
            )
        metadata["profile"]["semantic_digest_sha256"] = profile_digest
        metadata["plan"] = {
            "profile_digest_sha256": profile_digest,
            "requests": planned_requests,
            "scenario_digest_sha256": scenario_digest,
        }
        metadata["status"] = "planned"
        _write_json(metadata_path, metadata)

        _run_stage("generate", generate_command, campaign_dir, header_secrets)
        run_summary = _read_json(run_dir / "run.json", "agent-loadgen run summary")
        if run_summary.get("protocol_surface") != "chat_completions":
            raise CampaignError(
                "agent-loadgen run summary did not report the Chat Completions protocol surface"
            )
        if run_summary.get("passed") is not True:
            raise CampaignError(
                "agent-loadgen run summary did not pass request and dispatch checks"
            )
        if run_summary.get("request_count") != planned_requests:
            raise CampaignError(
                "agent-loadgen completed request count does not match the planned request count"
            )
        metadata["classification"] = _classification(args.intent, run_summary)
        metadata["run"] = {
            "capacity_performance_conclusions_allowed": run_summary.get(
                "capacity_performance_conclusions_allowed"
            ),
            "conclusion_blockers": run_summary.get("conclusion_blockers"),
            "passed": run_summary.get("passed"),
            "protocol_surface": run_summary.get("protocol_surface"),
            "request_count": run_summary.get("request_count"),
            "run_id": run_summary.get("run_id"),
        }
        metadata["status"] = "completed"
        _write_json(metadata_path, metadata)
    except CampaignError as error:
        metadata["error"] = _redact_text(str(error), header_secrets)
        metadata["status"] = "failed"
        _write_json(metadata_path, metadata)
        raise

    return campaign_dir


def main(argv: Sequence[str] | None = None) -> int:
    try:
        campaign_dir = run_campaign(argv)
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
