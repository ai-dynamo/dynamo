#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create or validate the immutable identity of a SWE-bench run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from runtime_binding import (  # noqa: E402
    canonical_sha256,
    make_wrapper,
    validate_deployment,
)
from source_provenance import verify_source_provenance  # noqa: E402
from verify_environment_lock import verify as verify_environment_lock  # noqa: E402


SWE_PIN_PREFIXES = ("MINI_SWE_", "SWEBENCH_")
SUITE_PIN_KEYS = {
    "verified": ("SWEBENCH_VERIFIED_REVISION", "SWEBENCH_VERIFIED_CASES"),
    "multilingual": (
        "SWEBENCH_MULTILINGUAL_REVISION",
        "SWEBENCH_MULTILINGUAL_CASES",
    ),
    "pro": ("SWEBENCH_PRO_REVISION", "SWEBENCH_PRO_PUBLIC_CASES"),
}
SOURCE_PIN_KEYS = {
    "mini_swe_agent": "MINI_SWE_AGENT_COMMIT",
    "swebench": "SWEBENCH_COMMIT",
    "swebench_pro": "SWEBENCH_PRO_COMMIT",
}
STACK_VARIANTS = {
    "dynamo-vllm": ("vllm", True),
    "vllm-serve": ("vllm", False),
    "dynamo-sglang": ("sglang", True),
    "sglang-serve": ("sglang", False),
}
EXPECTED_CONTEXT_WINDOW = 409600
EXPECTED_MAX_TOKENS = 32768


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def parse_named_path(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise ValueError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(raw_path)


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "endpoint must not contain credentials, a query, or a fragment"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def load_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"{path}:{line_number}: invalid pin assignment")
        if name.startswith(SWE_PIN_PREFIXES):
            pins[name] = value
    if not pins:
        raise ValueError(f"no SWE pins found in {path}")
    return dict(sorted(pins.items()))


def git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_identity(specs: list[str]) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for spec in specs:
        name, repo = parse_named_path(spec)
        if name in identities:
            raise ValueError(f"duplicate source repository name: {name}")
        head = git_output(repo, "rev-parse", "HEAD")
        dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            raise ValueError(f"managed source repository is not clean: {repo}")
        identities[name] = {"commit": head}
    return dict(sorted(identities.items()))


def config_identity(specs: list[str]) -> dict[str, Any]:
    files = []
    names: set[str] = set()
    for spec in specs:
        name, path = parse_named_path(spec)
        if name in names:
            raise ValueError(f"duplicate config name: {name}")
        names.add(name)
        files.append({"name": name, "sha256": sha256(path)})
    if not files:
        raise ValueError("at least one config file is required")
    return {"files": files, "sha256": canonical_sha256(files)}


def runtime_binding_payload(binding: dict[str, Any]) -> bytes:
    return (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode()


def validate_effective_config(
    config: dict[str, Any], endpoint: str, model: str, suite: str
) -> None:
    expected_cwd = "/app" if suite == "pro" else "/testbed"
    agent = config.get("agent")
    environment = config.get("environment")
    model_config = config.get("model")
    if not all(isinstance(value, dict) for value in (agent, environment, model_config)):
        raise ValueError(
            "effective config must contain agent/environment/model objects"
        )
    expected_agent = {
        "wall_time_limit_seconds": 14400,
        "step_limit": 250,
        "cost_limit": 3.0,
    }
    for key, expected in expected_agent.items():
        if agent.get(key) != expected:
            raise ValueError(f"effective agent.{key} is not {expected!r}")
    expected_environment = {
        "cwd": expected_cwd,
        "timeout": 900,
        "pull_timeout": 1800,
        "container_timeout": "4h",
        "interpreter": ["bash", "-c"],
        "run_args": ["--rm", "--entrypoint="] if suite == "pro" else ["--rm"],
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"effective environment.{key} is not {expected!r}")
    if model_config.get("model_name") != f"openai/{model}":
        raise ValueError("effective model.model_name does not match served model")
    if model_config.get("cost_tracking") != "ignore_errors":
        raise ValueError("effective model.cost_tracking must be ignore_errors")
    kwargs = model_config.get("model_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("effective model.model_kwargs must be an object")
    expected_kwargs = {
        "api_base": normalize_endpoint(endpoint),
        "custom_llm_provider": "openai",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": EXPECTED_MAX_TOKENS,
        "timeout": 1800,
        "drop_params": True,
        "parallel_tool_calls": True,
    }
    for key, expected in expected_kwargs.items():
        if kwargs.get(key) != expected:
            raise ValueError(f"effective model.model_kwargs.{key} is not {expected!r}")


def build_runtime_binding(
    args: argparse.Namespace, endpoint: str, model: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_family, expected_dynamo = STACK_VARIANTS[args.variant]
    if args.runtime_family != expected_family:
        raise ValueError(
            f"variant {args.variant} requires runtime family {expected_family}"
        )
    if args.dynamo_enabled != expected_dynamo:
        raise ValueError(
            f"variant {args.variant} requires dynamo_enabled={expected_dynamo}"
        )
    if args.context_window != EXPECTED_CONTEXT_WINDOW:
        raise ValueError(
            f"serving context must be {EXPECTED_CONTEXT_WINDOW}, got {args.context_window}"
        )
    if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", args.runtime_image):
        raise ValueError("runtime image must be pinned by sha256 digest")
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("model revision must be a 40-character commit")
    if not re.fullmatch(r"[0-9a-f]{40}", args.runtime_source_revision):
        raise ValueError("runtime source revision must be a 40-character commit")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor parallel size must be positive")
    deployment = validate_deployment(
        load_json_object(args.deployment_binding, "published deployment binding"),
        variant=args.variant,
        campaign_phase=args.campaign_phase,
        endpoint=normalize_endpoint(endpoint),
    )
    expected_deployment_fields = {
        "served_model_name": model,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "max_model_len": args.context_window,
        "image": args.runtime_image,
    }
    for field, expected in expected_deployment_fields.items():
        if deployment[field] != expected:
            raise ValueError(
                f"published deployment {field} differs from requested evaluator "
                f"identity: expected {expected!r}, got {deployment[field]!r}"
            )
    campaign_source = verify_source_provenance(
        args.campaign_source_metadata,
        args.campaign_source_root,
        deployment["recipe"]["source_commit"],
    )
    effective_config = load_json_object(args.effective_config, "effective config")
    validate_effective_config(effective_config, endpoint, model, args.suite)
    endpoint_evidence = load_json_object(args.endpoint_evidence, "endpoint evidence")
    if endpoint_evidence.get("requested_model") != model:
        raise ValueError("endpoint evidence model does not match run model")
    if endpoint_evidence.get("expected_context_window") != args.context_window:
        raise ValueError("endpoint evidence context does not match runtime context")
    selected_model = endpoint_evidence.get("selected_model_response")
    if not isinstance(selected_model, dict) or selected_model.get("id") != model:
        raise ValueError("endpoint evidence has no exact selected model response")
    if selected_model.get("context_window") != args.context_window:
        raise ValueError("endpoint selected model context_window drifted")
    evaluator = {
        "deployment_source_sha256": canonical_sha256(deployment),
        "runtime_family": args.runtime_family,
        "runtime_source_revision": args.runtime_source_revision,
        "dynamo_enabled": args.dynamo_enabled,
        "tensor_parallel_size": args.tensor_parallel_size,
        "generation": {
            "workers": args.generation_workers,
            "batch_size": args.generation_batch_size,
        },
        "evaluation": {
            "workers": args.evaluator_workers,
            "timeout_seconds": args.evaluator_timeout,
            "backend": (
                args.pro_eval_backend
                if args.suite == "pro"
                else "official-swebench-docker"
            ),
            "docker_platform": args.docker_platform if args.suite == "pro" else None,
        },
        "effective_config_sha256": sha256(args.effective_config),
        "effective_config_content_sha256": canonical_sha256(effective_config),
        "effective_config": effective_config,
        "endpoint_evidence": {
            "file_sha256": sha256(args.endpoint_evidence),
            "content_sha256": canonical_sha256(endpoint_evidence),
            "content": endpoint_evidence,
        },
        "campaign_source": campaign_source,
    }
    return (
        make_wrapper(
            deployment,
            evaluator=evaluator,
            variant=args.variant,
            campaign_phase=args.campaign_phase,
            endpoint=normalize_endpoint(endpoint),
        ),
        campaign_source,
    )


def build_identity(args: argparse.Namespace, endpoint: str) -> dict[str, Any]:
    provenance = load_json_object(args.dataset_provenance, "dataset provenance")
    datasets = provenance.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get(args.suite), dict):
        raise ValueError(f"dataset provenance has no {args.suite!r} entry")
    suite_provenance = datasets[args.suite]
    pins = load_pins(args.pins)
    revision_pin, cases_pin = SUITE_PIN_KEYS[args.suite]
    if suite_provenance.get("revision") != pins.get(revision_pin):
        raise ValueError(
            f"{args.suite} provenance revision does not match {revision_pin}"
        )
    try:
        pinned_cases = int(pins[cases_pin])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid or missing case-count pin {cases_pin}") from error
    if suite_provenance.get("expected") != pinned_cases:
        raise ValueError(
            f"{args.suite} provenance expected count does not match {cases_pin}"
        )
    if suite_provenance.get("rows") != pinned_cases:
        raise ValueError(f"{args.suite} provenance row count is not {pinned_cases}")
    source_lock = load_json_object(args.source_lock, "source lock")
    scope = load_json_object(args.scope, "run scope")
    if not re.search(
        rf"(^|[-._]){re.escape(args.campaign_phase)}($|[-._])", args.run_name
    ):
        raise ValueError("run name does not contain its delimited campaign phase")
    if not isinstance(scope.get("full_run"), bool):
        raise ValueError("run scope full_run must be boolean")
    if scope.get("full_run") is True and args.campaign_phase == "validation":
        raise ValueError("full runs require campaign phase ab or ba")
    if scope.get("full_run") is False and args.campaign_phase != "validation":
        raise ValueError("smoke runs require campaign phase validation")
    for field in (
        "generation_workers",
        "generation_batch_size",
        "evaluator_workers",
        "evaluator_timeout",
    ):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field.replace('_', '-')} must be a positive integer")
    if scope.get("full_run") is True:
        if args.generation_workers != 16:
            raise ValueError("full runs require exactly 16 generation workers")
        if args.generation_batch_size != 8:
            raise ValueError("full runs require generation batch size 8")
        if args.evaluator_workers != 8:
            raise ValueError("full runs require exactly 8 evaluator workers")
        if args.evaluator_timeout != 3600:
            raise ValueError("full runs require evaluator timeout 3600")
        if args.suite == "pro" and args.pro_eval_backend != "local":
            raise ValueError("full SWE-bench Pro runs require the local Docker backend")
        if args.suite == "pro" and args.docker_platform != "linux/amd64":
            raise ValueError(
                "full SWE-bench Pro runs require docker platform linux/amd64"
            )
    sources = source_identity(args.source_repo)
    normalized_requirements = verify_environment_lock(
        args.constraints_lock, args.environment_freeze
    )
    expected_normalized = "\n".join(normalized_requirements) + "\n"
    if args.normalized_environment_freeze.read_text() != expected_normalized:
        raise ValueError("persisted normalized environment freeze is not canonical")

    expected_source_commits = {
        "mini_swe_agent": source_lock.get("mini_swe_agent", {}).get("commit"),
        "swebench": source_lock.get("swebench", {}).get("commit"),
        "swebench_pro": source_lock.get("swebench_pro", {}).get("commit"),
    }
    if set(sources) != set(expected_source_commits):
        raise ValueError(
            "source repositories must be exactly "
            f"{sorted(expected_source_commits)}, got {sorted(sources)}"
        )
    for name, identity in sources.items():
        expected = expected_source_commits.get(name)
        if expected is None:
            raise ValueError(f"source lock has no commit for {name}")
        if identity["commit"] != expected:
            raise ValueError(
                f"source commit mismatch for {name}: expected {expected}, "
                f"got {identity['commit']}"
            )
        pin_name = SOURCE_PIN_KEYS[name]
        if expected != pins.get(pin_name):
            raise ValueError(f"source lock commit for {name} does not match {pin_name}")

    runtime_wrapper, campaign_source = build_runtime_binding(args, endpoint, args.model)
    runtime_binding = runtime_wrapper["content"]
    if runtime_wrapper["file"] != args.runtime_binding_output.name:
        raise ValueError(
            "runtime binding output must use canonical filename runtime-binding.json"
        )
    if (
        runtime_wrapper["deployment_sha256"]
        != runtime_binding["evaluator"]["deployment_source_sha256"]
    ):
        raise ValueError(
            "published deployment binding is not canonical or changed during capture"
        )

    return {
        "schema_version": 3,
        "run_name": args.run_name,
        "suite": args.suite,
        "campaign_phase": args.campaign_phase,
        "endpoint": normalize_endpoint(endpoint),
        "model": args.model,
        "scope_sha256": sha256(args.scope),
        "configuration": config_identity(args.config),
        "dataset": {
            "evaluator_jsonl_sha256": sha256(args.dataset),
            "provenance_sha256": sha256(args.dataset_provenance),
            "provenance": suite_provenance,
        },
        "pins": {"sha256": canonical_sha256(pins), "values": pins},
        "source": {
            "lock_sha256": sha256(args.source_lock),
            "lock": source_lock,
            "repositories": sources,
        },
        "python_environment": {
            "constraints_lock_sha256": sha256(args.constraints_lock),
            "freeze_sha256": sha256(args.environment_freeze),
            "normalized_freeze_sha256": sha256(args.normalized_environment_freeze),
            "normalized_requirement_count": len(normalized_requirements),
        },
        "campaign_source": campaign_source,
        "runtime_binding": runtime_wrapper,
    }


def describe_drift(current: Any, requested: Any, prefix: str = "") -> list[str]:
    if isinstance(current, dict) and isinstance(requested, dict):
        differences = []
        for key in sorted(current.keys() | requested.keys()):
            child = f"{prefix}.{key}" if prefix else key
            if key not in current or key not in requested:
                differences.append(child)
            else:
                differences.extend(describe_drift(current[key], requested[key], child))
        return differences
    return [] if current == requested else [prefix]


def write_exclusive(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise SystemExit(f"{label} appeared concurrently: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def prepare(args: argparse.Namespace) -> None:
    existing = None
    if args.output.exists():
        existing = load_json_object(args.output, "run metadata")
    elif args.require_existing:
        raise SystemExit(f"missing immutable run metadata: {args.output}")
    elif args.predictions is not None and args.predictions.exists():
        raise SystemExit(
            "refusing to adopt predictions that predate immutable run metadata; "
            "use a new run-name"
        )

    endpoint = args.endpoint
    model = args.model
    if existing is not None:
        endpoint = endpoint or existing.get("endpoint", "")
        model = model or existing.get("model", "")
    if not endpoint or not model:
        raise SystemExit("endpoint and model are required when creating run metadata")
    args.model = model
    requested = build_identity(args, endpoint)
    binding_payload = runtime_binding_payload(requested["runtime_binding"]["content"])
    if args.runtime_binding_output.exists():
        if args.runtime_binding_output.read_bytes() != binding_payload:
            raise SystemExit(
                "runtime binding differs from immutable run evidence; "
                "use a new run-name"
            )
    elif existing is not None:
        raise SystemExit(
            f"missing runtime binding for existing run: {args.runtime_binding_output}"
        )
    else:
        write_exclusive(args.runtime_binding_output, binding_payload, "runtime binding")

    if existing is not None:
        differences = describe_drift(existing, requested)
        if differences:
            raise SystemExit(
                "run metadata differs from the existing immutable identity; "
                f"use a new run-name (changed: {', '.join(differences)})"
            )
        return

    payload = (json.dumps(requested, indent=2) + "\n").encode()
    write_exclusive(args.output, payload, "run metadata")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prepare", choices=("prepare",))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--suite", required=True, choices=("verified", "multilingual", "pro")
    )
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--variant", required=True, choices=tuple(STACK_VARIANTS))
    parser.add_argument(
        "--campaign-phase", required=True, choices=("validation", "ab", "ba")
    )
    parser.add_argument("--deployment-binding", required=True, type=Path)
    parser.add_argument("--campaign-source-metadata", required=True, type=Path)
    parser.add_argument("--campaign-source-root", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--context-window", required=True, type=int)
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--runtime-family", required=True, choices=("vllm", "sglang"))
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--dynamo-enabled", required=True, type=parse_bool)
    parser.add_argument("--generation-workers", required=True, type=int)
    parser.add_argument("--generation-batch-size", required=True, type=int)
    parser.add_argument("--evaluator-workers", required=True, type=int)
    parser.add_argument("--evaluator-timeout", required=True, type=int)
    parser.add_argument("--pro-eval-backend", required=True, choices=("local", "modal"))
    parser.add_argument("--docker-platform", default="")
    parser.add_argument("--effective-config", required=True, type=Path)
    parser.add_argument("--endpoint-evidence", required=True, type=Path)
    parser.add_argument("--runtime-binding-output", required=True, type=Path)
    parser.add_argument("--config", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-provenance", required=True, type=Path)
    parser.add_argument("--pins", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--constraints-lock", required=True, type=Path)
    parser.add_argument("--environment-freeze", required=True, type=Path)
    parser.add_argument("--normalized-environment-freeze", required=True, type=Path)
    parser.add_argument(
        "--source-repo", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--require-existing", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
