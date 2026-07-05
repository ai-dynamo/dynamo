#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime_binding import make_wrapper  # noqa: E402
from source_provenance import verify_source_provenance  # noqa: E402


EXPECTED_TRACKED_DIFF_SHA256 = (
    "e075c864c9054095956a198475d09e179c7d2aa9d83f76ea51792fa5cf4650c4"
)
EXPECTED_BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
EXPECTED_NEW_HANDLER_SHA256 = (
    "f09a1999ef861c55cfd5a230a15e6536ce2f375b800fd837a5ac7a147b1920c5"
)
EXPECTED_SOURCE_STATUS = [
    " M berkeley-function-call-leaderboard/SUPPORTED_MODELS.md",
    " M berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py",
    " M berkeley-function-call-leaderboard/bfcl_eval/constants/supported_models.py",
    "?? berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/glm52_openai.py",
]
EXPECTED_CONSTRAINTS_SHA256 = (
    "1ae31cfcb689500018f8ce0239dfe9f43e471561c9d0b795b96fad96eaa83f04"
)
EXPECTED_PACKAGE_COUNT = 141
EXPECTED_FREEZE_SHA256 = (
    "829c4dc3b72a4ec6f160fc2cc681070147dc1ea8dbeeee283c92ff3e356287a7"
)
NEW_HANDLER = (
    "berkeley-function-call-leaderboard/bfcl_eval/model_handler/"
    "api_inference/glm52_openai.py"
)


def command(checkout: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(checkout), *arguments])


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_identity(checkout: Path) -> dict[str, object]:
    head = command(checkout, "rev-parse", "HEAD").decode().strip()
    status = (
        command(checkout, "status", "--porcelain=v1", "--untracked-files=all")
        .decode()
        .splitlines()
    )
    tracked_diff = command(checkout, "diff", "--binary", "--no-ext-diff", "HEAD")
    handler = checkout / NEW_HANDLER
    return {
        "head": head,
        "status": status,
        "tracked_diff_sha256": sha256_bytes(tracked_diff),
        "new_handler_sha256": (
            hashlib.sha256(handler.read_bytes()).hexdigest()
            if handler.is_file()
            else None
        ),
    }


def verify_source_identity(identity: dict[str, object], expected_commit: str) -> None:
    expected = {
        "head": expected_commit,
        "status": EXPECTED_SOURCE_STATUS,
        "tracked_diff_sha256": EXPECTED_TRACKED_DIFF_SHA256,
        "new_handler_sha256": EXPECTED_NEW_HANDLER_SHA256,
    }
    if identity != expected:
        raise RuntimeError(
            "BFCL checkout differs from pinned HEAD plus the exact campaign patch:\n"
            + json.dumps({"expected": expected, "actual": identity}, indent=2)
        )


def default_headers_sha256(mode: str, value: str | None) -> str | None:
    if mode == "full" and value is not None:
        raise RuntimeError(
            "Full BFCL baseline forbids GLM52_OPENAI_DEFAULT_HEADERS; "
            "remove endpoint routing/template overrides"
        )
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


def sanitized_url(raw_url: str) -> str:
    parts = urlsplit(raw_url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def validate_campaign_phase(mode: str, phase: str) -> None:
    allowed = {"ab", "ba"} if mode == "full" else {"ab", "ba", "validation"}
    if phase not in allowed:
        raise RuntimeError(
            f"{mode} BFCL campaign phase must be one of {sorted(allowed)}, got {phase!r}"
        )


def runtime_binding(
    path: Path,
    *,
    variant: str,
    campaign_phase: str,
    endpoint: str,
    evaluator: dict[str, object],
    campaign_source_metadata: Path,
    campaign_source_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    deployment = json.loads(path.read_text(encoding="utf-8"))
    recipe = deployment.get("recipe")
    if not isinstance(recipe, dict) or not isinstance(recipe.get("source_commit"), str):
        raise ValueError("runtime binding deployment has no campaign source commit")
    campaign_source = verify_source_provenance(
        campaign_source_metadata,
        campaign_source_root,
        recipe["source_commit"],
    )
    evaluator = {**evaluator, "campaign_source": campaign_source}
    return (
        make_wrapper(
            deployment,
            evaluator=evaluator,
            variant=variant,
            campaign_phase=campaign_phase,
            endpoint=endpoint,
        ),
        campaign_source,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--mode")
    parser.add_argument("--campaign-phase")
    parser.add_argument("--categories")
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--endpoint-models", type=Path)
    parser.add_argument("--runtime-binding", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--campaign-source-metadata", type=Path)
    parser.add_argument("--campaign-source-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    identity = source_identity(args.checkout)
    commit = str(identity["head"])
    verify_source_identity(identity, EXPECTED_BFCL_COMMIT)
    if args.verify_only:
        print(json.dumps(identity, indent=2, sort_keys=True))
        return
    for field in (
        "run_dir",
        "variant",
        "mode",
        "campaign_phase",
        "categories",
        "endpoint_models",
        "runtime_binding",
        "environment_lock",
        "campaign_source_metadata",
        "campaign_source_root",
    ):
        if getattr(args, field) is None:
            parser.error(
                f"--{field.replace('_', '-')} is required without --verify-only"
            )
    try:
        validate_campaign_phase(args.mode, args.campaign_phase)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    environment_lock = json.loads(args.environment_lock.read_text(encoding="utf-8"))
    if set(environment_lock) != {
        "schema_version",
        "constraints_sha256",
        "freeze_sha256",
        "package_count",
        "python",
    }:
        raise SystemExit("BFCL environment lock contains invalid fields")
    if (
        environment_lock.get("schema_version") != 1
        or environment_lock.get("constraints_sha256") != EXPECTED_CONSTRAINTS_SHA256
        or environment_lock.get("package_count") != EXPECTED_PACKAGE_COUNT
        or not isinstance(environment_lock.get("python"), str)
        or not environment_lock["python"]
    ):
        raise SystemExit("BFCL environment lock does not match campaign constraints")
    freeze_sha256 = environment_lock.get("freeze_sha256")
    if freeze_sha256 != EXPECTED_FREEZE_SHA256:
        raise SystemExit("BFCL environment freeze digest is invalid")
    patch_sha256 = hashlib.sha256(args.patch.read_bytes()).hexdigest()
    default_headers = os.getenv("GLM52_OPENAI_DEFAULT_HEADERS")
    try:
        default_headers_digest = default_headers_sha256(args.mode, default_headers)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    endpoint_models_raw = args.endpoint_models.read_bytes()
    endpoint_models = json.loads(endpoint_models_raw)
    matching_models = [
        entry
        for entry in endpoint_models.get("data", [])
        if isinstance(entry, dict) and entry.get("id") == "zai-org/GLM-5.2"
    ]
    if len(matching_models) != 1 or matching_models[0].get("context_window") != 409600:
        raise SystemExit("Endpoint model evidence is missing the 409600-token GLM-5.2")
    endpoint_model = {
        field: matching_models[0].get(field)
        for field in ("id", "object", "owned_by", "context_window")
    }
    endpoint = sanitized_url(os.environ["GLM52_OPENAI_BASE_URL"])
    evaluator_binding = {
        "harness": "bfcl-v4",
        "model_registry_name": os.getenv("BFCL_MODEL", "zai-org/GLM-5.2-FC"),
        "categories": [
            category.strip()
            for category in args.categories.split(",")
            if category.strip()
        ],
        "temperature": float(os.getenv("BFCL_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("BFCL_MAX_TOKENS", "64000")),
        "num_threads": int(os.getenv("BFCL_NUM_THREADS", "1")),
    }
    try:
        binding, campaign_source = runtime_binding(
            args.runtime_binding,
            variant=args.variant,
            campaign_phase=args.campaign_phase,
            endpoint=endpoint,
            evaluator=evaluator_binding,
            campaign_source_metadata=args.campaign_source_metadata,
            campaign_source_root=args.campaign_source_root,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    metadata = {
        "schema_version": 6,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "mode": args.mode,
        "campaign_phase": args.campaign_phase,
        "run_name": args.run_dir.name,
        "categories": [
            category.strip()
            for category in args.categories.split(",")
            if category.strip()
        ],
        "model_registry_name": os.getenv("BFCL_MODEL", "zai-org/GLM-5.2-FC"),
        "served_model_name": "zai-org/GLM-5.2",
        "endpoint": endpoint,
        "bfcl_gorilla_commit": commit,
        "bfcl_patch_sha256": patch_sha256,
        "bfcl_source_identity": identity,
        "endpoint_models_sha256": hashlib.sha256(endpoint_models_raw).hexdigest(),
        "endpoint_model": endpoint_model,
        "temperature": float(os.getenv("BFCL_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("BFCL_MAX_TOKENS", "64000")),
        "num_threads": int(os.getenv("BFCL_NUM_THREADS", "1")),
        "include_input_log": os.getenv("BFCL_INCLUDE_INPUT_LOG", "1") == "1",
        "glm52_openai_extra_body": os.getenv("GLM52_OPENAI_EXTRA_BODY"),
        "glm52_openai_default_headers_sha256": default_headers_digest,
        "runtime_binding": binding,
        "campaign_source": campaign_source,
        "python_environment": environment_lock,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    binding_path = args.run_dir / "runtime-binding.json"
    binding_payload = (
        json.dumps(binding["content"]["deployment"], indent=2, sort_keys=True) + "\n"
    )
    if (
        binding_path.exists()
        and binding_path.read_text(encoding="utf-8") != binding_payload
    ):
        raise SystemExit(
            "Refusing to resume: runtime-binding.json differs from immutable metadata"
        )
    if not binding_path.exists():
        binding_path.write_text(binding_payload, encoding="utf-8")
    metadata_path = args.run_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        immutable_fields = (
            "variant",
            "mode",
            "campaign_phase",
            "run_name",
            "categories",
            "model_registry_name",
            "served_model_name",
            "endpoint",
            "bfcl_gorilla_commit",
            "bfcl_patch_sha256",
            "bfcl_source_identity",
            "endpoint_models_sha256",
            "endpoint_model",
            "temperature",
            "max_tokens",
            "num_threads",
            "include_input_log",
            "glm52_openai_extra_body",
            "glm52_openai_default_headers_sha256",
            "runtime_binding",
            "campaign_source",
            "python_environment",
        )
        mismatches = {
            field: {"recorded": existing.get(field), "requested": metadata.get(field)}
            for field in immutable_fields
            if existing.get(field) != metadata.get(field)
        }
        if mismatches:
            raise SystemExit(
                "Refusing to change immutable generation metadata while resuming:\n"
                + json.dumps(mismatches, indent=2, sort_keys=True)
            )
        print(f"Reusing immutable generation metadata: {metadata_path}")
        return

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
