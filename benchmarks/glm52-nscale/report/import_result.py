#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import one validated full-run harness result into results/summary.json.

The importer intentionally has no partial-result mode.  A row is published only
after the harness-specific completeness evidence closes over the full official
population and contains no infrastructure/evaluation errors.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from generate import (
    RESULT_IMPORTER_ID,
    assert_pinned_report_sources,
    validate,
    validate_sidecars,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
if str(EVAL_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(EVAL_DIR))

from bfcl_endpoint import EndpointModelError, canonical_endpoint_model  # noqa: E402
from runtime_binding import (  # noqa: E402
    BindingError,
    canonical_sha256 as runtime_canonical_sha256,
    validate_continuity as validate_shared_continuity,
    validate_deployment,
)

DEFAULT_SUMMARY = ROOT / "results" / "summary.json"
SHA256_HEX_LENGTH = 64
CAMPAIGN_PHASES = ("ab", "ba")
EVAL_PINS = ROOT / "eval" / "pins.env"
BFCL_PATCH = (
    ROOT / "eval" / "bfcl" / "patches" / "0001-glm52-openai-chat-completions.patch"
)
SWE_CONFIG = ROOT / "eval" / "swebench" / "config" / "glm52.yaml"
SWE_PRO_CONFIG = ROOT / "eval" / "swebench" / "config" / "pro.yaml"
TERMINAL_PINS = ROOT / "eval" / "terminalbench" / "pins.env"
SWE_UPSTREAM_CONFIG_SHA256 = (
    "229f178a07faa109e3a020c75dfe603b6a24200e1f3aebe257b97098fd3a6fee"
)
SWE_PROVENANCE_SHA256 = (
    "716bc2eb8831cc4ed4c6ab71f29137e38e3896f7255d76664fbe0992fc850075"
)
SWE_CONSTRAINTS_LOCK_SHA256 = (
    "f3af163e40ef54b172c1436e79da7effa72a28ec0f8336243804c6604611e3c1"
)
SWE_NORMALIZED_FREEZE_SHA256 = (
    "224ccfacf94f8b60a2473f5eafaffcc3da76940c179c2c8b81c5392ac3df9aa6"
)
SWE_EDITABLE_REPOS = {"mini-swe-agent", "SWE-bench"}
SWE_DATASETS = {
    "verified": {
        "repo": "SWE-bench/SWE-bench_Verified",
        "jsonl_sha256": "7303cc5795e3707162f9b0ffcc5694f3fd67e20bd9d514cfdce63146fdebc196",
        "parquet_sha256": "074f9dc8317cbf8c822d145457f9393fa69b30793b946259a72a9036d6449e1d",
        "scope_sha256": "0bfa14d71aca1c18f5a944323225f9ad449cecccb7c136a019dc90cff9109c5e",
    },
    "multilingual": {
        "repo": "SWE-bench/SWE-bench_Multilingual",
        "jsonl_sha256": "2727ef4f232af972d4094fef5919efe44a2dd53b831e541a82fa3d8c464e0b56",
        "parquet_sha256": "1b1b10aead80c8ba0840e2a6c1caecd3cb3531c8b36e2d126fa65ff4a8f467c3",
        "scope_sha256": "818d5ec9f8b34a4ae6dd8cf994b26d97a57df2dac462bd9a18c7e600bc5cda6c",
    },
    "pro": {
        "repo": "ScaleAI/SWE-bench_Pro",
        "jsonl_sha256": "3184035833dbab5d5b884139f9accd7aa2e6fefe6c15c45d9f8e7ff6fa119e76",
        "parquet_sha256": "87aeb33d8450d785c2bc64bd2c158f123179553f908c483a80ca935b52d0e69a",
        "scope_sha256": "2e318f24560f904b611daab2edd4f9c09614784671f959890432618fa2829af1",
    },
}
TERMINAL_HARBOR_UV_LOCK_SHA256 = (
    "be9a8ef015cad49ba65d6d6bf8e34d7eafa2844a9d2f1e90aaf1441711410025"
)
TERMINAL_TASK_REFS_SHA256 = (
    "aa94ca6bd5361c7fbae8534b82b609e20163ebec8391c0fc8f36362ee5f781b3"
)
BFCL_GENERATED_IDS_SHA256 = (
    "75413b25cd8994c118b53d19f3de1df9c38c4399f746abedf5a0f7e27ee6f526"
)
BFCL_SCORED_IDS_SHA256 = (
    "e0ed6c1ac094b2dc6c831f59d59c7e57a02c589b535b0a1f0534751ee94e0417"
)
BFCL_SCORED_CATEGORY_COUNTS = {
    "irrelevance": 240,
    "live_irrelevance": 884,
    "live_multiple": 1053,
    "live_parallel": 16,
    "live_parallel_multiple": 24,
    "live_relevance": 16,
    "live_simple": 258,
    "memory_kv": 155,
    "memory_rec_sum": 155,
    "memory_vector": 155,
    "multi_turn_base": 200,
    "multi_turn_long_context": 200,
    "multi_turn_miss_func": 200,
    "multi_turn_miss_param": 200,
    "multiple": 200,
    "parallel": 200,
    "parallel_multiple": 200,
    "simple_java": 100,
    "simple_javascript": 50,
    "simple_python": 400,
    "web_search_base": 100,
    "web_search_no_snippet": 100,
}
BFCL_TRACKED_DIFF_SHA256 = (
    "e075c864c9054095956a198475d09e179c7d2aa9d83f76ea51792fa5cf4650c4"
)
BFCL_NEW_HANDLER_SHA256 = (
    "f09a1999ef861c55cfd5a230a15e6536ce2f375b800fd837a5ac7a147b1920c5"
)
BFCL_CONSTRAINTS_SHA256 = (
    "1ae31cfcb689500018f8ce0239dfe9f43e471561c9d0b795b96fad96eaa83f04"
)
BFCL_PACKAGE_COUNT = 141
BFCL_FREEZE_SHA256 = "829c4dc3b72a4ec6f160fc2cc681070147dc1ea8dbeeee283c92ff3e356287a7"
BFCL_SOURCE_STATUS = [
    " M berkeley-function-call-leaderboard/SUPPORTED_MODELS.md",
    " M berkeley-function-call-leaderboard/bfcl_eval/constants/model_config.py",
    " M berkeley-function-call-leaderboard/bfcl_eval/constants/supported_models.py",
    "?? berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/glm52_openai.py",
]
BFCL_OFFICIAL_MODEL = "GLM-5.2 Native FC OpenAI Chat Completions"
RUNTIME_MAX_MODEL_LEN = 409600
CAMPAIGN_SOURCE_FIELDS = {
    "schema_version",
    "source_commit",
    "source_clean",
    "source_changed_path_count",
    "bundle_sha256",
    "source_tree_sha256",
    "eval_tree_sha256",
    "campaign_env_sha256",
    "source_file_count",
    "eval_file_count",
}


class EvidenceError(ValueError):
    """Raised when a harness artifact does not prove a complete valid run."""


def load_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not name or name in assignments:
            raise EvidenceError(
                f"{path}:{line_number}: invalid or duplicate assignment"
            )
        if value[:1] in {'"', "'"} and value[-1:] == value[:1]:
            value = value[1:-1]
        assignments[name] = value
    return assignments


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def ids_sha256(values: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def jsonl_payload(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )


def attach_task_level(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    variant: str,
    suite: str,
    phase: str,
) -> dict[str, Any]:
    payload = jsonl_payload(records)
    row["task_level"] = {
        "path": f"results/task-level/{suite}/{phase}/{variant}.jsonl",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(records),
    }
    row["_task_level_payload"] = payload.decode()
    return row


def load_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise EvidenceError(f"{path}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}: evidence root must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EvidenceError(f"cannot hash evidence {path}: {error}") from error
    return digest.hexdigest()


def source(
    role: str, path: Path, *, variant: str, suite: str, phase: str
) -> dict[str, str]:
    resolved = path.resolve()
    logical_path = f"artifact://{suite}/{phase}/{variant}/{role}/{resolved.name}"
    return {"role": role, "path": logical_path, "sha256": sha256_file(resolved)}


def require_equal(value: Any, expected: Any, path: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise EvidenceError(f"{path} must be {expected!r}, found {value!r}")


def sha256_value(value: Any, path: str, expected: str | None = None) -> str:
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        raise EvidenceError(f"{path} must be a lowercase SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise EvidenceError(f"{path} must be a lowercase SHA-256 digest") from error
    if value.lower() != value:
        raise EvidenceError(f"{path} must be a lowercase SHA-256 digest")
    if expected is not None and value != expected:
        raise EvidenceError(f"{path} does not match campaign pin {expected}")
    return value


def require_http_v1_endpoint(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise EvidenceError(f"{path} must be an absolute HTTP(S) /v1 URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.path.rstrip("/").endswith("/v1")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise EvidenceError(
            f"{path} must be an absolute credential-free HTTP(S) /v1 URL"
        )
    return value


def unwrap_runtime_binding(value: Any) -> tuple[dict[str, Any], str, str]:
    wrapper = require_mapping(value, "runtime_binding")
    file_name = wrapper.get("file")
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
    ):
        raise EvidenceError("runtime_binding.file must be a path-free filename")
    content = require_mapping(wrapper.get("content"), "runtime_binding.content")
    if set(content) != {"deployment", "evaluator"}:
        raise EvidenceError(
            "runtime_binding.content must contain exactly deployment and evaluator"
        )
    deployment = require_mapping(
        content.get("deployment"), "runtime_binding.content.deployment"
    )
    deployment_digest = sha256_value(
        wrapper.get("deployment_sha256"), "runtime_binding.deployment_sha256"
    )
    expected_deployment_digest = canonical_sha256(deployment)
    if deployment_digest != expected_deployment_digest:
        raise EvidenceError(
            "runtime_binding.deployment_sha256 does not match canonical deployment"
        )
    content_digest = sha256_value(
        wrapper.get("content_sha256"), "runtime_binding.content_sha256"
    )
    if content_digest != canonical_sha256(content):
        raise EvidenceError(
            "runtime_binding.content_sha256 does not match canonical content"
        )
    return deployment, deployment_digest, content_digest


def validate_runtime_binding(
    value: Any,
    *,
    variant: str,
    served_model: str,
    image: str,
    harness_endpoint: str,
    phase: str,
    source_commit: str,
) -> tuple[dict[str, Any], str, str]:
    binding, deployment_sha256, content_sha256 = unwrap_runtime_binding(value)
    try:
        validate_deployment(
            binding,
            variant=variant,
            campaign_phase=phase,
            endpoint=harness_endpoint,
        )
    except BindingError as error:
        raise EvidenceError(str(error)) from error
    require_equal(
        binding.get("served_model_name"),
        served_model,
        "runtime_binding.served_model_name",
    )
    require_equal(binding.get("image"), image, "runtime_binding.image")
    recipe = require_mapping(binding.get("recipe"), "runtime_binding.recipe")
    require_string(
        recipe.get("source_commit"),
        source_commit,
        "runtime_binding.recipe.source_commit",
    )
    require_string(
        recipe.get("template_sha256"),
        sha256_file(ROOT / "deploy" / "templates" / f"{variant}.yaml"),
        "runtime_binding.recipe.template_sha256",
    )
    return binding, deployment_sha256, content_sha256


def validate_runtime_continuity(
    path: Path,
    *,
    variant: str,
    phase: str,
    binding: dict[str, Any],
    binding_sha256: str,
) -> None:
    continuity = load_object(path)
    require_equal(binding.get("variant"), variant, "runtime binding variant")
    require_equal(binding.get("campaign_phase"), phase, "runtime binding phase")
    require_equal(
        runtime_canonical_sha256(binding),
        binding_sha256,
        "runtime binding deployment digest",
    )
    try:
        validate_shared_continuity(continuity, binding)
    except BindingError as error:
        raise EvidenceError(str(error)) from error


def validate_campaign_source(
    value: Any, *, source_commit: str, path: str
) -> dict[str, Any]:
    identity = require_mapping(value, path)
    if set(identity) != CAMPAIGN_SOURCE_FIELDS:
        raise EvidenceError(f"{path} contains invalid fields")
    require_integer(identity.get("schema_version"), 1, f"{path}.schema_version")
    require_string(
        identity.get("source_commit"), source_commit, f"{path}.source_commit"
    )
    require_equal(identity.get("source_clean"), True, f"{path}.source_clean")
    require_integer(
        identity.get("source_changed_path_count"),
        0,
        f"{path}.source_changed_path_count",
    )
    for field in (
        "bundle_sha256",
        "source_tree_sha256",
        "eval_tree_sha256",
        "campaign_env_sha256",
    ):
        sha256_value(identity.get(field), f"{path}.{field}")
    source_count = integer(
        identity.get("source_file_count"), f"{path}.source_file_count"
    )
    eval_count = integer(identity.get("eval_file_count"), f"{path}.eval_file_count")
    if eval_count < 1 or source_count != eval_count + 1:
        raise EvidenceError(f"{path} file counts are inconsistent")
    return identity


def runtime_identity(
    binding: dict[str, Any], deployment_sha256: str, content_sha256: str
) -> dict[str, Any]:
    controller = require_mapping(
        binding.get("controller"), "runtime binding controller"
    )
    pods = require_mapping(binding.get("pods"), "runtime binding pods")
    capture = require_mapping(binding.get("capture"), "runtime binding capture")
    recipe = require_mapping(binding.get("recipe"), "runtime binding recipe")
    hardware = require_mapping(binding.get("hardware"), "runtime binding hardware")
    return {
        "deployment_sha256": deployment_sha256,
        "content_sha256": content_sha256,
        "captured_at": capture["captured_at"],
        "controller_uid_sha256": controller["uid_sha256"],
        "pod_uid_sha256_by_role": {
            role: require_mapping(pod, f"runtime binding pod {role}")["uid_sha256"]
            for role, pod in sorted(pods.items())
        },
        "capture_sha256": capture["sha256"],
        "recipe": recipe,
        "hardware": hardware,
        "control_plane": binding.get("control_plane"),
    }


def require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{path} must be an object")
    return value


def require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceError(f"{path} must be an array")
    return value


def require_empty_list(value: Any, path: str) -> None:
    values = require_list(value, path)
    if values:
        raise EvidenceError(f"{path} must be empty; found {len(values)} item(s)")


def string_set(value: Any, path: str) -> set[str]:
    values = require_list(value, path)
    if not all(isinstance(item, str) and item for item in values):
        raise EvidenceError(f"{path} must contain only non-empty strings")
    result = set(values)
    if len(result) != len(values):
        raise EvidenceError(f"{path} must not contain duplicate IDs")
    return result


def require_bool(value: Any, expected: bool, path: str) -> None:
    if value is not expected:
        raise EvidenceError(f"{path} must be {str(expected).lower()}")


def require_string(value: Any, expected: str, path: str) -> None:
    if value != expected:
        raise EvidenceError(f"{path} must be {expected!r}, found {value!r}")


def integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{path} must be a non-negative integer")
    return value


def require_integer(value: Any, expected: int, path: str) -> int:
    actual = integer(value, path)
    if actual != expected:
        raise EvidenceError(f"{path} must be {expected}, found {actual}")
    return actual


def finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{path} must be finite")
    return result


def fraction(value: Any, path: str) -> float:
    result = finite_number(value, path)
    if not 0.0 <= result <= 1.0:
        raise EvidenceError(f"{path} must be in [0, 1]")
    return result


def close(actual: float, expected: float, path: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise EvidenceError(f"{path}={actual!r}, expected {expected!r}")


def parse_official_percent(value: Any, path: str) -> float:
    had_percent = False
    if isinstance(value, str):
        text = value.strip()
        had_percent = text.endswith("%")
        if had_percent:
            text = text[:-1].strip()
        try:
            numeric = float(text)
        except ValueError as error:
            raise EvidenceError(f"{path} is not numeric: {value!r}") from error
    else:
        numeric = finite_number(value, path)
    if had_percent or numeric > 1.0:
        numeric /= 100.0
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise EvidenceError(f"{path} must encode a percentage in [0, 100]")
    return numeric


def bfcl_official_overall(categories: dict[str, Any]) -> float:
    """Reproduce the pinned BFCL v4 leaderboard's official weighted score."""

    def category_accuracy(name: str) -> float:
        stats = require_mapping(categories.get(name), f"BFCL category {name}")
        total = integer(stats.get("total_count"), f"BFCL {name}.total_count")
        correct = integer(stats.get("correct_count"), f"BFCL {name}.correct_count")
        if total == 0 or correct > total:
            raise EvidenceError(f"BFCL {name} has invalid official score counts")
        return correct / total

    def unweighted(names: tuple[str, ...]) -> float:
        return sum(category_accuracy(name) for name in names) / len(names)

    simple_non_live = unweighted(("simple_python", "simple_java", "simple_javascript"))
    non_live = (
        sum(
            (
                simple_non_live,
                category_accuracy("multiple"),
                category_accuracy("parallel"),
                category_accuracy("parallel_multiple"),
            )
        )
        / 4
    )

    live_names = (
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
    )
    live_total = sum(categories[name]["total_count"] for name in live_names)
    live = (
        sum(
            category_accuracy(name) * categories[name]["total_count"]
            for name in live_names
        )
        / live_total
    )
    irrelevance = unweighted(("irrelevance", "live_irrelevance"))
    multi_turn = unweighted(
        (
            "multi_turn_base",
            "multi_turn_miss_func",
            "multi_turn_miss_param",
            "multi_turn_long_context",
        )
    )
    web_search = unweighted(("web_search_base", "web_search_no_snippet"))
    memory = unweighted(("memory_kv", "memory_vector", "memory_rec_sum"))
    agentic = (web_search + memory) / 2
    return (
        0.10 * non_live
        + 0.10 * live
        + 0.10 * irrelevance
        + 0.30 * multi_turn
        + 0.40 * agentic
    )


def suite_definition(summary: dict[str, Any], suite_id: str) -> dict[str, Any]:
    matches = [
        suite for suite in summary.get("suites", []) if suite.get("id") == suite_id
    ]
    if len(matches) != 1:
        raise EvidenceError(f"summary does not define suite {suite_id!r} exactly once")
    return matches[0]


def variant_definition(summary: dict[str, Any], variant_id: str) -> dict[str, Any]:
    matches = [
        variant
        for variant in summary.get("variants", [])
        if variant.get("id") == variant_id
    ]
    if len(matches) != 1:
        raise EvidenceError(
            f"summary does not define variant {variant_id!r} exactly once"
        )
    return matches[0]


def result_row(
    variant: str,
    suite: str,
    phase: str,
    generated: int,
    evaluated: int,
    trials: int,
    metrics: dict[str, int | float],
    sources: list[dict[str, str]],
    runtime: dict[str, Any],
    campaign_source: dict[str, Any],
    wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant,
        "suite": suite,
        "phase": phase,
        "run_type": "full",
        "status": "complete",
        "completeness": {
            "generated_units": generated,
            "evaluated_units": evaluated,
            "completed_trials": trials,
        },
        "metrics": metrics,
        "runtime_identity": runtime,
        "campaign_source": campaign_source,
        "evidence": {"importer": RESULT_IMPORTER_ID, "sources": sources},
    }
    if wall_time_seconds is not None:
        row["wall_time_seconds"] = wall_time_seconds
    return row


def load_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvidenceError(
                        f"{label} {path}:{line_number} is invalid JSON: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise EvidenceError(
                        f"{label} {path}:{line_number} must be a JSON object"
                    )
                rows.append(row)
    except OSError as error:
        raise EvidenceError(f"cannot read {label} {path}: {error}") from error
    return rows


def bfcl_task_records(
    run_dir: Path,
    *,
    expected_commit: str,
    expected_scored: int,
    correct: int,
    failed: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    manifest_path = run_dir / "expected-ids.json"
    failures_path = run_dir / "failures.jsonl"
    manifest = load_object(manifest_path)
    require_integer(
        manifest.get("schema_version"), 1, "BFCL expected IDs.schema_version"
    )
    require_string(
        manifest.get("bfcl_gorilla_commit"),
        expected_commit,
        "BFCL expected IDs.bfcl_gorilla_commit",
    )
    scored = require_mapping(manifest.get("scored"), "BFCL expected IDs.scored")
    require_integer(
        scored.get("count"), expected_scored, "BFCL expected IDs.scored.count"
    )
    by_category = require_mapping(
        scored.get("ids_by_category"), "BFCL expected IDs.scored.ids_by_category"
    )
    if set(by_category) != set(BFCL_SCORED_CATEGORY_COUNTS):
        raise EvidenceError(
            "BFCL expected IDs categories do not match the pinned population"
        )
    id_category: dict[str, str] = {}
    for category, expected_count in BFCL_SCORED_CATEGORY_COUNTS.items():
        values = require_list(
            by_category.get(category), f"BFCL expected IDs category {category}"
        )
        if len(values) != expected_count or values != sorted(values):
            raise EvidenceError(
                f"BFCL expected IDs category {category} must contain "
                f"{expected_count} sorted IDs"
            )
        for value in values:
            if not isinstance(value, str) or not value or value in id_category:
                raise EvidenceError(
                    "BFCL expected IDs contain an invalid or duplicate ID"
                )
            id_category[value] = category
    all_ids = set(id_category)
    if len(all_ids) != expected_scored:
        raise EvidenceError("BFCL expected IDs do not cover the scored population")
    expected_ids_digest = sha256_value(
        scored.get("ids_sha256"), "BFCL expected IDs.scored.ids_sha256"
    )
    if ids_sha256(all_ids) != expected_ids_digest:
        raise EvidenceError("BFCL expected IDs digest does not match its ID population")
    require_equal(
        expected_ids_digest,
        BFCL_SCORED_IDS_SHA256,
        "BFCL expected IDs pinned digest",
    )

    failure_by_id: dict[str, list[str]] = {}
    for index, failure in enumerate(load_jsonl_objects(failures_path, "BFCL failures")):
        case_id = failure.get("id")
        category = failure.get("category")
        if not isinstance(case_id, str) or case_id not in id_category:
            raise EvidenceError(f"BFCL failure row {index} has an unexpected ID")
        if case_id in failure_by_id:
            raise EvidenceError(f"BFCL failure row {index} duplicates ID {case_id!r}")
        if category != id_category[case_id]:
            raise EvidenceError(
                f"BFCL failure row {index} category does not match expected IDs"
            )
        raw_error_types = failure.get("error_type", "unknown")
        if isinstance(raw_error_types, list):
            if not raw_error_types or not all(
                isinstance(value, str) and value for value in raw_error_types
            ):
                raise EvidenceError(f"BFCL failure row {index} has invalid error types")
            error_types = sorted(set(raw_error_types))
        elif isinstance(raw_error_types, str) and raw_error_types:
            error_types = [raw_error_types]
        else:
            raise EvidenceError(f"BFCL failure row {index} has an invalid error type")
        failure_by_id[case_id] = error_types
    if len(failure_by_id) != failed or expected_scored - len(failure_by_id) != correct:
        raise EvidenceError("BFCL task outcomes do not reconcile with aggregate counts")
    records = [
        {
            "id": case_id,
            "category": id_category[case_id],
            "outcome": "failed" if case_id in failure_by_id else "passed",
            **(
                {"error_types": failure_by_id[case_id]}
                if case_id in failure_by_id
                else {}
            ),
        }
        for case_id in sorted(all_ids)
    ]
    return records, [manifest_path, failures_path]


def swe_task_records(
    target_ids: set[str],
    passed_ids: set[str],
    unresolved_ids: set[str],
    empty_patch_ids: set[str],
) -> list[dict[str, Any]]:
    records = []
    for instance_id in sorted(target_ids):
        if instance_id in passed_ids:
            records.append({"id": instance_id, "outcome": "passed"})
        else:
            failure_kind = (
                "empty_patch" if instance_id in empty_patch_ids else "unresolved"
            )
            if instance_id not in unresolved_ids | empty_patch_ids:
                raise EvidenceError(f"SWE task outcome missing for {instance_id!r}")
            records.append(
                {
                    "id": instance_id,
                    "outcome": "failed",
                    "failure_kind": failure_kind,
                }
            )
    return records


def terminal_task_records(
    path: Path,
    *,
    expected_task_names: list[Any],
    expected_attempts: int,
) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise EvidenceError(
            f"cannot read Terminal trial evidence {path}: {error}"
        ) from error
    expected_count = len(expected_task_names) * expected_attempts
    if len(raw_rows) != expected_count:
        raise EvidenceError(
            f"Terminal trials.csv must contain {expected_count} attempts, "
            f"found {len(raw_rows)}"
        )
    required_fields = {
        "task_name",
        "trial_name",
        "status",
        "primary_reward",
        "result_sha256",
    }
    if not raw_rows or not required_fields.issubset(raw_rows[0]):
        raise EvidenceError("Terminal trials.csv is missing required columns")
    grouped: dict[str, list[dict[str, str]]] = {
        str(name): [] for name in expected_task_names
    }
    seen_trials: set[str] = set()
    for row in raw_rows:
        task_name = row.get("task_name", "")
        trial_name = row.get("trial_name", "")
        status = row.get("status")
        if task_name not in grouped or not trial_name or trial_name in seen_trials:
            raise EvidenceError(
                "Terminal trials.csv contains an invalid task/trial identity"
            )
        if status not in {"passed", "failed"}:
            raise EvidenceError("Terminal trials.csv contains an incomplete attempt")
        seen_trials.add(trial_name)
        grouped[task_name].append(row)
    records: list[dict[str, Any]] = []
    for task_name in expected_task_names:
        rows = sorted(grouped[str(task_name)], key=lambda row: row["trial_name"])
        if len(rows) != expected_attempts:
            raise EvidenceError(
                f"Terminal trials.csv task {task_name!r} must have {expected_attempts} attempts"
            )
        for attempt, row in enumerate(rows, start=1):
            raw_reward = row.get("primary_reward")
            try:
                reward = float(raw_reward) if raw_reward not in {None, ""} else None
            except ValueError as error:
                raise EvidenceError(
                    "Terminal trials.csv contains a non-numeric reward"
                ) from error
            if reward is None or not math.isfinite(reward):
                raise EvidenceError(
                    "Terminal trials.csv contains a missing/non-finite reward"
                )
            sha256_value(
                row.get("result_sha256"),
                f"Terminal trial {row['trial_name']} result_sha256",
            )
            records.append(
                {
                    "id": str(task_name),
                    "attempt": attempt,
                    "outcome": row["status"],
                    "reward": reward,
                }
            )
    return records


def bfcl_paths(
    artifact: Path, payload: dict[str, Any]
) -> tuple[Path, Path, Path, Path]:
    if payload.get("phase") == "complete" and "generation" in payload:
        return (
            artifact.parent / "summary.json",
            artifact,
            artifact.parent / "metadata.json",
            artifact.parent / "endpoint-models.json",
        )
    if "totals" in payload and "official_overall_csv_row" in payload:
        return (
            artifact,
            artifact.parent / "complete-validation.json",
            artifact.parent / "metadata.json",
            artifact.parent / "endpoint-models.json",
        )
    raise EvidenceError(
        "BFCL artifact must be summary.json or complete-validation.json"
    )


def import_bfcl(
    variant: str,
    phase: str,
    suite: dict[str, Any],
    artifact: Path,
    payload: dict[str, Any],
    served_model: str,
    image: str,
    source_commit: str,
) -> dict[str, Any]:
    compact_path, validation_path, metadata_path, endpoint_models_path = bfcl_paths(
        artifact, payload
    )
    compact = load_object(compact_path)
    proof = load_object(validation_path)
    metadata = load_object(metadata_path)
    endpoint_models = load_object(endpoint_models_path)
    environment_lock_path = compact_path.parent / "environment-lock.json"
    environment_freeze_path = compact_path.parent / "environment.freeze.txt"
    environment_lock = load_object(environment_lock_path)
    if set(environment_lock) != {
        "schema_version",
        "constraints_sha256",
        "freeze_sha256",
        "package_count",
        "python",
    }:
        raise EvidenceError("BFCL environment lock contains invalid fields")
    require_integer(
        environment_lock.get("schema_version"), 1, "BFCL environment schema_version"
    )
    require_string(
        environment_lock.get("constraints_sha256"),
        BFCL_CONSTRAINTS_SHA256,
        "BFCL environment constraints_sha256",
    )
    require_integer(
        environment_lock.get("package_count"),
        BFCL_PACKAGE_COUNT,
        "BFCL environment package_count",
    )
    sha256_value(
        environment_lock.get("freeze_sha256"),
        "BFCL environment freeze_sha256",
        BFCL_FREEZE_SHA256,
    )
    require_string(
        sha256_file(environment_freeze_path),
        environment_lock["freeze_sha256"],
        "BFCL environment freeze file",
    )
    if (
        not isinstance(environment_lock.get("python"), str)
        or not environment_lock["python"]
    ):
        raise EvidenceError("BFCL environment Python version must be non-empty")
    expected_generated = suite.get("generation_units", suite["units"])
    expected_scored = suite["units"]
    campaign_pins = load_assignments(EVAL_PINS)
    expected_commit = campaign_pins["BFCL_COMMIT"]
    expected_patch_sha256 = sha256_file(BFCL_PATCH)
    campaign_source = validate_campaign_source(
        metadata.get("campaign_source"),
        source_commit=source_commit,
        path="BFCL metadata.campaign_source",
    )

    require_integer(compact.get("schema_version"), 1, "BFCL summary.schema_version")
    require_string(compact.get("variant"), variant, "BFCL summary.variant")
    require_string(compact.get("mode"), "full", "BFCL summary.mode")
    require_string(compact.get("campaign_phase"), phase, "BFCL summary.campaign_phase")
    require_string(
        compact.get("run_name"), compact_path.parent.name, "BFCL summary.run_name"
    )
    require_string(
        compact.get("bfcl_gorilla_commit"),
        expected_commit,
        "BFCL summary.bfcl_gorilla_commit",
    )
    require_integer(proof.get("schema_version"), 2, "BFCL validation.schema_version")
    require_string(proof.get("phase"), "complete", "BFCL validation.phase")
    require_string(proof.get("campaign_phase"), phase, "BFCL validation.campaign_phase")
    require_string(proof.get("status"), "pass", "BFCL validation.status")
    require_empty_list(proof.get("errors"), "BFCL validation.errors")

    expected_identity = {
        "schema_version": 6,
        "variant": variant,
        "mode": "full",
        "campaign_phase": phase,
        "run_name": metadata.get("run_name"),
        "categories": ["all_scoring"],
        "model_registry_name": "zai-org/GLM-5.2-FC",
        "served_model_name": "zai-org/GLM-5.2",
        "endpoint": metadata.get("endpoint"),
        "bfcl_gorilla_commit": expected_commit,
        "bfcl_patch_sha256": expected_patch_sha256,
        "bfcl_source_identity": {
            "head": expected_commit,
            "status": BFCL_SOURCE_STATUS,
            "tracked_diff_sha256": BFCL_TRACKED_DIFF_SHA256,
            "new_handler_sha256": BFCL_NEW_HANDLER_SHA256,
        },
        "endpoint_models_sha256": sha256_file(endpoint_models_path),
        "endpoint_model": metadata.get("endpoint_model"),
        "temperature": 0.0,
        "max_tokens": 64000,
        "num_threads": 16,
        "include_input_log": True,
        "glm52_openai_extra_body": None,
        "glm52_openai_default_headers_sha256": None,
        "runtime_binding": metadata.get("runtime_binding"),
        "campaign_source": campaign_source,
        "python_environment": environment_lock,
    }
    if not identity_contains_token(metadata.get("run_name"), phase):
        raise EvidenceError(
            f"BFCL metadata.run_name must contain campaign phase {phase!r}"
        )
    require_equal(
        metadata.get("run_name"), compact_path.parent.name, "BFCL metadata.run_name"
    )
    require_http_v1_endpoint(metadata.get("endpoint"), "BFCL metadata.endpoint")
    binding, deployment_sha256, content_sha256 = validate_runtime_binding(
        metadata.get("runtime_binding"),
        variant=variant,
        served_model=served_model,
        image=image,
        harness_endpoint=metadata["endpoint"],
        phase=phase,
        source_commit=source_commit,
    )
    bfcl_wrapper = require_mapping(
        metadata.get("runtime_binding"), "BFCL runtime binding"
    )
    bfcl_wrapper_content = require_mapping(
        bfcl_wrapper.get("content"), "BFCL runtime binding content"
    )
    require_equal(
        bfcl_wrapper_content.get("evaluator"),
        {
            "harness": "bfcl-v4",
            "model_registry_name": "zai-org/GLM-5.2-FC",
            "categories": ["all_scoring"],
            "temperature": 0.0,
            "max_tokens": 64000,
            "num_threads": 16,
            "campaign_source": campaign_source,
        },
        "BFCL runtime binding evaluator",
    )
    continuity_path = compact_path.parent / "runtime-continuity.json"
    validate_runtime_continuity(
        continuity_path,
        variant=variant,
        phase=phase,
        binding=binding,
        binding_sha256=deployment_sha256,
    )
    matching_endpoint_models = [
        row
        for row in endpoint_models.get("data", [])
        if isinstance(row, dict) and row.get("id") == served_model
    ]
    if len(matching_endpoint_models) != 1:
        raise EvidenceError("BFCL endpoint evidence must contain the served model once")
    try:
        endpoint_model_identity = canonical_endpoint_model(
            matching_endpoint_models[0], RUNTIME_MAX_MODEL_LEN
        )
    except EndpointModelError as error:
        raise EvidenceError(
            f"BFCL endpoint model context is invalid: {error}"
        ) from error
    require_equal(
        endpoint_model_identity,
        metadata.get("endpoint_model"),
        "BFCL metadata.endpoint_model",
    )
    require_equal(
        endpoint_model_identity.get("context_window"),
        RUNTIME_MAX_MODEL_LEN,
        "BFCL endpoint model context_window",
    )
    for field, expected in expected_identity.items():
        require_equal(metadata.get(field), expected, f"BFCL metadata.{field}")
    run_identity = require_mapping(
        proof.get("run_identity"), "BFCL validation.run_identity"
    )
    sha256_value(
        run_identity.get("metadata_sha256"),
        "BFCL validation.run_identity.metadata_sha256",
        sha256_file(metadata_path),
    )
    require_equal(
        run_identity.get("immutable"),
        expected_identity,
        "BFCL validation.run_identity.immutable",
    )

    generation = require_mapping(proof.get("generation"), "BFCL validation.generation")
    require_string(generation.get("status"), "pass", "BFCL generation.status")
    require_empty_list(generation.get("errors"), "BFCL generation.errors")
    require_integer(
        generation.get("expected_count"),
        expected_generated,
        "BFCL generation.expected_count",
    )
    require_integer(
        generation.get("actual_count"),
        expected_generated,
        "BFCL generation.actual_count",
    )
    require_integer(
        generation.get("actual_entry_count"),
        expected_generated,
        "BFCL generation.actual_entry_count",
    )
    for field in (
        "missing_ids",
        "extra_ids",
        "duplicate_ids",
        "missing_categories",
        "extra_categories",
        "inference_error_ids",
    ):
        value = generation.get(field)
        if field == "duplicate_ids" and isinstance(value, dict):
            if value:
                raise EvidenceError("BFCL generation.duplicate_ids must be empty")
        else:
            require_empty_list(value, f"BFCL generation.{field}")
    sha256_value(
        generation.get("expected_ids_sha256"),
        "BFCL generation.expected_ids_sha256",
        BFCL_GENERATED_IDS_SHA256,
    )
    sha256_value(
        generation.get("actual_ids_sha256"),
        "BFCL generation.actual_ids_sha256",
        BFCL_GENERATED_IDS_SHA256,
    )

    scores = require_mapping(proof.get("scores"), "BFCL validation.scores")
    require_string(scores.get("status"), "pass", "BFCL scores.status")
    require_empty_list(scores.get("errors"), "BFCL scores.errors")
    require_integer(
        scores.get("expected_count"), expected_scored, "BFCL scores.expected_count"
    )
    require_integer(
        scores.get("scored_count"), expected_scored, "BFCL scores.scored_count"
    )
    sha256_value(
        scores.get("expected_ids_sha256"),
        "BFCL scores.expected_ids_sha256",
        BFCL_SCORED_IDS_SHA256,
    )
    require_empty_list(
        scores.get("missing_categories"), "BFCL scores.missing_categories"
    )
    require_empty_list(scores.get("extra_categories"), "BFCL scores.extra_categories")

    categories = require_mapping(scores.get("categories"), "BFCL scores.categories")
    if set(categories) != set(BFCL_SCORED_CATEGORY_COUNTS):
        raise EvidenceError(
            "BFCL scores.categories do not match the pinned all_scoring category set"
        )
    correct = 0
    failed = 0
    scored = 0
    for category, raw_stats in categories.items():
        stats = require_mapping(raw_stats, f"BFCL scores.categories[{category!r}]")
        expected = require_integer(
            stats.get("expected_count"),
            BFCL_SCORED_CATEGORY_COUNTS[category],
            f"BFCL {category}.expected_count",
        )
        require_integer(
            stats.get("total_count"), expected, f"BFCL {category}.total_count"
        )
        category_correct = integer(
            stats.get("correct_count"), f"BFCL {category}.correct_count"
        )
        category_failed = integer(
            stats.get("failure_count"), f"BFCL {category}.failure_count"
        )
        if category_correct + category_failed != expected:
            raise EvidenceError(
                f"BFCL {category} outcomes do not sum to expected_count"
            )
        correct += category_correct
        failed += category_failed
        scored += expected
    if scored != expected_scored or correct + failed != expected_scored:
        raise EvidenceError(
            "BFCL category outcomes do not cover the official scored population"
        )

    totals = require_mapping(compact.get("totals"), "BFCL summary.totals")
    for field, expected in (
        ("generated_count", expected_generated),
        ("scored_count", expected_scored),
        ("correct_count", correct),
        ("failure_count", failed),
        ("inference_error_count", 0),
    ):
        require_integer(totals.get(field), expected, f"BFCL summary.totals.{field}")
    official = require_mapping(
        compact.get("official_overall_csv_row"), "BFCL summary.official_overall_csv_row"
    )
    require_string(
        official.get("Model"),
        BFCL_OFFICIAL_MODEL,
        "BFCL summary.official_overall_csv_row['Model']",
    )
    if "Overall Acc" not in official:
        raise EvidenceError("BFCL official overall CSV row is missing 'Overall Acc'")
    expected_overall = bfcl_official_overall(categories)
    expected_display = f"{expected_overall * 100:.2f}%"
    require_string(
        official["Overall Acc"],
        expected_display,
        "BFCL summary.official_overall_csv_row['Overall Acc']",
    )
    overall = parse_official_percent(
        official["Overall Acc"], "BFCL summary.official_overall_csv_row['Overall Acc']"
    )
    task_records, task_evidence_paths = bfcl_task_records(
        compact_path.parent,
        expected_commit=expected_commit,
        expected_scored=expected_scored,
        correct=correct,
        failed=failed,
    )
    row = result_row(
        variant,
        suite["id"],
        phase,
        expected_generated,
        expected_scored,
        expected_scored,
        {
            "overall_accuracy": overall,
            "correct_cases": correct,
            "failed_cases": failed,
            "inference_errors": 0,
        },
        [
            source(
                "bfcl-summary",
                compact_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-complete-validation",
                validation_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-run-metadata",
                metadata_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-endpoint-models",
                endpoint_models_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "runtime-continuity",
                continuity_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-expected-ids",
                task_evidence_paths[0],
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-failures",
                task_evidence_paths[1],
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-environment-lock",
                environment_lock_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "bfcl-environment-freeze",
                environment_freeze_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
        ],
        runtime_identity(binding, deployment_sha256, content_sha256),
        campaign_source,
    )
    row["suite_identity"] = {
        "python_environment": environment_lock,
        "campaign_source": campaign_source,
    }
    return attach_task_level(
        row, task_records, variant=variant, suite=suite["id"], phase=phase
    )


SWE_SUITE_NAMES = {
    "swebench-verified": "verified",
    "swebench-pro": "pro",
    "swebench-multilingual": "multilingual",
}


def swe_paths(artifact: Path, payload: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    if "benchmark_score" in payload and "passed_instances" in payload:
        score_path = artifact
    elif "predictions" in payload and "exit_statuses" in payload:
        score_path = artifact.parent / "score.json"
    else:
        raise EvidenceError(
            "SWE-bench artifact must be score.json or generation-summary.json"
        )
    return (
        score_path,
        score_path.parent / "generation-summary.json",
        score_path.parent / "run-metadata.json",
        score_path.parent / "run-scope.json",
    )


def identity_matches_variant(run_name: Any, variant: str) -> bool:
    if not isinstance(run_name, str):
        return False
    return run_name == variant or any(
        run_name.startswith(variant + separator) for separator in "-_."
    )


def identity_contains_token(run_name: Any, token: str) -> bool:
    if not isinstance(run_name, str):
        return False
    normalized = run_name.replace("_", "-").replace(".", "-")
    return token in normalized.split("-")


def expected_swe_pins() -> dict[str, str]:
    return dict(
        sorted(
            (name, value)
            for name, value in load_assignments(EVAL_PINS).items()
            if name.startswith(("MINI_SWE_", "SWEBENCH_"))
        )
    )


def expected_swe_source_lock(pins: dict[str, str]) -> dict[str, Any]:
    return {
        "mini_swe_agent": {
            "version": pins["MINI_SWE_AGENT_VERSION"],
            "commit": pins["MINI_SWE_AGENT_COMMIT"],
        },
        "swebench": {
            "version": pins["SWEBENCH_VERSION"],
            "commit": pins["SWEBENCH_COMMIT"],
        },
        "swebench_pro": {"commit": pins["SWEBENCH_PRO_COMMIT"]},
    }


def serialized_source_lock_sha256(source_lock: dict[str, Any]) -> str:
    payload = (json.dumps(source_lock, indent=2) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def validate_swe_run_identity(
    metadata: dict[str, Any],
    *,
    variant: str,
    suite_name: str,
    expected_instances: int,
    served_model: str,
    phase: str,
    source_commit: str,
) -> dict[str, Any]:
    pins = expected_swe_pins()
    dataset_pin_names = {
        "verified": ("SWEBENCH_VERIFIED_REVISION", "SWEBENCH_VERIFIED_CASES"),
        "multilingual": (
            "SWEBENCH_MULTILINGUAL_REVISION",
            "SWEBENCH_MULTILINGUAL_CASES",
        ),
        "pro": ("SWEBENCH_PRO_REVISION", "SWEBENCH_PRO_PUBLIC_CASES"),
    }
    revision_pin, count_pin = dataset_pin_names[suite_name]
    require_integer(
        metadata.get("schema_version"), 3, "SWE run metadata.schema_version"
    )
    require_string(metadata.get("suite"), suite_name, "SWE run metadata.suite")
    require_string(metadata.get("model"), served_model, "SWE run metadata.model")
    require_string(
        metadata.get("campaign_phase"), phase, "SWE run metadata.campaign_phase"
    )
    require_http_v1_endpoint(metadata.get("endpoint"), "SWE run metadata.endpoint")
    if not identity_matches_variant(metadata.get("run_name"), variant):
        raise EvidenceError(
            f"SWE run metadata.run_name must begin with variant {variant!r} and a separator"
        )
    if not identity_contains_token(metadata.get("run_name"), phase):
        raise EvidenceError(
            f"SWE run metadata.run_name must contain campaign phase {phase!r}"
        )
    require_equal(
        metadata.get("scope_sha256"),
        SWE_DATASETS[suite_name]["scope_sha256"],
        "SWE run metadata.scope_sha256",
    )

    config_files = [
        {"name": "upstream-swebench", "sha256": SWE_UPSTREAM_CONFIG_SHA256},
        {"name": "glm52", "sha256": sha256_file(SWE_CONFIG)},
    ]
    if suite_name == "pro":
        config_files.append({"name": "pro", "sha256": sha256_file(SWE_PRO_CONFIG)})
    expected_configuration = {
        "files": config_files,
        "sha256": canonical_sha256(config_files),
    }
    require_equal(
        metadata.get("configuration"),
        expected_configuration,
        "SWE run metadata.configuration",
    )

    expected_pin_identity = {"sha256": canonical_sha256(pins), "values": pins}
    require_equal(metadata.get("pins"), expected_pin_identity, "SWE run metadata.pins")
    source_lock = expected_swe_source_lock(pins)
    expected_source = {
        "lock_sha256": serialized_source_lock_sha256(source_lock),
        "lock": source_lock,
        "repositories": {
            "mini_swe_agent": {"commit": pins["MINI_SWE_AGENT_COMMIT"]},
            "swebench": {"commit": pins["SWEBENCH_COMMIT"]},
            "swebench_pro": {"commit": pins["SWEBENCH_PRO_COMMIT"]},
        },
    }
    require_equal(metadata.get("source"), expected_source, "SWE run metadata.source")

    dataset = require_mapping(metadata.get("dataset"), "SWE run metadata.dataset")
    sha256_value(
        dataset.get("evaluator_jsonl_sha256"),
        "SWE run metadata.dataset.evaluator_jsonl_sha256",
        SWE_DATASETS[suite_name]["jsonl_sha256"],
    )
    sha256_value(
        dataset.get("provenance_sha256"),
        "SWE run metadata.dataset.provenance_sha256",
        SWE_PROVENANCE_SHA256,
    )
    provenance = require_mapping(
        dataset.get("provenance"), "SWE run metadata.dataset.provenance"
    )
    expected_provenance = {
        "repo": SWE_DATASETS[suite_name]["repo"],
        "revision": pins[revision_pin],
        "expected": expected_instances,
        "rows": expected_instances,
        "jsonl_sha256": SWE_DATASETS[suite_name]["jsonl_sha256"],
        "parquet_sha256": SWE_DATASETS[suite_name]["parquet_sha256"],
    }
    require_integer(
        int(pins[count_pin]), expected_instances, f"campaign pin {count_pin}"
    )
    for field, expected in expected_provenance.items():
        require_equal(
            provenance.get(field),
            expected,
            f"SWE run metadata.dataset.provenance.{field}",
        )
    for field, suffix in (
        ("agent_dataset", f"/agent/{suite_name}"),
        ("evaluator_dataset", f"/evaluator/{suite_name}.jsonl"),
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.endswith(suffix):
            raise EvidenceError(
                f"SWE run metadata.dataset.provenance.{field} must end with {suffix!r}"
            )
    return validate_campaign_source(
        metadata.get("campaign_source"),
        source_commit=source_commit,
        path="SWE run metadata.campaign_source",
    )


def validate_swe_runtime_evidence(
    metadata: dict[str, Any],
    generation: dict[str, Any],
    *,
    variant: str,
    suite_name: str,
    target_ids: set[str],
) -> dict[str, Any]:
    expected_environment = {
        "constraints_lock_sha256": SWE_CONSTRAINTS_LOCK_SHA256,
        "normalized_freeze_sha256": SWE_NORMALIZED_FREEZE_SHA256,
        "normalized_requirement_count": 101,
    }
    environment = require_mapping(
        metadata.get("python_environment"), "SWE run metadata.python_environment"
    )
    if set(environment) != {
        "constraints_lock_sha256",
        "freeze_sha256",
        "normalized_freeze_sha256",
        "normalized_requirement_count",
    }:
        raise EvidenceError("SWE Python environment identity contains invalid fields")
    for field, expected in expected_environment.items():
        require_equal(
            environment.get(field), expected, f"SWE python_environment.{field}"
        )
    sha256_value(
        environment.get("freeze_sha256"), "SWE python_environment.freeze_sha256"
    )
    require_equal(
        generation.get("python_environment"),
        environment,
        "SWE generation.python_environment",
    )
    require_equal(
        generation.get("runtime_binding"),
        metadata.get("runtime_binding"),
        "SWE generation.runtime_binding",
    )
    deployment, deployment_sha256, content_sha256 = unwrap_runtime_binding(
        metadata.get("runtime_binding")
    )
    wrapper = require_mapping(metadata.get("runtime_binding"), "SWE runtime binding")
    content = require_mapping(wrapper.get("content"), "SWE runtime binding content")
    evaluator = require_mapping(content.get("evaluator"), "SWE runtime evaluator")
    expected_evaluator_fields = {
        "deployment_source_sha256",
        "runtime_family",
        "runtime_source_revision",
        "dynamo_enabled",
        "tensor_parallel_size",
        "generation",
        "evaluation",
        "effective_config_sha256",
        "effective_config_content_sha256",
        "effective_config",
        "endpoint_evidence",
        "campaign_source",
    }
    if set(evaluator) != expected_evaluator_fields:
        raise EvidenceError("SWE runtime evaluator identity contains invalid fields")
    require_equal(
        evaluator.get("campaign_source"),
        metadata.get("campaign_source"),
        "SWE evaluator.campaign_source",
    )
    require_string(
        evaluator.get("deployment_source_sha256"),
        deployment_sha256,
        "SWE evaluator.deployment_source_sha256",
    )
    runtime_family = (
        "vllm" if variant.endswith("vllm") or "vllm" in variant else "sglang"
    )
    require_string(
        evaluator.get("runtime_family"), runtime_family, "SWE evaluator.runtime_family"
    )
    runtime_revision = evaluator.get("runtime_source_revision")
    if (
        not isinstance(runtime_revision, str)
        or len(runtime_revision) != 40
        or any(character not in "0123456789abcdef" for character in runtime_revision)
    ):
        raise EvidenceError(
            "SWE evaluator.runtime_source_revision must be a Git commit"
        )
    require_bool(
        evaluator.get("dynamo_enabled"),
        variant.startswith("dynamo-"),
        "SWE evaluator.dynamo_enabled",
    )
    require_integer(
        evaluator.get("tensor_parallel_size"), 4, "SWE evaluator.tensor_parallel_size"
    )
    require_equal(
        evaluator.get("generation"),
        {"workers": 16, "batch_size": 8},
        "SWE evaluator.generation",
    )
    expected_evaluation = {
        "workers": 8,
        "timeout_seconds": 3600,
        "backend": "local" if suite_name == "pro" else "official-swebench-docker",
        "docker_platform": "linux/amd64" if suite_name == "pro" else None,
    }
    require_equal(
        evaluator.get("evaluation"), expected_evaluation, "SWE evaluator.evaluation"
    )
    effective_config = require_mapping(
        evaluator.get("effective_config"), "SWE evaluator.effective_config"
    )
    effective_file_sha256 = sha256_value(
        evaluator.get("effective_config_sha256"),
        "SWE evaluator.effective_config_sha256",
    )
    effective_content_sha256 = sha256_value(
        evaluator.get("effective_config_content_sha256"),
        "SWE evaluator.effective_config_content_sha256",
        canonical_sha256(effective_config),
    )
    require_string(
        generation.get("effective_config_sha256"),
        effective_content_sha256,
        "SWE generation.effective_config_sha256",
    )
    require_equal(
        generation.get("trajectory_effective_config_sha256s"),
        [effective_content_sha256],
        "SWE generation.trajectory_effective_config_sha256s",
    )
    endpoint_evidence = require_mapping(
        evaluator.get("endpoint_evidence"), "SWE evaluator.endpoint_evidence"
    )
    if set(endpoint_evidence) != {"file_sha256", "content_sha256", "content"}:
        raise EvidenceError("SWE endpoint evidence contains invalid fields")
    sha256_value(endpoint_evidence.get("file_sha256"), "SWE endpoint file SHA-256")
    endpoint_content = require_mapping(
        endpoint_evidence.get("content"), "SWE endpoint evidence content"
    )
    sha256_value(
        endpoint_evidence.get("content_sha256"),
        "SWE endpoint content SHA-256",
        canonical_sha256(endpoint_content),
    )
    selected_model = require_mapping(
        endpoint_content.get("selected_model_response"),
        "SWE endpoint selected model",
    )
    require_string(selected_model.get("id"), "zai-org/GLM-5.2", "SWE endpoint model")
    require_integer(
        selected_model.get("context_window"),
        RUNTIME_MAX_MODEL_LEN,
        "SWE endpoint context_window",
    )

    for field in (
        "missing_task_image_ids",
        "unexpected_task_image_ids",
        "invalid_task_image_ids",
    ):
        require_empty_list(generation.get(field), f"SWE generation.{field}")
    task_evidence = require_mapping(
        generation.get("task_image_evidence"), "SWE generation.task_image_evidence"
    )
    task_evidence_sha256 = sha256_value(
        task_evidence.get("sha256"), "SWE task image evidence SHA-256"
    )
    images = require_mapping(task_evidence.get("images"), "SWE task image map")
    if set(images) != target_ids:
        raise EvidenceError("SWE task image map does not match the pinned run scope")
    for instance_id, raw_identity in images.items():
        identity = require_mapping(raw_identity, f"SWE task image {instance_id}")
        if set(identity) != {
            "requested_ref",
            "image_id",
            "repo_digests",
            "content_identity_sha256",
        }:
            raise EvidenceError(f"SWE task image {instance_id} contains invalid fields")
        if (
            not isinstance(identity.get("requested_ref"), str)
            or not identity["requested_ref"]
        ):
            raise EvidenceError(
                f"SWE task image {instance_id} requested_ref is invalid"
            )
        image_id = identity.get("image_id")
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise EvidenceError(f"SWE task image {instance_id} image_id is invalid")
        sha256_value(
            image_id.removeprefix("sha256:"), f"SWE task image {instance_id} image_id"
        )
        repo_digests = require_list(
            identity.get("repo_digests"), f"SWE task image {instance_id} repo_digests"
        )
        if (
            not repo_digests
            or not all(isinstance(digest, str) for digest in repo_digests)
            or repo_digests != sorted(set(repo_digests))
        ):
            raise EvidenceError(
                f"SWE task image {instance_id} repo_digests are invalid"
            )
        for digest in repo_digests:
            if not isinstance(digest, str) or "@sha256:" not in digest:
                raise EvidenceError(
                    f"SWE task image {instance_id} repo digest is invalid"
                )
            sha256_value(
                digest.rsplit("@sha256:", 1)[1],
                f"SWE task image {instance_id} repo digest",
            )
        sha256_value(
            identity.get("content_identity_sha256"),
            f"SWE task image {instance_id} content identity",
            canonical_sha256({"image_id": image_id, "repo_digests": repo_digests}),
        )

    fairness_config = json.loads(json.dumps(effective_config))
    model_config = fairness_config.get("model")
    if isinstance(model_config, dict) and isinstance(
        model_config.get("model_kwargs"), dict
    ):
        model_config["model_kwargs"]["api_base"] = "<campaign-endpoint>"
    return {
        "python_environment": environment,
        "effective_config_file_sha256": effective_file_sha256,
        "effective_config_content_sha256": effective_content_sha256,
        "fairness_config_sha256": canonical_sha256(fairness_config),
        "task_image_evidence_sha256": task_evidence_sha256,
        "task_image_map_sha256": canonical_sha256(images),
        "generation": evaluator["generation"],
        "evaluation": evaluator["evaluation"],
        "runtime_source_revision": runtime_revision,
        "runtime_family": runtime_family,
        "runtime_deployment_sha256": deployment_sha256,
        "runtime_content_sha256": content_sha256,
    }


def validate_swe_environment_files(
    run_dir: Path, environment: dict[str, Any]
) -> tuple[Path, Path]:
    freeze_path = run_dir / "environment.freeze.txt"
    normalized_path = run_dir / "environment.normalized.freeze.txt"
    require_string(
        sha256_file(freeze_path),
        environment["freeze_sha256"],
        "SWE environment.freeze.txt SHA-256",
    )
    require_string(
        sha256_file(normalized_path),
        environment["normalized_freeze_sha256"],
        "SWE environment.normalized.freeze.txt SHA-256",
    )

    lock_lines = sorted(
        line.strip()
        for line in (ROOT / "eval/swebench/constraints.lock").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(lock_lines) != len(set(lock_lines)):
        raise EvidenceError("SWE constraints lock contains duplicate requirements")
    raw_requirements: list[str] = []
    editable_repos: list[str] = []
    for raw_line in freeze_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-e file://"):
            editable_repos.append(Path(line.removeprefix("-e file://")).name)
        elif line.startswith("-e "):
            raise EvidenceError(f"SWE freeze contains unsupported editable: {line}")
        else:
            raw_requirements.append(line)
    if set(editable_repos) != SWE_EDITABLE_REPOS or len(editable_repos) != len(
        SWE_EDITABLE_REPOS
    ):
        raise EvidenceError(
            "SWE freeze does not contain the exact editable repositories"
        )
    if len(raw_requirements) != len(set(raw_requirements)):
        raise EvidenceError("SWE freeze contains duplicate requirements")
    if sorted(raw_requirements) != lock_lines:
        raise EvidenceError("SWE freeze requirements differ from constraints.lock")
    expected_normalized = "\n".join(lock_lines) + "\n"
    if normalized_path.read_text() != expected_normalized:
        raise EvidenceError(
            "SWE normalized freeze is not the canonical constraints-lock environment"
        )
    require_integer(
        environment.get("normalized_requirement_count"),
        len(lock_lines),
        "SWE normalized requirement count",
    )
    return freeze_path, normalized_path


def import_swe(
    variant: str,
    phase: str,
    suite: dict[str, Any],
    artifact: Path,
    payload: dict[str, Any],
    served_model: str,
    image: str,
    source_commit: str,
) -> dict[str, Any]:
    score_path, generation_path, metadata_path, scope_path = swe_paths(
        artifact, payload
    )
    score = load_object(score_path)
    generation = load_object(generation_path)
    metadata = load_object(metadata_path)
    scope = load_object(scope_path)
    expected = suite["units"]
    expected_name = SWE_SUITE_NAMES[suite["id"]]

    require_integer(scope.get("schema_version"), 1, "SWE run scope.schema_version")
    require_string(scope.get("scope"), "full", "SWE run scope.scope")
    require_bool(scope.get("full_run"), True, "SWE run scope.full_run")
    require_integer(
        scope.get("dataset_instances"), expected, "SWE run scope.dataset_instances"
    )
    require_integer(
        scope.get("target_instances"), expected, "SWE run scope.target_instances"
    )
    require_equal(scope.get("instance_filter"), None, "SWE run scope.instance_filter")
    require_equal(scope.get("instance_slice"), None, "SWE run scope.instance_slice")
    target_ids = string_set(scope.get("target_ids"), "SWE run scope.target_ids")
    if len(target_ids) != expected:
        raise EvidenceError(
            "SWE run scope target_ids do not cover the official dataset"
        )
    sha256_value(
        sha256_file(scope_path),
        "SWE run scope SHA-256",
        SWE_DATASETS[expected_name]["scope_sha256"],
    )

    require_string(score.get("suite"), expected_name, "SWE score.suite")
    require_string(score.get("scope"), "full", "SWE score.scope")
    require_bool(score.get("full_run"), True, "SWE score.full_run")
    require_bool(score.get("complete"), True, "SWE score.complete")
    require_empty_list(score.get("gate_failures"), "SWE score.gate_failures")
    for field in ("dataset_instances", "expected_instances", "target_instances"):
        require_integer(score.get(field), expected, f"SWE score.{field}")
    require_integer(
        score.get("excluded_dataset_instances"),
        0,
        "SWE score.excluded_dataset_instances",
    )
    submitted = require_integer(
        score.get("submitted_instances"), expected, "SWE score.submitted_instances"
    )
    completed = integer(
        score.get("completed_instances"), "SWE score.completed_instances"
    )
    passed = integer(score.get("passed_instances"), "SWE score.passed_instances")
    failed = integer(score.get("failed_instances"), "SWE score.failed_instances")
    missing = require_integer(
        score.get("missing_instances"), 0, "SWE score.missing_instances"
    )
    if passed + failed != expected:
        raise EvidenceError(
            "SWE passed_instances + failed_instances must equal dataset size"
        )
    for field in (
        "missing_ids",
        "unexpected_ids",
        "incomplete_evaluation_ids",
        "missing_evaluation_ids",
        "unexpected_evaluation_ids",
        "evaluation_error_ids",
    ):
        require_empty_list(score.get(field), f"SWE score.{field}")
    passed_ids = string_set(score.get("passed_ids"), "SWE score.passed_ids")
    failed_ids = string_set(score.get("failed_ids"), "SWE score.failed_ids")
    unresolved_ids = string_set(score.get("unresolved_ids"), "SWE score.unresolved_ids")
    empty_patch_ids = string_set(
        score.get("empty_patch_ids"), "SWE score.empty_patch_ids"
    )
    if len(passed_ids) != passed or len(failed_ids) != failed:
        raise EvidenceError("SWE outcome ID lists do not match passed/failed counts")
    if passed_ids & failed_ids:
        raise EvidenceError("SWE passed_ids and failed_ids overlap")
    if passed_ids | failed_ids != target_ids:
        raise EvidenceError("SWE score outcome IDs do not match the pinned run scope")
    if unresolved_ids & empty_patch_ids:
        raise EvidenceError("SWE unresolved_ids and empty_patch_ids overlap")
    if failed_ids != unresolved_ids | empty_patch_ids:
        raise EvidenceError(
            "SWE failed_ids must equal unresolved_ids plus valid empty-patch outcomes"
        )
    if completed != len(passed_ids | unresolved_ids):
        raise EvidenceError(
            "SWE completed_instances does not match resolved/unresolved evaluator outcomes"
        )
    benchmark_score = fraction(
        score.get("benchmark_score"), "SWE score.benchmark_score"
    )
    submitted_score = fraction(
        score.get("score_on_submitted"), "SWE score.score_on_submitted"
    )
    close(benchmark_score, passed / expected, "SWE score.benchmark_score")
    close(submitted_score, passed / submitted, "SWE score.score_on_submitted")

    require_string(generation.get("scope"), "full", "SWE generation.scope")
    require_bool(generation.get("full_run"), True, "SWE generation.full_run")
    require_bool(generation.get("complete"), True, "SWE generation.complete")
    require_empty_list(generation.get("gate_failures"), "SWE generation.gate_failures")
    for field in (
        "dataset_instances",
        "expected_instances",
        "target_instances",
        "predictions",
    ):
        require_integer(generation.get(field), expected, f"SWE generation.{field}")
    require_integer(
        generation.get("excluded_dataset_instances"),
        0,
        "SWE generation.excluded_dataset_instances",
    )
    require_integer(
        generation.get("missing_predictions"), 0, "SWE generation.missing_predictions"
    )
    for field in (
        "missing_prediction_ids",
        "unexpected_prediction_ids",
        "infrastructure_error_ids",
    ):
        require_empty_list(generation.get(field), f"SWE generation.{field}")
    instances = require_list(generation.get("instances"), "SWE generation.instances")
    if len(instances) != expected:
        raise EvidenceError("SWE generation.instances does not cover the run scope")
    generation_ids: set[str] = set()
    for index, raw_instance in enumerate(instances):
        instance = require_mapping(raw_instance, f"SWE generation.instances[{index}]")
        instance_id = instance.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id in generation_ids
        ):
            raise EvidenceError(
                f"SWE generation.instances[{index}].instance_id is invalid or duplicate"
            )
        generation_ids.add(instance_id)
        require_bool(
            instance.get("valid_model_result"),
            True,
            f"SWE generation instance {instance_id}.valid_model_result",
        )
        require_empty_list(
            instance.get("validation_failures"),
            f"SWE generation instance {instance_id}.validation_failures",
        )
        calls = integer(
            instance.get("api_calls"),
            f"SWE generation instance {instance_id}.api_calls",
        )
        if calls < 1:
            raise EvidenceError(
                f"SWE generation instance {instance_id}.api_calls must be positive"
            )
    if generation_ids != target_ids:
        raise EvidenceError(
            "SWE generation instance IDs do not match the pinned run scope"
        )

    campaign_source = validate_swe_run_identity(
        metadata,
        variant=variant,
        suite_name=expected_name,
        expected_instances=expected,
        served_model=served_model,
        phase=phase,
        source_commit=source_commit,
    )
    binding, deployment_sha256, content_sha256 = validate_runtime_binding(
        metadata.get("runtime_binding"),
        variant=variant,
        served_model=served_model,
        image=image,
        harness_endpoint=metadata["endpoint"],
        phase=phase,
        source_commit=source_commit,
    )
    suite_identity = validate_swe_runtime_evidence(
        metadata,
        generation,
        variant=variant,
        suite_name=expected_name,
        target_ids=target_ids,
    )
    environment_paths = validate_swe_environment_files(
        score_path.parent, suite_identity["python_environment"]
    )
    images = generation["task_image_evidence"]["images"]
    for index, raw_instance in enumerate(instances):
        instance = require_mapping(raw_instance, f"SWE generation.instances[{index}]")
        instance_id = instance["instance_id"]
        require_string(
            instance.get("effective_config_sha256"),
            suite_identity["effective_config_content_sha256"],
            f"SWE generation instance {instance_id}.effective_config_sha256",
        )
        require_equal(
            instance.get("task_image"),
            images[instance_id],
            f"SWE generation instance {instance_id}.task_image",
        )
        require_string(
            instance.get("runtime_deployment_sha256"),
            suite_identity["runtime_deployment_sha256"],
            f"SWE generation instance {instance_id}.runtime_deployment_sha256",
        )
        require_string(
            instance.get("runtime_content_sha256"),
            suite_identity["runtime_content_sha256"],
            f"SWE generation instance {instance_id}.runtime_content_sha256",
        )
    continuity_path = score_path.parent / "runtime-continuity.json"
    validate_runtime_continuity(
        continuity_path,
        variant=variant,
        phase=phase,
        binding=binding,
        binding_sha256=deployment_sha256,
    )
    row = result_row(
        variant,
        suite["id"],
        phase,
        expected,
        expected,
        expected,
        {
            "benchmark_score": benchmark_score,
            "score_on_submitted": submitted_score,
            "passed_instances": passed,
            "failed_instances": failed,
            "missing_instances": missing,
        },
        [
            source(
                "swe-score",
                score_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "swe-generation-validation",
                generation_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "swe-run-metadata",
                metadata_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "swe-run-scope",
                scope_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "runtime-continuity",
                continuity_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "swe-environment-freeze",
                environment_paths[0],
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "swe-environment-normalized-freeze",
                environment_paths[1],
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
        ],
        runtime_identity(binding, deployment_sha256, content_sha256),
        campaign_source,
    )
    row["suite_identity"] = suite_identity
    return attach_task_level(
        row,
        swe_task_records(target_ids, passed_ids, unresolved_ids, empty_patch_ids),
        variant=variant,
        suite=suite["id"],
        phase=phase,
    )


def command_values(command: list[Any], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(command):
        if value == option:
            if index + 1 >= len(command) or not isinstance(command[index + 1], str):
                raise EvidenceError(f"Terminal command {option} has no value")
            values.append(command[index + 1])
    return values


def require_command_value(command: list[Any], option: str, expected: str) -> None:
    values = command_values(command, option)
    if values != [expected]:
        raise EvidenceError(
            f"Terminal command {option} must be exactly {expected!r}, found {values!r}"
        )


def validate_terminal_command(
    command: Any, run_spec: dict[str, Any], pins: dict[str, str]
) -> None:
    values = require_list(command, "Terminal invocation.command")
    if not values or not all(isinstance(value, str) for value in values):
        raise EvidenceError(
            "Terminal invocation.command must be a non-empty string array"
        )
    require_command_value(values, "--dataset", pins["TERMINALBENCH_DATASET"])
    require_command_value(values, "--agent", pins["TERMINUS_AGENT"])
    require_command_value(values, "--model", f"openai/{run_spec['served_model']}")
    require_command_value(
        values, "--n-attempts", pins["TERMINALBENCH_OFFICIAL_ATTEMPTS"]
    )
    require_command_value(values, "--n-concurrent", "4")
    require_command_value(values, "--max-retries", "0")
    require_command_value(
        values, "--timeout-multiplier", pins["TERMINALBENCH_TIMEOUT_MULTIPLIER"]
    )
    require_command_value(values, "--env", "docker")
    if values.count("--delete") != 1 or values.count("--yes") != 1:
        raise EvidenceError(
            "Terminal command must include --delete and --yes exactly once"
        )
    if "--n-tasks" in values:
        raise EvidenceError("Terminal full command must not limit the task population")

    max_context = int(pins["TERMINALBENCH_MAX_CONTEXT_TOKENS"])
    max_output = int(pins["TERMINALBENCH_MAX_OUTPUT_TOKENS"])
    model_info = json.dumps(
        {
            "max_tokens": max_context,
            "max_input_tokens": max_context - max_output,
            "max_output_tokens": max_output,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        separators=(",", ":"),
    )
    call_kwargs = json.dumps(
        {"max_tokens": max_output, "top_p": float(pins["TERMINALBENCH_TOP_P"])},
        separators=(",", ":"),
    )
    expected_agent_kwargs = {
        f"api_base={run_spec['api_base']}",
        "parser_name=json",
        f"max_turns={pins['TERMINALBENCH_MAX_TURNS']}",
        f"temperature={pins['TERMINALBENCH_TEMPERATURE']}",
        f"model_info={model_info}",
        f"llm_call_kwargs={call_kwargs}",
    }
    actual_agent_kwargs = command_values(values, "--agent-kwarg")
    if (
        len(actual_agent_kwargs) != len(expected_agent_kwargs)
        or set(actual_agent_kwargs) != expected_agent_kwargs
    ):
        raise EvidenceError(
            "Terminal command --agent-kwarg values do not match the pinned parser/model recipe"
        )


def validate_terminal_run_identity(
    run_metadata: dict[str, Any],
    *,
    variant: str,
    served_model: str,
    campaign: dict[str, Any],
    expected_task_names: list[Any],
    phase: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pins = load_assignments(TERMINAL_PINS)
    require_integer(
        run_metadata.get("schema_version"), 2, "Terminal run_metadata.schema_version"
    )
    run_spec = require_mapping(
        run_metadata.get("run_spec"), "Terminal run_metadata.run_spec"
    )
    expected_spec = {
        "mode": "full",
        "label": variant,
        "campaign_phase": phase,
        "dataset": pins["TERMINALBENCH_DATASET"],
        "dataset_revision": int(pins["TERMINALBENCH_DATASET_REVISION"]),
        "dataset_content_hash": pins["TERMINALBENCH_DATASET_CONTENT_HASH"],
        "dataset_version_id": pins["TERMINALBENCH_DATASET_VERSION_ID"],
        "expected_tasks": int(pins["TERMINALBENCH_TASK_COUNT"]),
        "attempts_per_task": int(pins["TERMINALBENCH_OFFICIAL_ATTEMPTS"]),
        "expected_trials": int(pins["TERMINALBENCH_TASK_COUNT"])
        * int(pins["TERMINALBENCH_OFFICIAL_ATTEMPTS"]),
        "agent": pins["TERMINUS_AGENT"],
        "litellm_model": f"openai/{served_model}",
        "served_model": served_model,
        "n_concurrent": 4,
        "temperature": float(pins["TERMINALBENCH_TEMPERATURE"]),
        "top_p": float(pins["TERMINALBENCH_TOP_P"]),
        "max_turns": int(pins["TERMINALBENCH_MAX_TURNS"]),
        "max_context_tokens": int(pins["TERMINALBENCH_MAX_CONTEXT_TOKENS"]),
        "max_output_tokens": int(pins["TERMINALBENCH_MAX_OUTPUT_TOKENS"]),
        "timeout_multiplier": float(pins["TERMINALBENCH_TIMEOUT_MULTIPLIER"]),
    }
    for field, expected in expected_spec.items():
        require_equal(run_spec.get(field), expected, f"Terminal run_spec.{field}")
    require_http_v1_endpoint(run_spec.get("api_base"), "Terminal run_spec.api_base")
    if not identity_contains_token(run_spec.get("job_name"), phase):
        raise EvidenceError(
            f"Terminal run_spec.job_name must contain campaign phase {phase!r}"
        )

    metadata_pins = require_mapping(
        run_metadata.get("pins"), "Terminal run_metadata.pins"
    )
    expected_pin_fields: dict[str, Any] = {
        "harbor_repository": pins["HARBOR_REPOSITORY"],
        "harbor_version": pins["HARBOR_VERSION"],
        "harbor_commit": pins["HARBOR_COMMIT"],
        "harbor_uv_lock_sha256": TERMINAL_HARBOR_UV_LOCK_SHA256,
        "dataset": pins["TERMINALBENCH_DATASET"],
        "dataset_revision": int(pins["TERMINALBENCH_DATASET_REVISION"]),
        "dataset_content_hash": pins["TERMINALBENCH_DATASET_CONTENT_HASH"],
        "dataset_version_id": pins["TERMINALBENCH_DATASET_VERSION_ID"],
    }
    for field, expected in expected_pin_fields.items():
        require_equal(
            metadata_pins.get(field), expected, f"Terminal run_metadata.pins.{field}"
        )
    resolved = require_mapping(
        metadata_pins.get("resolved_dataset"),
        "Terminal run_metadata.pins.resolved_dataset",
    )
    resolved_expected = {
        "requested_ref": pins["TERMINALBENCH_DATASET"],
        "name": pins["TERMINALBENCH_DATASET"].rsplit("@", 1)[0],
        "version": f"sha256:{pins['TERMINALBENCH_DATASET_CONTENT_HASH']}",
        "dataset_version_id": pins["TERMINALBENCH_DATASET_VERSION_ID"],
        "content_hash": pins["TERMINALBENCH_DATASET_CONTENT_HASH"],
        "task_count": int(pins["TERMINALBENCH_TASK_COUNT"]),
    }
    for field, expected in resolved_expected.items():
        require_equal(
            resolved.get(field),
            expected,
            f"Terminal run_metadata.pins.resolved_dataset.{field}",
        )
    task_refs = require_list(
        resolved.get("task_refs"),
        "Terminal run_metadata.pins.resolved_dataset.task_refs",
    )
    if len(task_refs) != len(expected_task_names):
        raise EvidenceError(
            "Terminal resolved dataset task refs do not cover every task"
        )
    sha256_value(
        canonical_sha256(task_refs),
        "Terminal resolved dataset task_refs SHA-256",
        TERMINAL_TASK_REFS_SHA256,
    )
    ref_names: list[str] = []
    seen_ref_names: set[str] = set()
    for index, raw_ref in enumerate(task_refs):
        ref = require_mapping(raw_ref, f"Terminal resolved task ref[{index}]")
        name = ref.get("name")
        if not isinstance(name, str) or not name:
            raise EvidenceError(f"Terminal resolved task ref[{index}] has invalid name")
        require_string(
            ref.get("org"), "terminal-bench", f"Terminal task ref {name}.org"
        )
        qualified_name = f"terminal-bench/{name}"
        if qualified_name in seen_ref_names:
            raise EvidenceError(
                f"Terminal resolved task ref[{index}] has duplicate name"
            )
        seen_ref_names.add(qualified_name)
        ref_names.append(qualified_name)
        digest = ref.get("ref")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise EvidenceError(f"Terminal task ref {name}.ref is not a SHA-256 ref")
        sha256_value(digest.removeprefix("sha256:"), f"Terminal task ref {name}.ref")
    if ref_names != expected_task_names:
        raise EvidenceError(
            "Terminal resolved dataset task names do not match validation"
        )

    source_identity = require_mapping(
        run_metadata.get("source"), "Terminal run_metadata.source"
    )
    source_commit = campaign.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise EvidenceError("campaign.source_commit must be a lowercase Git commit ID")
    campaign_source = validate_campaign_source(
        run_metadata.get("campaign_source"),
        source_commit=source_commit,
        path="Terminal run_metadata.campaign_source",
    )
    require_integer(
        source_identity.get("schema_version"), 2, "Terminal source.schema_version"
    )
    require_string(
        source_identity.get("source_commit"),
        source_commit,
        "Terminal source.source_commit",
    )
    require_string(
        source_identity.get("source_branch"),
        campaign["branch"],
        "Terminal source.source_branch",
    )
    require_equal(
        source_identity.get("bundle_contents"),
        ["campaign.env", "eval"],
        "Terminal source.bundle_contents",
    )
    sha256_value(source_identity.get("bundle_sha256"), "Terminal source.bundle_sha256")
    require_equal(
        source_identity.get("source_clean"), True, "Terminal source.source_clean"
    )
    require_integer(
        source_identity.get("source_changed_path_count"),
        0,
        "Terminal source.source_changed_path_count",
    )
    for field in CAMPAIGN_SOURCE_FIELDS - {
        "schema_version",
        "source_commit",
        "source_clean",
        "source_changed_path_count",
    }:
        require_equal(
            source_identity.get(field),
            campaign_source[field],
            f"Terminal source.{field}",
        )

    wrapper = require_mapping(
        run_metadata.get("runtime_binding"), "Terminal runtime binding"
    )
    content = require_mapping(
        wrapper.get("content"), "Terminal runtime binding content"
    )
    require_equal(
        content.get("evaluator"),
        {"campaign_source": campaign_source},
        "Terminal runtime binding evaluator",
    )

    harbor_environment = require_mapping(
        run_metadata.get("harbor_environment"),
        "Terminal run_metadata.harbor_environment",
    )
    if set(harbor_environment) != {
        "uv_sync_check",
        "python",
        "package_count",
        "packages_sha256",
        "packages",
    }:
        raise EvidenceError("Terminal harbor_environment contains invalid fields")
    require_string(
        harbor_environment.get("uv_sync_check"),
        "passed",
        "Terminal harbor_environment.uv_sync_check",
    )
    if (
        not isinstance(harbor_environment.get("python"), str)
        or not harbor_environment["python"]
    ):
        raise EvidenceError("Terminal harbor_environment.python is invalid")
    packages = require_list(
        harbor_environment.get("packages"), "Terminal harbor_environment.packages"
    )
    normalized_names: set[str] = set()
    canonical_packages: list[list[str]] = []
    for index, raw_package in enumerate(packages):
        package = require_list(
            raw_package, f"Terminal harbor_environment.packages[{index}]"
        )
        if len(package) != 2 or not all(
            isinstance(value, str) and value for value in package
        ):
            raise EvidenceError(
                f"Terminal harbor_environment.packages[{index}] is invalid"
            )
        normalized_name = re.sub(r"[-_.]+", "-", package[0]).casefold()
        if normalized_name in normalized_names:
            raise EvidenceError(
                "Terminal harbor_environment contains duplicate normalized package names"
            )
        normalized_names.add(normalized_name)
        canonical_packages.append(package)
    sorted_packages = sorted(
        canonical_packages,
        key=lambda item: (
            re.sub(r"[-_.]+", "-", item[0]).casefold(),
            item[0],
            item[1],
        ),
    )
    if not canonical_packages or canonical_packages != sorted_packages:
        raise EvidenceError("Terminal harbor_environment packages are not canonical")
    require_integer(
        harbor_environment.get("package_count"),
        len(canonical_packages),
        "Terminal harbor_environment.package_count",
    )
    packages_payload = json.dumps(
        canonical_packages, sort_keys=True, separators=(",", ":")
    ).encode()
    sha256_value(
        harbor_environment.get("packages_sha256"),
        "Terminal harbor_environment.packages_sha256",
        hashlib.sha256(packages_payload).hexdigest(),
    )

    invocations = require_list(
        run_metadata.get("invocations"), "Terminal run_metadata.invocations"
    )
    if not invocations:
        raise EvidenceError("Terminal run_metadata.invocations must not be empty")
    for index, raw_invocation in enumerate(invocations):
        invocation = require_mapping(raw_invocation, f"Terminal invocation[{index}]")
        validate_terminal_command(invocation.get("command"), run_spec, pins)
    return campaign_source, harbor_environment


def terminal_wall_time(run_metadata: dict[str, Any]) -> float | None:
    invocations = require_list(
        run_metadata.get("invocations"), "Terminal run_metadata.invocations"
    )
    if not invocations:
        raise EvidenceError("Terminal run_metadata.invocations must not be empty")
    elapsed = 0.0
    for index, raw_invocation in enumerate(invocations):
        invocation = require_mapping(raw_invocation, f"Terminal invocation[{index}]")
        if not invocation.get("finished_at"):
            raise EvidenceError(f"Terminal invocation[{index}] is not finished")
        elapsed_value = invocation.get("elapsed_seconds")
        if elapsed_value is None:
            return None
        seconds = finite_number(
            elapsed_value, f"Terminal invocation[{index}].elapsed_seconds"
        )
        if seconds < 0:
            raise EvidenceError(
                f"Terminal invocation[{index}].elapsed_seconds is negative"
            )
        elapsed += seconds
    last = require_mapping(invocations[-1], "Terminal final invocation")
    require_integer(
        last.get("harbor_exit_code"), 0, "Terminal final invocation.harbor_exit_code"
    )
    return elapsed


def validate_terminal_task_images(
    path: Path,
    embedded: Any,
    *,
    expected_sha256: Any,
    expected_task_names: list[Any],
    expected_task_refs: list[Any],
    expected_attempts: int,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceError(
            f"Terminal task-image evidence must be a regular file: {path}"
        )
    document = load_object(path)
    require_equal(document, embedded, "Terminal embedded task_images")
    sha256_value(
        expected_sha256,
        "Terminal input_hashes.task_images_sha256",
        sha256_file(path),
    )
    if set(document) != {"schema_version", "task_count", "trial_count", "tasks"}:
        raise EvidenceError("Terminal task-images.json contains invalid root fields")
    require_integer(document.get("schema_version"), 1, "Terminal task images schema")
    require_integer(
        document.get("task_count"),
        len(expected_task_names),
        "Terminal task image count",
    )
    require_integer(
        document.get("trial_count"),
        len(expected_task_names) * expected_attempts,
        "Terminal task image trial count",
    )
    tasks = require_list(document.get("tasks"), "Terminal task images tasks")
    if len(tasks) != len(expected_task_names):
        raise EvidenceError(
            "Terminal task image rows do not cover the full task population"
        )
    if len(expected_task_refs) != len(expected_task_names):
        raise EvidenceError("Terminal task refs do not cover the full task population")

    compact_map: dict[str, Any] = {}
    for index, (raw_task, expected_name, raw_expected_ref) in enumerate(
        zip(tasks, expected_task_names, expected_task_refs)
    ):
        task = require_mapping(raw_task, f"Terminal task image[{index}]")
        expected_fields = {
            "task_name",
            "task_ref",
            "task_checksum",
            "task_toml_sha256",
            "requested_ref",
            "image_id",
            "repo_digests",
        }
        if set(task) != expected_fields:
            raise EvidenceError(f"Terminal task image[{index}] contains invalid fields")
        require_string(
            task.get("task_name"),
            str(expected_name),
            f"Terminal task image[{index}].task_name",
        )
        expected_ref = require_mapping(
            raw_expected_ref, f"Terminal expected task ref[{index}]"
        )
        task_ref = require_mapping(
            task.get("task_ref"), f"Terminal task image[{index}].task_ref"
        )
        if set(task_ref) != {"org", "name", "ref"}:
            raise EvidenceError(f"Terminal task image[{index}].task_ref is invalid")
        require_equal(task_ref, expected_ref, f"Terminal task image[{index}].task_ref")
        require_string(
            task_ref.get("org"),
            "terminal-bench",
            f"Terminal task image[{index}].task_ref.org",
        )
        require_string(
            f"{task_ref['org']}/{task_ref['name']}",
            str(expected_name),
            f"Terminal task image[{index}] qualified task ref",
        )
        ref = task_ref.get("ref")
        if not isinstance(ref, str) or not ref.startswith("sha256:"):
            raise EvidenceError(f"Terminal task image[{index}].task_ref.ref is invalid")
        sha256_value(
            ref.removeprefix("sha256:"), f"Terminal task image[{index}].task_ref.ref"
        )
        for field in ("task_checksum", "task_toml_sha256"):
            sha256_value(task.get(field), f"Terminal task image[{index}].{field}")
        requested_ref = task.get("requested_ref")
        if (
            not isinstance(requested_ref, str)
            or not requested_ref
            or any(
                character.isspace() or character == "\0" for character in requested_ref
            )
        ):
            raise EvidenceError(
                f"Terminal task image[{index}].requested_ref is invalid"
            )
        image_id = task.get("image_id")
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise EvidenceError(f"Terminal task image[{index}].image_id is invalid")
        sha256_value(
            image_id.removeprefix("sha256:"), f"Terminal task image[{index}].image_id"
        )
        repo_digests = require_list(
            task.get("repo_digests"), f"Terminal task image[{index}].repo_digests"
        )
        if (
            not repo_digests
            or not all(isinstance(value, str) for value in repo_digests)
            or repo_digests != sorted(set(repo_digests))
        ):
            raise EvidenceError(f"Terminal task image[{index}].repo_digests is invalid")
        for digest in repo_digests:
            repository, separator, sha256 = digest.rpartition("@sha256:")
            if not separator or not repository:
                raise EvidenceError(
                    f"Terminal task image[{index}] contains an invalid RepoDigest"
                )
            sha256_value(
                sha256,
                f"Terminal task image[{index}] RepoDigest",
            )
        compact_map[str(expected_name)] = {
            field: task[field] for field in sorted(expected_fields - {"task_name"})
        }
    return {
        "task_count": len(tasks),
        "trial_count": len(expected_task_names) * expected_attempts,
        "task_image_map_sha256": canonical_sha256(compact_map),
    }


def import_terminal(
    variant: str,
    phase: str,
    suite: dict[str, Any],
    artifact: Path,
    payload: dict[str, Any],
    served_model: str,
    campaign: dict[str, Any],
    image: str,
) -> dict[str, Any]:
    require_integer(payload.get("schema_version"), 2, "Terminal summary.schema_version")
    validation = require_mapping(payload.get("validation"), "Terminal validation")
    require_bool(validation.get("strict"), True, "Terminal validation.strict")
    require_bool(validation.get("complete"), True, "Terminal validation.complete")
    require_empty_list(validation.get("errors"), "Terminal validation.errors")
    expected_tasks = suite["units"]
    expected_attempts = suite["attempts"]
    expected_trials = expected_tasks * expected_attempts
    for field, expected in (
        ("expected_tasks", expected_tasks),
        ("expected_attempts_per_task", expected_attempts),
        ("expected_trials", expected_trials),
        ("observed_tasks", expected_tasks),
        ("observed_trials", expected_trials),
    ):
        require_integer(validation.get(field), expected, f"Terminal validation.{field}")
    task_names = require_list(
        validation.get("expected_task_names"), "Terminal expected_task_names"
    )
    if len(task_names) != expected_tasks or len(set(task_names)) != expected_tasks:
        raise EvidenceError(
            "Terminal expected_task_names must contain every task exactly once"
        )

    score = require_mapping(payload.get("score"), "Terminal score")
    passed = integer(score.get("passed_attempts"), "Terminal score.passed_attempts")
    failed = integer(score.get("failed_attempts"), "Terminal score.failed_attempts")
    errored = require_integer(
        score.get("errored_attempts"), 0, "Terminal score.errored_attempts"
    )
    no_reward = require_integer(
        score.get("no_reward_attempts"), 0, "Terminal score.no_reward_attempts"
    )
    if passed + failed + errored + no_reward != expected_trials:
        raise EvidenceError("Terminal attempt outcomes do not sum to expected trials")
    pass_at_k = require_mapping(score.get("pass_at_k"), "Terminal score.pass_at_k")
    pass_metrics: dict[str, float] = {}
    for k in range(1, expected_attempts + 1):
        pass_metrics[f"pass_at_{k}"] = fraction(
            pass_at_k.get(str(k)), f"Terminal score.pass_at_k[{k}]"
        )
    ordered = [pass_metrics[f"pass_at_{k}"] for k in range(1, expected_attempts + 1)]
    if ordered != sorted(ordered):
        raise EvidenceError("Terminal pass@k values must be non-decreasing")

    tasks = require_list(payload.get("tasks"), "Terminal tasks")
    if len(tasks) != expected_tasks:
        raise EvidenceError(f"Terminal tasks must contain {expected_tasks} rows")
    observed_names: set[str] = set()
    task_pass_sums = [0.0] * expected_attempts
    outcome_sums = [0, 0, 0, 0]
    for index, raw_task in enumerate(tasks):
        task = require_mapping(raw_task, f"Terminal tasks[{index}]")
        task_name = task.get("task_name")
        if (
            not isinstance(task_name, str)
            or not task_name
            or task_name in observed_names
        ):
            raise EvidenceError(
                f"Terminal tasks[{index}].task_name is invalid or duplicate"
            )
        observed_names.add(task_name)
        require_integer(
            task.get("attempts"),
            expected_attempts,
            f"Terminal task {task_name}.attempts",
        )
        task_outcomes = [
            integer(task.get(field), f"Terminal task {task_name}.{field}")
            for field in (
                "passed_attempts",
                "failed_attempts",
                "errored_attempts",
                "no_reward_attempts",
            )
        ]
        if sum(task_outcomes) != expected_attempts or task_outcomes[2:] != [0, 0]:
            raise EvidenceError(
                f"Terminal task {task_name} has invalid attempt outcomes"
            )
        outcome_sums = [
            left + right for left, right in zip(outcome_sums, task_outcomes)
        ]
        for k in range(1, expected_attempts + 1):
            task_pass_sums[k - 1] += fraction(
                task.get(f"pass_at_{k}"), f"Terminal task {task_name}.pass_at_{k}"
            )
    if observed_names != set(task_names):
        raise EvidenceError("Terminal task rows do not match expected_task_names")
    if outcome_sums != [passed, failed, errored, no_reward]:
        raise EvidenceError(
            "Terminal task outcomes do not reconcile with aggregate outcomes"
        )
    for k, total in enumerate(task_pass_sums, start=1):
        close(
            pass_metrics[f"pass_at_{k}"],
            round(total / expected_tasks, 8),
            f"Terminal score.pass_at_k[{k}]",
            tolerance=1e-8,
        )

    run_metadata = require_mapping(payload.get("run_metadata"), "Terminal run_metadata")
    campaign_source, harbor_environment = validate_terminal_run_identity(
        run_metadata,
        variant=variant,
        served_model=served_model,
        campaign=campaign,
        expected_task_names=task_names,
        phase=phase,
    )
    terminal_spec = require_mapping(
        run_metadata.get("run_spec"), "Terminal run_metadata.run_spec"
    )
    binding, deployment_sha256, content_sha256 = validate_runtime_binding(
        run_metadata.get("runtime_binding"),
        variant=variant,
        served_model=served_model,
        image=image,
        harness_endpoint=terminal_spec["api_base"],
        phase=phase,
        source_commit=campaign["source_commit"],
    )
    terminal_wrapper = require_mapping(
        run_metadata.get("runtime_binding"), "Terminal runtime binding"
    )
    terminal_content = require_mapping(
        terminal_wrapper.get("content"), "Terminal runtime binding content"
    )
    require_equal(
        terminal_content.get("evaluator"),
        {"campaign_source": campaign_source},
        "Terminal runtime binding evaluator",
    )
    require_string(
        terminal_spec.get("runtime_deployment_sha256"),
        deployment_sha256,
        "Terminal run_spec.runtime_deployment_sha256",
    )
    continuity_path = artifact.parent / "runtime-continuity.json"
    validate_runtime_continuity(
        continuity_path,
        variant=variant,
        phase=phase,
        binding=binding,
        binding_sha256=deployment_sha256,
    )
    hashes = require_mapping(payload.get("input_hashes"), "Terminal input_hashes")
    for field in (
        "job_result_sha256",
        "job_config_sha256",
        "job_lock_sha256",
        "dataset_metadata_sha256",
        "task_images_sha256",
        "run_metadata_sha256",
    ):
        sha256_value(hashes.get(field), f"Terminal input_hashes.{field}")
    serialized_metadata = (
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n"
    ).encode()
    expected_metadata_sha256 = hashlib.sha256(serialized_metadata).hexdigest()
    sha256_value(
        hashes.get("run_metadata_sha256"),
        "Terminal input_hashes.run_metadata_sha256",
        expected_metadata_sha256,
    )

    metrics: dict[str, int | float] = {
        **pass_metrics,
        "passed_attempts": passed,
        "failed_attempts": failed,
        "errored_attempts": errored,
        "no_reward_attempts": no_reward,
    }
    mean_reward = score.get("mean_reward_all_trials")
    if mean_reward is not None:
        metrics["mean_reward_all_trials"] = finite_number(
            mean_reward, "Terminal score.mean_reward_all_trials"
        )
    trials_path = artifact.parent / "trials.csv"
    task_records = terminal_task_records(
        trials_path,
        expected_task_names=task_names,
        expected_attempts=expected_attempts,
    )
    resolved_dataset = require_mapping(
        require_mapping(run_metadata.get("pins"), "Terminal run_metadata.pins").get(
            "resolved_dataset"
        ),
        "Terminal run_metadata.pins.resolved_dataset",
    )
    task_images_path = artifact.parent / "task-images.json"
    task_image_identity = validate_terminal_task_images(
        task_images_path,
        payload.get("task_images"),
        expected_sha256=hashes.get("task_images_sha256"),
        expected_task_names=task_names,
        expected_task_refs=require_list(
            resolved_dataset.get("task_refs"),
            "Terminal run_metadata.pins.resolved_dataset.task_refs",
        ),
        expected_attempts=expected_attempts,
    )
    row = result_row(
        variant,
        suite["id"],
        phase,
        expected_tasks,
        expected_tasks,
        expected_trials,
        metrics,
        [
            source(
                "terminalbench-summary",
                artifact,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "runtime-continuity",
                continuity_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "terminalbench-trials",
                trials_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
            source(
                "terminalbench-task-images",
                task_images_path,
                variant=variant,
                suite=suite["id"],
                phase=phase,
            ),
        ],
        runtime_identity(binding, deployment_sha256, content_sha256),
        campaign_source,
        terminal_wall_time(run_metadata),
    )
    row["suite_identity"] = {
        "harbor_environment": harbor_environment,
        "task_images": task_image_identity,
    }
    return attach_task_level(
        row, task_records, variant=variant, suite=suite["id"], phase=phase
    )


def build_result(
    summary: dict[str, Any],
    variant: str,
    suite_id: str,
    phase: str,
    artifact: Path,
    *,
    include_task_payload: bool = False,
) -> dict[str, Any]:
    if phase not in CAMPAIGN_PHASES:
        raise EvidenceError(f"phase must be one of {CAMPAIGN_PHASES}, found {phase!r}")
    variant_row = variant_definition(summary, variant)
    suite = suite_definition(summary, suite_id)
    artifact = artifact.resolve()
    payload = load_object(artifact)
    kind = suite.get("kind")
    served_model = summary.get("campaign", {}).get("served_model_name")
    if not isinstance(served_model, str) or not served_model:
        raise EvidenceError("campaign.served_model_name is missing")
    campaign = require_mapping(summary.get("campaign"), "campaign")
    source_commit = campaign.get("source_commit")
    if not isinstance(source_commit, str):
        raise EvidenceError("campaign.source_commit is required for result import")
    importers: dict[str, Callable[..., dict[str, Any]]] = {
        "bfcl": import_bfcl,
        "swebench": import_swe,
        "terminalbench": import_terminal,
    }
    if kind not in importers:
        raise EvidenceError(f"unsupported suite kind: {kind!r}")
    if kind == "bfcl":
        row = import_bfcl(
            variant,
            phase,
            suite,
            artifact,
            payload,
            served_model,
            variant_row["image"],
            source_commit,
        )
    elif kind == "swebench":
        row = import_swe(
            variant,
            phase,
            suite,
            artifact,
            payload,
            served_model,
            variant_row["image"],
            source_commit,
        )
    else:
        row = import_terminal(
            variant,
            phase,
            suite,
            artifact,
            payload,
            served_model,
            campaign,
            variant_row["image"],
        )
    if not include_task_payload:
        row.pop("_task_level_payload", None)
    return row


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    atomic_write_bytes(path, payload)


def task_payload_path(results_dir: Path, logical_path: str) -> Path:
    parts = Path(logical_path).parts
    if len(parts) < 3 or parts[:2] != ("results", "task-level"):
        raise EvidenceError("task-level path is outside results/task-level")
    return results_dir.joinpath(*parts[1:])


def task_record_index(
    payload: bytes, label: str
) -> dict[tuple[str, int | None], dict[str, Any]]:
    index: dict[tuple[str, int | None], dict[str, Any]] = {}
    try:
        lines = payload.decode().splitlines()
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label} is not UTF-8 JSONL") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{label}:{line_number} is invalid JSON") from error
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise EvidenceError(f"{label}:{line_number} has an invalid task ID")
        attempt = record.get("attempt")
        if attempt is not None and (
            isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1
        ):
            raise EvidenceError(f"{label}:{line_number} has an invalid attempt")
        if record.get("outcome") not in {"passed", "failed"}:
            raise EvidenceError(f"{label}:{line_number} has an invalid outcome")
        key = (record["id"], attempt)
        if key in index:
            raise EvidenceError(f"{label} contains a duplicate task/attempt key")
        index[key] = record
    return index


def build_paired_disagreements(
    summary: dict[str, Any],
    results_dir: Path,
    payload_overrides: dict[str, bytes],
) -> tuple[list[dict[str, Any]], dict[Path, bytes]]:
    rows = {
        (row.get("variant"), row.get("suite"), row.get("phase")): row
        for row in summary.get("results", [])
        if row.get("status") == "complete"
    }
    entries: list[dict[str, Any]] = []
    outputs: dict[Path, bytes] = {}
    for suite in summary.get("suites", []):
        for phase in CAMPAIGN_PHASES:
            for pair in summary.get("pairs", []):
                dynamo = rows.get((pair["dynamo_variant"], suite["id"], phase))
                native = rows.get((pair["native_variant"], suite["id"], phase))
                if dynamo is None or native is None:
                    continue

                def payload_for(row: dict[str, Any]) -> bytes:
                    logical = row["task_level"]["path"]
                    if logical in payload_overrides:
                        return payload_overrides[logical]
                    path = task_payload_path(results_dir, logical)
                    try:
                        payload = path.read_bytes()
                    except OSError as error:
                        raise EvidenceError(
                            f"missing task-level evidence for paired comparison: {path}"
                        ) from error
                    require_equal(
                        hashlib.sha256(payload).hexdigest(),
                        row["task_level"]["sha256"],
                        f"task-level digest for {logical}",
                    )
                    return payload

                dynamo_index = task_record_index(
                    payload_for(dynamo), f"{pair['dynamo_variant']} task-level"
                )
                native_index = task_record_index(
                    payload_for(native), f"{pair['native_variant']} task-level"
                )
                if set(dynamo_index) != set(native_index):
                    raise EvidenceError(
                        f"paired task populations differ for {suite['id']}/{phase}/{pair['id']}"
                    )
                disagreements: list[dict[str, Any]] = []
                for task_id, attempt in sorted(
                    dynamo_index, key=lambda key: (key[0], key[1] or 0)
                ):
                    dynamo_outcome = dynamo_index[(task_id, attempt)]["outcome"]
                    native_outcome = native_index[(task_id, attempt)]["outcome"]
                    if dynamo_outcome == native_outcome:
                        continue
                    record: dict[str, Any] = {
                        "id": task_id,
                        "dynamo_outcome": dynamo_outcome,
                        "native_outcome": native_outcome,
                    }
                    if attempt is not None:
                        record["attempt"] = attempt
                    disagreements.append(record)
                payload = jsonl_payload(disagreements)
                logical_path = (
                    "results/paired-disagreements/"
                    f"{suite['id']}/{phase}/{pair['id']}.jsonl"
                )
                physical_path = results_dir.joinpath(*Path(logical_path).parts[1:])
                outputs[physical_path] = payload
                entries.append(
                    {
                        "suite": suite["id"],
                        "phase": phase,
                        "pair": pair["id"],
                        "path": logical_path,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "compared_records": len(dynamo_index),
                        "disagreement_records": len(disagreements),
                    }
                )
    return entries, outputs


def update_summary(
    summary_path: Path,
    variant: str,
    suite_id: str,
    phase: str,
    artifact: Path,
    *,
    replace: bool = False,
) -> tuple[dict[str, Any], bool]:
    summary_path = summary_path.resolve()
    lock_path = summary_path.with_suffix(summary_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summary = load_object(summary_path)
        assert_pinned_report_sources(summary)
        validate(summary)
        new_row = build_result(
            summary,
            variant,
            suite_id,
            phase,
            artifact,
            include_task_payload=True,
        )
        task_payload = new_row.pop("_task_level_payload").encode()
        task_metadata = require_mapping(new_row.get("task_level"), "task_level")
        task_output = task_payload_path(summary_path.parent, task_metadata["path"])
        require_equal(
            hashlib.sha256(task_payload).hexdigest(),
            task_metadata["sha256"],
            "task-level payload digest",
        )
        matching = [
            (index, row)
            for index, row in enumerate(summary["results"])
            if row.get("variant") == variant
            and row.get("suite") == suite_id
            and row.get("phase") == phase
        ]
        if len(matching) > 1:
            raise EvidenceError(
                f"summary contains duplicate {variant}/{suite_id}/{phase} rows"
            )
        row_changed = False
        if matching:
            index, existing = matching[0]
            if existing != new_row:
                if not replace:
                    raise EvidenceError(
                        "summary already contains "
                        f"{variant}/{suite_id}/{phase}; pass --replace to replace it"
                    )
                summary["results"][index] = new_row
                row_changed = True
        else:
            summary["results"].append(new_row)
            row_changed = True
        variant_order = {
            row["id"]: index for index, row in enumerate(summary["variants"])
        }
        suite_order = {row["id"]: index for index, row in enumerate(summary["suites"])}
        phase_order = {
            phase_id: index for index, phase_id in enumerate(CAMPAIGN_PHASES)
        }
        summary["results"].sort(
            key=lambda row: (
                variant_order[row["variant"]],
                suite_order[row["suite"]],
                phase_order[row["phase"]],
            )
        )
        disagreement_entries, disagreement_outputs = build_paired_disagreements(
            summary,
            summary_path.parent,
            {task_metadata["path"]: task_payload},
        )
        disagreements_changed = (
            summary.get("paired_disagreements") != disagreement_entries
        )
        summary["paired_disagreements"] = disagreement_entries
        validate(summary)
        sidecar_overrides = {task_metadata["path"]: task_payload}
        sidecar_overrides.update(
            {
                entry["path"]: disagreement_outputs[
                    summary_path.parent.joinpath(*Path(entry["path"]).parts[1:])
                ]
                for entry in disagreement_entries
            }
        )
        validate_sidecars(
            summary,
            summary_path.parent,
            payload_overrides=sidecar_overrides,
        )
        changed = row_changed or disagreements_changed
        sidecar_writes = {task_output: task_payload, **disagreement_outputs}
        previous_sidecars: dict[Path, tuple[int, bytes] | None] = {}
        for output_path in sidecar_writes:
            if output_path.exists():
                if output_path.is_symlink() or not output_path.is_file():
                    raise EvidenceError(
                        f"sidecar output is not a regular file: {output_path}"
                    )
                previous_sidecars[output_path] = (
                    output_path.stat().st_mode & 0o777,
                    output_path.read_bytes(),
                )
            else:
                previous_sidecars[output_path] = None
        try:
            for output_path, output_payload in sidecar_writes.items():
                atomic_write_bytes(output_path, output_payload)
            if changed:
                atomic_write_json(summary_path, summary)
        except Exception:
            for output_path, previous in reversed(previous_sidecars.items()):
                if previous is None:
                    output_path.unlink(missing_ok=True)
                else:
                    mode, payload = previous
                    atomic_write_bytes(output_path, payload)
                    output_path.chmod(mode)
            raise
        return summary, changed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--variant", required=True)
    result.add_argument("--suite", required=True)
    result.add_argument("--phase", required=True, choices=CAMPAIGN_PHASES)
    result.add_argument("--artifact", required=True, type=Path)
    result.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    result.add_argument(
        "--replace",
        action="store_true",
        help=(
            "replace an existing variant/suite/phase row after fully validating "
            "new evidence"
        ),
    )
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        _, changed = update_summary(
            args.summary,
            args.variant,
            args.suite,
            args.phase,
            args.artifact,
            replace=args.replace,
        )
    except (EvidenceError, ValueError) as error:
        print(f"result import failed: {error}", file=os.sys.stderr)
        return 2
    action = "updated" if changed else "already current"
    print(f"{action}: {args.variant}/{args.suite}/{args.phase} in {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
