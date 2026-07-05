#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate summary.json and generate the self-contained GLM-5.2 report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "summary.json"
DEFAULT_OUTPUT = ROOT / "report" / "glm52-nscale-comparison.html"

METRIC_FORMATS = {"integer", "number", "percent"}
METRIC_DIRECTIONS = {"higher", "lower"}
CAMPAIGN_STATUSES = {"pending", "in_progress", "blocked", "failed", "complete"}
RESULT_STATUSES = {
    "pending",
    "starting",
    "running",
    "in_progress",
    "partial",
    "blocked",
    "failed",
    "complete",
}
VARIANT_STATUSES = RESULT_STATUSES | {"ready"}
CAMPAIGN_PHASE_IDS = ("ab", "ba")
EXPECTED_FULL_RESULT_COUNT = 40
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RESULT_IMPORTER_ID = "glm52-result-import/v1"
PINNED_SOURCE_PATHS = (
    "campaign.env",
    "eval",
    "report/import_result.py",
    "report/generate.py",
)


class SchemaError(ValueError):
    """Raised when summary.json could produce an ambiguous or stale report."""


def esc(value: Any) -> str:
    if value is None:
        return "—"
    return html.escape(str(value))


def number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return esc(value)


def format_metric(value: Any, definition: dict[str, Any]) -> str:
    if value is None:
        return "—"
    metric_format = definition["format"]
    if metric_format == "percent":
        return f"{value * 100:,.2f}%"
    if metric_format == "integer":
        return f"{value:,}"
    return number(value)


def format_delta(delta: float | int, definition: dict[str, Any]) -> str:
    if definition["format"] == "percent":
        return f"{delta * 100:+,.2f} pp"
    if definition["format"] == "integer":
        return f"{delta:+,d}"
    return f"{delta:+,.2f}"


def status_badge(status: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]", "-", status.lower().replace("_", "-"))
    return f'<span class="badge badge-{normalized}">{esc(status)}</span>'


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{path} must be an array")
    return value


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{path} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{path} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    value = _require_nonnegative_int(value, path)
    if value == 0:
        raise SchemaError(f"{path} must be greater than zero")
    return value


def _require_timestamp(value: Any, path: str) -> datetime:
    value = _require_string(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SchemaError(f"{path} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SchemaError(f"{path} must include a UTC offset")
    return parsed


def _unique_by_id(rows: list[Any], path: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for position, raw_row in enumerate(rows):
        row = _require_mapping(raw_row, f"{path}[{position}]")
        row_id = _require_string(row.get("id"), f"{path}[{position}].id")
        if row_id in index:
            raise SchemaError(f"duplicate {path} id: {row_id!r}")
        index[row_id] = row
    return index


def _validate_metric_value(value: Any, definition: dict[str, Any], path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be numeric")
    if not math.isfinite(value):
        raise SchemaError(f"{path} must be finite")
    if definition["format"] == "integer":
        if not isinstance(value, int) or value < 0:
            raise SchemaError(f"{path} must be a non-negative integer")
    elif definition["format"] == "percent" and not 0 <= value <= 1:
        raise SchemaError(f"{path} must be a fraction in [0, 1]")


def summary_sha256(data: dict[str, Any]) -> str:
    """Return a stable digest of the parsed summary represented by the report."""

    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_pinned_report_sources(
    data: dict[str, Any],
    *,
    campaign_root: Path = ROOT,
    allow_unpinned_scaffold: bool = False,
) -> None:
    """Require all evaluator/import/report behavior to equal campaign.source_commit."""

    campaign = _require_mapping(data.get("campaign"), "campaign")
    source_commit = campaign.get("source_commit")
    if source_commit is None:
        if allow_unpinned_scaffold and not data.get("results"):
            return
        raise SchemaError(
            "campaign.source_commit may be null only for a result-free scaffold"
        )
    if (
        not isinstance(source_commit, str)
        or GIT_SHA_RE.fullmatch(source_commit) is None
    ):
        raise SchemaError("campaign.source_commit must be a lowercase Git commit")

    campaign_root = campaign_root.resolve()
    try:
        repository = Path(
            subprocess.run(
                ["git", "-C", str(campaign_root), "rev-parse", "--show-toplevel"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ).stdout.strip()
        ).resolve()
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "cat-file",
                "-e",
                f"{source_commit}^{{commit}}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                source_commit,
                "HEAD",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SchemaError(
            f"campaign source commit is unavailable or not an ancestor: {source_commit}"
        ) from error

    try:
        campaign_relative = campaign_root.relative_to(repository)
    except ValueError as error:
        raise SchemaError("campaign directory is outside its Git repository") from error
    guarded_paths = [str(campaign_relative / path) for path in PINNED_SOURCE_PATHS]
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--name-only",
            "--no-ext-diff",
            source_commit,
            "--",
            *guarded_paths,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *guarded_paths,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    drift = sorted(set(tracked + untracked))
    if drift:
        raise SchemaError(
            "campaign evaluator/import/report sources differ from "
            f"campaign.source_commit {source_commit}: {drift}"
        )


def _validate_evidence(
    value: Any, path: str, *, variant: str, suite: str, phase: str
) -> None:
    evidence = _require_mapping(value, path)
    importer = _require_string(evidence.get("importer"), f"{path}.importer")
    if importer != RESULT_IMPORTER_ID:
        raise SchemaError(f"{path}.importer must be {RESULT_IMPORTER_ID!r}")
    sources = _require_list(evidence.get("sources"), f"{path}.sources")
    if not sources:
        raise SchemaError(f"{path}.sources must not be empty")
    roles: set[str] = set()
    for index, raw_source in enumerate(sources):
        source = _require_mapping(raw_source, f"{path}.sources[{index}]")
        role = _require_string(source.get("role"), f"{path}.sources[{index}].role")
        if role in roles:
            raise SchemaError(f"{path}.sources contains duplicate role {role!r}")
        roles.add(role)
        source_path = _require_string(
            source.get("path"), f"{path}.sources[{index}].path"
        )
        logical_parts = source_path.removeprefix("artifact://").split("/")
        if (
            not source_path.startswith("artifact://")
            or "\\" in source_path
            or "/../" in source_path
            or source_path.endswith("/..")
            or len(logical_parts) != 5
        ):
            raise SchemaError(
                f"{path}.sources[{index}].path must be a logical artifact URI"
            )
        if logical_parts[:4] != [suite, phase, variant, role]:
            raise SchemaError(
                f"{path}.sources[{index}].path identity does not match its result row"
            )
        digest = _require_string(
            source.get("sha256"), f"{path}.sources[{index}].sha256"
        )
        if SHA256_RE.fullmatch(digest) is None:
            raise SchemaError(
                f"{path}.sources[{index}].sha256 must be a lowercase SHA-256 digest"
            )


def _validate_task_level(value: Any, path: str, expected_records: int) -> None:
    task_level = _require_mapping(value, path)
    logical_path = _require_string(task_level.get("path"), f"{path}.path")
    if not logical_path.startswith("results/task-level/") or ".." in logical_path:
        raise SchemaError(f"{path}.path must be under results/task-level")
    digest = _require_string(task_level.get("sha256"), f"{path}.sha256")
    if SHA256_RE.fullmatch(digest) is None:
        raise SchemaError(f"{path}.sha256 must be a lowercase SHA-256 digest")
    records = _require_nonnegative_int(task_level.get("records"), f"{path}.records")
    if records != expected_records:
        raise SchemaError(
            f"{path}.records must cover {expected_records} task-level outcomes"
        )


def _validate_runtime_identity(value: Any, path: str) -> None:
    identity = _require_mapping(value, path)
    expected_fields = {
        "deployment_sha256",
        "content_sha256",
        "captured_at",
        "controller_uid_sha256",
        "pod_uid_sha256_by_role",
        "capture_sha256",
        "recipe",
        "hardware",
        "control_plane",
    }
    if set(identity) != expected_fields:
        raise SchemaError(f"{path} must contain the exact public runtime identity")
    for field in (
        "deployment_sha256",
        "content_sha256",
        "controller_uid_sha256",
        "capture_sha256",
    ):
        digest = _require_string(identity.get(field), f"{path}.{field}")
        if SHA256_RE.fullmatch(digest) is None:
            raise SchemaError(f"{path}.{field} must be a lowercase SHA-256 digest")
    _require_timestamp(identity.get("captured_at"), f"{path}.captured_at")
    pod_uids = _require_mapping(
        identity.get("pod_uid_sha256_by_role"), f"{path}.pod_uid_sha256_by_role"
    )
    if not pod_uids:
        raise SchemaError(f"{path}.pod_uid_sha256_by_role must not be empty")
    for role, digest in pod_uids.items():
        _require_string(role, f"{path}.pod_uid_sha256_by_role role")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise SchemaError(
                f"{path}.pod_uid_sha256_by_role[{role!r}] must be a SHA-256 digest"
            )
    recipe = _require_mapping(identity.get("recipe"), f"{path}.recipe")
    if set(recipe) != {
        "source_commit",
        "template_sha256",
        "rendered_manifest_sha256",
    }:
        raise SchemaError(f"{path}.recipe contains invalid fields")
    if (
        not isinstance(recipe.get("source_commit"), str)
        or GIT_SHA_RE.fullmatch(recipe["source_commit"]) is None
    ):
        raise SchemaError(f"{path}.recipe.source_commit must be a Git commit")
    for field in ("template_sha256", "rendered_manifest_sha256"):
        if (
            not isinstance(recipe.get(field), str)
            or SHA256_RE.fullmatch(recipe[field]) is None
        ):
            raise SchemaError(f"{path}.recipe.{field} must be a SHA-256 digest")
    hardware = _require_mapping(identity.get("hardware"), f"{path}.hardware")
    if hardware.get("gpu_count") != 4 or hardware.get("gpu_model") != "NVIDIA B200":
        raise SchemaError(f"{path}.hardware must identify 4x NVIDIA B200")


def _validate_campaign_source(value: Any, path: str, source_commit: str | None) -> None:
    identity = _require_mapping(value, path)
    expected_fields = {
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
    if set(identity) != expected_fields:
        raise SchemaError(f"{path} contains invalid fields")
    if identity.get("schema_version") != 1:
        raise SchemaError(f"{path}.schema_version must be 1")
    if source_commit is None or identity.get("source_commit") != source_commit:
        raise SchemaError(f"{path}.source_commit differs from campaign")
    if identity.get("source_clean") is not True:
        raise SchemaError(f"{path}.source_clean must be true")
    if identity.get("source_changed_path_count") != 0:
        raise SchemaError(f"{path}.source_changed_path_count must be zero")
    for field in (
        "bundle_sha256",
        "source_tree_sha256",
        "eval_tree_sha256",
        "campaign_env_sha256",
    ):
        digest = _require_string(identity.get(field), f"{path}.{field}")
        if SHA256_RE.fullmatch(digest) is None:
            raise SchemaError(f"{path}.{field} must be a SHA-256 digest")
    source_count = _require_positive_int(
        identity.get("source_file_count"), f"{path}.source_file_count"
    )
    eval_count = _require_positive_int(
        identity.get("eval_file_count"), f"{path}.eval_file_count"
    )
    if source_count != eval_count + 1:
        raise SchemaError(f"{path} file counts are inconsistent")


def _validate_harbor_environment(value: Any, path: str) -> None:
    identity = _require_mapping(value, path)
    if set(identity) != {
        "uv_sync_check",
        "python",
        "package_count",
        "packages_sha256",
        "packages",
    }:
        raise SchemaError(f"{path} contains invalid fields")
    if identity.get("uv_sync_check") != "passed":
        raise SchemaError(f"{path}.uv_sync_check must be passed")
    _require_string(identity.get("python"), f"{path}.python")
    packages = _require_list(identity.get("packages"), f"{path}.packages")
    normalized_names: set[str] = set()
    canonical: list[list[str]] = []
    for index, raw_package in enumerate(packages):
        package = _require_list(raw_package, f"{path}.packages[{index}]")
        if len(package) != 2 or not all(
            isinstance(item, str) and item for item in package
        ):
            raise SchemaError(f"{path}.packages[{index}] is invalid")
        normalized = re.sub(r"[-_.]+", "-", package[0]).casefold()
        if normalized in normalized_names:
            raise SchemaError(f"{path} contains duplicate normalized package names")
        normalized_names.add(normalized)
        canonical.append(package)
    expected_order = sorted(
        canonical,
        key=lambda item: (
            re.sub(r"[-_.]+", "-", item[0]).casefold(),
            item[0],
            item[1],
        ),
    )
    if not canonical or canonical != expected_order:
        raise SchemaError(f"{path}.packages is not canonical")
    if _require_positive_int(
        identity.get("package_count"), f"{path}.package_count"
    ) != len(canonical):
        raise SchemaError(f"{path}.package_count does not match packages")
    digest = _require_string(identity.get("packages_sha256"), f"{path}.packages_sha256")
    expected_digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != expected_digest:
        raise SchemaError(f"{path}.packages_sha256 does not match packages")


def _validate_terminal_task_images_identity(
    value: Any, path: str, *, expected_tasks: int, expected_trials: int
) -> None:
    identity = _require_mapping(value, path)
    if set(identity) != {"task_count", "trial_count", "task_image_map_sha256"}:
        raise SchemaError(f"{path} contains invalid fields")
    if (
        _require_positive_int(identity.get("task_count"), f"{path}.task_count")
        != expected_tasks
    ):
        raise SchemaError(f"{path}.task_count must be {expected_tasks}")
    if (
        _require_positive_int(identity.get("trial_count"), f"{path}.trial_count")
        != expected_trials
    ):
        raise SchemaError(f"{path}.trial_count must be {expected_trials}")
    digest = _require_string(
        identity.get("task_image_map_sha256"), f"{path}.task_image_map_sha256"
    )
    if SHA256_RE.fullmatch(digest) is None:
        raise SchemaError(f"{path}.task_image_map_sha256 must be a SHA-256 digest")


def _validate_swe_suite_identity(value: Any, path: str) -> None:
    identity = _require_mapping(value, path)
    expected_fields = {
        "python_environment",
        "effective_config_file_sha256",
        "effective_config_content_sha256",
        "fairness_config_sha256",
        "task_image_evidence_sha256",
        "task_image_map_sha256",
        "generation",
        "evaluation",
        "runtime_source_revision",
        "runtime_family",
        "runtime_deployment_sha256",
        "runtime_content_sha256",
    }
    if set(identity) != expected_fields:
        raise SchemaError(f"{path} contains invalid SWE identity fields")
    for field in (
        "effective_config_file_sha256",
        "effective_config_content_sha256",
        "fairness_config_sha256",
        "task_image_evidence_sha256",
        "task_image_map_sha256",
        "runtime_deployment_sha256",
        "runtime_content_sha256",
    ):
        digest = _require_string(identity.get(field), f"{path}.{field}")
        if SHA256_RE.fullmatch(digest) is None:
            raise SchemaError(f"{path}.{field} must be a SHA-256 digest")
    revision = _require_string(
        identity.get("runtime_source_revision"), f"{path}.runtime_source_revision"
    )
    if GIT_SHA_RE.fullmatch(revision) is None:
        raise SchemaError(f"{path}.runtime_source_revision must be a Git commit")
    if identity.get("runtime_family") not in {"vllm", "sglang"}:
        raise SchemaError(f"{path}.runtime_family is invalid")
    _require_mapping(identity.get("python_environment"), f"{path}.python_environment")
    _require_mapping(identity.get("generation"), f"{path}.generation")
    _require_mapping(identity.get("evaluation"), f"{path}.evaluation")


def _strict_json_object(line: str, path: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SchemaError(f"{path} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(line, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as error:
        raise SchemaError(f"{path} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise SchemaError(f"{path} must be a JSON object")
    return value


def _read_jsonl_payload(payload: bytes, path: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SchemaError(f"{path} must be UTF-8 JSONL") from error
    if text and not text.endswith("\n"):
        raise SchemaError(f"{path} must end with a newline")
    lines = text.splitlines()
    if any(not line for line in lines):
        raise SchemaError(f"{path} must not contain blank JSONL records")
    records = [
        _strict_json_object(line, f"{path}:{index}")
        for index, line in enumerate(lines, start=1)
    ]
    canonical = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode()
    if payload != canonical:
        raise SchemaError(f"{path} must use canonical JSONL encoding")
    return records


def _safe_sidecar_path(results_dir: Path, logical_path: str, category: str) -> Path:
    logical = PurePosixPath(logical_path)
    if (
        logical.is_absolute()
        or ".." in logical.parts
        or "." in logical.parts
        or logical.parts[:2] != ("results", category)
        or logical.suffix != ".jsonl"
    ):
        raise SchemaError(
            f"sidecar path must be a logical results/{category} JSONL path: "
            f"{logical_path!r}"
        )
    results_dir = results_dir.resolve()
    physical = results_dir.joinpath(*logical.parts[1:])
    if not physical.resolve().is_relative_to(results_dir):
        raise SchemaError(f"sidecar path escapes results directory: {logical_path!r}")
    return physical


def _validate_task_records(
    records: list[dict[str, Any]], suite: dict[str, Any], path: str
) -> dict[tuple[str, int | None], dict[str, Any]]:
    index: dict[tuple[str, int | None], dict[str, Any]] = {}
    for position, record in enumerate(records):
        prefix = f"{path}[{position}]"
        task_id = _require_string(record.get("id"), f"{prefix}.id")
        if any(character in task_id for character in "\r\n\0"):
            raise SchemaError(f"{prefix}.id contains a control character")
        outcome = _require_string(record.get("outcome"), f"{prefix}.outcome")
        if outcome not in {"passed", "failed"}:
            raise SchemaError(f"{prefix}.outcome must be passed or failed")

        kind = suite["kind"]
        attempt: int | None = None
        if kind == "bfcl":
            allowed = {"id", "category", "outcome", "error_types"}
            category = _require_string(record.get("category"), f"{prefix}.category")
            if any(character in category for character in "\r\n\0"):
                raise SchemaError(f"{prefix}.category contains a control character")
            error_types = record.get("error_types")
            if outcome == "passed" and error_types is not None:
                raise SchemaError(f"{prefix} passed result must not have error_types")
            if outcome == "failed":
                values = _require_list(error_types, f"{prefix}.error_types")
                if (
                    not values
                    or not all(isinstance(value, str) and value for value in values)
                    or values != sorted(set(values))
                ):
                    raise SchemaError(f"{prefix}.error_types is invalid")
        elif kind == "swebench":
            allowed = {"id", "outcome", "failure_kind"}
            failure_kind = record.get("failure_kind")
            if outcome == "passed" and failure_kind is not None:
                raise SchemaError(f"{prefix} passed result must not have failure_kind")
            if outcome == "failed" and failure_kind not in {
                "empty_patch",
                "unresolved",
            }:
                raise SchemaError(f"{prefix}.failure_kind is invalid")
        elif kind == "terminalbench":
            allowed = {"id", "attempt", "outcome", "reward"}
            attempt = _require_positive_int(record.get("attempt"), f"{prefix}.attempt")
            if attempt > suite["attempts"]:
                raise SchemaError(f"{prefix}.attempt exceeds suite attempts")
            reward = record.get("reward")
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                raise SchemaError(f"{prefix}.reward must be finite")
            if (outcome == "passed") != (reward > 0):
                raise SchemaError(f"{prefix}.reward does not match outcome")
        else:
            raise SchemaError(f"{prefix} has unsupported suite kind {kind!r}")
        if not set(record) <= allowed:
            raise SchemaError(f"{prefix} contains non-sanitized task fields")
        key = (task_id, attempt)
        if key in index:
            raise SchemaError(f"{path} contains duplicate task/attempt {key!r}")
        index[key] = record
    return index


def validate_sidecars(
    data: dict[str, Any],
    results_dir: Path,
    *,
    payload_overrides: dict[str, bytes] | None = None,
) -> None:
    """Verify every committed task/disagreement sidecar represented by summary.json."""

    payload_overrides = payload_overrides or {}
    results_dir = results_dir.resolve()
    suite_index = {suite["id"]: suite for suite in data["suites"]}
    pair_index = {pair["id"]: pair for pair in data["pairs"]}
    complete_rows = {
        (row["variant"], row["suite"], row["phase"]): row
        for row in data["results"]
        if row["status"] == "complete"
    }
    task_indexes: dict[
        tuple[str, str, str], dict[tuple[str, int | None], dict[str, Any]]
    ] = {}
    expected_task_paths: set[str] = set()

    def payload_for(logical_path: str, category: str) -> bytes:
        physical = _safe_sidecar_path(results_dir, logical_path, category)
        if physical.exists() and (physical.is_symlink() or not physical.is_file()):
            raise SchemaError(f"missing or non-regular sidecar: {logical_path}")
        if logical_path in payload_overrides:
            return payload_overrides[logical_path]
        if physical.is_symlink() or not physical.is_file():
            raise SchemaError(f"missing or non-regular sidecar: {logical_path}")
        try:
            return physical.read_bytes()
        except OSError as error:
            raise SchemaError(f"cannot read sidecar {logical_path}: {error}") from error

    for key, row in complete_rows.items():
        task = row["task_level"]
        logical_path = task["path"]
        expected_task_paths.add(logical_path)
        payload = payload_for(logical_path, "task-level")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != task["sha256"]:
            raise SchemaError(f"task sidecar digest mismatch: {logical_path}")
        records = _read_jsonl_payload(payload, logical_path)
        if len(records) != task["records"]:
            raise SchemaError(f"task sidecar record count mismatch: {logical_path}")
        task_indexes[key] = _validate_task_records(
            records, suite_index[row["suite"]], logical_path
        )

    disagreement_index = {
        (entry["suite"], entry["phase"], entry["pair"]): entry
        for entry in data["paired_disagreements"]
    }
    expected_disagreement_keys: set[tuple[str, str, str]] = set()
    expected_disagreement_paths: set[str] = set()
    for suite_id in suite_index:
        for phase in CAMPAIGN_PHASE_IDS:
            for pair_id, pair in pair_index.items():
                dynamo_key = (pair["dynamo_variant"], suite_id, phase)
                native_key = (pair["native_variant"], suite_id, phase)
                if dynamo_key not in complete_rows or native_key not in complete_rows:
                    continue
                disagreement_key = (suite_id, phase, pair_id)
                expected_disagreement_keys.add(disagreement_key)
                entry = disagreement_index.get(disagreement_key)
                if entry is None:
                    raise SchemaError(
                        f"complete pair is missing disagreement sidecar: {disagreement_key}"
                    )
                logical_path = entry["path"]
                expected_disagreement_paths.add(logical_path)
                payload = payload_for(logical_path, "paired-disagreements")
                if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise SchemaError(
                        f"paired-disagreement digest mismatch: {logical_path}"
                    )
                records = _read_jsonl_payload(payload, logical_path)
                if len(records) != entry["disagreement_records"]:
                    raise SchemaError(
                        f"paired-disagreement record count mismatch: {logical_path}"
                    )

                dynamo_tasks = task_indexes[dynamo_key]
                native_tasks = task_indexes[native_key]
                if set(dynamo_tasks) != set(native_tasks):
                    raise SchemaError(
                        f"paired task populations differ: {suite_id}/{phase}/{pair_id}"
                    )
                expected_records: list[dict[str, Any]] = []
                for task_id, attempt in sorted(
                    dynamo_tasks, key=lambda value: (value[0], value[1] or 0)
                ):
                    dynamo_outcome = dynamo_tasks[(task_id, attempt)]["outcome"]
                    native_outcome = native_tasks[(task_id, attempt)]["outcome"]
                    if dynamo_outcome == native_outcome:
                        continue
                    expected: dict[str, Any] = {
                        "id": task_id,
                        "dynamo_outcome": dynamo_outcome,
                        "native_outcome": native_outcome,
                    }
                    if attempt is not None:
                        expected["attempt"] = attempt
                    expected_records.append(expected)
                if records != expected_records:
                    raise SchemaError(
                        f"paired-disagreement content mismatch: {logical_path}"
                    )
                if entry["compared_records"] != len(dynamo_tasks):
                    raise SchemaError(
                        f"paired-disagreement compared count mismatch: {logical_path}"
                    )

    if set(disagreement_index) != expected_disagreement_keys:
        unexpected = sorted(set(disagreement_index) - expected_disagreement_keys)
        missing = sorted(expected_disagreement_keys - set(disagreement_index))
        raise SchemaError(
            "paired-disagreement summary entries differ from complete pairs: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for category, expected_paths in (
        ("task-level", expected_task_paths),
        ("paired-disagreements", expected_disagreement_paths),
    ):
        root = results_dir / category
        actual_paths: set[str] = set()
        if root.exists():
            if root.is_symlink() or not root.is_dir():
                raise SchemaError(f"results/{category} must be a regular directory")
            for path in root.rglob("*.jsonl"):
                if path.is_symlink() or not path.is_file():
                    raise SchemaError(
                        f"results/{category} contains a non-regular sidecar"
                    )
                actual_paths.add(
                    PurePosixPath(
                        "results", *path.relative_to(results_dir).parts
                    ).as_posix()
                )
        actual_paths.update(
            logical
            for logical in payload_overrides
            if PurePosixPath(logical).parts[:2] == ("results", category)
        )
        if actual_paths != expected_paths:
            raise SchemaError(
                f"results/{category} sidecars differ from summary: "
                f"missing={sorted(expected_paths - actual_paths)}, "
                f"orphaned={sorted(actual_paths - expected_paths)}"
            )


def validate(data: dict[str, Any]) -> None:
    """Validate report invariants, including complete-run and pair semantics."""

    _require_mapping(data, "root")
    if data.get("schema_version") != 3:
        raise SchemaError("schema_version must be 3")
    campaign = _require_mapping(data.get("campaign"), "campaign")
    for field in (
        "title",
        "model",
        "served_model_name",
        "branch",
        "branch_base",
        "namespace",
        "hardware",
        "execution_policy",
    ):
        _require_string(campaign.get(field), f"campaign.{field}")
    model_revision = _require_string(
        campaign.get("model_revision"), "campaign.model_revision"
    )
    if GIT_SHA_RE.fullmatch(model_revision) is None:
        raise SchemaError("campaign.model_revision must be a lowercase 40-hex revision")
    if "source_commit" not in campaign:
        raise SchemaError("campaign.source_commit must be present")
    source_commit = campaign.get("source_commit")
    if source_commit is not None and (
        not isinstance(source_commit, str)
        or GIT_SHA_RE.fullmatch(source_commit) is None
    ):
        raise SchemaError(
            "campaign.source_commit must be null or a lowercase Git commit"
        )
    _require_positive_int(
        campaign.get("serving_context_tokens"), "campaign.serving_context_tokens"
    )
    _require_positive_int(
        campaign.get("terminalbench_context_tokens"),
        "campaign.terminalbench_context_tokens",
    )
    campaign_status = _require_string(campaign.get("status"), "campaign.status")
    if campaign_status not in CAMPAIGN_STATUSES:
        raise SchemaError(f"campaign.status must be one of {sorted(CAMPAIGN_STATUSES)}")
    started_at = _require_timestamp(campaign.get("started_at"), "campaign.started_at")
    if "completed_at" not in campaign:
        raise SchemaError("campaign.completed_at must be present")
    completed_at_raw = campaign["completed_at"]
    completed_at = (
        _require_timestamp(completed_at_raw, "campaign.completed_at")
        if completed_at_raw is not None
        else None
    )
    if campaign_status == "complete" and completed_at is None:
        raise SchemaError("complete campaign requires campaign.completed_at")
    if campaign_status == "complete" and source_commit is None:
        raise SchemaError("complete campaign requires campaign.source_commit")
    if campaign_status in {"pending", "in_progress"} and completed_at is not None:
        raise SchemaError(
            f"{campaign_status} campaign must not set campaign.completed_at"
        )
    if completed_at is not None and completed_at < started_at:
        raise SchemaError("campaign.completed_at must not precede campaign.started_at")

    variants = _require_list(data.get("variants"), "variants")
    variant_index = _unique_by_id(variants, "variants")
    if not variant_index:
        raise SchemaError("variants must not be empty")
    for variant_id, variant in variant_index.items():
        _require_string(variant.get("label"), f"variants[{variant_id!r}].label")
        _require_string(variant.get("framework"), f"variants[{variant_id!r}].framework")
        _require_string(variant.get("image"), f"variants[{variant_id!r}].image")
        if not isinstance(variant.get("dynamo"), bool):
            raise SchemaError(f"variants[{variant_id!r}].dynamo must be boolean")
        variant_status = _require_string(
            variant.get("status"), f"variants[{variant_id!r}].status"
        )
        if variant_status not in VARIANT_STATUSES:
            raise SchemaError(
                f"variants[{variant_id!r}].status must be one of "
                f"{sorted(VARIANT_STATUSES)}"
            )

    suites = _require_list(data.get("suites"), "suites")
    suite_index = _unique_by_id(suites, "suites")
    if not suite_index:
        raise SchemaError("suites must not be empty")
    metric_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for suite_id, suite in suite_index.items():
        prefix = f"suites[{suite_id!r}]"
        _require_string(suite.get("label"), f"{prefix}.label")
        _require_string(suite.get("kind"), f"{prefix}.kind")
        _require_string(suite.get("unit_label"), f"{prefix}.unit_label")
        units = _require_positive_int(suite.get("units"), f"{prefix}.units")
        generation_units = _require_positive_int(
            suite.get("generation_units", units), f"{prefix}.generation_units"
        )
        if generation_units < units:
            raise SchemaError(f"{prefix}.generation_units must be at least units")
        _require_positive_int(suite.get("attempts"), f"{prefix}.attempts")

        metric_rows = _require_list(suite.get("metrics"), f"{prefix}.metrics")
        metrics = _unique_by_id(metric_rows, f"{prefix}.metrics")
        if not metrics:
            raise SchemaError(f"{prefix}.metrics must not be empty")
        for metric_id, metric in metrics.items():
            metric_prefix = f"{prefix}.metrics[{metric_id!r}]"
            _require_string(metric.get("label"), f"{metric_prefix}.label")
            if metric.get("format") not in METRIC_FORMATS:
                raise SchemaError(
                    f"{metric_prefix}.format must be one of {sorted(METRIC_FORMATS)}"
                )
            if "direction" in metric and metric["direction"] not in METRIC_DIRECTIONS:
                raise SchemaError(
                    f"{metric_prefix}.direction must be one of {sorted(METRIC_DIRECTIONS)}"
                )

        primary_metric = _require_string(
            suite.get("primary_metric"), f"{prefix}.primary_metric"
        )
        if primary_metric not in metrics:
            raise SchemaError(f"{prefix}.primary_metric references unknown metric")
        comparison_metrics = _require_list(
            suite.get("comparison_metrics"), f"{prefix}.comparison_metrics"
        )
        if not comparison_metrics:
            raise SchemaError(f"{prefix}.comparison_metrics must not be empty")
        if len(comparison_metrics) != len(set(comparison_metrics)):
            raise SchemaError(f"{prefix}.comparison_metrics contains duplicates")
        for metric_id in comparison_metrics:
            if metric_id not in metrics:
                raise SchemaError(
                    f"{prefix}.comparison_metrics references unknown metric {metric_id!r}"
                )
            if metrics[metric_id].get("direction") not in METRIC_DIRECTIONS:
                raise SchemaError(
                    f"{prefix}.metrics[{metric_id!r}] needs direction for pair deltas"
                )
        required_metrics = _require_list(
            suite.get("required_metrics"), f"{prefix}.required_metrics"
        )
        if not required_metrics:
            raise SchemaError(f"{prefix}.required_metrics must not be empty")
        if len(required_metrics) != len(set(required_metrics)):
            raise SchemaError(f"{prefix}.required_metrics contains duplicates")
        for metric_id in required_metrics:
            if metric_id not in metrics:
                raise SchemaError(
                    f"{prefix}.required_metrics references unknown metric {metric_id!r}"
                )
        if primary_metric not in required_metrics:
            raise SchemaError(f"{prefix}.required_metrics must include primary_metric")
        missing_comparisons = sorted(set(comparison_metrics) - set(required_metrics))
        if missing_comparisons:
            raise SchemaError(
                f"{prefix}.required_metrics omits comparison metrics: "
                f"{missing_comparisons}"
            )
        metric_indexes[suite_id] = metrics

    pairs = _require_list(data.get("pairs"), "pairs")
    pair_index = _unique_by_id(pairs, "pairs")
    paired_variants: set[str] = set()
    for pair_id, pair in pair_index.items():
        prefix = f"pairs[{pair_id!r}]"
        _require_string(pair.get("label"), f"{prefix}.label")
        dynamo_id = _require_string(
            pair.get("dynamo_variant"), f"{prefix}.dynamo_variant"
        )
        native_id = _require_string(
            pair.get("native_variant"), f"{prefix}.native_variant"
        )
        if dynamo_id == native_id:
            raise SchemaError(f"{prefix} must reference two distinct variants")
        if dynamo_id not in variant_index or native_id not in variant_index:
            raise SchemaError(f"{prefix} references an unknown variant")
        if not variant_index[dynamo_id]["dynamo"] or variant_index[native_id]["dynamo"]:
            raise SchemaError(f"{prefix} must order Dynamo then native variants")
        if (
            variant_index[dynamo_id]["framework"]
            != variant_index[native_id]["framework"]
        ):
            raise SchemaError(f"{prefix} variants must use the same framework")
        for variant_id in (dynamo_id, native_id):
            if variant_id in paired_variants:
                raise SchemaError(
                    f"variant {variant_id!r} appears in more than one pair"
                )
            paired_variants.add(variant_id)
    unpaired_variants = sorted(set(variant_index) - paired_variants)
    if unpaired_variants:
        raise SchemaError(
            f"pairs must cover every variant exactly once; unpaired: {unpaired_variants}"
        )

    phases = _require_list(data.get("phases"), "phases")
    phase_index = _unique_by_id(phases, "phases")
    if tuple(phase_index) != CAMPAIGN_PHASE_IDS:
        raise SchemaError(f"phases must be ordered exactly {CAMPAIGN_PHASE_IDS}")
    ab_order = [
        variant_id
        for pair in pairs
        for variant_id in (pair["dynamo_variant"], pair["native_variant"])
    ]
    ba_order = [
        variant_id
        for pair in pairs
        for variant_id in (pair["native_variant"], pair["dynamo_variant"])
    ]
    for phase_id, expected_order in (("ab", ab_order), ("ba", ba_order)):
        phase = phase_index[phase_id]
        _require_string(phase.get("label"), f"phases[{phase_id!r}].label")
        if phase.get("variant_order") != expected_order:
            raise SchemaError(
                f"phases[{phase_id!r}].variant_order must be {expected_order}"
            )
        if phase.get("fresh_deployment_required") is not True:
            raise SchemaError(
                f"phases[{phase_id!r}].fresh_deployment_required must be true"
            )

    results = _require_list(data.get("results"), "results")
    result_keys: set[tuple[str, str, str]] = set()
    for position, raw_result in enumerate(results):
        result = _require_mapping(raw_result, f"results[{position}]")
        prefix = f"results[{position}]"
        variant_id = _require_string(result.get("variant"), f"{prefix}.variant")
        suite_id = _require_string(result.get("suite"), f"{prefix}.suite")
        phase_id = _require_string(result.get("phase"), f"{prefix}.phase")
        key = (variant_id, suite_id, phase_id)
        if key in result_keys:
            raise SchemaError(
                "duplicate result row for variant/suite/phase: "
                f"variant={variant_id!r}, suite={suite_id!r}, phase={phase_id!r}"
            )
        result_keys.add(key)
        if variant_id not in variant_index:
            raise SchemaError(f"{prefix}.variant references unknown variant")
        if suite_id not in suite_index:
            raise SchemaError(f"{prefix}.suite references unknown suite")
        if phase_id not in phase_index:
            raise SchemaError(f"{prefix}.phase references unknown phase")

        status = _require_string(result.get("status"), f"{prefix}.status")
        if status not in RESULT_STATUSES:
            raise SchemaError(
                f"{prefix}.status must be one of {sorted(RESULT_STATUSES)}"
            )
        if result.get("run_type", "full") != "full":
            raise SchemaError(
                f"{prefix}.run_type must be 'full'; smoke runs are not report rows"
            )

        completeness = _require_mapping(
            result.get("completeness"), f"{prefix}.completeness"
        )
        suite = suite_index[suite_id]
        expected_units = suite["units"]
        expected_generation_units = suite.get("generation_units", expected_units)
        expected_trials = expected_units * suite["attempts"]
        generated = _require_nonnegative_int(
            completeness.get("generated_units"),
            f"{prefix}.completeness.generated_units",
        )
        evaluated = _require_nonnegative_int(
            completeness.get("evaluated_units"),
            f"{prefix}.completeness.evaluated_units",
        )
        completed_trials = _require_nonnegative_int(
            completeness.get("completed_trials"),
            f"{prefix}.completeness.completed_trials",
        )
        if generated > expected_generation_units:
            raise SchemaError(
                f"{prefix}.completeness exceeds {expected_generation_units} "
                "expected generation units"
            )
        if evaluated > expected_units:
            raise SchemaError(
                f"{prefix}.completeness exceeds {expected_units} expected "
                "evaluation units"
            )
        if evaluated > generated:
            raise SchemaError(
                f"{prefix}.completeness evaluated_units exceeds generated_units"
            )
        if completed_trials > expected_trials:
            raise SchemaError(
                f"{prefix}.completeness exceeds {expected_trials} expected trials"
            )

        metric_values = _require_mapping(result.get("metrics"), f"{prefix}.metrics")
        definitions = metric_indexes[suite_id]
        unknown_metrics = sorted(set(metric_values) - set(definitions))
        if unknown_metrics:
            raise SchemaError(
                f"{prefix}.metrics contains unknown metrics: {unknown_metrics}"
            )
        for metric_id, value in metric_values.items():
            _validate_metric_value(
                value, definitions[metric_id], f"{prefix}.metrics.{metric_id}"
            )

        wall_time = result.get("wall_time_seconds")
        if wall_time is not None and (
            isinstance(wall_time, bool)
            or not isinstance(wall_time, (int, float))
            or not math.isfinite(wall_time)
            or wall_time < 0
        ):
            raise SchemaError(f"{prefix}.wall_time_seconds must be non-negative")

        if "evidence" in result:
            _validate_evidence(
                result["evidence"],
                f"{prefix}.evidence",
                variant=variant_id,
                suite=suite_id,
                phase=phase_id,
            )

        if "runtime_identity" in result:
            _validate_runtime_identity(
                result["runtime_identity"], f"{prefix}.runtime_identity"
            )
            if (
                source_commit is not None
                and result["runtime_identity"]["recipe"]["source_commit"]
                != source_commit
            ):
                raise SchemaError(
                    f"{prefix}.runtime_identity recipe source differs from campaign"
                )

        if status == "complete":
            if "evidence" not in result:
                raise SchemaError(
                    f"{prefix} is complete but has no validated evidence lineage"
                )
            if "runtime_identity" not in result:
                raise SchemaError(f"{prefix} is complete but has no runtime identity")
            _validate_campaign_source(
                result.get("campaign_source"),
                f"{prefix}.campaign_source",
                source_commit,
            )
            if suite["kind"] == "swebench":
                _validate_swe_suite_identity(
                    result.get("suite_identity"), f"{prefix}.suite_identity"
                )
            elif suite["kind"] == "bfcl":
                suite_identity = _require_mapping(
                    result.get("suite_identity"), f"{prefix}.suite_identity"
                )
                if set(suite_identity) != {"python_environment", "campaign_source"}:
                    raise SchemaError(
                        f"{prefix}.suite_identity contains invalid BFCL fields"
                    )
                _require_mapping(
                    suite_identity.get("python_environment"),
                    f"{prefix}.suite_identity.python_environment",
                )
                _validate_campaign_source(
                    suite_identity.get("campaign_source"),
                    f"{prefix}.suite_identity.campaign_source",
                    source_commit,
                )
            elif suite["kind"] == "terminalbench":
                suite_identity = _require_mapping(
                    result.get("suite_identity"), f"{prefix}.suite_identity"
                )
                if set(suite_identity) != {"harbor_environment", "task_images"}:
                    raise SchemaError(
                        f"{prefix}.suite_identity contains invalid Terminal fields"
                    )
                _validate_harbor_environment(
                    suite_identity.get("harbor_environment"),
                    f"{prefix}.suite_identity.harbor_environment",
                )
                _validate_terminal_task_images_identity(
                    suite_identity.get("task_images"),
                    f"{prefix}.suite_identity.task_images",
                    expected_tasks=expected_units,
                    expected_trials=expected_trials,
                )
            expected_task_records = (
                expected_trials if suite["kind"] == "terminalbench" else expected_units
            )
            _validate_task_level(
                result.get("task_level"),
                f"{prefix}.task_level",
                expected_task_records,
            )
            expected_task_path = (
                f"results/task-level/{suite_id}/{phase_id}/{variant_id}.jsonl"
            )
            if result["task_level"]["path"] != expected_task_path:
                raise SchemaError(
                    f"{prefix}.task_level.path must be {expected_task_path!r}"
                )
            if (generated, evaluated, completed_trials) != (
                expected_generation_units,
                expected_units,
                expected_trials,
            ):
                raise SchemaError(
                    f"{prefix} is complete but completeness is not "
                    f"{expected_generation_units} generated units, "
                    f"{expected_units} evaluated units, and {expected_trials} trials"
                )
            required_metrics = set(suite["required_metrics"])
            missing_metrics = sorted(required_metrics - set(metric_values))
            if missing_metrics:
                raise SchemaError(
                    f"{prefix} is complete but metrics are missing: {missing_metrics}"
                )
            if not missing_metrics:
                if suite["kind"] == "bfcl":
                    outcomes = (
                        metric_values["correct_cases"] + metric_values["failed_cases"]
                    )
                    if outcomes != evaluated:
                        raise SchemaError(
                            f"{prefix} BFCL correct_cases + failed_cases must "
                            "equal evaluated_units"
                        )
                    if metric_values["inference_errors"] != 0:
                        raise SchemaError(
                            f"{prefix} BFCL complete result must have zero "
                            "inference_errors"
                        )
                elif suite["kind"] == "swebench":
                    outcomes = (
                        metric_values["passed_instances"]
                        + metric_values["failed_instances"]
                    )
                    if outcomes != generated:
                        raise SchemaError(
                            f"{prefix} SWE passed_instances + failed_instances "
                            "must equal generated_units"
                        )
                    if metric_values["missing_instances"] != expected_units - generated:
                        raise SchemaError(
                            f"{prefix} SWE missing_instances does not match completeness"
                        )
                    expected_score = metric_values["passed_instances"] / expected_units
                    if not math.isclose(
                        metric_values["benchmark_score"],
                        expected_score,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise SchemaError(
                            f"{prefix} SWE benchmark_score must equal "
                            "passed_instances / dataset units"
                        )
                    submitted = expected_units - metric_values["missing_instances"]
                    expected_submitted_score = (
                        metric_values["passed_instances"] / submitted
                        if submitted
                        else 0.0
                    )
                    if not math.isclose(
                        metric_values["score_on_submitted"],
                        expected_submitted_score,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise SchemaError(
                            f"{prefix} SWE score_on_submitted must equal "
                            "passed_instances / submitted instances"
                        )
                elif suite["kind"] == "terminalbench":
                    outcomes = sum(
                        metric_values[metric_id]
                        for metric_id in (
                            "passed_attempts",
                            "failed_attempts",
                            "errored_attempts",
                            "no_reward_attempts",
                        )
                    )
                    if outcomes != completed_trials:
                        raise SchemaError(
                            f"{prefix} Terminal-Bench attempt outcomes must equal "
                            "completed_trials"
                        )
                    if metric_values["errored_attempts"] != 0:
                        raise SchemaError(
                            f"{prefix} Terminal-Bench complete result must have zero "
                            "errored_attempts"
                        )
                    if metric_values["no_reward_attempts"] != 0:
                        raise SchemaError(
                            f"{prefix} Terminal-Bench complete result must have zero "
                            "no_reward_attempts"
                        )
                    expected_pass_at_1 = (
                        metric_values["passed_attempts"] / completed_trials
                    )
                    if not math.isclose(
                        metric_values["pass_at_1"],
                        expected_pass_at_1,
                        rel_tol=0.0,
                        # The harness rounds task-mean pass@k to eight decimals.
                        abs_tol=1e-8,
                    ):
                        raise SchemaError(
                            f"{prefix} Terminal-Bench pass_at_1 must equal "
                            "passed_attempts / completed_trials"
                        )
                    pass_at_k = [
                        metric_values[f"pass_at_{k}"]
                        for k in range(1, suite["attempts"] + 1)
                    ]
                    if pass_at_k != sorted(pass_at_k):
                        raise SchemaError(
                            f"{prefix} Terminal-Bench pass@k must be non-decreasing"
                        )

    deployment_identities: dict[tuple[str, str, str], dict[str, Any]] = {}
    deployment_digests: dict[str, tuple[str, str, str]] = {}
    controller_uids: dict[str, tuple[str, str, str]] = {}
    pod_uids: dict[str, tuple[str, str, str, str]] = {}
    for result in results:
        if result.get("status") != "complete":
            continue
        identity = result["runtime_identity"]
        key = (result["variant"], result["suite"], result["phase"])
        deployment_identities[key] = identity
        deployment_digest = identity["deployment_sha256"]
        previous_deployment = deployment_digests.setdefault(deployment_digest, key)
        if previous_deployment != key:
            raise SchemaError(
                "every complete suite/variant/phase cell must use a fresh deployment; "
                f"{key} reuses {previous_deployment}"
            )
        controller_uid = identity["controller_uid_sha256"]
        previous_controller = controller_uids.setdefault(controller_uid, key)
        if previous_controller != key:
            raise SchemaError(
                "every complete suite/variant/phase cell must use a fresh controller; "
                f"{key} reuses {previous_controller}"
            )
        for role, pod_uid in identity["pod_uid_sha256_by_role"].items():
            pod_key = (*key, role)
            previous_pod = pod_uids.setdefault(pod_uid, pod_key)
            if previous_pod != pod_key:
                raise SchemaError(
                    "every complete suite/variant/phase cell must use fresh pods; "
                    f"{pod_key} reuses {previous_pod}"
                )

    for variant_id in variant_index:
        variant_identities = [
            identity
            for (row_variant, _, _), identity in deployment_identities.items()
            if row_variant == variant_id
        ]
        if (
            len(
                {
                    json.dumps(
                        identity["recipe"], sort_keys=True, separators=(",", ":")
                    )
                    for identity in variant_identities
                }
            )
            > 1
        ):
            raise SchemaError(f"{variant_id} campaign cells must use the same recipe")
        if (
            len(
                {
                    json.dumps(
                        identity["control_plane"], sort_keys=True, separators=(",", ":")
                    )
                    for identity in variant_identities
                }
            )
            > 1
        ):
            raise SchemaError(
                f"{variant_id} campaign cells must use the same control-plane identity"
            )

    for suite_id in suite_index:
        for pair_id, pair in pair_index.items():
            dynamo_id = pair["dynamo_variant"]
            native_id = pair["native_variant"]
            ab_dynamo = deployment_identities.get((dynamo_id, suite_id, "ab"))
            ab_native = deployment_identities.get((native_id, suite_id, "ab"))
            ba_dynamo = deployment_identities.get((dynamo_id, suite_id, "ba"))
            ba_native = deployment_identities.get((native_id, suite_id, "ba"))

            if ab_dynamo is not None and ab_native is not None:
                if _require_timestamp(
                    ab_dynamo["captured_at"], "runtime_identity.captured_at"
                ) >= _require_timestamp(
                    ab_native["captured_at"], "runtime_identity.captured_at"
                ):
                    raise SchemaError(
                        f"{suite_id}/{pair_id} ab must deploy Dynamo before native"
                    )
            if ba_dynamo is not None and ba_native is not None:
                if _require_timestamp(
                    ba_native["captured_at"], "runtime_identity.captured_at"
                ) >= _require_timestamp(
                    ba_dynamo["captured_at"], "runtime_identity.captured_at"
                ):
                    raise SchemaError(
                        f"{suite_id}/{pair_id} ba must deploy native before Dynamo"
                    )
            if None not in (ab_dynamo, ab_native, ba_dynamo, ba_native):
                ab_finished = max(
                    _require_timestamp(
                        identity["captured_at"], "runtime_identity.captured_at"
                    )
                    for identity in (ab_dynamo, ab_native)
                )
                ba_started = min(
                    _require_timestamp(
                        identity["captured_at"], "runtime_identity.captured_at"
                    )
                    for identity in (ba_native, ba_dynamo)
                )
                if ab_finished >= ba_started:
                    raise SchemaError(
                        f"{suite_id}/{pair_id} ba deployments must follow ab deployments"
                    )
    hardware_identities = {
        json.dumps(identity["hardware"], sort_keys=True, separators=(",", ":"))
        for identity in deployment_identities.values()
    }
    if len(hardware_identities) > 1:
        raise SchemaError("all campaign stacks/phases must use identical hardware")
    dynamo_control_planes = {
        json.dumps(identity["control_plane"], sort_keys=True, separators=(",", ":"))
        for (variant_id, _, _), identity in deployment_identities.items()
        if variant_index[variant_id]["dynamo"]
    }
    if len(dynamo_control_planes) > 1:
        raise SchemaError(
            "all Dynamo campaign runs must use one control-plane identity"
        )
    campaign_source_identities = {
        json.dumps(result["campaign_source"], sort_keys=True, separators=(",", ":"))
        for result in results
        if result.get("status") == "complete"
    }
    if len(campaign_source_identities) > 1:
        raise SchemaError(
            "all complete campaign runs must use one campaign source tree"
        )

    for suite_id, suite in suite_index.items():
        if suite["kind"] == "bfcl":
            identities = {
                json.dumps(
                    result["suite_identity"]["python_environment"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for result in results
                if result.get("suite") == suite_id
                and result.get("status") == "complete"
            }
            if len(identities) > 1:
                raise SchemaError(
                    "BFCL must use one exact Python environment across all runs"
                )
            continue
        if suite["kind"] == "terminalbench":
            harbor_identities = {
                json.dumps(
                    result["suite_identity"]["harbor_environment"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for result in results
                if result.get("suite") == suite_id
                and result.get("status") == "complete"
            }
            if len(harbor_identities) > 1:
                raise SchemaError(
                    "Terminal-Bench must use one exact Harbor environment across all runs"
                )
            task_image_identities = {
                json.dumps(
                    result["suite_identity"]["task_images"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for result in results
                if result.get("suite") == suite_id
                and result.get("status") == "complete"
            }
            if len(task_image_identities) > 1:
                raise SchemaError(
                    "Terminal-Bench must use one exact task-image map across all runs"
                )
            continue
        if suite["kind"] != "swebench":
            continue
        suite_rows = [
            result
            for result in results
            if result.get("suite") == suite_id and result.get("status") == "complete"
        ]
        if not suite_rows:
            continue
        fairness_fields = (
            "python_environment",
            "fairness_config_sha256",
            "task_image_evidence_sha256",
            "task_image_map_sha256",
            "generation",
            "evaluation",
        )
        for field in fairness_fields:
            identities = {
                json.dumps(
                    result["suite_identity"][field],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for result in suite_rows
            }
            if len(identities) > 1:
                raise SchemaError(
                    f"SWE suite {suite_id} must use identical {field} across all runs"
                )
        for variant_id in variant_index:
            variant_rows = [
                result for result in suite_rows if result.get("variant") == variant_id
            ]
            for field in (
                "effective_config_file_sha256",
                "effective_config_content_sha256",
                "runtime_source_revision",
                "runtime_family",
            ):
                values = {result["suite_identity"][field] for result in variant_rows}
                if len(values) > 1:
                    raise SchemaError(
                        f"SWE suite {suite_id}/{variant_id} phase identity drifted: {field}"
                    )

    disagreement_rows = _require_list(
        data.get("paired_disagreements"), "paired_disagreements"
    )
    disagreement_keys: set[tuple[str, str, str]] = set()
    for index, raw_entry in enumerate(disagreement_rows):
        entry = _require_mapping(raw_entry, f"paired_disagreements[{index}]")
        suite_id = _require_string(
            entry.get("suite"), f"paired_disagreements[{index}].suite"
        )
        phase_id = _require_string(
            entry.get("phase"), f"paired_disagreements[{index}].phase"
        )
        pair_id = _require_string(
            entry.get("pair"), f"paired_disagreements[{index}].pair"
        )
        key = (suite_id, phase_id, pair_id)
        if key in disagreement_keys:
            raise SchemaError(f"duplicate paired disagreement entry: {key}")
        disagreement_keys.add(key)
        if (
            suite_id not in suite_index
            or phase_id not in phase_index
            or pair_id not in pair_index
        ):
            raise SchemaError(
                f"paired disagreement entry references unknown identity: {key}"
            )
        expected_path = (
            f"results/paired-disagreements/{suite_id}/{phase_id}/{pair_id}.jsonl"
        )
        if entry.get("path") != expected_path:
            raise SchemaError(
                f"paired_disagreements[{index}].path must be {expected_path!r}"
            )
        digest = _require_string(
            entry.get("sha256"), f"paired_disagreements[{index}].sha256"
        )
        if SHA256_RE.fullmatch(digest) is None:
            raise SchemaError(
                f"paired_disagreements[{index}].sha256 must be a SHA-256 digest"
            )
        suite = suite_index[suite_id]
        expected_records = (
            suite["units"] * suite["attempts"]
            if suite["kind"] == "terminalbench"
            else suite["units"]
        )
        compared = _require_nonnegative_int(
            entry.get("compared_records"),
            f"paired_disagreements[{index}].compared_records",
        )
        disagreements = _require_nonnegative_int(
            entry.get("disagreement_records"),
            f"paired_disagreements[{index}].disagreement_records",
        )
        if compared != expected_records or disagreements > compared:
            raise SchemaError(
                f"paired_disagreements[{index}] has invalid population counts"
            )
        pair = pair_index[pair_id]
        for variant_id in (pair["dynamo_variant"], pair["native_variant"]):
            if (variant_id, suite_id, phase_id) not in result_keys:
                raise SchemaError(
                    f"paired disagreement {key} has no complete paired result rows"
                )

    if campaign_status == "complete":
        expected_result_keys = {
            (variant_id, suite_id, phase_id)
            for variant_id in variant_index
            for suite_id in suite_index
            for phase_id in phase_index
        }
        if len(expected_result_keys) != EXPECTED_FULL_RESULT_COUNT:
            raise SchemaError(
                "complete campaign must define exactly "
                f"{EXPECTED_FULL_RESULT_COUNT} variant/suite combinations; "
                f"found {len(expected_result_keys)}"
            )
        missing_results = sorted(expected_result_keys - result_keys)
        if missing_results:
            formatted = [
                f"{variant}/{suite}/{phase}"
                for variant, suite, phase in missing_results
            ]
            raise SchemaError(
                f"complete campaign is missing variant/suite/phase results: {formatted}"
            )
        noncomplete_results = sorted(
            f"{result['variant']}/{result['suite']}/{result['phase']}"
            for result in results
            if result["status"] != "complete"
        )
        if noncomplete_results:
            raise SchemaError(
                "complete campaign contains non-complete results: "
                f"{noncomplete_results}"
            )
        noncomplete_variants = sorted(
            variant_id
            for variant_id, variant in variant_index.items()
            if variant["status"] != "complete"
        )
        if noncomplete_variants:
            raise SchemaError(
                "complete campaign contains non-complete variants: "
                f"{noncomplete_variants}"
            )
        expected_disagreement_keys = {
            (suite_id, phase_id, pair_id)
            for suite_id in suite_index
            for phase_id in phase_index
            for pair_id in pair_index
        }
        if disagreement_keys != expected_disagreement_keys:
            missing = sorted(expected_disagreement_keys - disagreement_keys)
            raise SchemaError(
                f"complete campaign is missing paired disagreements: {missing}"
            )


def _render_completeness(result: dict[str, Any], suite: dict[str, Any]) -> str:
    completeness = result["completeness"]
    expected_units = suite["units"]
    expected_generation_units = suite.get("generation_units", expected_units)
    unit_label = esc(suite["unit_label"])
    parts = [
        "generated "
        f"{number(completeness['generated_units'], 0)}/"
        f"{number(expected_generation_units, 0)} {unit_label}",
        "evaluated "
        f"{number(completeness['evaluated_units'], 0)}/"
        f"{number(expected_units, 0)} {unit_label}",
    ]
    if suite["attempts"] > 1:
        expected_trials = expected_units * suite["attempts"]
        parts.append(
            f"trials {number(completeness['completed_trials'], 0)}/{number(expected_trials, 0)}"
        )
    return "<br>".join(parts)


def _render_metric_list(
    result: dict[str, Any], suite: dict[str, Any], *, omit: set[str] | None = None
) -> str:
    definitions = {metric["id"]: metric for metric in suite["metrics"]}
    values = result["metrics"]
    rows = []
    for metric in suite["metrics"]:
        metric_id = metric["id"]
        if metric_id not in values or (omit and metric_id in omit):
            continue
        rows.append(
            '<span class="metric-detail">'
            f"<span>{esc(metric['label'])}</span> "
            f"<strong>{format_metric(values[metric_id], definitions[metric_id])}</strong>"
            "</span>"
        )
    return "".join(rows) or "—"


def render(data: dict[str, Any]) -> str:
    validate(data)
    source_digest = summary_sha256(data)
    campaign = data["campaign"]
    variants = data["variants"]
    suites = data["suites"]
    pairs = data["pairs"]
    phases = data["phases"]
    result_index = {
        (row["variant"], row["suite"], row["phase"]): row for row in data["results"]
    }

    deployment_rows = "".join(
        "<tr>"
        f"<td>{esc(variant['label'])}</td>"
        f"<td>{esc(variant['framework'])}</td>"
        f"<td>{'yes' if variant['dynamo'] else 'no'}</td>"
        f"<td><code>{esc(variant['image'])}</code></td>"
        f"<td>{status_badge(variant['status'])}</td>"
        "</tr>"
        for variant in variants
    )

    score_rows: list[str] = []
    for suite in suites:
        definitions = {metric["id"]: metric for metric in suite["metrics"]}
        primary_id = suite["primary_metric"]
        for phase in phases:
            for variant in variants:
                result = result_index.get((variant["id"], suite["id"], phase["id"]))
                if result is None:
                    cells = [status_badge("pending"), "—", "—", "—", "—"]
                else:
                    publish_metrics = result["status"] == "complete"
                    primary_value = (
                        result["metrics"].get(primary_id) if publish_metrics else None
                    )
                    primary_cell = "—"
                    if primary_value is not None:
                        primary_cell = (
                            '<span class="metric-name">'
                            f"{esc(definitions[primary_id]['label'])}</span>"
                            '<strong class="primary-value">'
                            f"{format_metric(primary_value, definitions[primary_id])}</strong>"
                        )
                    cells = [
                        status_badge(result["status"]),
                        _render_completeness(result, suite),
                        primary_cell,
                        _render_metric_list(result, suite, omit={primary_id})
                        if publish_metrics
                        else "—",
                        number(result.get("wall_time_seconds")),
                    ]
                score_rows.append(
                    "<tr>"
                    f"<td>{esc(suite['label'])}</td>"
                    f"<td>{esc(phase['label'])}</td>"
                    f"<td>{esc(variant['label'])}</td>"
                    + "".join(f"<td>{cell}</td>" for cell in cells)
                    + "</tr>"
                )

    delta_rows: list[str] = []
    for suite in suites:
        definitions = {metric["id"]: metric for metric in suite["metrics"]}
        for pair in pairs:
            for metric_id in suite["comparison_metrics"]:
                definition = definitions[metric_id]
                phase_values: list[tuple[float | int, float | int]] = []
                for phase in phases:
                    dynamo_result = result_index.get(
                        (pair["dynamo_variant"], suite["id"], phase["id"])
                    )
                    native_result = result_index.get(
                        (pair["native_variant"], suite["id"], phase["id"])
                    )
                    pair_complete = (
                        dynamo_result is not None
                        and native_result is not None
                        and dynamo_result["status"] == "complete"
                        and native_result["status"] == "complete"
                    )
                    if pair_complete:
                        dynamo_value = dynamo_result["metrics"][metric_id]
                        native_value = native_result["metrics"][metric_id]
                        phase_values.append((dynamo_value, native_value))
                        cells = [
                            format_metric(dynamo_value, definition),
                            format_metric(native_value, definition),
                            format_delta(dynamo_value - native_value, definition),
                            status_badge("complete"),
                        ]
                    else:
                        cells = ["—", "—", "—", status_badge("waiting")]
                    delta_rows.append(
                        "<tr>"
                        f"<td>{esc(suite['label'])}</td>"
                        f"<td>{esc(pair['label'])}</td>"
                        f"<td>{esc(phase['label'])}</td>"
                        f"<td>{esc(definition['label'])}</td>"
                        + "".join(f"<td>{cell}</td>" for cell in cells)
                        + "</tr>"
                    )
                if len(phase_values) == len(phases):
                    dynamo_mean = sum(value[0] for value in phase_values) / len(phases)
                    native_mean = sum(value[1] for value in phase_values) / len(phases)
                    delta = dynamo_mean - native_mean
                    favorable = (
                        delta >= 0
                        if definition["direction"] == "higher"
                        else delta <= 0
                    )
                    delta_class = "delta-positive" if favorable else "delta-negative"
                    cells = [
                        format_metric(dynamo_mean, definition),
                        format_metric(native_mean, definition),
                        f'<strong class="{delta_class}">{format_delta(delta, definition)}</strong>',
                        status_badge("complete"),
                    ]
                else:
                    cells = ["—", "—", "—", status_badge("waiting")]
                delta_rows.append(
                    "<tr>"
                    f"<td>{esc(suite['label'])}</td>"
                    f"<td>{esc(pair['label'])}</td>"
                    "<td>Balanced mean</td>"
                    f"<td>{esc(definition['label'])}</td>"
                    + "".join(f"<td>{cell}</td>" for cell in cells)
                    + "</tr>"
                )

    issue_rows = (
        "".join(
            "<tr>"
            f"<td><code>{esc(issue['id'])}</code></td>"
            f"<td>{esc(issue.get('variant'))}</td>"
            f"<td>{esc(issue.get('severity'))}</td>"
            f"<td>{status_badge(issue.get('status', 'unknown'))}</td>"
            f"<td>{esc(issue.get('error'))}</td>"
            f"<td>{esc(issue.get('impact'))}</td>"
            f"<td>{esc(issue.get('fix'))}</td>"
            f"<td>{esc(issue.get('verification'))}</td>"
            "</tr>"
            for issue in data.get("issues", [])
        )
        or '<tr><td colspan="8">No issues recorded.</td></tr>'
    )

    disagreement_rows = (
        "".join(
            "<tr>"
            f"<td>{esc(entry['suite'])}</td>"
            f"<td>{esc(entry['phase'])}</td>"
            f"<td>{esc(entry['pair'])}</td>"
            f"<td>{number(entry['compared_records'], 0)}</td>"
            f"<td>{number(entry['disagreement_records'], 0)}</td>"
            f"<td><code>{esc(entry['path'])}</code></td>"
            "</tr>"
            for entry in data["paired_disagreements"]
        )
        or '<tr><td colspan="6">Waiting for complete paired task-level results.</td></tr>'
    )

    suite_cards = "".join(
        '<article class="metric">'
        f'<div class="metric-value">{number(suite["units"] * suite["attempts"], 0)}</div>'
        f'<div class="metric-label">{esc(suite["label"])} trials per stack</div>'
        "</article>"
        for suite in suites
    )
    completed_results = sum(
        result["status"] == "complete" for result in data["results"]
    )
    expected_results = len(variants) * len(suites) * len(phases)
    notes = "".join(f"<li>{esc(note)}</li>" for note in data.get("notes", []))

    return f"""<!doctype html>
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="glm52-summary-sha256" content="{source_digest}">
<title>{esc(campaign["title"])}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b0d10; --panel:#14181d; --line:#2b323a; --text:#edf1f5;
  --muted:#9ca8b4; --green:#76b900; --cyan:#54d2e0; --amber:#ffc857; --red:#ff6b6b; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
main {{ width:min(1500px,94vw); margin:0 auto; padding:56px 0 96px; }}
h1 {{ font-size:clamp(32px,5vw,64px); line-height:1.02; margin:0 0 16px; max-width:1100px; }}
h2 {{ margin:48px 0 16px; font-size:24px; }}
p,li {{ color:var(--muted); }}
.eyebrow {{ color:var(--green); text-transform:uppercase; letter-spacing:.12em; font-weight:700; }}
.lede {{ font-size:19px; max-width:900px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:24px 0; }}
.metric,.panel {{ background:linear-gradient(145deg,#171c22,#11151a); border:1px solid var(--line); border-radius:12px; }}
.metric {{ padding:20px; }} .metric-value {{ font-size:28px; color:var(--cyan); font-variant-numeric:tabular-nums; }}
.metric-label {{ color:var(--muted); margin-top:4px; }} .panel {{ padding:20px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; min-width:900px; }}
th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:10px 12px; vertical-align:top; }}
th {{ position:sticky; top:0; background:#171c22; color:#d8e0e8; }}
code {{ color:#bfe69a; font-size:12px; overflow-wrap:anywhere; }}
.badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:2px 8px; font-size:12px; }}
.badge-complete,.badge-passed,.badge-ready {{ color:var(--green); border-color:#527c13; }}
.badge-failed,.badge-error,.badge-blocked {{ color:var(--red); border-color:#783838; }}
.badge-in-progress,.badge-starting,.badge-running,.badge-observed,.badge-partial {{ color:var(--amber); border-color:#77602c; }}
.badge-pending,.badge-waiting {{ color:var(--muted); }}
.meta {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted); font-size:13px; }}
.warning {{ border-left:4px solid var(--amber); padding:12px 16px; background:#211c12; color:#ead9a9; }}
.metric-detail {{ display:block; white-space:nowrap; }} .metric-detail span {{ color:var(--muted); }}
.metric-name {{ display:block; color:var(--muted); }} .primary-value {{ display:block; font-size:18px; color:var(--cyan); }}
.delta-positive {{ color:var(--green); }} .delta-negative {{ color:var(--red); }}
footer {{ margin-top:56px; color:var(--muted); font-size:13px; }}
</style>
</head>
<body><main>
<div class="eyebrow">NScale evaluation campaign · {status_badge(campaign["status"])}</div>
<h1>{esc(campaign["title"])}</h1>
<p class="lede">A controlled accuracy and reliability comparison of Dynamo-backed and native
engine serving for GLM-5.2 across function calling, repository repair, multilingual repair, and
terminal-agent workloads.</p>
<div class="meta">
  <span>Branch <code>{esc(campaign["branch"])}</code></span>
  <span>Model <code>{esc(campaign["model"])}</code></span>
  <span>Revision <code>{esc(campaign["model_revision"])}</code></span>
  <span>Serving context {number(campaign["serving_context_tokens"])} tokens</span>
  <span>Terminal-Bench context {number(campaign["terminalbench_context_tokens"])} tokens</span>
  <span>Hardware {esc(campaign["hardware"])}</span>
  <span>Started {esc(campaign["started_at"])}</span>
</div>
<div class="grid">{suite_cards}</div>
<div class="warning">{completed_results}/{expected_results} full stack/suite results are complete.
Headline metrics and pair deltas appear only after exact full-run completeness and official scoring.</div>

<h2>Deployment identity</h2>
<div class="panel"><table><thead><tr><th>Stack</th><th>Engine</th><th>Dynamo</th><th>Image</th><th>Status</th></tr></thead>
<tbody>{deployment_rows}</tbody></table></div>

<h2>Suite results and completeness</h2>
<div class="panel"><table><thead><tr><th>Suite</th><th>Phase</th><th>Stack</th><th>Status</th><th>Completeness</th><th>Primary metric</th><th>Additional suite metrics</th><th>Wall time (s)</th></tr></thead>
<tbody>{"".join(score_rows)}</tbody></table></div>

<h2>Dynamo versus native deltas</h2>
<div class="panel"><table><thead><tr><th>Suite</th><th>Engine pair</th><th>Phase</th><th>Metric</th><th>Dynamo</th><th>Native</th><th>Delta (Dynamo − native)</th><th>Status</th></tr></thead>
<tbody>{"".join(delta_rows)}</tbody></table></div>

<h2>Methodology</h2>
<div class="panel"><ul>
<li>{esc(campaign["execution_policy"])}.</li>
<li>Identical checkpoint revision, TP4 allocation, serving context limit, cache dtype, and framework image within each pair.</li>
<li>SWE-bench uses a {number(campaign["serving_context_tokens"])}-token serving context; Terminal-Bench advertises its published {number(campaign["terminalbench_context_tokens"])}-token context explicitly.</li>
<li>Task-level outputs and paired disagreements are retained; aggregate score alone is not treated as sufficient evidence.</li>
<li>BFCL uses its official overall accuracy. SWE-bench reports resolved instances divided by the full dataset. Terminal-Bench reports task-mean pass@1 through pass@5 over five attempts.</li>
<li>Official evaluators score SWE-bench and Terminal-Bench artifacts after generation.</li>
<li>Every deployment captures applied YAML, image IDs, package versions, effective commands, GPU placement, logs, and teardown proof.</li>
</ul></div>

<h2>Paired task disagreements</h2>
<div class="panel"><table><thead><tr><th>Suite</th><th>Phase</th><th>Engine pair</th><th>Compared outcomes</th><th>Disagreements</th><th>Sanitized artifact</th></tr></thead>
<tbody>{disagreement_rows}</tbody></table></div>

<h2>Issues, fixes, and re-verification</h2>
<div class="panel"><table><thead><tr><th>ID</th><th>Stack</th><th>Severity</th><th>Status</th><th>Error</th><th>Impact</th><th>Fix</th><th>Verification</th></tr></thead>
<tbody>{issue_rows}</tbody></table></div>

<h2>Notes and constraints</h2>
<div class="panel"><ul>{notes}</ul></div>

<h2>Reproduction artifacts</h2>
<div class="panel"><p>Deployment recipes and lifecycle scripts are under
<code>benchmarks/glm52-nscale/deploy/</code>. Pinned harness adapters and exact commands are under
<code>benchmarks/glm52-nscale/eval/</code>. Machine-readable summaries are under
<code>benchmarks/glm52-nscale/results/</code>.</p></div>

<footer>Generated from validated schema v{data["schema_version"]} <code>results/summary.json</code>
at source digest <code>{source_digest}</code>. This report is self-contained and uses no external assets.</footer>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="validate summary.json without writing HTML",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing HTML is stale relative to summary.json",
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    validate(data)
    assert_pinned_report_sources(data, allow_unpinned_scaffold=True)
    validate_sidecars(data, args.input.parent)
    if args.validate_only:
        print(f"valid: {args.input}")
        return
    output = render(data)
    if args.check:
        try:
            current = args.output.read_text()
        except OSError as error:
            raise SystemExit(f"stale: cannot read {args.output}: {error}") from error
        if current != output:
            raise SystemExit(
                f"stale: {args.output} does not match {args.input} "
                f"(summary digest {summary_sha256(data)})"
            )
        print(f"current: {args.output} (summary digest {summary_sha256(data)})")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(args.output)


if __name__ == "__main__":
    main()
