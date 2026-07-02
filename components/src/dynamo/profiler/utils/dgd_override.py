# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply DGD overrides through the operator-owned Go implementation."""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DGD_OVERRIDE_BINARY_ENV = "DYNAMO_DGD_APPLY_OVERRIDES_BIN"
DGD_OVERRIDE_BINARY_NAME = "dgd-apply-overrides"
DGD_OVERRIDE_PROTOCOL_VERSION = "1"

_DGD_KIND = "DynamoGraphDeployment"
_LEGACY_DGD_API_VERSION = "nvidia.com/v1alpha1"
_COMMAND_TIMEOUT_SECONDS = 30


class DGDOverrideError(RuntimeError):
    """Raised when the external DGD override operation cannot be completed."""


def _validated_executable(path: str, source: str) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise DGDOverrideError(f"{source} does not point to a file: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise DGDOverrideError(f"{source} is not executable: {candidate}")
    return str(candidate.resolve())


def resolve_dgd_override_binary(binary_path: str | None = None) -> str:
    """Resolve the override CLI without downloading or modifying the environment."""
    if binary_path:
        return _validated_executable(binary_path, "DGD override binary path")

    configured_path = os.environ.get(DGD_OVERRIDE_BINARY_ENV)
    if configured_path:
        return _validated_executable(
            configured_path,
            f"${DGD_OVERRIDE_BINARY_ENV}",
        )

    discovered_path = shutil.which(DGD_OVERRIDE_BINARY_NAME)
    if discovered_path:
        return _validated_executable(discovered_path, "DGD override binary on PATH")

    raise DGDOverrideError(
        f"could not find {DGD_OVERRIDE_BINARY_NAME!r}; set "
        f"${DGD_OVERRIDE_BINARY_ENV} or add it to PATH"
    )


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise DGDOverrideError(
            f"failed to execute DGD override binary {command[0]!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DGDOverrideError(
            f"DGD override binary timed out after {_COMMAND_TIMEOUT_SECONDS} seconds"
        ) from exc


@lru_cache(maxsize=None)
def _verify_protocol(binary_path: str) -> None:
    result = _run_cli([binary_path, "--protocol-version"])
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise DGDOverrideError(
            f"failed to query DGD override protocol version: {detail}"
        )

    actual_version = result.stdout.strip()
    if actual_version != DGD_OVERRIDE_PROTOCOL_VERSION:
        raise DGDOverrideError(
            "incompatible DGD override binary protocol: "
            f"expected {DGD_OVERRIDE_PROTOCOL_VERSION!r}, got {actual_version!r}"
        )


def _versioned_override(overrides: object) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise DGDOverrideError("DGD override must be a YAML/JSON object")
    api_version = overrides.get("apiVersion")
    kind = overrides.get("kind")
    if api_version is None and kind is None:
        logger.warning(
            "DGD override is missing apiVersion and kind; treating it as %s %s. "
            "Add explicit type metadata because this compatibility behavior will "
            "be removed in a future release.",
            _LEGACY_DGD_API_VERSION,
            _DGD_KIND,
        )
        versioned = copy.deepcopy(overrides)
        versioned["apiVersion"] = _LEGACY_DGD_API_VERSION
        versioned["kind"] = _DGD_KIND
        return versioned
    if api_version is None or kind is None:
        raise DGDOverrideError(
            "DGD override must specify both apiVersion and kind, or neither for "
            "legacy v1alpha1 compatibility"
        )
    return copy.deepcopy(overrides)


def _validate_dgd(value: object, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DGDOverrideError(f"{role} must be a YAML/JSON object")
    if not value.get("apiVersion") or not value.get("kind"):
        raise DGDOverrideError(f"{role} must specify apiVersion and kind")
    if value["kind"] != _DGD_KIND:
        raise DGDOverrideError(
            f"{role} kind must be {_DGD_KIND!r}, got {value['kind']!r}"
        )
    return value


def apply_dgd_overrides(
    dgd_config: dict[str, Any],
    overrides: dict[str, Any],
    *,
    binary_path: str | None = None,
) -> dict[str, Any]:
    """Return a new DGD with ``overrides`` applied by the shared Go engine."""
    blueprint = _validate_dgd(dgd_config, "generated DGD blueprint")
    versioned_override = _versioned_override(overrides)
    executable = resolve_dgd_override_binary(binary_path)
    _verify_protocol(executable)

    with tempfile.TemporaryDirectory(prefix="dgd-override-") as directory:
        work_dir = Path(directory)
        blueprint_path = work_dir / "blueprint.json"
        override_path = work_dir / "override.json"
        output_path = work_dir / "effective.yaml"
        blueprint_path.write_text(json.dumps(blueprint), encoding="utf-8")
        override_path.write_text(json.dumps(versioned_override), encoding="utf-8")

        result = _run_cli(
            [
                executable,
                "--blueprint",
                str(blueprint_path),
                "--override",
                str(override_path),
                "--output",
                str(output_path),
            ]
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no error output"
            raise DGDOverrideError(f"failed to apply DGD override: {detail}")

        for diagnostic in result.stderr.splitlines():
            diagnostic = diagnostic.strip()
            if not diagnostic:
                continue
            if diagnostic.startswith("warning: "):
                logger.warning("DGD override: %s", diagnostic.removeprefix("warning: "))
            else:
                logger.info("DGD override CLI: %s", diagnostic)

        try:
            effective = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise DGDOverrideError(
                f"failed to read effective DGD from override binary: {exc}"
            ) from exc

    return _validate_dgd(effective, "effective DGD")
