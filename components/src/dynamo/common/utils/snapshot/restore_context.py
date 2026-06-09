# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore-time context capture and reload helpers for Dynamo snapshot."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, NoReturn

from dynamo.common.utils.namespace import get_worker_namespace

from .constants import (
    KUBERNETES_OPTIONAL_PODINFO_FILES,
    KUBERNETES_REQUIRED_PODINFO_FILES,
    PODINFO_ROOT,
    RESTORE_RUNTIME_ENV_NAMES,
    SNAPSHOT_CONTROL_DIR,
    SNAPSHOT_CONTROL_DIR_ENV,
    SNAPSHOT_RESTORE_CONTEXT_FILE,
    SNAPSHOT_RESTORE_PLACEHOLDER_ENV,
)

logger = logging.getLogger(__name__)


def apply_snapshot_restore_config(config: Any) -> None:
    """Apply restore-time env to ``os.environ`` and a backend config object.

    CRIU restores the checkpoint-time process environment. The restore
    placeholder captures the restore pod's non-secret environment into the
    snapshot-control volume before snapshot-agent restores the process. Apply
    that environment before constructing ``DistributedRuntime`` so restored
    workers do not use stale checkpoint-job env such as ``NATS_SERVER=localhost``
    or a missing ``DYN_SYSTEM_PORT``.
    """

    restore_env, source, env_applied = _load_snapshot_restore_env()

    refreshed_discovery_backend = _restore_env_value(
        restore_env,
        env_name="DYN_DISCOVERY_BACKEND",
        fallback=config.discovery_backend,
    )
    if refreshed_discovery_backend != "kubernetes":
        logger.info(
            "Snapshot restore reusing configured discovery backend",
            extra={
                "dynamo_namespace": config.namespace,
                "discovery_backend": refreshed_discovery_backend,
            },
        )
        config.discovery_backend = refreshed_discovery_backend
        _apply_restore_planes(config, restore_env)
        return

    os.environ["DYN_DISCOVERY_BACKEND"] = "kubernetes"
    if not env_applied:
        _apply_restore_env(restore_env, source=source)
    _validate_kubernetes_restore_env()
    config.namespace = get_worker_namespace()
    config.discovery_backend = "kubernetes"
    _apply_restore_planes(config, restore_env)


def _apply_restore_planes(config: Any, restore_env: dict[str, str | None]) -> None:
    request_plane = _restore_env_value(
        restore_env,
        env_name="DYN_REQUEST_PLANE",
        fallback=config.request_plane,
    )
    if request_plane is not None:
        config.request_plane = request_plane
    config.event_plane = _restore_env_value(
        restore_env,
        env_name="DYN_EVENT_PLANE",
        fallback=config.event_plane,
    )


def _restore_env_value(
    restore_env: dict[str, str | None],
    env_name: str,
    fallback: str | None,
) -> str | None:
    if env_name in restore_env:
        value = restore_env[env_name]
        if value is None:
            return fallback
        return value
    return os.environ.get(env_name, fallback)


def write_snapshot_restore_context(control_dir: str | None = None) -> None:
    """Capture restore-time environment into the snapshot-control volume.

    The restore placeholder runs in the new Pod before CRIU restores the old
    process image. Capturing here lets Kubernetes resolve all env sources
    (literal env, Downward API, ConfigMap, and Secret refs) without teaching the
    operator how to copy runtime env values.

    Args:
        control_dir: Optional snapshot-control directory override. Defaults to
            ``DYN_SNAPSHOT_CONTROL_DIR`` or ``/snapshot-control``.
    """

    control_path = Path(
        control_dir or os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    )
    control_path.mkdir(parents=True, exist_ok=True)
    context = {
        "version": 1,
        "env": _capture_restore_env(),
    }
    context_file = control_path / SNAPSHOT_RESTORE_CONTEXT_FILE
    _write_text_atomic(
        context_file,
        json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n",
    )
    logger.info("Captured snapshot restore context at %s", context_file)


def _capture_restore_env() -> dict[str, str | None]:
    return {
        name: os.environ.get(name) if name in os.environ else None
        for name in sorted(_supported_restore_env_names())
    }


def _write_text_atomic(path: Path, contents: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(contents, encoding="utf-8")
    tmp_path.replace(path)


def _load_snapshot_restore_env() -> tuple[dict[str, str | None], str, bool]:
    for context_path in _snapshot_restore_context_paths():
        if not context_path.is_file():
            continue
        payload = context_path.read_text(encoding="utf-8").strip()
        if not payload:
            continue
        return _parse_and_apply_snapshot_restore_env(
            payload,
            source=str(context_path),
        )

    return _read_legacy_kubernetes_podinfo(), PODINFO_ROOT, False


def _snapshot_restore_context_paths() -> list[Path]:
    control_dir = os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    return [Path(control_dir) / SNAPSHOT_RESTORE_CONTEXT_FILE]


def _parse_and_apply_snapshot_restore_env(
    payload: str,
    source: str,
) -> tuple[dict[str, str | None], str, bool]:
    try:
        restore_context = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid snapshot restore context from {source}: {exc}"
        ) from exc

    if not isinstance(restore_context, dict):
        raise RuntimeError("snapshot restore context requires an object payload")
    version = restore_context.get("version")
    if version != 1:
        raise RuntimeError(f"unsupported snapshot restore context version {version!r}")

    env_config = restore_context.get("env")
    if not isinstance(env_config, dict):
        raise RuntimeError("snapshot restore context requires an object env field")

    return _apply_restore_env(env_config, source=source), source, True


def _apply_restore_env(
    env_config: Mapping[str, object],
    source: str,
) -> dict[str, str | None]:
    applied = []
    cleared = []
    restored_env: dict[str, str | None] = {}
    for env_name, value in env_config.items():
        if env_name not in _supported_restore_env_names():
            logger.warning("Ignoring unsupported snapshot restore env %s", env_name)
            continue
        if value is None:
            os.environ.pop(env_name, None)
            cleared.append(env_name)
            restored_env[env_name] = None
            continue
        if not isinstance(value, str):
            raise RuntimeError(
                f"snapshot restore runtime env {env_name} must be a string or null"
            )
        os.environ[env_name] = value
        applied.append(env_name)
        restored_env[env_name] = value

    logger.info(
        "Applied snapshot restore context runtime env",
        extra={
            "source": source,
            "applied_env": sorted(applied),
            "cleared_env": sorted(cleared),
        },
    )
    return restored_env


def _supported_restore_env_names() -> set[str]:
    return {
        *KUBERNETES_REQUIRED_PODINFO_FILES,
        *KUBERNETES_OPTIONAL_PODINFO_FILES,
        *RESTORE_RUNTIME_ENV_NAMES,
    }


def _read_legacy_kubernetes_podinfo() -> dict[str, str | None]:
    legacy_env = {}
    for env_name, podinfo_file in {
        **KUBERNETES_REQUIRED_PODINFO_FILES,
        **KUBERNETES_OPTIONAL_PODINFO_FILES,
    }.items():
        podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
        if not os.path.isfile(podinfo_path):
            legacy_env[env_name] = None
            continue
        with open(podinfo_path, encoding="utf-8") as podinfo:
            value = podinfo.read().strip()
        legacy_env[env_name] = value or None
    return legacy_env


def _validate_kubernetes_restore_env() -> None:
    for env_name in KUBERNETES_REQUIRED_PODINFO_FILES:
        if not os.environ.get(env_name):
            raise RuntimeError(
                "snapshot restore context requires a non-empty "
                f"{env_name} for kubernetes discovery"
            )


def is_restore_placeholder_mode() -> bool:
    """Return whether this process should act as a restore placeholder."""

    return os.environ.get(SNAPSHOT_RESTORE_PLACEHOLDER_ENV) == "1"


def run_restore_placeholder() -> NoReturn:
    """Capture restore env, then hold the container for snapshot-agent."""

    write_snapshot_restore_context()
    os.execvp("sleep", ["sleep", "infinity"])
