# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore-time context capture and reload helpers for Dynamo snapshot."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

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

_SUPPORTED_RESTORE_ENV_NAMES = {
    *KUBERNETES_REQUIRED_PODINFO_FILES,
    *KUBERNETES_OPTIONAL_PODINFO_FILES,
    *RESTORE_RUNTIME_ENV_NAMES,
}


def apply_snapshot_restore_config(config: Any) -> None:
    """Apply restore-time env to ``os.environ`` and a backend config object.

    CRIU restores the checkpoint-time process environment. The restore
    placeholder captures the restore pod's non-secret environment into the
    snapshot-control volume before snapshot-agent restores the process. Apply
    that environment before constructing ``DistributedRuntime`` so restored
    workers do not use stale checkpoint-job env such as ``NATS_SERVER=localhost``
    or a missing ``DYN_SYSTEM_PORT``.
    """

    # Prefer the restore-context JSON captured by the placeholder. It contains
    # the target container's actual restore-time env after Kubernetes resolved
    # literals, Downward API values, ConfigMaps, and Secrets.
    control_dir = os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    context_path = Path(control_dir) / SNAPSHOT_RESTORE_CONTEXT_FILE
    restore_env: dict[str, str | None]
    if context_path.is_file() and (
        payload := context_path.read_text(encoding="utf-8").strip()
    ):
        source = str(context_path)
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

        restore_env = _apply_restore_env(env_config, source=source)
        env_applied = True
    else:
        # Legacy fallback: older restore pods only projected Kubernetes identity
        # through /etc/podinfo. Keep that path so old restore shaping still works
        # for identity refresh, even though it cannot refresh runtime endpoints.
        source = PODINFO_ROOT
        restore_env = {}
        for env_name, podinfo_file in {
            **KUBERNETES_REQUIRED_PODINFO_FILES,
            **KUBERNETES_OPTIONAL_PODINFO_FILES,
        }.items():
            podinfo_path = os.path.join(PODINFO_ROOT, podinfo_file)
            if not os.path.isfile(podinfo_path):
                restore_env[env_name] = None
                continue
            with open(podinfo_path, encoding="utf-8") as podinfo:
                value = podinfo.read().strip()
            restore_env[env_name] = value or None
        env_applied = False

    # Refresh the parsed config fields that were originally derived from env.
    # Null entries in restore-context mean "unset in the restore pod", so keep
    # the already-parsed CLI/config fallback for these fields.
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

    # Kubernetes discovery depends on env that is read during registration, so
    # make sure os.environ is updated before create_runtime() is called.
    os.environ["DYN_DISCOVERY_BACKEND"] = "kubernetes"
    if not env_applied:
        _apply_restore_env(restore_env, source=source)
    for env_name in KUBERNETES_REQUIRED_PODINFO_FILES:
        if not os.environ.get(env_name):
            raise RuntimeError(
                "snapshot restore context requires a non-empty "
                f"{env_name} for kubernetes discovery"
            )
    config.namespace = get_worker_namespace()
    config.discovery_backend = "kubernetes"
    _apply_restore_planes(config, restore_env)


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

    # Capture only the non-secret env names Dynamo needs after restore. Missing
    # values are written as null so stale checkpoint-time env can be cleared.
    context = {
        "version": 1,
        "env": {
            name: os.environ.get(name) if name in os.environ else None
            for name in sorted(_SUPPORTED_RESTORE_ENV_NAMES)
        },
    }

    # Write atomically into the shared snapshot-control volume before exec'ing
    # the inert placeholder process that snapshot-agent restores into.
    control_path = Path(
        control_dir or os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    )
    control_path.mkdir(parents=True, exist_ok=True)
    context_file = control_path / SNAPSHOT_RESTORE_CONTEXT_FILE
    tmp_path = context_file.with_name(f".{context_file.name}.tmp")
    tmp_path.write_text(
        json.dumps(context, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(context_file)
    logger.info("Captured snapshot restore context at %s", context_file)


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


def _apply_restore_env(
    env_config: Mapping[str, object],
    source: str,
) -> dict[str, str | None]:
    applied = []
    cleared = []
    restored_env: dict[str, str | None] = {}
    for env_name, value in env_config.items():
        if env_name not in _SUPPORTED_RESTORE_ENV_NAMES:
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


def maybe_run_restore_placeholder_mode() -> None:
    """Capture restore env and sleep when restore-placeholder mode is enabled."""

    if os.environ.get(SNAPSHOT_RESTORE_PLACEHOLDER_ENV) != "1":
        return

    write_snapshot_restore_context()
    os.execvp("sleep", ["sleep", "infinity"])
