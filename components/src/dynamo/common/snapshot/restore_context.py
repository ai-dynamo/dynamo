# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore-time context capture and reload helpers for Dynamo snapshot."""

import json
import logging
import os
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import Awaitable, Callable, Mapping, TypeVar

from dynamo.common.snapshot.constants import (
    KUBERNETES_OPTIONAL_ENV_NAMES,
    KUBERNETES_REQUIRED_ENV_NAMES,
    RESTORE_RUNTIME_ENV_NAMES,
    SNAPSHOT_CONTROL_DIR,
    SNAPSHOT_CONTROL_DIR_ENV,
    SNAPSHOT_RESTORE_CONTEXT_FILE,
    SNAPSHOT_RESTORE_STANDBY_ENV,
)

logger = logging.getLogger(__name__)
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class RestoredRuntimeConfig:
    """Unified Worker identity/transport settings after restore."""

    namespace: str
    discovery_backend: str
    request_plane: str
    event_plane: str | None = None


_RESTORE_RUNTIME_CONFIG_FIELDS = (
    "namespace",
    "discovery_backend",
    "request_plane",
    "event_plane",
)

_SUPPORTED_RESTORE_ENV_NAMES = {
    *KUBERNETES_REQUIRED_ENV_NAMES,
    *KUBERNETES_OPTIONAL_ENV_NAMES,
    *RESTORE_RUNTIME_ENV_NAMES,
}


async def refresh_snapshot_restore_config(
    config: ConfigT,
    parse_config: Callable[[], object | Awaitable[object]],
    runtime_config: Callable[[object], object] | None = None,
) -> ConfigT:
    """Apply restore env, then rebuild backend config through normal parsing.

    The restore-context file is created by the restore standby process in the
    new Pod before CRIU resumes the snapshotted process. Once resumed, apply
    that env first and then re-run the runtime parser so fields derived from env
    (namespace, discovery backend, request plane, event plane, etc.) follow the
    same CLI/env precedence as a cold start. This avoids brittle ad hoc patches
    to an already-parsed config object.

    Args:
        parse_config: Zero-argument callable that returns a config object after
            reparsing runtime arguments against the updated environment.
        runtime_config: Selector for the Dynamo runtime config object. Backends
            that embed runtime config under a field (for example SGLang's
            ``config.dynamo_args``) can provide a selector while preserving the
            existing backend config and pre-created engine arguments.

    Returns:
        The original config object with runtime fields refreshed.
    """

    apply_snapshot_restore_env()
    parsed = parse_config()
    if isawaitable(parsed):
        parsed_config = await parsed
    else:
        parsed_config = parsed

    target_runtime_config = runtime_config(config) if runtime_config else config
    parsed_runtime_config = (
        runtime_config(parsed_config) if runtime_config else parsed_config
    )
    _copy_restore_runtime_config(target_runtime_config, parsed_runtime_config)
    _validate_kubernetes_restore_env_for_config(target_runtime_config)
    logger.info(
        "Refreshed snapshot restore runtime config",
        extra={
            "dynamo_namespace": getattr(target_runtime_config, "namespace", None),
            "discovery_backend": getattr(
                target_runtime_config, "discovery_backend", None
            ),
            "request_plane": getattr(target_runtime_config, "request_plane", None),
            "event_plane": getattr(target_runtime_config, "event_plane", None),
        },
    )
    return config


def load_restored_runtime_config(
    argv: list[str] | None,
) -> RestoredRuntimeConfig:
    """Apply restore context and return unified Worker replacement settings.

    This module owns the small result record so restore-standby and unit-test
    paths do not need to import the native backend bindings.
    """

    apply_snapshot_restore_env()
    parsed = parse_snapshot_restore_runtime_config(argv)
    _validate_kubernetes_restore_env_for_config(parsed)
    return RestoredRuntimeConfig(
        namespace=parsed.namespace,
        discovery_backend=parsed.discovery_backend,
        request_plane=parsed.request_plane,
        event_plane=parsed.event_plane,
    )


def parse_snapshot_restore_runtime_config(argv: list[str] | None) -> object:
    """Parse Dynamo runtime args after restore env has been applied.

    This uses the same ``DynamoRuntimeArgGroup`` env/CLI handling as normal
    backend startup, but avoids reparsing backend engine args or redoing
    backend-specific side effects such as model fetching.
    """

    import argparse

    from dynamo.common.configuration.groups.runtime_args import (
        DynamoRuntimeArgGroup,
        DynamoRuntimeConfig,
    )

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    DynamoRuntimeArgGroup().add_arguments(parser)
    args, _ = parser.parse_known_args(argv)
    config = DynamoRuntimeConfig.from_cli_args(args)
    config.validate()
    return config


def apply_snapshot_restore_env() -> dict[str, str | None]:
    """Load restore-context JSON and apply its runtime env to ``os.environ``."""

    # Load the restore-context JSON captured by the restore standby process. It
    # contains the target container's actual restore-time env after Kubernetes
    # resolved literals, Downward API values, ConfigMaps, and Secrets.
    control_dir = os.environ.get(SNAPSHOT_CONTROL_DIR_ENV, SNAPSHOT_CONTROL_DIR)
    context_path = Path(control_dir) / SNAPSHOT_RESTORE_CONTEXT_FILE
    if not context_path.is_file():
        raise RuntimeError(f"snapshot restore context file not found: {context_path}")

    source = str(context_path)
    try:
        restore_context = json.loads(context_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"invalid snapshot restore context from {source}: {exc}"
        ) from exc

    if not isinstance(restore_context, dict):
        raise RuntimeError("snapshot restore context requires an object payload")
    env_config = restore_context.get("env")
    if not isinstance(env_config, dict):
        raise RuntimeError("snapshot restore context requires an object env field")
    return _apply_restore_env(env_config, source=source)


def write_snapshot_restore_context(control_dir: str | None = None) -> None:
    """Capture restore-time environment into the snapshot-control volume.

    The restore standby process runs in the new Pod before CRIU restores the old
    process image. Capturing here lets Kubernetes resolve all env sources
    (literal env, Downward API, ConfigMap, and Secret refs) without teaching the
    operator how to copy runtime env values.

    Args:
        control_dir: Optional snapshot-control directory override. Defaults to
            ``DYN_SNAPSHOT_CONTROL_DIR`` or ``/snapshot-control``.
    """

    # Capture only the non-secret env names Dynamo needs after restore. Missing
    # values are written as null so stale snapshot-time env can be cleared.
    context = {
        "env": {
            name: os.environ.get(name) if name in os.environ else None
            for name in sorted(_SUPPORTED_RESTORE_ENV_NAMES)
        },
    }

    # Write atomically into the shared snapshot-control volume before exec'ing
    # the inert standby process that snapshot-agent restores into.
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


def _validate_kubernetes_restore_env_for_config(config: object) -> None:
    if getattr(config, "discovery_backend", None) == "kubernetes":
        _validate_kubernetes_restore_env()


def _copy_restore_runtime_config(target: object, source: object) -> None:
    for name in _RESTORE_RUNTIME_CONFIG_FIELDS:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))


def _validate_kubernetes_restore_env() -> None:
    for env_name in KUBERNETES_REQUIRED_ENV_NAMES:
        if not os.environ.get(env_name):
            raise RuntimeError(
                "snapshot restore context requires a non-empty "
                f"{env_name} for kubernetes discovery"
            )


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


def maybe_run_restore_standby_mode() -> None:
    """Capture restore env and sleep when restore standby mode is enabled."""

    if os.environ.get(SNAPSHOT_RESTORE_STANDBY_ENV) != "1":
        return

    write_snapshot_restore_context()
    os.execvp("sleep", ["sleep", "infinity"])
