# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-facing snapshot controller wrapper."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .checkpoint import CheckpointConfig
from .restore_context import RestoreRuntimeConfig, reload_snapshot_restore_config

EngineT = TypeVar("EngineT")


@dataclass
class EngineSnapshotController(Generic[EngineT]):
    engine: EngineT
    pause_controller: Any
    checkpoint_config: CheckpointConfig
    pause_args: tuple[object, ...] = ()

    async def wait_for_restore(self) -> bool:
        return await self.checkpoint_config.run_lifecycle(
            self.pause_controller,
            *self.pause_args,
        )

    def reload_restore_identity(
        self,
        namespace: str,
        discovery_backend: str,
    ) -> tuple[str, str]:
        restored = reload_snapshot_restore_config(
            namespace=namespace,
            discovery_backend=discovery_backend,
        )
        return restored.namespace, restored.discovery_backend

    def reload_restore_config(
        self,
        namespace: str,
        discovery_backend: str,
        request_plane: str | None = None,
        event_plane: str | None = None,
    ) -> RestoreRuntimeConfig:
        return reload_snapshot_restore_config(
            namespace=namespace,
            discovery_backend=discovery_backend,
            request_plane=request_plane,
            event_plane=event_plane,
        )
