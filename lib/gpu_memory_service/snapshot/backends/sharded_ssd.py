# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded SSD storage-layout backend for GMS snapshot restore."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from gpu_memory_service.snapshot.backends.nixl_staging import (
    NixlFileGroup,
    NixlPosixStagingTransferBackend,
)
from gpu_memory_service.snapshot.transfer import (
    SHARDED_SSD_TRANSFER_BACKEND,
    FileTransferSource,
    GMSSnapshotConfig,
    group_sources_by_path,
)

logger = logging.getLogger(__name__)

GMS_SHARDED_SSD_ROOTS_ENV = "GMS_SHARDED_SSD_ROOTS"
SHARDED_SSD_ROOTS_CONFIG_KEY = "sharded_ssd_roots"


def parse_sharded_ssd_roots(value: str | None) -> List[str]:
    if not value:
        return []
    return _normalize_roots(value.split(","))


def device_sharded_ssd_roots(
    checkpoint_dir: str,
    device: int,
    roots: Sequence[str],
) -> List[str]:
    suffix = _checkpoint_suffix(checkpoint_dir) / f"device-{device}"
    return [str(Path(root) / suffix) for root in _normalize_roots(roots)]


def _checkpoint_suffix(checkpoint_dir: str) -> Path:
    parts = Path(checkpoint_dir).parts
    if "versions" in parts:
        idx = parts.index("versions")
        if idx > 0 and idx + 1 < len(parts):
            return Path(parts[idx - 1]) / "versions" / parts[idx + 1]
    return Path(checkpoint_dir.strip(os.sep).replace(os.sep, "_"))


def _normalize_roots(values: Sequence[str]) -> List[str]:
    return [
        os.path.abspath(str(value).strip()) for value in values if str(value).strip()
    ]


def _roots_from_config(config: Mapping[str, Any]) -> List[str]:
    configured = config.get(SHARDED_SSD_ROOTS_CONFIG_KEY)
    if configured is None:
        return parse_sharded_ssd_roots(os.environ.get(GMS_SHARDED_SSD_ROOTS_ENV))
    if isinstance(configured, str):
        return parse_sharded_ssd_roots(configured)
    return _normalize_roots(configured)


def _match_root(file_path: str, roots: Sequence[str]) -> Optional[str]:
    abs_path = os.path.abspath(file_path)
    for root in roots:
        try:
            if os.path.commonpath([root, abs_path]) == root:
                return root
        except ValueError:
            continue
    return None


def _group_sources_by_root(
    sources: Sequence[FileTransferSource],
    roots: Sequence[str],
) -> Mapping[str, List[NixlFileGroup]]:
    groups_by_path = group_sources_by_path(sources)
    groups: dict[str, List[NixlFileGroup]] = {}
    for file_path, grouped_sources in groups_by_path.items():
        root = _match_root(file_path, roots)
        if root is None:
            raise RuntimeError(
                f"{SHARDED_SSD_TRANSFER_BACKEND} source path {file_path!r} is not "
                f"under any {GMS_SHARDED_SSD_ROOTS_ENV} root: {list(roots)}"
            )
        groups.setdefault(root, []).append((file_path, grouped_sources))
    for grouped_paths in groups.values():
        grouped_paths.sort(key=lambda item: item[0])
    return groups


class ShardedSSDTransferBackend(NixlPosixStagingTransferBackend):
    """Same-node sharded SSD restore using NIXL POSIX direct I/O."""

    def __init__(
        self,
        *,
        config: GMSSnapshotConfig,
    ) -> None:
        self._roots = _roots_from_config(config.backend_config)
        if not self._roots:
            raise RuntimeError(
                f"{SHARDED_SSD_TRANSFER_BACKEND} requires "
                f"{GMS_SHARDED_SSD_ROOTS_ENV}=<root0>,<root1>,..."
            )
        logger.info(
            "%s initialized with %d root(s) using NIXL POSIX staging",
            SHARDED_SSD_TRANSFER_BACKEND,
            len(self._roots),
        )
        super().__init__(
            config=config,
            backend_name=SHARDED_SSD_TRANSFER_BACKEND,
            group_sources=self._group_sources,
            group_kind="ssd_root",
            warn_under_parallelized=True,
        )

    def _group_sources(
        self,
        sources: Sequence[FileTransferSource],
    ) -> Mapping[str, List[NixlFileGroup]]:
        return _group_sources_by_root(sources, self._roots)
