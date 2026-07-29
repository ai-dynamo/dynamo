# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sharded SSD storage-layout backend for GMS snapshot restore."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Mapping, Sequence

from gpu_memory_service.snapshot.backends.nixl_common import (
    NixlFileGroup,
    split_work_groups,
)
from gpu_memory_service.snapshot.backends.nixl_staging import (
    NixlPosixStagingTransferBackend,
)
from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    TransferBackendKind,
    group_sources_by_path,
)

logger = logging.getLogger(__name__)


def parse_sharded_ssd_roots(value: str | None) -> List[str]:
    if not value:
        return []
    return [os.path.abspath(root.strip()) for root in value.split(",") if root.strip()]


def device_sharded_ssd_roots(
    checkpoint_dir: str,
    device: int,
    roots: Sequence[str],
) -> List[str]:
    suffix = _checkpoint_suffix(checkpoint_dir) / f"device-{device}"
    return [
        str(Path(os.path.abspath(str(root).strip())) / suffix)
        for root in roots
        if str(root).strip()
    ]


def _checkpoint_suffix(checkpoint_dir: str) -> Path:
    parts = Path(checkpoint_dir).parts
    if "versions" in parts:
        idx = parts.index("versions")
        if idx > 0 and idx + 1 < len(parts):
            return Path(parts[idx - 1]) / "versions" / parts[idx + 1]
    return Path(checkpoint_dir.strip(os.sep).replace(os.sep, "_"))


def _group_sources_by_root(
    sources: Sequence[FileTransferSource],
    roots: Sequence[str],
    queues_per_root: int = 2,
) -> Mapping[str, List[NixlFileGroup]]:
    groups_by_path = group_sources_by_path(sources)
    groups_by_root: dict[str, List[NixlFileGroup]] = {}
    for file_path, grouped_sources in groups_by_path.items():
        abs_path = os.path.abspath(file_path)
        root = None
        for candidate in roots:
            try:
                if os.path.commonpath([candidate, abs_path]) == candidate:
                    root = candidate
                    break
            except ValueError:
                continue
        if root is None:
            raise RuntimeError(
                f"{TransferBackendKind.SHARDED_SSD.value} source path "
                f"{file_path!r} is not under any configured sharded SSD root: "
                f"{list(roots)}"
            )
        groups_by_root.setdefault(root, []).append((file_path, grouped_sources))
    for grouped_paths in groups_by_root.values():
        grouped_paths.sort(key=lambda item: item[0])

    queued_groups: dict[str, List[NixlFileGroup]] = {}
    for root, file_groups in groups_by_root.items():
        buckets = [
            sorted(bucket_file_groups, key=lambda item: item[0])
            for _group_name, bucket_file_groups in split_work_groups(
                [(file_group[0], [file_group]) for file_group in file_groups],
                queues_per_root,
            )
        ]
        if len(buckets) == 1:
            queued_groups[root] = buckets[0]
            continue

        for index, bucket in enumerate(buckets):
            queued_groups[f"{root}#q{index}"] = bucket
    return queued_groups


class ShardedSSDTransferBackend(NixlPosixStagingTransferBackend):
    """Same-node sharded SSD restore using NIXL POSIX direct I/O."""

    def __init__(
        self,
        *,
        config: GMSSnapshotConfig,
    ) -> None:
        configured_roots = config.backend_config.get("sharded_ssd_roots")
        if configured_roots is None:
            self._roots = []
        elif isinstance(configured_roots, str):
            self._roots = parse_sharded_ssd_roots(configured_roots)
        else:
            self._roots = [
                os.path.abspath(str(root).strip())
                for root in configured_roots
                if str(root).strip()
            ]
        self._queues_per_root = int(
            config.backend_config.get("sharded_ssd_queues_per_root") or 2
        )
        if not self._roots:
            raise RuntimeError(
                f"{TransferBackendKind.SHARDED_SSD.value} requires "
                "sharded_ssd_roots=<root0>,<root1>,..."
            )
        logger.info(
            "%s initialized with %d root(s), %d queue(s)/root using NIXL POSIX staging",
            TransferBackendKind.SHARDED_SSD.value,
            len(self._roots),
            self._queues_per_root,
        )
        super().__init__(
            config=config,
            backend_name=TransferBackendKind.SHARDED_SSD.value,
            group_sources=self._group_sources,
            group_kind="ssd_root",
            warn_under_parallelized=True,
        )

    def _group_sources(
        self,
        sources: Sequence[FileTransferSource],
    ) -> Mapping[str, List[NixlFileGroup]]:
        return _group_sources_by_root(
            sources,
            self._roots,
            queues_per_root=self._queues_per_root,
        )
