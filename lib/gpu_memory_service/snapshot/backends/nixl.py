# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default NIXL transfer backend for GMS snapshot restore."""

from __future__ import annotations

from typing import List, Mapping, Sequence

from gpu_memory_service.snapshot.backends.nixl_common import NixlFileGroup
from gpu_memory_service.snapshot.backends.nixl_staging import (
    NixlPosixStagingTransferBackend,
)
from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    TransferBackendKind,
    group_sources_by_path,
)


def _group_sources_by_file(
    sources: Sequence[FileTransferSource],
) -> Mapping[str, List[NixlFileGroup]]:
    # Grouping by file preserves O_DIRECT-friendly, offset-sorted reads from
    # each checkpoint shard.  The shared NIXL/POSIX staging runner coalesces
    # these logical groups into at most max_workers agent buckets at execution
    # time, so large PVC checkpoints do not create one NIXL agent per shard.
    return {
        file_path: [(file_path, grouped_sources)]
        for file_path, grouped_sources in group_sources_by_path(sources).items()
    }


class NixlTransferBackend(NixlPosixStagingTransferBackend):
    """NIXL POSIX backend for checkpoint shard restore without GDS."""

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        super().__init__(
            config=config,
            backend_name=TransferBackendKind.NIXL.value,
            group_sources=_group_sources_by_file,
            group_kind="file",
        )
