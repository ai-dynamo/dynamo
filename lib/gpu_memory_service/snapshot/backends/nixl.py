# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default NIXL transfer backend for GMS snapshot restore."""

from __future__ import annotations

from typing import List, Mapping, Sequence

from gpu_memory_service.snapshot.backends.nixl_staging import (
    NixlFileGroup,
    NixlPosixStagingTransferBackend,
)
from gpu_memory_service.snapshot.transfer import (
    NIXL_TRANSFER_BACKEND,
    FileTransferSource,
    GMSSnapshotConfig,
    group_sources_by_path,
)


def _group_sources_by_file(
    sources: Sequence[FileTransferSource],
) -> Mapping[str, List[NixlFileGroup]]:
    return {
        file_path: [(file_path, grouped_sources)]
        for file_path, grouped_sources in group_sources_by_path(sources).items()
    }


class NixlTransferBackend(NixlPosixStagingTransferBackend):
    """NIXL POSIX backend for checkpoint shard restore without GDS."""

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        super().__init__(
            config=config,
            backend_name=NIXL_TRANSFER_BACKEND,
            group_sources=_group_sources_by_file,
            group_kind="file",
        )
