# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin GMS adapter for ModelExpress snapshot restore."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Mapping, Sequence

from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    TransferSession,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)

_STRATEGY_CHAIN = "rdma->gds->posix"
_GMS_RESTORE_API_REQUIREMENT = (
    "The ModelExpress GMS snapshot backend requires a ModelExpress version "
    "newer than 0.4.0 that provides modelexpress.gds_loader read descriptors "
    "and modelexpress.restore_strategy. ModelExpress 0.4.0 does not provide "
    "this API; install the coordinated ModelExpress release containing it "
    "before selecting --transfer-backend=modelexpress."
)

try:
    from modelexpress.gds_loader import MxDeviceReadTarget, MxFileReadSource
    from modelexpress.restore_strategy import (
        GmsRestoreContext,
        MxGmsRestoreStrategyChain,
    )
except ImportError as exc:
    raise ImportError(_GMS_RESTORE_API_REQUIREMENT) from exc


def _group_restore_pairs(
    sources: Sequence[MxFileReadSource],
    targets: Mapping[str, MxDeviceReadTarget],
) -> tuple[dict[str, list[tuple[MxFileReadSource, MxDeviceReadTarget]]], int,]:
    """Group validated restore pairs by path in file-offset order."""
    grouped: dict[str, list[tuple[MxFileReadSource, MxDeviceReadTarget]]] = defaultdict(
        list
    )
    total_bytes = 0
    for source in sources:
        grouped[str(source.file_path)].append((source, targets[source.allocation_id]))
        total_bytes += int(source.byte_count)

    for pairs in grouped.values():
        pairs.sort(key=lambda pair: int(pair[0].file_offset))
    return dict(grouped), total_bytes


class ModelExpressTransferBackend:
    """GMS restore backend that delegates transfers to ModelExpress."""

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        self._device = config.device
        self._max_workers = config.max_workers
        self._backend_config = dict(config.backend_config or {})
        logger.info(
            "ModelExpress GMS backend initialized: device=%d max_workers=%d "
            "strategy_chain=%s gds_chunk_size=%s gds_max_inflight=%s",
            self._device,
            self._max_workers,
            _STRATEGY_CHAIN,
            self._backend_config.get("mx_gds_chunk_size_bytes"),
            self._backend_config.get("mx_gds_max_inflight_batches"),
        )

    def start_restore(self, sources: Sequence[FileTransferSource]) -> TransferSession:
        return MxTransferSession(
            device=self._device,
            max_workers=self._max_workers,
            backend_config=self._backend_config,
            sources=sources,
        )

    def close(self) -> None:
        pass


class MxTransferSession:
    """Convert GMS transfer descriptors and invoke ModelExpress restore."""

    def __init__(
        self,
        *,
        device: int,
        max_workers: int,
        backend_config: Mapping[str, object],
        sources: Sequence[FileTransferSource],
    ) -> None:
        self._device = int(device)
        self._max_workers = max(1, int(max_workers))
        self._backend_config = dict(backend_config)
        self._sources = list(sources)

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._device)

        try:
            mx_sources = [
                MxFileReadSource(
                    allocation_id=source.allocation_id,
                    file_path=source.file_path,
                    file_offset=source.file_offset,
                    byte_count=source.byte_count,
                )
                for source in self._sources
            ]
            mx_targets = {
                allocation_id: MxDeviceReadTarget(
                    allocation_id=target.allocation_id,
                    va=target.va,
                    device=target.device,
                    byte_count=target.byte_count,
                )
                for allocation_id, target in targets.items()
            }
            grouped_sources, _ = _group_restore_pairs(mx_sources, mx_targets)

            ctx = GmsRestoreContext.from_env(
                sources=mx_sources,
                targets=mx_targets,
                grouped_sources=grouped_sources,
                device=self._device,
                max_workers=self._max_workers,
                backend_config=self._backend_config,
                gds_chunk_size=self._backend_config.get("mx_gds_chunk_size_bytes"),
                gds_max_inflight=self._backend_config.get(
                    "mx_gds_max_inflight_batches"
                ),
            )
            stats = MxGmsRestoreStrategyChain.run(ctx)
        except Exception as exc:
            raise RuntimeError(
                "MX GMS restore failed: "
                f"device={self._device} strategy_chain={_STRATEGY_CHAIN} "
                f"sources={len(self._sources)} error={exc}"
            ) from exc

        logger.info(
            "MX restore complete: bytes=%s elapsed=%ss strategy=%s files=%s",
            stats.get("total_bytes", "-"),
            stats.get("elapsed_s", "-"),
            stats.get("selected_strategy", "-"),
            stats.get("file_count", "-"),
        )

    def close(self) -> None:
        pass
