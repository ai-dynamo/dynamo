# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS v0 implementation of the persistent-pool ownership contract."""

from __future__ import annotations

import logging
import os
import time

from gpu_memory_service.client.rpc import GMS_ERR_CLAIM_CONFLICT, GmsRemoteError
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.persistent_pool import (
    PersistentPoolAllocation,
    PersistentPoolKey,
)

logger = logging.getLogger(__name__)


class V0PersistentPoolBackend:
    """Adapt persistent-allocation RPCs on a connected GMS v0 session."""

    def __init__(self, session: _GMSClientSession) -> None:
        self._session = session

    def claim(
        self,
        key: PersistentPoolKey,
        aligned_size: int,
        *,
        shared: bool = False,
    ) -> PersistentPoolAllocation:
        retry_secs = float(os.environ.get("GMS_PERSISTENT_CLAIM_RETRY_SECS", "2.0"))
        deadline = time.monotonic() + max(0.0, retry_secs)
        delay = 0.05
        while True:
            try:
                response = self._session.claim_persistent(
                    engine_id=key.engine_id,
                    tag=key.tag,
                    size=aligned_size,
                    shared=shared,
                )
                return PersistentPoolAllocation(
                    key=key,
                    allocation_id=response.allocation_id,
                    size=int(getattr(response, "size", aligned_size)),
                    aligned_size=int(response.aligned_size),
                    reattached=bool(response.reattached),
                )
            except GmsRemoteError as exc:
                if exc.code != GMS_ERR_CLAIM_CONFLICT or time.monotonic() >= deadline:
                    raise
                logger.warning(
                    "GMS persistent claim conflict for %s/%s; retrying in %.2fs "
                    "(likely a prior connection's claim not yet cleaned up)",
                    key.engine_id,
                    key.tag,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 0.5)

    def export(self, key: PersistentPoolKey) -> int:
        _, fd = self._session.export_persistent(
            engine_id=key.engine_id,
            tag=key.tag,
        )
        return fd

    def inventory(
        self,
        engine_id: str | None = None,
        *,
        include_unclaimed: bool = False,
    ) -> list[PersistentPoolAllocation]:
        return [
            PersistentPoolAllocation(
                key=PersistentPoolKey(item.engine_id, item.tag),
                allocation_id=item.allocation_id,
                size=int(item.size),
                aligned_size=int(item.aligned_size),
                claimed=bool(item.claimed),
            )
            for item in self._session.list_persistent(
                engine_id=engine_id,
                include_unclaimed=include_unclaimed,
            )
        ]

    def destroy(self, key: PersistentPoolKey) -> bool:
        return self._session.release_persistent(
            engine_id=key.engine_id,
            tag=key.tag,
        )
