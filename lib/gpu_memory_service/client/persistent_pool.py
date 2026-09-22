# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS v0 implementation of the persistent-pool ownership contract."""

from __future__ import annotations

from gpu_memory_service.client.rpc import (
    GMS_ERR_CLAIM_CONFLICT,
    GMS_ERR_OUT_OF_MEMORY,
    GmsRemoteError,
)
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.persistent_pool import (
    PersistentPoolAllocation,
    PersistentPoolKey,
    retry_persistent_claim,
)


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
        try:
            response = retry_persistent_claim(
                lambda: self._session.claim_persistent(
                    engine_id=key.engine_id,
                    tag=key.tag,
                    size=aligned_size,
                    shared=shared,
                ),
                lambda exc: isinstance(exc, GmsRemoteError)
                and exc.code == GMS_ERR_CLAIM_CONFLICT,
            )
        except GmsRemoteError as exc:
            if exc.code == GMS_ERR_OUT_OF_MEMORY:
                raise MemoryError(str(exc)) from exc
            raise
        return PersistentPoolAllocation(
            key=key,
            allocation_id=response.allocation_id,
            size=int(getattr(response, "size", aligned_size)),
            aligned_size=int(response.aligned_size),
            reattached=bool(response.reattached),
        )

    def unclaim(self, key: PersistentPoolKey) -> bool:
        return self._session.unclaim_persistent(engine_id=key.engine_id, tag=key.tag)

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
