# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.persistent_pool import (
    PersistentPoolAllocation,
    PersistentPoolKey,
    retry_persistent_claim,
)
from gpu_memory_service.v1.client.session import GMSV1RemoteError, _GMSClientSession
from gpu_memory_service.v1.protocol import ERROR_CLAIM_CONFLICT


class V1PersistentPoolBackend:
    """Adapt typed persistent-pool RPCs on a connected GMS v1 session.

    The session must use ``RW_PERSISTENT``. This namespace is independent of
    v1's transactional weight/KV allocation epochs: disconnect drops claims,
    while backing survives until :meth:`destroy` explicitly retires it.
    """

    def __init__(self, session: _GMSClientSession) -> None:
        if session.lock_type is not GrantedLockType.RW_PERSISTENT:
            raise ValueError(
                "v1 persistent-pool backend requires an RW_PERSISTENT session"
            )
        self._session = session

    def claim(
        self,
        key: PersistentPoolKey,
        aligned_size: int,
        *,
        shared: bool = False,
    ) -> PersistentPoolAllocation:
        response = retry_persistent_claim(
            lambda: self._session.claim_persistent(
                key.engine_id, key.tag, aligned_size, shared=shared
            ),
            lambda exc: isinstance(exc, GMSV1RemoteError)
            and exc.code == ERROR_CLAIM_CONFLICT,
        )
        allocation = response.allocation
        return PersistentPoolAllocation(
            key=key,
            allocation_id=allocation.allocation_id,
            size=int(allocation.size),
            aligned_size=int(allocation.aligned_size),
            reattached=bool(response.reattached),
            claimed=bool(allocation.claimed),
        )

    def unclaim(self, key: PersistentPoolKey) -> bool:
        return self._session.unclaim_persistent(key.engine_id, key.tag)

    def export(self, key: PersistentPoolKey) -> int:
        return self._session.export_persistent(key.engine_id, key.tag)

    def inventory(
        self,
        engine_id: str | None = None,
        *,
        include_unclaimed: bool = False,
    ) -> list[PersistentPoolAllocation]:
        response = self._session.list_persistent(
            engine_id,
            include_unclaimed=include_unclaimed,
        )
        return [
            PersistentPoolAllocation(
                key=PersistentPoolKey(item.engine_id, item.tag),
                allocation_id=item.allocation_id,
                size=int(item.size),
                aligned_size=int(item.aligned_size),
                claimed=bool(item.claimed),
            )
            for item in response.allocations
        ]

    def destroy(self, key: PersistentPoolKey) -> bool:
        return self._session.destroy_persistent(key.engine_id, key.tag)
