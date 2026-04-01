# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal GPU Memory Service client session."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from gpu_memory_service.client.rpc import _GMSRPCTransport
from gpu_memory_service.common.protocol.messages import (
    AllocateRequest,
    AllocateResponse,
    CommitRequest,
    CommitResponse,
    ExportAllocationRequest,
    ExportAllocationResponse,
    FreeAllocationRequest,
    FreeAllocationResponse,
    GetAllocationRequest,
    GetAllocationResponse,
    GetAllocationStateRequest,
    GetAllocationStateResponse,
    GetLockStateRequest,
    GetLockStateResponse,
    GetStateHashRequest,
    GetStateHashResponse,
    HandshakeResponse,
    ListAllocationsRequest,
    ListAllocationsResponse,
    MetadataDeleteRequest,
    MetadataDeleteResponse,
    MetadataGetRequest,
    MetadataGetResponse,
    MetadataListRequest,
    MetadataListResponse,
    MetadataPutRequest,
    MetadataPutResponse,
)
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

logger = logging.getLogger(__name__)


class _GMSClientSession:
    """Connected GMS client session with granted lock state."""

    def __init__(
        self,
        socket_path: str,
        lock_type: RequestedLockType,
        timeout_ms: Optional[int],
    ):
        self._requested_lock_type = lock_type
        self._transport = _GMSRPCTransport(socket_path)
        self._transport.connect()
        try:
            response = self._transport.handshake(lock_type, timeout_ms)
        except Exception:
            try:
                self._transport.close()
            except Exception:
                pass
            raise
        self._initialize_from_handshake(response)

    def _initialize_from_handshake(self, response: HandshakeResponse) -> None:
        if not response.success:
            self._transport.close()
            raise TimeoutError("Timeout waiting for lock")

        self._committed = response.committed
        if response.granted_lock_type is None:
            self._transport.close()
            raise RuntimeError("HandshakeResponse omitted granted_lock_type")
        self._granted_lock_type = response.granted_lock_type

        logger.info(
            "Connected with %s lock (granted=%s), committed=%s",
            self._requested_lock_type.value,
            self._granted_lock_type.value,
            self._committed,
        )

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def lock_type(self) -> GrantedLockType:
        return self._granted_lock_type

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    def get_lock_state(self) -> GetLockStateResponse:
        return self._transport.request(GetLockStateRequest(), GetLockStateResponse)

    def get_allocation_state(self) -> GetAllocationStateResponse:
        return self._transport.request(
            GetAllocationStateRequest(), GetAllocationStateResponse
        )

    def is_ready(self) -> bool:
        return self.committed

    def commit(self) -> bool:
        response = self._transport.request(CommitRequest(), CommitResponse)
        if not response.success:
            raise RuntimeError("GMS commit returned failure")
        self._committed = True
        try:
            self.close()
        except ConnectionError as exc:
            logger.warning("Commit succeeded but closing transport failed: %s", exc)
        logger.info("Committed weights and released RW connection")
        return True

    def allocate_info(self, size: int, tag: str = "default") -> AllocateResponse:
        return self._transport.request(
            AllocateRequest(size=size, tag=tag), AllocateResponse
        )

    def allocate(self, size: int, tag: str = "default") -> Tuple[str, int]:
        response = self.allocate_info(size=size, tag=tag)
        return response.allocation_id, response.aligned_size

    def export(self, allocation_id: str) -> int:
        response, fd = self._transport.request_with_fd(
            ExportAllocationRequest(allocation_id=allocation_id),
            ExportAllocationResponse,
        )
        if fd < 0:
            raise RuntimeError(
                f"GMS export returned no FD for allocation_id={allocation_id}"
            )
        return fd

    def get_allocation(self, allocation_id: str) -> GetAllocationResponse:
        return self._transport.request(
            GetAllocationRequest(allocation_id=allocation_id),
            GetAllocationResponse,
        )

    def list_allocations(
        self, tag: Optional[str] = None
    ) -> List[GetAllocationResponse]:
        return self._transport.request(
            ListAllocationsRequest(tag=tag),
            ListAllocationsResponse,
        ).allocations

    def free(self, allocation_id: str) -> bool:
        return self._transport.request(
            FreeAllocationRequest(allocation_id=allocation_id),
            FreeAllocationResponse,
        ).success

    def metadata_put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> bool:
        return self._transport.request(
            MetadataPutRequest(
                key=key,
                allocation_id=allocation_id,
                offset_bytes=offset_bytes,
                value=value,
            ),
            MetadataPutResponse,
        ).success

    def metadata_get(self, key: str) -> Optional[tuple[str, int, bytes]]:
        response = self._transport.request(
            MetadataGetRequest(key=key), MetadataGetResponse
        )
        if not response.found:
            return None
        return response.allocation_id, response.offset_bytes, response.value

    def metadata_delete(self, key: str) -> bool:
        return self._transport.request(
            MetadataDeleteRequest(key=key), MetadataDeleteResponse
        ).deleted

    def metadata_list(self, prefix: str = "") -> List[str]:
        return self._transport.request(
            MetadataListRequest(prefix=prefix), MetadataListResponse
        ).keys

    def get_memory_layout_hash(self) -> str:
        return self._transport.request(
            GetStateHashRequest(), GetStateHashResponse
        ).memory_layout_hash

    def close(self) -> None:
        self._transport.close()
        logger.info("Closed %s connection", self._granted_lock_type.value)

    def __enter__(self) -> "_GMSClientSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
