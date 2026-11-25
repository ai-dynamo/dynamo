# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct NIXL delivery of encoded artifacts (`mode: "nixl_read"`).

Instead of uploading, the worker copies the encoded artifact into host memory it
registers with NIXL and returns `dynamo.nixl_connect` read metadata in the receipt.
The consumer pulls the bytes once, straight from worker memory, into its own host
or GPU buffer; nothing passes through the frontend or object storage.

The worker releases the buffer when the read completes or its lease expires,
whichever comes first. Held bytes count against a process budget, so a consumer
that never reads cannot grow worker memory without bound.

The receipt's metadata is a capability: any NIXL agent that holds it can read the
artifact until the lease ends. Deliver receipts only to trusted consumers.

Enable it on a worker with `DYNAMO_ARTIFACT_NIXL_BYTES` (the held-byte budget).
`DYNAMO_ARTIFACT_NIXL_MAX_LEASE_SECONDS` caps the lease a request may ask for.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

MODE = "nixl_read"
_DEFAULT_MAX_LEASE_SECONDS = 300
_RECEIPT_FIELDS = ("kind", "mode", "format", "contents")


class NixlArtifactCapacityError(RuntimeError):
    """Held NIXL artifacts already use this worker's byte budget."""


def _load_nixl_connect():
    import dynamo.nixl_connect as nixl_connect

    return nixl_connect


@dataclass
class _Held:
    operation: Any
    buffer: Any
    nbytes: int
    task: asyncio.Task | None = field(default=None, repr=False)


class NixlArtifactServer:
    """Holds encoded artifacts for one NIXL read each. Use from one event loop."""

    def __init__(
        self,
        *,
        max_held_bytes: int,
        max_lease_seconds: int = _DEFAULT_MAX_LEASE_SECONDS,
        nixl_connect: Any = None,
    ):
        if type(max_held_bytes) is not int or max_held_bytes <= 0:
            raise ValueError("NIXL artifact budget must be a positive byte count")
        if type(max_lease_seconds) is not int or not 1 <= max_lease_seconds <= 3600:
            raise ValueError("NIXL artifact lease cap must be 1 to 3600 seconds")
        self.max_held_bytes = max_held_bytes
        self.max_lease_seconds = max_lease_seconds
        self._nixl = nixl_connect if nixl_connect is not None else _load_nixl_connect()
        self._connector = None
        self._held: dict[int, _Held] = {}
        self._held_bytes = 0
        self._next_id = 0
        self._closed = False

    @property
    def held_bytes(self) -> int:
        return self._held_bytes

    def validate(self, destination: dict[str, Any], mode: str) -> None:
        """Check a request's destination before inference."""
        if self._closed:
            raise NixlArtifactCapacityError("NIXL artifact delivery is closed")
        if mode != MODE:
            raise ValueError("NIXL artifact delivery requires nixl_read mode")
        if (
            not isinstance(destination, dict)
            or destination.keys() != {"kind", "lease_seconds"}
            or destination.get("kind") != MODE
        ):
            raise ValueError("invalid nixl_read destination")
        lease = destination["lease_seconds"]
        if type(lease) is not int or not 1 <= lease <= self.max_lease_seconds:
            raise ValueError(
                f"nixl_read lease must be 1 to {self.max_lease_seconds} seconds"
            )
        if self._held_bytes >= self.max_held_bytes:
            raise NixlArtifactCapacityError("NIXL artifact budget exhausted")

    async def submit(
        self,
        data: bytes,
        contents: list[str],
        destination: dict[str, Any],
        *,
        mode: str = MODE,
    ) -> dict[str, Any]:
        """Hold `data` for one read and return the receipt the consumer needs."""
        self.validate(destination, mode)
        if not isinstance(data, bytes) or not data:
            raise ValueError("NIXL artifact requires nonempty bytes")
        nbytes = len(data)
        if self._held_bytes + nbytes > self.max_held_bytes:
            raise NixlArtifactCapacityError("NIXL artifact budget exhausted")
        self._held_bytes += nbytes
        try:
            buffer = _host_buffer(data)
            if self._connector is None:
                self._connector = self._nixl.Connector()
            operation = await self._connector.create_readable(
                self._nixl.Descriptor(buffer)
            )
            metadata = operation.metadata().model_dump(mode="json")
        except BaseException:
            self._held_bytes -= nbytes
            raise
        lease = destination["lease_seconds"]
        held = _Held(operation, buffer, nbytes)
        held_id = self._next_id
        self._next_id += 1
        self._held[held_id] = held
        held.task = asyncio.create_task(self._hold(held_id, held, lease))
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease)
        return {
            "kind": "generation_artifact",
            "mode": MODE,
            "format": "generation_artifact_v1",
            "contents": list(contents),
            "state": "ready",
            "byte_count": nbytes,
            "sha256": hashlib.sha256(data).hexdigest(),
            "rdma_metadata": metadata,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }

    async def _hold(self, held_id: int, held: _Held, lease: int) -> None:
        started = time.monotonic()
        try:
            await asyncio.wait_for(held.operation.wait_for_completion(), lease)
            logger.debug(
                "NIXL artifact read after %.3fs (%d bytes)",
                time.monotonic() - started,
                held.nbytes,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "NIXL artifact lease expired unread after %ds (%d bytes)",
                lease,
                held.nbytes,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("NIXL artifact hold failed")
        finally:
            self._release(held_id)

    def _release(self, held_id: int) -> None:
        held = self._held.pop(held_id, None)
        if held is None:
            return
        try:
            # Exiting the operation deregisters its memory from NIXL.
            held.operation.__exit__(None, None, None)
        finally:
            held.buffer = None
            self._held_bytes -= held.nbytes

    async def close(self) -> None:
        """Release every held artifact, read or not."""
        self._closed = True
        tasks = []
        for held_id, held in tuple(self._held.items()):
            if held.task is not None and not held.task.done():
                held.task.cancel()
                tasks.append(held.task)
            self._release(held_id)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _host_buffer(data: bytes):
    """Copy into a uint8 tensor NIXL can register, pinned when CUDA allows it."""
    import torch

    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if torch.cuda.is_available():
        try:
            return source.pin_memory()
        except RuntimeError:
            logger.debug("Pinned memory unavailable; serving NIXL artifact pageable")
    return source


@cache
def configured_nixl_artifact_server() -> NixlArtifactServer | None:
    """The worker's NIXL artifact server, or None when not enabled."""
    configured = os.environ.get("DYNAMO_ARTIFACT_NIXL_BYTES")
    if not configured:
        return None
    lease = int(
        os.environ.get(
            "DYNAMO_ARTIFACT_NIXL_MAX_LEASE_SECONDS", str(_DEFAULT_MAX_LEASE_SECONDS)
        )
    )
    return NixlArtifactServer(max_held_bytes=int(configured), max_lease_seconds=lease)


async def close_configured_nixl_artifact_server() -> None:
    if os.environ.get("DYNAMO_ARTIFACT_NIXL_BYTES"):
        server = configured_nixl_artifact_server()
        if server is not None:
            await server.close()


async def read_nixl_artifact(
    receipt: dict[str, Any],
    *,
    device: Any = "cpu",
    timeout_seconds: float = 60.0,
    connector: Any = None,
    nixl_connect: Any = None,
):
    """Pull one `nixl_read` artifact into a uint8 tensor on `device`.

    On the CPU, the bytes are checked against the receipt's sha256. Decode them with
    `dynamo.artifacts.vllm.format.decode_generation_artifact(tensor.numpy())`.
    """
    import torch

    if not isinstance(receipt, dict) or any(
        receipt.get(name) is None for name in (*_RECEIPT_FIELDS, "rdma_metadata")
    ):
        raise ValueError("not a nixl_read artifact receipt")
    if receipt.get("mode") != MODE or receipt.get("state") != "ready":
        raise ValueError("artifact receipt is not a ready nixl_read delivery")
    nbytes = receipt.get("byte_count")
    if type(nbytes) is not int or nbytes <= 0:
        raise ValueError("invalid nixl_read byte count")
    nixl = nixl_connect if nixl_connect is not None else _load_nixl_connect()
    connector = connector if connector is not None else nixl.Connector()
    metadata = nixl.RdmaMetadata.model_validate(receipt["rdma_metadata"])
    target = torch.empty(nbytes, dtype=torch.uint8, device=device)
    operation = await connector.begin_read(metadata, nixl.Descriptor(target))
    with operation:
        await asyncio.wait_for(operation.wait_for_completion(), timeout_seconds)
    if target.device.type == "cpu":
        digest = hashlib.sha256(target.numpy().tobytes()).hexdigest()
        if digest != receipt.get("sha256"):
            raise ValueError("nixl_read artifact digest differs from receipt")
    return target
