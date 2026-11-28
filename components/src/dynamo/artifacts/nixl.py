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

An artifact with chunked hidden states (wire 1.2) is held as the object plus one
buffer per distinct chunk, all in one readable operation. Its receipt lists the
chunks under `parts`, in descriptor order; read it with `read_parts()`.

Consumers read with `NixlArtifactReader`, or `read_nixl_artifact()`, which uses a
shared one. A process's first NIXL agent takes about a second to start (UCX and CUDA
initialization); later agents and reads are milliseconds. Call `warm()` at startup
to move that cost off the first request.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

MODE = "nixl_read"
_DEFAULT_MAX_LEASE_SECONDS = 300
_DEFAULT_MAX_REMOTES = 64
_RECEIPT_FIELDS = ("kind", "mode", "format", "contents")


class NixlArtifactCapacityError(RuntimeError):
    """Held NIXL artifacts already use this worker's byte budget."""


def _load_nixl_connect():
    import dynamo.nixl_connect as nixl_connect

    return nixl_connect


async def _shared_connection(connector: Any) -> Any:
    """The connector's single NIXL agent, started on first use (about a second)."""
    # nixl_connect starts its shared agent lazily inside its first operation and
    # has no public accessor for it; this is that same idempotent call.
    return await connector._create_connection()


@dataclass
class _Held:
    operation: Any
    buffers: list[Any]
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

    @property
    def max_artifact_bytes(self) -> int:
        """Largest artifact, object plus chunks, this server could ever hold."""
        return self.max_held_bytes

    async def warm(self) -> None:
        """Start the NIXL agent now instead of on the first request."""
        if self._connector is None:
            self._connector = self._nixl.Connector()
        await _shared_connection(self._connector)

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
        parts: dict[str, bytes] | None = None,
    ) -> dict[str, Any]:
        """Hold `data`, and any chunk `parts` by SHA-256, for one read.

        Returns the receipt the consumer needs.
        """
        self.validate(destination, mode)
        if not isinstance(data, bytes) or not data:
            raise ValueError("NIXL artifact requires nonempty bytes")
        chunks = list((parts or {}).items())
        # Digests come from the capture that hashed these bytes; the reader
        # checks every buffer against them, so they are not recomputed here.
        if any(not isinstance(blob, bytes) or not blob for _, blob in chunks):
            raise ValueError("NIXL artifact chunks require nonempty bytes")
        nbytes = len(data) + sum(len(blob) for _, blob in chunks)
        if self._held_bytes + nbytes > self.max_held_bytes:
            raise NixlArtifactCapacityError("NIXL artifact budget exhausted")
        self._held_bytes += nbytes
        try:
            buffers = [_host_buffer(data), *(_host_buffer(blob) for _, blob in chunks)]
            if self._connector is None:
                self._connector = self._nixl.Connector()
            descriptors = [self._nixl.Descriptor(buffer) for buffer in buffers]
            operation = await self._connector.create_readable(
                descriptors if chunks else descriptors[0]
            )
            metadata = operation.metadata().model_dump(mode="json")
        except BaseException:
            self._held_bytes -= nbytes
            raise
        lease = destination["lease_seconds"]
        held = _Held(operation, buffers, nbytes)
        held_id = self._next_id
        self._next_id += 1
        self._held[held_id] = held
        held.task = asyncio.create_task(self._hold(held_id, held, lease))
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=lease)
        receipt = {
            "kind": "generation_artifact",
            "mode": MODE,
            "format": "generation_artifact_v1",
            "contents": list(contents),
            "state": "ready",
            "byte_count": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "rdma_metadata": metadata,
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }
        if chunks:
            receipt["parts"] = [
                {"sha256": digest, "byte_count": len(blob)} for digest, blob in chunks
            ]
        return receipt

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
            held.buffers = []
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


class NixlArtifactReader:
    """Reads `nixl_read` artifacts through one NIXL agent, kept for its lifetime.

    A new agent per read leaks memory (about 1 MiB a read in testing), and adding and
    removing a worker's agent around every read leaks a file descriptor each time.
    The reader keeps each worker it has read from connected, evicting the least
    recently used beyond `max_remotes`. Use it from one event loop at a time.
    """

    def __init__(
        self,
        *,
        nixl_connect: Any = None,
        connector: Any = None,
        max_remotes: int = _DEFAULT_MAX_REMOTES,
    ):
        if type(max_remotes) is not int or max_remotes < 1:
            raise ValueError("max_remotes must be a positive integer")
        self._nixl = nixl_connect if nixl_connect is not None else _load_nixl_connect()
        self._connector = connector if connector is not None else self._nixl.Connector()
        self._max_remotes = max_remotes
        # Keyed by the worker agent's metadata, which changes when it restarts.
        self._remotes: OrderedDict[str, Any] = OrderedDict()

    @property
    def connected_remotes(self) -> int:
        return len(self._remotes)

    async def warm(self) -> None:
        """Start the NIXL agent now instead of on the first read."""
        await _shared_connection(self._connector)

    async def _pin_remote(self, nixl_metadata: str) -> None:
        if nixl_metadata in self._remotes:
            self._remotes.move_to_end(nixl_metadata)
            return
        connection = await _shared_connection(self._connector)
        self._remotes[nixl_metadata] = self._nixl.Remote(connection, nixl_metadata)
        while len(self._remotes) > self._max_remotes:
            _, evicted = self._remotes.popitem(last=False)
            evicted.__exit__(None, None, None)

    async def read(
        self,
        receipt: dict[str, Any],
        *,
        device: Any = "cpu",
        timeout_seconds: float = 60.0,
    ):
        """Pull one artifact into a uint8 tensor on `device`.

        On the CPU, the bytes are checked against the receipt's sha256. Decode them
        with `dynamo.artifacts.vllm.format.decode_generation_artifact`. A receipt
        with chunk `parts` needs `read_parts()`.
        """
        if receipt.get("parts") if isinstance(receipt, dict) else None:
            raise ValueError("chunked nixl_read artifact: use read_parts()")
        artifact, _ = await self.read_parts(
            receipt, device=device, timeout_seconds=timeout_seconds
        )
        return artifact

    async def read_parts(
        self,
        receipt: dict[str, Any],
        *,
        device: Any = "cpu",
        timeout_seconds: float = 60.0,
    ):
        """Pull an artifact and its chunks in one read.

        Returns the object as a uint8 tensor and a dict of chunk tensors by
        SHA-256, empty when the artifact has no chunks. On the CPU every buffer is
        checked against its digest. Rebuild hidden states with
        `dynamo.artifacts.vllm.format.assemble_hidden_states(component,
        lambda digest: chunks[digest].numpy())`.
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
        expected = [(receipt.get("sha256"), nbytes)]
        parts = receipt.get("parts") or []
        if not isinstance(parts, list):
            raise ValueError("invalid nixl_read parts")
        for part in parts:
            if (
                not isinstance(part, dict)
                or not isinstance(part.get("sha256"), str)
                or type(part.get("byte_count")) is not int
                or part["byte_count"] <= 0
            ):
                raise ValueError("invalid nixl_read part")
            expected.append((part["sha256"], part["byte_count"]))
        metadata = self._nixl.RdmaMetadata.model_validate(receipt["rdma_metadata"])
        nixl_metadata = getattr(metadata, "nixl_metadata", None)
        if isinstance(nixl_metadata, str) and nixl_metadata:
            await self._pin_remote(nixl_metadata)
        targets = [
            torch.empty(size, dtype=torch.uint8, device=device) for _, size in expected
        ]
        descriptors = [self._nixl.Descriptor(target) for target in targets]
        operation = await self._connector.begin_read(
            metadata, descriptors if parts else descriptors[0]
        )
        with operation:
            await asyncio.wait_for(operation.wait_for_completion(), timeout_seconds)
        if targets[0].device.type == "cpu":
            for target, (digest, _) in zip(targets, expected):
                if hashlib.sha256(target.numpy().tobytes()).hexdigest() != digest:
                    raise ValueError("nixl_read artifact digest differs from receipt")
        return targets[0], {
            digest: target for target, (digest, _) in zip(targets[1:], expected[1:])
        }

    def close(self) -> None:
        """Disconnect from every worker this reader has read from."""
        while self._remotes:
            _, remote = self._remotes.popitem()
            remote.__exit__(None, None, None)


_default_reader: NixlArtifactReader | None = None


def default_nixl_artifact_reader() -> NixlArtifactReader:
    """The process-wide reader `read_nixl_artifact` uses."""
    global _default_reader
    if _default_reader is None:
        _default_reader = NixlArtifactReader()
    return _default_reader


async def read_nixl_artifact(
    receipt: dict[str, Any],
    *,
    device: Any = "cpu",
    timeout_seconds: float = 60.0,
    reader: NixlArtifactReader | None = None,
):
    """Pull one `nixl_read` artifact; see `NixlArtifactReader.read`."""
    reader = reader if reader is not None else default_nixl_artifact_reader()
    return await reader.read(receipt, device=device, timeout_seconds=timeout_seconds)
