# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit NIXL tensor carrier and producer-side lease ownership."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from dynamo.workflow.runtime import WorkflowExecutionError

logger = logging.getLogger(__name__)

NIXL_TENSOR_SCHEMA = "dynamo.workflow.nixl_tensor"
NIXL_TENSOR_FANOUT_SCHEMA = "dynamo.workflow.nixl_tensor_fanout"
NIXL_TENSOR_VERSION = 0
DEFAULT_NIXL_LEASE_TIMEOUT_S = 60.0


def _check_keys(data: Mapping[str, Any], required: set[str]) -> None:
    keys = set(data)
    missing = required - keys
    unknown = keys - required
    if missing:
        raise WorkflowExecutionError(f"NIXL metadata missing fields: {sorted(missing)}")
    if unknown:
        raise WorkflowExecutionError(
            f"NIXL metadata has unknown fields: {sorted(unknown)}"
        )


def _validate_wire_version(data: Mapping[str, Any], schema: str) -> None:
    if data["schema"] != schema:
        raise WorkflowExecutionError(f"unsupported NIXL schema {data['schema']!r}")
    version = data["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != NIXL_TENSOR_VERSION
    ):
        raise WorkflowExecutionError(f"unsupported NIXL version {version!r}")


@dataclass(frozen=True)
class NixlTensorRef:
    """Serializable reference to one tensor-readable NIXL operation."""

    transfer_id: str
    lease_id: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    rdma_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("transfer_id", self.transfer_id),
            ("lease_id", self.lease_id),
            ("dtype", self.dtype),
            ("device", self.device),
        ):
            if not isinstance(value, str) or not value:
                raise WorkflowExecutionError(
                    f"NIXL tensor {field_name} must be a non-empty string"
                )
        shape = tuple(self.shape)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in shape
        ):
            raise WorkflowExecutionError(
                "NIXL tensor shape must contain non-negative integers"
            )
        if not isinstance(self.rdma_metadata, Mapping):
            raise WorkflowExecutionError("NIXL rdma_metadata must be an object")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self, "rdma_metadata", MappingProxyType(dict(self.rdma_metadata))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NIXL_TENSOR_SCHEMA,
            "version": NIXL_TENSOR_VERSION,
            "transfer_id": self.transfer_id,
            "lease_id": self.lease_id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "rdma_metadata": dict(self.rdma_metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NixlTensorRef":
        if not isinstance(data, Mapping):
            raise WorkflowExecutionError("NIXL tensor reference must be an object")
        _check_keys(
            data,
            {
                "schema",
                "version",
                "transfer_id",
                "lease_id",
                "shape",
                "dtype",
                "device",
                "rdma_metadata",
            },
        )
        _validate_wire_version(data, NIXL_TENSOR_SCHEMA)
        shape = data["shape"]
        if not isinstance(shape, list):
            raise WorkflowExecutionError("NIXL tensor shape must be an array")
        return cls(
            transfer_id=data["transfer_id"],
            lease_id=data["lease_id"],
            shape=tuple(shape),
            dtype=data["dtype"],
            device=data["device"],
            rdma_metadata=data["rdma_metadata"],
        )


@dataclass(frozen=True)
class NixlTensorFanout:
    """Per-consumer NIXL references for one logical tensor output."""

    transfers: Mapping[str, NixlTensorRef]

    def __post_init__(self) -> None:
        if not isinstance(self.transfers, Mapping) or not self.transfers:
            raise WorkflowExecutionError("NIXL tensor fanout requires transfers")
        transfers: dict[str, NixlTensorRef] = {}
        for transfer_id, reference in sorted(self.transfers.items()):
            if not isinstance(reference, NixlTensorRef):
                raise WorkflowExecutionError(
                    "NIXL tensor fanout values must use NixlTensorRef"
                )
            if transfer_id != reference.transfer_id:
                raise WorkflowExecutionError(
                    "NIXL tensor fanout key does not match transfer id"
                )
            transfers[transfer_id] = reference
        object.__setattr__(self, "transfers", MappingProxyType(transfers))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": NIXL_TENSOR_FANOUT_SCHEMA,
            "version": NIXL_TENSOR_VERSION,
            "transfers": {
                transfer_id: reference.to_dict()
                for transfer_id, reference in self.transfers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NixlTensorFanout":
        if not isinstance(data, Mapping):
            raise WorkflowExecutionError("NIXL tensor fanout must be an object")
        _check_keys(data, {"schema", "version", "transfers"})
        _validate_wire_version(data, NIXL_TENSOR_FANOUT_SCHEMA)
        transfers = data["transfers"]
        if not isinstance(transfers, Mapping):
            raise WorkflowExecutionError(
                "NIXL tensor fanout transfers must be an object"
            )
        return cls(
            {
                transfer_id: NixlTensorRef.from_dict(reference)
                for transfer_id, reference in transfers.items()
            }
        )

    def for_transfer(self, transfer_id: str) -> NixlTensorRef:
        try:
            return self.transfers[transfer_id]
        except KeyError as error:
            raise WorkflowExecutionError(
                f"NIXL tensor fanout has no transfer {transfer_id!r}"
            ) from error


@dataclass
class _Lease:
    operations: tuple[Any, ...]
    value: Any
    task: asyncio.Task[None]
    on_release: Optional[Callable[[], None]]


class NixlLeaseRegistry:
    """Keep producer memory registered until every read is confirmed complete."""

    def __init__(self, timeout_s: float = DEFAULT_NIXL_LEASE_TIMEOUT_S) -> None:
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("NIXL lease timeout must be a finite positive number")
        self._timeout_s = float(timeout_s)
        self._leases: dict[str, _Lease] = {}

    @property
    def active_count(self) -> int:
        return len(self._leases)

    def track(self, lease_id: str, operation: Any, value: Any) -> None:
        self.track_fanout({lease_id: operation}, value)

    def track_fanout(
        self,
        operations: Mapping[str, Any],
        value: Any,
        on_release: Optional[Callable[[], None]] = None,
    ) -> None:
        """Keep one shared tensor alive until every consumer read is terminal."""

        if not isinstance(operations, Mapping) or not operations:
            raise WorkflowExecutionError("NIXL fanout lease requires operations")
        if on_release is not None and not callable(on_release):
            raise TypeError("NIXL fanout release callback must be callable")
        lease_ids = tuple(operations)
        duplicate = next(
            (lease_id for lease_id in lease_ids if lease_id in self._leases), None
        )
        if duplicate is not None:
            raise WorkflowExecutionError(f"duplicate NIXL lease {duplicate!r}")
        operation_values = tuple(operations.values())
        task = asyncio.create_task(
            self._wait_and_release(lease_ids, operation_values, on_release),
            name=f"workflow-nixl-lease:{lease_ids[0]}",
        )
        lease = _Lease(operation_values, value, task, on_release)
        for lease_id in lease_ids:
            self._leases[lease_id] = lease

    async def _wait_and_release(
        self,
        lease_ids: tuple[str, ...],
        operations: tuple[Any, ...],
        on_release: Optional[Callable[[], None]],
    ) -> None:
        async def wait_for_all() -> None:
            results = await asyncio.gather(
                *(operation.wait_for_completion() for operation in operations),
                return_exceptions=True,
            )
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if failures:
                raise WorkflowExecutionError(
                    f"NIXL workflow leases {lease_ids!r} have uncertain read state"
                ) from failures[0]

        completion = asyncio.create_task(
            wait_for_all(),
            name=f"workflow-nixl-completion:{lease_ids[0]}",
        )
        try:
            done, _ = await asyncio.wait({completion}, timeout=self._timeout_s)
            if not done:
                logger.warning(
                    "NIXL workflow leases %s were not all read within %.1fs; "
                    "retaining producer memory until late completion or process exit",
                    lease_ids,
                    self._timeout_s,
                )
            await asyncio.shield(completion)
        except asyncio.CancelledError:
            logger.warning(
                "NIXL workflow lease monitor for %s was cancelled; retaining "
                "producer memory because read completion is uncertain",
                lease_ids,
            )
            raise
        except Exception:
            logger.warning(
                "NIXL workflow leases %s entered quarantine because read completion "
                "is uncertain; retaining producer memory until process exit",
                lease_ids,
                exc_info=True,
            )
            return

        for lease_id in lease_ids:
            self._leases.pop(lease_id, None)
        for lease_id, operation in zip(lease_ids, operations):
            try:
                operation.__exit__(None, None, None)
            except Exception:
                logger.warning(
                    "failed to release completed NIXL workflow lease %s",
                    lease_id,
                    exc_info=True,
                )
        if on_release is not None:
            on_release()

    async def close(self) -> None:
        if self._leases:
            logger.warning(
                "closing NIXL workflow registry with %d active leases; retaining "
                "producer memory until late completion or process exit",
                len(self._leases),
            )
        await asyncio.sleep(0)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        result = value.model_dump()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise WorkflowExecutionError("NIXL metadata is not serializable")
    if not isinstance(result, dict):
        raise WorkflowExecutionError("NIXL metadata must serialize as an object")
    return result


def _limit_transfer_metadata(metadata: Any, transfer_bytes: int) -> dict[str, Any]:
    """Describe only the live prefix of a larger registered pool slot."""

    result = _model_dump(metadata)
    descriptors = result.get("descriptors")
    if descriptors is None:
        return result
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        raise WorkflowExecutionError(
            "NIXL pooled tensor export requires exactly one descriptor"
        )
    descriptor = descriptors[0]
    if hasattr(descriptor, "model_dump"):
        descriptor = descriptor.model_dump()
    elif isinstance(descriptor, Mapping):
        descriptor = dict(descriptor)
    else:
        raise WorkflowExecutionError("NIXL descriptor metadata must be an object")
    registered_bytes = descriptor.get("size")
    if (
        isinstance(registered_bytes, bool)
        or not isinstance(registered_bytes, int)
        or transfer_bytes > registered_bytes
    ):
        raise WorkflowExecutionError(
            "NIXL tensor exceeds its registered transfer descriptor"
        )
    descriptor["size"] = transfer_bytes
    result["descriptors"] = [descriptor]
    return result


class NixlTensorCarrier:
    """Export and import torch tensors with explicit NIXL READ operations."""

    def __init__(
        self,
        *,
        receive_device: Optional[str] = None,
        lease_timeout_s: float = DEFAULT_NIXL_LEASE_TIMEOUT_S,
        connector: Any = None,
        nixl_module: Any = None,
        torch_module: Any = None,
        send_pool_capacity: int = 0,
        send_pool_bytes: int = 0,
        enable_progress_thread: bool = False,
    ) -> None:
        if nixl_module is None:
            try:
                from dynamo import nixl_connect as default_nixl_module
            except ImportError as error:
                raise RuntimeError(
                    "NIXL tensor carrier requested but dynamo.nixl_connect is unavailable"
                ) from error
            nixl_module = default_nixl_module
        if torch_module is None:
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("NIXL tensor carrier requires PyTorch") from error
            torch_module = torch
        if receive_device is not None and (
            not isinstance(receive_device, str) or not receive_device
        ):
            raise ValueError("NIXL receive_device must be non-empty when set")
        for field_name, value in (
            ("send_pool_capacity", send_pool_capacity),
            ("send_pool_bytes", send_pool_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if bool(send_pool_capacity) != bool(send_pool_bytes):
            raise ValueError(
                "send_pool_capacity and send_pool_bytes must both be zero or positive"
            )
        if not isinstance(enable_progress_thread, bool):
            raise TypeError("enable_progress_thread must be a bool")
        self._nixl = nixl_module
        self._torch = torch_module
        if connector is None:
            self._connector = nixl_module.Connector(
                enable_progress_thread=enable_progress_thread
            )
            self._export_connector_factory = lambda: nixl_module.Connector(
                enable_progress_thread=enable_progress_thread
            )
        else:
            self._connector = connector
            self._export_connector_factory = None
        self._receive_device = receive_device
        self._leases = NixlLeaseRegistry(lease_timeout_s)
        self._send_pool_capacity = send_pool_capacity
        self._send_pool_bytes = send_pool_bytes
        self._send_pool_lock = asyncio.Lock()
        self._send_pool_available: asyncio.Queue[int] = asyncio.Queue(
            maxsize=send_pool_capacity
        )
        self._send_pool_backing: Any = None
        self._send_pool_descriptors: list[Any] = []
        self._send_pool_connection: Any = None

    @property
    def active_leases(self) -> int:
        return self._leases.active_count

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Mapping[str, Any]:
        references = await self.export_tensor_fanout(tensor, (transfer_id,))
        return references[transfer_id]

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Export one shared tensor through one independently addressed read per consumer."""

        if not isinstance(tensor, self._torch.Tensor):
            raise WorkflowExecutionError("NIXL carrier can export torch.Tensor only")
        if not isinstance(transfer_ids, tuple) or not transfer_ids:
            raise WorkflowExecutionError("NIXL tensor export requires transfer ids")
        if any(
            not isinstance(transfer_id, str) or not transfer_id
            for transfer_id in transfer_ids
        ):
            raise WorkflowExecutionError(
                "NIXL tensor transfer ids must be non-empty strings"
            )
        if len(set(transfer_ids)) != len(transfer_ids):
            raise WorkflowExecutionError("NIXL tensor transfer ids must be unique")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if self._send_pool_capacity:
            return await self._export_from_send_pool(tensor, transfer_ids)
        readables: dict[str, Any] = {}
        try:
            references: dict[str, NixlTensorRef] = {}
            for transfer_id in transfer_ids:
                # A NIXL agent's remote metadata cannot be extended safely while
                # that agent has active transfers. Give every exported edge an
                # immutable agent/registration pair instead of mutating one
                # process-wide agent for each request. A supplied connector is
                # retained for deterministic tests and explicit integrations.
                connector = (
                    self._connector
                    if self._export_connector_factory is None
                    else self._export_connector_factory()
                )
                descriptor = self._nixl.Descriptor(tensor)
                readable = await connector.create_readable(descriptor)
                lease_id = uuid.uuid4().hex
                readables[lease_id] = readable
                references[transfer_id] = NixlTensorRef(
                    transfer_id=transfer_id,
                    lease_id=lease_id,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).rsplit(".", 1)[-1],
                    device=str(tensor.device),
                    rdma_metadata=_model_dump(readable.metadata()),
                )
            self._leases.track_fanout(readables, tensor)
        except BaseException:
            for readable in readables.values():
                readable.__exit__(None, None, None)
            raise
        return {
            transfer_id: reference.to_dict()
            for transfer_id, reference in references.items()
        }

    async def _ensure_send_pool(self, tensor: Any) -> None:
        if self._send_pool_backing is not None:
            if tensor.device != self._send_pool_backing.device:
                raise WorkflowExecutionError(
                    "NIXL send pool cannot mix tensor devices: "
                    f"{tensor.device} != {self._send_pool_backing.device}"
                )
            return
        async with self._send_pool_lock:
            if self._send_pool_backing is not None:
                return
            connection = await self._connector._create_connection()
            backing = self._torch.empty(
                (self._send_pool_capacity, self._send_pool_bytes),
                dtype=self._torch.uint8,
                device=tensor.device,
            )
            descriptors = []
            try:
                for slot in range(self._send_pool_capacity):
                    descriptor = self._nixl.Descriptor(backing[slot])
                    descriptor.register_with_connector(connection)
                    descriptors.append(descriptor)
                    self._send_pool_available.put_nowait(slot)
            except BaseException:
                for descriptor in descriptors:
                    if descriptor.is_registered:
                        descriptor.deregister_with_connector(connection)
                raise
            self._send_pool_connection = connection
            self._send_pool_backing = backing
            self._send_pool_descriptors = descriptors

    async def _export_from_send_pool(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes > self._send_pool_bytes:
            raise WorkflowExecutionError(
                "NIXL tensor exceeds configured send-pool slot: "
                f"{tensor_bytes} > {self._send_pool_bytes} bytes"
            )
        await self._ensure_send_pool(tensor)
        slot = await self._send_pool_available.get()
        readables: dict[str, Any] = {}
        try:
            storage = self._send_pool_backing[slot]
            storage[:tensor_bytes].view(tensor.dtype).view(tensor.shape).copy_(tensor)
            descriptor = self._send_pool_descriptors[slot]
            references: dict[str, NixlTensorRef] = {}
            for transfer_id in transfer_ids:
                readable = await self._connector.create_readable(descriptor)
                lease_id = uuid.uuid4().hex
                readables[lease_id] = readable
                references[transfer_id] = NixlTensorRef(
                    transfer_id=transfer_id,
                    lease_id=lease_id,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).rsplit(".", 1)[-1],
                    device=str(tensor.device),
                    rdma_metadata=_limit_transfer_metadata(
                        readable.metadata(), tensor_bytes
                    ),
                )
            self._leases.track_fanout(
                readables,
                storage,
                on_release=lambda: self._send_pool_available.put_nowait(slot),
            )
        except BaseException:
            for readable in readables.values():
                readable.__exit__(None, None, None)
            self._send_pool_available.put_nowait(slot)
            raise
        return {
            transfer_id: reference.to_dict()
            for transfer_id, reference in references.items()
        }

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        parsed = NixlTensorRef.from_dict(reference)
        dtype = getattr(self._torch, parsed.dtype, None)
        if dtype is None:
            raise WorkflowExecutionError(
                f"unsupported NIXL tensor dtype {parsed.dtype!r}"
            )
        device = self._receive_device or parsed.device
        rdma_metadata = self._nixl.RdmaMetadata.model_validate(
            dict(parsed.rdma_metadata)
        )
        tensor_bytes = (
            math.prod(parsed.shape) * self._torch.empty((), dtype=dtype).element_size()
        )
        remote_descriptors = getattr(rdma_metadata, "descriptors", None)
        if remote_descriptors is None:
            # Test doubles and alternate carrier adapters may expose only their
            # opaque metadata. Exact-size allocation preserves the old path.
            transfer_bytes = tensor_bytes
        else:
            if len(remote_descriptors) != 1:
                raise WorkflowExecutionError(
                    "NIXL tensor carrier requires exactly one remote descriptor"
                )
            transfer_bytes = remote_descriptors[0].size
        if tensor_bytes > transfer_bytes:
            raise WorkflowExecutionError(
                "NIXL tensor shape exceeds the remote transfer buffer"
            )
        storage = self._torch.empty(
            transfer_bytes, dtype=self._torch.uint8, device=device
        )
        descriptor = self._nixl.Descriptor(storage)
        operation = await self._connector.begin_read(rdma_metadata, descriptor)
        try:
            await operation.wait_for_completion()
        finally:
            operation.__exit__(None, None, None)
        return storage[:tensor_bytes].view(dtype).view(parsed.shape)

    async def close(self) -> None:
        await self._leases.close()
        if self._leases.active_count == 0 and self._send_pool_connection is not None:
            for descriptor in self._send_pool_descriptors:
                if descriptor.is_registered:
                    descriptor.deregister_with_connector(self._send_pool_connection)
            self._send_pool_descriptors.clear()
            self._send_pool_backing = None
