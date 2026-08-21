# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit NIXL tensor carrier and producer-side lease ownership."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Union

from dynamo.experimental.workflow.perf import WORKFLOW_PERF_TRACE
from dynamo.experimental.workflow.runtime import WorkflowExecutionError

logger = logging.getLogger(__name__)

NIXL_TENSOR_SCHEMA = "dynamo.experimental.workflow.nixl_tensor"
NIXL_TENSOR_FANOUT_SCHEMA = "dynamo.experimental.workflow.nixl_tensor_fanout"
NIXL_TENSOR_VERSION = 0
DEFAULT_NIXL_LEASE_TIMEOUT_S = 60.0
_NIXL_PROGRESS_THREAD_ENV = "DYN_NIXL_PROGRESS_THREAD"


def _progress_thread_from_environment() -> bool:
    value = os.environ.get(_NIXL_PROGRESS_THREAD_ENV, "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{_NIXL_PROGRESS_THREAD_ENV} must be a boolean value, got {value!r}"
    )


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
class EmbeddingTransferRef:
    """Existing multimodal ``TransferRequest`` carried on one workflow edge."""

    shape: tuple[int, ...]
    dtype: str
    serialized_request: Any
    transfer_id: Optional[str] = None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in shape
        ):
            raise WorkflowExecutionError(
                "embedding transfer shape must contain non-negative integers"
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise WorkflowExecutionError(
                "embedding transfer dtype must be a non-empty string"
            )
        if self.transfer_id is not None and (
            not isinstance(self.transfer_id, str) or not self.transfer_id
        ):
            raise WorkflowExecutionError(
                "embedding transfer id must be a non-empty string when set"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", self.dtype.removeprefix("torch."))

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings_shape": list(self.shape),
            "embedding_dtype_str": self.dtype,
            "serialized_request": self.serialized_request,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        transfer_id: Optional[str] = None,
    ) -> "EmbeddingTransferRef":
        if not isinstance(data, Mapping):
            raise WorkflowExecutionError(
                "embedding transfer reference must be an object"
            )
        _check_keys(
            data,
            {
                "embeddings_shape",
                "embedding_dtype_str",
                "serialized_request",
            },
        )
        shape = data["embeddings_shape"]
        if not isinstance(shape, list):
            raise WorkflowExecutionError("embedding transfer shape must be an array")
        return cls(
            shape=tuple(shape),
            dtype=data["embedding_dtype_str"],
            serialized_request=data["serialized_request"],
            transfer_id=transfer_id,
        )


TensorTransferRef = Union[NixlTensorRef, EmbeddingTransferRef]


def tensor_transfer_ref_from_dict(
    data: Mapping[str, Any],
    *,
    transfer_id: Optional[str] = None,
) -> TensorTransferRef:
    """Parse either the legacy READ reference or a multimodal transfer request."""

    if isinstance(data, Mapping) and data.get("schema") == NIXL_TENSOR_SCHEMA:
        reference = NixlTensorRef.from_dict(data)
        if transfer_id is not None and reference.transfer_id != transfer_id:
            raise WorkflowExecutionError(
                "NIXL tensor fanout key does not match transfer id"
            )
        return reference
    return EmbeddingTransferRef.from_dict(data, transfer_id=transfer_id)


@dataclass(frozen=True)
class NixlTensorFanout:
    """Per-consumer NIXL references for one logical tensor output."""

    transfers: Mapping[str, TensorTransferRef]

    def __post_init__(self) -> None:
        if not isinstance(self.transfers, Mapping) or not self.transfers:
            raise WorkflowExecutionError("NIXL tensor fanout requires transfers")
        transfers: dict[str, TensorTransferRef] = {}
        for transfer_id, reference in sorted(self.transfers.items()):
            if not isinstance(reference, (NixlTensorRef, EmbeddingTransferRef)):
                raise WorkflowExecutionError(
                    "NIXL tensor fanout values must use a tensor transfer reference"
                )
            if (
                reference.transfer_id is not None
                and transfer_id != reference.transfer_id
            ):
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
                transfer_id: tensor_transfer_ref_from_dict(
                    reference, transfer_id=transfer_id
                )
                for transfer_id, reference in transfers.items()
            }
        )

    def for_transfer(self, transfer_id: str) -> TensorTransferRef:
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
        tracked_ns = time.perf_counter_ns()
        task = asyncio.create_task(
            self._wait_and_release(
                lease_ids,
                operation_values,
                on_release,
                tracked_ns,
            ),
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
        tracked_ns: int,
    ) -> None:
        async def wait_for_one(lease_id: str, operation: Any) -> None:
            started_ns = time.perf_counter_ns()
            try:
                await operation.wait_for_completion()
            except BaseException:
                WORKFLOW_PERF_TRACE.emit(
                    logger,
                    "nixl.producer_wait",
                    lease_id,
                    status="error",
                    wait_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
                )
                raise
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.producer_wait",
                lease_id,
                status="complete",
                wait_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            )

        async def wait_for_all() -> None:
            results = await asyncio.gather(
                *(
                    wait_for_one(lease_id, operation)
                    for lease_id, operation in zip(lease_ids, operations)
                ),
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
        residence_ms = (time.perf_counter_ns() - tracked_ns) / 1_000_000
        for lease_id in lease_ids:
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.lease_release",
                lease_id,
                active_leases=self.active_count,
                fanout=len(lease_ids),
                residence_ms=residence_ms,
            )

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
        enable_progress_thread: Optional[bool] = None,
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
        if enable_progress_thread is None:
            enable_progress_thread = _progress_thread_from_environment()
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
        self._enable_progress_thread = enable_progress_thread
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

    def can_export(self, value: Any) -> bool:
        """Whether ``value`` is a complete top-level tensor port value."""

        return isinstance(value, self._torch.Tensor)

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Mapping[str, Any]:
        references = await self.export_tensor_fanout(tensor, (transfer_id,))
        return references[transfer_id]

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Export one shared tensor through one independently addressed read per consumer."""

        if not self.can_export(tensor):
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
        started_ns = time.perf_counter_ns()
        lease_ids = {transfer_id: uuid.uuid4().hex for transfer_id in transfer_ids}
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
                lease_id = lease_ids[transfer_id]
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
        wire_references = {
            transfer_id: reference.to_dict()
            for transfer_id, reference in references.items()
        }
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        tensor_bytes = tensor.numel() * tensor.element_size()
        for lease_id in lease_ids.values():
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.export",
                lease_id,
                active_leases=self.active_leases,
                bytes=tensor_bytes,
                elapsed_ms=elapsed_ms,
                fanout=len(transfer_ids),
                pooled=False,
                progress_thread=self._enable_progress_thread,
            )
        return wire_references

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
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.pool_init",
                "send-pool",
                force=True,
                bytes_per_slot=self._send_pool_bytes,
                capacity=self._send_pool_capacity,
                device=str(tensor.device),
                progress_thread=self._enable_progress_thread,
                total_bytes=self._send_pool_capacity * self._send_pool_bytes,
            )

    async def _export_from_send_pool(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        started_ns = time.perf_counter_ns()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes > self._send_pool_bytes:
            raise WorkflowExecutionError(
                "NIXL tensor exceeds configured send-pool slot: "
                f"{tensor_bytes} > {self._send_pool_bytes} bytes"
            )
        await self._ensure_send_pool(tensor)
        lease_ids = {transfer_id: uuid.uuid4().hex for transfer_id in transfer_ids}
        pool_wait_started_ns = time.perf_counter_ns()
        slot = await self._send_pool_available.get()
        pool_wait_ms = (time.perf_counter_ns() - pool_wait_started_ns) / 1_000_000
        available_slots = self._send_pool_available.qsize()
        readables: dict[str, Any] = {}
        try:
            copy_started_ns = time.perf_counter_ns()
            storage = self._send_pool_backing[slot]
            storage[:tensor_bytes].view(tensor.dtype).view(tensor.shape).copy_(tensor)
            copy_ms = (time.perf_counter_ns() - copy_started_ns) / 1_000_000
            descriptor = self._send_pool_descriptors[slot]
            references: dict[str, NixlTensorRef] = {}
            readable_started_ns = time.perf_counter_ns()
            for transfer_id in transfer_ids:
                readable = await self._connector.create_readable(descriptor)
                lease_id = lease_ids[transfer_id]
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
            readable_ms = (time.perf_counter_ns() - readable_started_ns) / 1_000_000
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
        wire_references = {
            transfer_id: reference.to_dict()
            for transfer_id, reference in references.items()
        }
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        for lease_id in lease_ids.values():
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.export",
                lease_id,
                active_leases=self.active_leases,
                available_slots=available_slots,
                bytes=tensor_bytes,
                copy_ms=copy_ms,
                elapsed_ms=elapsed_ms,
                fanout=len(transfer_ids),
                pool_capacity=self._send_pool_capacity,
                pool_wait_ms=pool_wait_ms,
                pooled=True,
                progress_thread=self._enable_progress_thread,
                readable_ms=readable_ms,
            )
        return wire_references

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        started_ns = time.perf_counter_ns()
        parsed = NixlTensorRef.from_dict(reference)
        parse_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        dtype = getattr(self._torch, parsed.dtype, None)
        if dtype is None:
            raise WorkflowExecutionError(
                f"unsupported NIXL tensor dtype {parsed.dtype!r}"
            )
        device = self._receive_device or parsed.device
        metadata_started_ns = time.perf_counter_ns()
        rdma_metadata = self._nixl.RdmaMetadata.model_validate(
            dict(parsed.rdma_metadata)
        )
        metadata_ms = (time.perf_counter_ns() - metadata_started_ns) / 1_000_000
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
        allocation_started_ns = time.perf_counter_ns()
        storage = self._torch.empty(
            transfer_bytes, dtype=self._torch.uint8, device=device
        )
        descriptor = self._nixl.Descriptor(storage)
        allocation_ms = (time.perf_counter_ns() - allocation_started_ns) / 1_000_000
        begin_read_started_ns = time.perf_counter_ns()
        operation = await self._connector.begin_read(rdma_metadata, descriptor)
        begin_read_ms = (time.perf_counter_ns() - begin_read_started_ns) / 1_000_000
        wait_started_ns = time.perf_counter_ns()
        try:
            await operation.wait_for_completion()
        finally:
            operation.__exit__(None, None, None)
        wait_ms = (time.perf_counter_ns() - wait_started_ns) / 1_000_000
        result = storage[:tensor_bytes].view(dtype).view(parsed.shape)
        WORKFLOW_PERF_TRACE.emit(
            logger,
            "nixl.import",
            parsed.lease_id,
            allocation_ms=allocation_ms,
            begin_read_ms=begin_read_ms,
            bytes=tensor_bytes,
            device=str(device),
            elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            metadata_ms=metadata_ms,
            parse_ms=parse_ms,
            progress_thread=self._enable_progress_thread,
            transfer_bytes=transfer_bytes,
            wait_ms=wait_ms,
        )
        return result

    async def close(self) -> None:
        await self._leases.close()
        if self._leases.active_count == 0 and self._send_pool_connection is not None:
            for descriptor in self._send_pool_descriptors:
                if descriptor.is_registered:
                    descriptor.deregister_with_connector(self._send_pool_connection)
            self._send_pool_descriptors.clear()
            self._send_pool_backing = None
