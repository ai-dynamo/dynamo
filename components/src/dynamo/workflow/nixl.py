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
from typing import Any, Mapping, Optional

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


class NixlLeaseRegistry:
    """Keep producer memory registered until read completion or bounded expiry."""

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

    def track_fanout(self, operations: Mapping[str, Any], value: Any) -> None:
        """Keep one shared tensor alive until every consumer read is terminal."""

        if not isinstance(operations, Mapping) or not operations:
            raise WorkflowExecutionError("NIXL fanout lease requires operations")
        lease_ids = tuple(operations)
        duplicate = next(
            (lease_id for lease_id in lease_ids if lease_id in self._leases), None
        )
        if duplicate is not None:
            raise WorkflowExecutionError(f"duplicate NIXL lease {duplicate!r}")
        operation_values = tuple(operations.values())
        task = asyncio.create_task(
            self._wait_and_release(lease_ids, operation_values),
            name=f"workflow-nixl-lease:{lease_ids[0]}",
        )
        lease = _Lease(operation_values, value, task)
        for lease_id in lease_ids:
            self._leases[lease_id] = lease

    async def _wait_and_release(
        self, lease_ids: tuple[str, ...], operations: tuple[Any, ...]
    ) -> None:
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *(operation.wait_for_completion() for operation in operations)
                ),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "NIXL workflow leases %s were not all read within %.1fs; releasing",
                lease_ids,
                self._timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "NIXL workflow leases %s failed before completion",
                lease_ids,
                exc_info=True,
            )
        finally:
            for lease_id in lease_ids:
                self._leases.pop(lease_id, None)
            for lease_id, operation in zip(lease_ids, operations):
                try:
                    operation.__exit__(None, None, None)
                except Exception:
                    logger.warning(
                        "failed to release NIXL workflow lease %s",
                        lease_id,
                        exc_info=True,
                    )

    async def close(self) -> None:
        tasks = list(
            {id(lease.task): lease.task for lease in self._leases.values()}.values()
        )
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


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
    ) -> None:
        if nixl_module is None:
            try:
                from dynamo import nixl_connect as nixl_module
            except ImportError as error:
                raise RuntimeError(
                    "NIXL tensor carrier requested but dynamo.nixl_connect is unavailable"
                ) from error
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
        self._nixl = nixl_module
        self._torch = torch_module
        self._connector = connector or nixl_module.Connector()
        self._receive_device = receive_device
        self._leases = NixlLeaseRegistry(lease_timeout_s)

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
        descriptor = self._nixl.Descriptor(tensor)
        readables: dict[str, Any] = {}
        try:
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

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        parsed = NixlTensorRef.from_dict(reference)
        dtype = getattr(self._torch, parsed.dtype, None)
        if dtype is None:
            raise WorkflowExecutionError(
                f"unsupported NIXL tensor dtype {parsed.dtype!r}"
            )
        device = self._receive_device or parsed.device
        tensor = self._torch.empty(parsed.shape, dtype=dtype, device=device)
        descriptor = self._nixl.Descriptor(tensor)
        rdma_metadata = self._nixl.RdmaMetadata.model_validate(
            dict(parsed.rdma_metadata)
        )
        operation = await self._connector.begin_read(rdma_metadata, descriptor)
        try:
            await operation.wait_for_completion()
        finally:
            operation.__exit__(None, None, None)
        return tensor

    async def close(self) -> None:
        await self._leases.close()
