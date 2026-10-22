# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw tensor adapter for Custom fields, independent of modality semantics.

Only objects containing the reserved ``$dynamo_tensor`` key are interpreted.
Other JSON passes unchanged. This is not vLLM's processed MultiModalKwargsItem
codec, a media descriptor, or an encoder/disaggregation handoff.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    TypeAdapter,
    ValidationError,
)

from dynamo.common.multimodal.shm_bytes import read_shm_bytes, write_shm_bytes

TENSOR_KEY = "$dynamo_tensor"
MAX_TENSOR_BYTES = 64 * 1024 * 1024
MAX_REQUEST_TENSOR_BYTES = 128 * 1024 * 1024
MAX_REQUEST_TENSORS = 32
MAX_CONCURRENT_RESOLUTIONS = 2

# Process-wide across processors/LoRA models and event loops. Reject excess work
# before submission: ThreadPoolExecutor's own work queue is otherwise unbounded.
_RESOLVER_POOL = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_RESOLUTIONS, thread_name_prefix="custom-tensors"
)
_RESOLVER_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_RESOLUTIONS)


class CustomTensorCapacityError(RuntimeError):
    """All process-wide custom-input preparation slots are occupied."""


DType = Literal[
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int8",
    "uint8",
    "int16",
    "int32",
    "int64",
    "bool",
]


class TensorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    dtype: DType
    shape: list[Annotated[StrictInt, Field(ge=0, le=2**63 - 1)]] = Field(
        max_length=32
    )
    order: Literal["C"]

    def byte_size(self) -> int:
        if sys.byteorder != "little":
            raise ValueError("custom tensor transport requires a little-endian host")
        size = math.prod(self.shape) * getattr(torch, self.dtype).itemsize
        if size > MAX_TENSOR_BYTES:
            raise ValueError("custom tensor exceeds the 64 MiB per-tensor limit")
        return size


class InlineTensor(TensorSpec):
    transport: Literal["inline"]
    data_base64: str


class ShmTensor(TensorSpec):
    transport: Literal["shm"]
    name: str = Field(pattern=r"^dynamo_mm_[a-f0-9]{32}$")
    size: Annotated[StrictInt, Field(ge=0, le=MAX_TENSOR_BYTES)]
    placement: str
    worker_id: Annotated[StrictInt, Field(ge=0, le=2**64 - 1)]
    expires_at: float = Field(allow_inf_nan=False)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


_SPEC = TypeAdapter(
    Annotated[InlineTensor | ShmTensor, Field(discriminator="transport")]
)


def shm_placement() -> str:
    """Identify this boot, IPC namespace and shared-memory mount on Linux."""
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return f"{boot}:{os.stat('/proc/self/ns/ipc').st_ino}:{os.stat('/dev/shm').st_dev}"


def _tensor_bytes(tensor: torch.Tensor) -> tuple[dict[str, Any], bytes]:
    dtype = str(tensor.dtype).removeprefix("torch.")
    spec = TensorSpec(version=1, dtype=dtype, shape=list(tensor.shape), order="C")
    spec.byte_size()
    # GPU -> CPU (when needed), contiguous staging, then one owned byte copy.
    data = (
        tensor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return spec.model_dump(), data


def inline_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Create the supported inline raw-tensor envelope."""
    spec, data = _tensor_bytes(tensor)
    return {
        TENSOR_KEY: {
            **spec,
            "transport": "inline",
            "data_base64": base64.b64encode(data).decode("ascii"),
        }
    }


class ShmTensorPublisher:
    """Producer-owned lease for one tensor, reusable until close or expiry.

    Keep the context open across the HTTP request and any retries. Closing on
    response, cancellation or failure unlinks the name. A reader that already
    opened it can finish its copy; model inputs own independent storage. A daemon
    timer also unlinks abandoned leases. No worker unlinks producer resources.
    """

    def __init__(self, tensor: torch.Tensor, *, worker_id: int, ttl: float = 60):
        if not math.isfinite(ttl) or not 0 < ttl <= 3600:
            raise ValueError("tensor lease TTL must be in (0, 3600] seconds")
        spec, data = _tensor_bytes(tensor)
        # Validate all externally supplied metadata before allocating resources.
        self.spec = ShmTensor(
            **spec,
            transport="shm",
            name="dynamo_mm_" + "0" * 32,
            size=len(data),
            placement=shm_placement(),
            worker_id=worker_id,
            expires_at=time.time() + ttl,
            sha256=hashlib.sha256(data).hexdigest(),
        )
        self._lock = threading.Lock()
        self._segment = write_shm_bytes(data)
        self.spec.name = self._segment.name
        self._closed = False
        self._timer = threading.Timer(ttl, self.close)
        self._timer.daemon = True
        try:
            self._timer.start()
        except BaseException:
            self.close()
            raise

    def envelope(self) -> dict[str, Any]:
        with self._lock:
            if self._closed or time.time() >= self.spec.expires_at:
                raise ValueError("custom tensor reference expired or released")
            return {TENSOR_KEY: self.spec.model_dump()}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._timer.cancel()
                self._segment.close()
                try:
                    self._segment.unlink()
                except FileNotFoundError:
                    # Another owner cleanup (e.g. interpreter shutdown) won.
                    pass
                self._closed = True

    def __enter__(self) -> ShmTensorPublisher:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _plan_custom_tensors(
    value: Any, worker_id: int | None
) -> list[tuple[TensorSpec, int]]:
    """Validate the entire request before any bulk decode, SHM read or allocation.

    Count occurrences, including aliases/repeated references and empty tensors.
    Keep descriptors in traversal order so materialization cannot reset the
    budget for nested containers or separate modalities.
    """
    specs: list[tuple[TensorSpec, int]] = []
    total_bytes = 0

    def visit(item: Any) -> None:
        nonlocal total_bytes
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            if TENSOR_KEY not in item:
                for child in item.values():
                    visit(child)
                return
            if len(item) != 1:
                raise ValueError("tensor envelope cannot have sibling fields")
            if len(specs) >= MAX_REQUEST_TENSORS:
                raise ValueError("custom tensor request exceeds the tensor count limit")
            try:
                spec = _SPEC.validate_python(item[TENSOR_KEY])
            except ValidationError as exc:
                raise ValueError("invalid custom tensor descriptor") from exc
            size = spec.byte_size()
            total_bytes += size
            if total_bytes > MAX_REQUEST_TENSOR_BYTES:
                raise ValueError("custom tensor request exceeds the total byte limit")
            if isinstance(spec, InlineTensor):
                if len(spec.data_base64) != 4 * ((size + 2) // 3):
                    raise ValueError(
                        "inline tensor byte count does not match shape and dtype"
                    )
            else:
                _validate_reference(spec, worker_id)
                if spec.size != size:
                    raise ValueError(
                        "tensor reference size does not match shape and dtype"
                    )
            specs.append((spec, size))

    visit(value)
    return specs


def _validate_reference(spec: ShmTensor, worker_id: int | None) -> None:
    if spec.worker_id != worker_id:
        raise ValueError("shared-memory tensor requires its pinned worker")
    if spec.placement != shm_placement():
        raise ValueError("shared-memory tensor placement mismatch")
    if time.time() >= spec.expires_at:
        raise ValueError("custom tensor reference expired")


def _resolve_tensor(spec: TensorSpec, size: int, worker_id: int | None) -> torch.Tensor:
    if isinstance(spec, InlineTensor):
        try:
            # binascii's decoder holds the GIL. Bound each call and explicitly
            # yield it so off-thread work cannot starve the serving event loop.
            data = bytearray()
            for offset in range(0, len(spec.data_base64), 1024 * 1024):
                chunk = spec.data_base64[offset : offset + 1024 * 1024]
                if offset + len(chunk) < len(spec.data_base64) and "=" in chunk:
                    raise binascii.Error("base64 padding before final block")
                data.extend(base64.b64decode(chunk, validate=True))
                time.sleep(0)
        except binascii.Error as exc:
            raise ValueError("invalid tensor base64") from exc
    else:
        # Earlier tensors may take time to decode; recheck the lease immediately
        # before opening each segment. Producer release during a read is safe:
        # unlink does not invalidate the reader's already-open mapping.
        _validate_reference(spec, worker_id)
        try:
            data = read_shm_bytes(spec.name, spec.size)
        except FileNotFoundError as exc:
            raise ValueError("custom tensor reference expired or released") from exc
        except OSError as exc:
            raise ValueError("custom tensor reference is not accessible") from exc
        if hashlib.sha256(data).hexdigest() != spec.sha256:
            raise ValueError("tensor reference checksum mismatch")
    if len(data) != size:
        raise ValueError("tensor byte count does not match shape and dtype")
    dtype = getattr(torch, spec.dtype)
    # The tensor retains this private writable bytearray, never a SHM view.
    try:
        if size == 0:
            return torch.empty(spec.shape, dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).reshape(spec.shape)
    except RuntimeError as exc:
        raise ValueError("custom tensor shape cannot be represented") from exc


def resolve_custom_tensors(value: Any, *, worker_id: int | None = None) -> Any:
    """Resolve one complete request's custom fields with a shared count/byte budget.

    Callers must not split the request by modality or mutate it during resolution.
    Serving code must use the bounded async entry point below.
    """
    specs = iter(_plan_custom_tensors(value, worker_id))

    def materialize(item: Any) -> Any:
        if isinstance(item, list):
            return [materialize(child) for child in item]
        if isinstance(item, dict):
            if TENSOR_KEY in item:
                spec, size = next(specs)
                return _resolve_tensor(spec, size, worker_id)
            return {key: materialize(child) for key, child in item.items()}
        return item

    # Serial materialization limits transient storage to one tensor in addition
    # to the admitted output budget. No per-tensor fanout or parallel reads.
    return materialize(value)


async def resolve_custom_tensors_async(
    value: Any, *, worker_id: int | None = None
) -> Any:
    """Offload complete request preparation without an unbounded executor queue.

    A cancelled waiter cannot stop a running thread. Its slot stays occupied
    until the concurrent future completes and drops any unconsumed result. No
    asyncio task/callback retains a resolved tensor after caller cancellation.
    """
    if not _RESOLVER_SLOTS.acquire(blocking=False):
        raise CustomTensorCapacityError("custom tensor resolution capacity exhausted")
    try:
        future = _RESOLVER_POOL.submit(
            resolve_custom_tensors, value, worker_id=worker_id
        )
    except BaseException:
        _RESOLVER_SLOTS.release()
        raise
    future.add_done_callback(lambda _: _RESOLVER_SLOTS.release())
    return await asyncio.wrap_future(future)
