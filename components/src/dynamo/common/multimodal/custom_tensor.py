# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw tensor adapter for Custom fields, independent of modality semantics.

Only objects containing the reserved ``$dynamo_tensor`` key are interpreted.
Other JSON passes unchanged. This is not vLLM's processed MultiModalKwargsItem
codec, a media descriptor, or an encoder/disaggregation handoff.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import os
import sys
import threading
import time
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


def resolve_custom_tensors(value: Any, *, worker_id: int | None = None) -> Any:
    """Resolve reserved envelopes recursively; retain JSON/item structure.

    worker_id is the strict routing target carried by the routed request; the
    frontend must require equality with every SHM descriptor before dispatch.
    """
    if isinstance(value, list):
        return [resolve_custom_tensors(item, worker_id=worker_id) for item in value]
    if not isinstance(value, dict):
        return value
    if TENSOR_KEY not in value:
        return {
            key: resolve_custom_tensors(item, worker_id=worker_id)
            for key, item in value.items()
        }
    if len(value) != 1:
        raise ValueError("tensor envelope cannot have sibling fields")
    try:
        spec = _SPEC.validate_python(value[TENSOR_KEY])
    except ValidationError as exc:
        raise ValueError("invalid custom tensor descriptor") from exc
    size = spec.byte_size()
    if isinstance(spec, InlineTensor):
        if len(spec.data_base64) != 4 * ((size + 2) // 3):
            raise ValueError("inline tensor byte count does not match shape and dtype")
        try:
            data = bytearray(base64.b64decode(spec.data_base64, validate=True))
        except binascii.Error as exc:
            raise ValueError("invalid tensor base64") from exc
    else:
        if spec.worker_id != worker_id:
            raise ValueError("shared-memory tensor requires its pinned worker")
        if spec.placement != shm_placement():
            raise ValueError("shared-memory tensor placement mismatch")
        if time.time() >= spec.expires_at:
            raise ValueError("custom tensor reference expired")
        if spec.size != size:
            raise ValueError("tensor reference size does not match shape and dtype")
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
    # Tensor owns a reference to this private writable bytearray. No SHM view
    # survives resolution, so producer release cannot invalidate model inputs.
    try:
        if size == 0:
            return torch.empty(spec.shape, dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).reshape(spec.shape)
    except RuntimeError as exc:
        raise ValueError("custom tensor shape cannot be represented") from exc
