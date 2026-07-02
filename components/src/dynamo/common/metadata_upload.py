# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import cache
from typing import Any, cast


def _backend_metadata_upload_settings(request: dict[str, Any]) -> dict[str, Any] | None:
    nvext = request.get("nvext")
    extra_args = request.get("extra_args") or {}
    extra_nvext = extra_args.get("nvext") if isinstance(extra_args, dict) else None

    for source in (nvext, extra_nvext):
        if not isinstance(source, dict):
            continue
        if "metadata_upload" not in source:
            continue
        settings = source["metadata_upload"]
        if settings is None:
            continue
        if not isinstance(settings, dict):
            raise ValueError("metadata_upload must be an object")
        return settings
    return None


async def _upload_bytes(url: str, storage_path: str, data: bytes) -> str:
    try:
        from dynamo.common.storage import get_fs, upload_to_fs
    except ImportError as exc:
        raise RuntimeError(
            "Metadata upload requires fsspec support. "
            "Install fsspec and the backend extra, for example `fsspec[s3]` for S3."
        ) from exc

    return await upload_to_fs(get_fs(url), storage_path, data)


def _serialize_metadata(metadata: dict[str, Any]) -> bytes:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "Metadata upload requires zstandard. "
            "Install ai-dynamo with the selected backend extra or add the zstandard package."
        ) from exc

    try:
        import msgspec
    except ImportError as exc:
        raise RuntimeError("Metadata upload requires msgspec.") from exc

    payload = {
        "schema_version": 1,
        "metadata": _normalize_for_upload(metadata),
    }
    raw = msgspec.msgpack.encode(payload)
    return zstd.ZstdCompressor().compress(raw)


@cache
def _torch_tensor_type() -> type | None:
    try:
        import torch
    except ImportError:
        return None

    return torch.Tensor


@cache
def _numpy_array_type() -> type | None:
    try:
        import numpy as np
    except ImportError:
        return None

    return np.ndarray


def _torch_tensor_to_payload(value: Any) -> dict[str, Any] | None:
    tensor_type = _torch_tensor_type()
    if tensor_type is None or not isinstance(value, tensor_type):
        return None

    import torch

    tensor = cast(Any, value).detach().cpu().contiguous()
    try:
        data = tensor.numpy().tobytes()
    except TypeError:
        data = tensor.view(torch.uint8).numpy().tobytes()

    return {
        "type": "tensor",
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": data,
    }


def _numpy_array_to_payload(value: Any) -> dict[str, Any] | None:
    array_type = _numpy_array_type()
    if array_type is None or not isinstance(value, array_type):
        return None

    import numpy as np

    array = np.ascontiguousarray(value)
    return {
        "type": "ndarray",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tobytes(),
    }


def _normalize_for_upload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, dict):
        return {key: _normalize_for_upload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_upload(item) for item in value]

    tensor_payload = _torch_tensor_to_payload(value)
    if tensor_payload is not None:
        return tensor_payload

    array_payload = _numpy_array_to_payload(value)
    if array_payload is not None:
        return array_payload

    return value


@dataclass(frozen=True)
class MetadataUploader:
    url: str

    def __post_init__(self) -> None:
        if not isinstance(self.url, str):
            raise ValueError("metadata_upload.url must be a string")
        url = self.url.strip()
        if not url:
            raise ValueError("metadata_upload.url must not be empty")
        object.__setattr__(self, "url", url)

    @classmethod
    def from_settings(cls, settings: dict[str, Any] | None) -> MetadataUploader | None:
        if settings is None:
            return None
        unexpected = settings.keys() - {"url"}
        if unexpected:
            field = min(unexpected)
            raise ValueError(f"metadata_upload.{field} is not supported")
        if "url" not in settings:
            raise ValueError("metadata_upload.url is required")
        return cls(url=settings["url"])

    @classmethod
    def from_backend_request(cls, request: dict[str, Any]) -> MetadataUploader | None:
        return cls.from_settings(_backend_metadata_upload_settings(request))

    async def upload_choice(self, choice_index: int, metadata: dict[str, Any]) -> None:
        if not metadata:
            return

        storage_path = f"choice_{choice_index}.msgpack.zst"
        data = await asyncio.to_thread(_serialize_metadata, metadata)
        try:
            await _upload_bytes(self.url, storage_path, data)
        finally:
            del data
