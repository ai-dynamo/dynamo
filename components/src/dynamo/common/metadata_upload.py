# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_FORMAT = "msgpack"
_FORMATS = {"json", "msgpack"}


def _backend_metadata_upload_settings(request: dict[str, Any]) -> dict[str, Any] | None:
    nvext = request.get("nvext")
    extra_args = request.get("extra_args") or {}
    extra_nvext = extra_args.get("nvext") if isinstance(extra_args, dict) else None

    for source in (nvext, extra_nvext):
        if not isinstance(source, dict):
            continue
        settings = source.get("metadata_upload")
        if settings:
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


def _serialize_payload(payload: dict[str, Any], payload_format: str) -> bytes:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "Metadata upload requires zstandard. "
            "Install ai-dynamo with the selected backend extra or add the zstandard package."
        ) from exc

    if payload_format == "json":
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    else:
        try:
            import msgspec
        except ImportError as exc:
            raise RuntimeError("Metadata msgpack upload requires msgspec.") from exc
        raw = msgspec.msgpack.encode(payload)

    try:
        return zstd.ZstdCompressor().compress(raw)
    finally:
        del raw


@dataclass
class ChoiceMetadata:
    choice_index: int
    log_probs: list[float] = field(default_factory=list)
    top_logprobs: list[list[dict[str, Any]]] = field(default_factory=list)
    routed_experts: Any = None

    def add_logprobs(
        self,
        log_probs: list[float] | None,
        top_logprobs: list[list[dict[str, Any]]] | None,
    ) -> None:
        if log_probs:
            self.log_probs.extend(log_probs)
        if top_logprobs:
            self.top_logprobs.extend(top_logprobs)

    def has_payload(self) -> bool:
        return bool(
            self.log_probs or self.top_logprobs or self.routed_experts is not None
        )

    def release_payload(self) -> None:
        self.log_probs.clear()
        self.top_logprobs.clear()
        self.routed_experts = None

    def to_payload(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.log_probs:
            metadata["log_probs"] = self.log_probs
        if self.top_logprobs:
            metadata["top_logprobs"] = self.top_logprobs
        if self.routed_experts is not None:
            metadata["routed_experts"] = self.routed_experts

        return {
            "schema_version": 1,
            "metadata": metadata,
        }


@dataclass(frozen=True)
class MetadataUploader:
    url: str
    payload_format: str = _DEFAULT_FORMAT

    def __post_init__(self) -> None:
        url = self.url.strip()
        if not url:
            raise ValueError("metadata_upload.url must not be empty")

        payload_format = self.payload_format.strip().lower()
        if payload_format not in _FORMATS:
            raise ValueError("metadata_upload.format must be one of: json, msgpack")
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "payload_format", payload_format)

    @classmethod
    def from_settings(cls, settings: dict[str, Any] | None) -> MetadataUploader | None:
        if not settings or not settings.get("url"):
            return None
        return cls(
            url=settings["url"],
            payload_format=settings.get("format", _DEFAULT_FORMAT),
        )

    @classmethod
    def from_backend_request(cls, request: dict[str, Any]) -> MetadataUploader | None:
        return cls.from_settings(_backend_metadata_upload_settings(request))

    async def upload_choice(self, choice: ChoiceMetadata) -> None:
        if not choice.has_payload():
            return None

        storage_path = f"choice_{choice.choice_index}.{self.payload_format}.zst"
        payload = choice.to_payload()
        data = await asyncio.to_thread(_serialize_payload, payload, self.payload_format)
        try:
            await _upload_bytes(self.url, storage_path, data)
        finally:
            del data
            del payload
