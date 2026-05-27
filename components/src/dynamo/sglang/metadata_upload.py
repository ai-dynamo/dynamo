# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dynamo._core import Context

_SAFE_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._=-]+")


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _first_str(raw: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _as_str(raw.get(key))
        if value is not None:
            return value.strip()
    return None


def _request_scopes(request: dict[str, Any]):
    yield request
    nvext = request.get("nvext")
    if isinstance(nvext, dict):
        yield nvext

    extra_args = request.get("extra_args")
    if isinstance(extra_args, dict):
        yield extra_args

        extra_nvext = extra_args.get("nvext")
        if isinstance(extra_nvext, dict):
            yield extra_nvext


def _find_upload_config(request: dict[str, Any]) -> dict[str, Any] | None:
    for scope in _request_scopes(request):
        candidate = scope.get("metadata_upload")
        if isinstance(candidate, dict):
            return candidate
    return None


def _find_request_id(request: dict[str, Any]) -> str | None:
    for scope in _request_scopes(request):
        request_id = _first_str(scope, "request_id", "x_request_id", "id")
        if request_id is not None:
            return request_id
    return None


def _sanitize_path_component(value: str, default: str = "unknown") -> str:
    sanitized = _SAFE_PATH_COMPONENT_RE.sub("_", value.strip())
    sanitized = sanitized.strip("._-/")
    return sanitized[:160] or default


def _join_storage_path(*parts: str | None) -> str:
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)


def _sanitize_storage_prefix(value: str) -> str:
    return "/".join(
        _sanitize_path_component(part)
        for part in value.split("/")
        if part.strip() and part.strip() not in (".", "..")
    )


def _context_request_id(context: "Context") -> str | None:
    try:
        headers = context.trace_headers()
    except Exception:
        headers = None

    if isinstance(headers, dict):
        for key in ("x-request-id", "request-id"):
            value = _as_str(headers.get(key))
            if value is not None:
                return value

    try:
        return _as_str(context.id())
    except Exception:
        return None


async def _upload_bytes(fs_url: str, storage_path: str, data: bytes) -> str:
    try:
        from dynamo.common.storage import get_fs, upload_to_fs
    except ImportError as exc:
        raise RuntimeError(
            "SGLang metadata upload requires fsspec support. "
            "Install fsspec and the backend extra, for example `fsspec[s3]` for S3."
        ) from exc

    return await upload_to_fs(get_fs(fs_url), storage_path, data)


def _serialize_zstd_json(payload: dict[str, Any]) -> bytes:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "SGLang metadata upload requires zstandard. "
            "Install ai-dynamo[sglang] or add the zstandard package."
        ) from exc

    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    try:
        return zstd.ZstdCompressor().compress(raw)
    finally:
        del raw


@dataclass(frozen=True)
class MetadataUploadConfig:
    fs_url: str
    base_path: str = ""
    request_id: str | None = None

    @classmethod
    def from_request(cls, request: dict[str, Any]) -> MetadataUploadConfig | None:
        raw = _find_upload_config(request)
        if raw is None:
            return None
        if raw.get("enabled") is False:
            return None

        fs_url = _first_str(raw, "s3_url", "fs_url", "url")
        if fs_url is None:
            return None

        return cls(
            fs_url=fs_url,
            base_path=_sanitize_storage_prefix(
                _first_str(raw, "s3_path", "path", "prefix") or ""
            ),
            request_id=_first_str(raw, "request_id", "x_request_id", "id")
            or _find_request_id(request),
        )

    def uploader_for_context(self, context: "Context") -> MetadataUploader:
        request_id = self.request_id or _context_request_id(context) or "unknown"
        try:
            context_id = _as_str(context.id())
        except Exception:
            context_id = None
        return MetadataUploader(
            fs_url=self.fs_url,
            base_path=self.base_path,
            request_id=_sanitize_path_component(request_id),
            context_id=context_id,
        )


@dataclass
class ChoiceMetadata:
    choice_index: int
    sglang_request_id: str | None = None
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

    def to_payload(self, request_id: str, context_id: str | None) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.log_probs:
            metadata["log_probs"] = self.log_probs
        if self.top_logprobs:
            metadata["top_logprobs"] = self.top_logprobs
        if self.routed_experts is not None:
            metadata["routed_experts"] = self.routed_experts

        return {
            "schema_version": 1,
            "request_id": request_id,
            "context_id": context_id,
            "choice_index": self.choice_index,
            "sglang_request_id": self.sglang_request_id,
            "metadata": metadata,
        }


@dataclass(frozen=True)
class MetadataUploader:
    fs_url: str
    base_path: str
    request_id: str
    context_id: str | None = None

    def storage_path_for_choice(self, choice_index: int) -> str:
        choice_name = f"choice_{choice_index}.json.zst"
        return _join_storage_path(self.base_path, self.request_id, choice_name)

    async def upload_choice(self, choice: ChoiceMetadata) -> dict[str, Any] | None:
        if not choice.has_payload():
            return None

        storage_path = self.storage_path_for_choice(choice.choice_index)
        payload = choice.to_payload(self.request_id, self.context_id)
        data = await asyncio.to_thread(_serialize_zstd_json, payload)
        try:
            url = await _upload_bytes(self.fs_url, storage_path, data)
        finally:
            del data
            del payload
        return {
            "url": url,
            "path": storage_path,
            "request_id": self.request_id,
            "choice_index": choice.choice_index,
            "compression": "zstd",
        }


def metadata_upload_requested(request: dict[str, Any]) -> bool:
    raw = _find_upload_config(request)
    return (
        raw is not None
        and raw.get("enabled") is not False
        and _first_str(raw, "s3_url", "fs_url", "url") is not None
    )
