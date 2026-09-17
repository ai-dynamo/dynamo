# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-object fsspec writers for generation artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, cast
from urllib.parse import urlsplit

import aiohttp
from fsspec.implementations.http import HTTPFileSystem

_HTTP_CONNECT_TIMEOUT_SECONDS = 10
_HTTP_TOTAL_TIMEOUT_SECONDS = 60
_DEFAULT_MAX_PRESIGNED_TTL_SECONDS = 3600
_MAX_URL_BYTES = 8192
_MAX_OBJECT_ID_BYTES = 512
_MAX_HEADER_COUNT = 16
_MAX_HEADER_NAME_BYTES = 256
_MAX_HEADER_VALUE_BYTES = 4096
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
_DENIED_HEADERS = frozenset(
    {
        "authorization",
        "connection",
        "content-length",
        "cookie",
        "destination",
        "expect",
        "forwarded",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-forwarded-uri",
        "x-http-method-override",
        "x-method-override",
        "x-original-url",
        "x-rewrite-url",
    }
)


class ArtifactStorageError(RuntimeError):
    """Safe, provider-independent object delivery failure."""


class _ExactHttpPutFileSystem(HTTPFileSystem):
    async def _pipe_file(self, path, value, mode="overwrite", **kwargs):
        del mode
        url = self._strip_protocol(path)
        headers = dict(kwargs.pop("headers", {}))
        headers["Content-Length"] = str(len(value))
        session = await self.set_session()
        async with session.put(url, data=value, headers=headers, **kwargs) as response:
            if not 200 <= response.status < 300:
                raise ArtifactStorageError("presigned artifact PUT was not accepted")

    async def aclose(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()


@dataclass(frozen=True)
class PresignedHttpPutTarget:
    url: str = field(repr=False)
    max_bytes: int
    object_id: str
    required_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    expires_at: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        url = self.url.strip() if isinstance(self.url, str) else ""
        object_id = self.object_id.strip() if isinstance(self.object_id, str) else ""
        if not url or not object_id:
            raise ArtifactStorageError(
                "presigned target URL and object_id are required"
            )
        parsed = urlsplit(url)
        insecure_test_target = _insecure_http_allowed(parsed)
        if parsed.scheme != "https" and not insecure_test_target:
            raise ArtifactStorageError("presigned artifact URLs must use HTTPS")
        if (
            not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ArtifactStorageError("presigned artifact URL authority is invalid")
        if parsed.fragment or _INVALID_PERCENT_ESCAPE.search(url):
            raise ArtifactStorageError("presigned artifact URL is invalid")
        if len(url.encode()) > _MAX_URL_BYTES or _has_control_characters(url):
            raise ArtifactStorageError("presigned artifact URL is too large or invalid")
        if not insecure_test_target and not _presigned_host_allowed(parsed):
            raise ArtifactStorageError("presigned artifact host is not allowlisted")
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes <= 0
        ):
            raise ArtifactStorageError("presigned target max_bytes must be positive")
        if not isinstance(self.required_headers, Mapping):
            raise ArtifactStorageError("presigned target headers must be an object")
        if len(self.required_headers) > _MAX_HEADER_COUNT:
            raise ArtifactStorageError("presigned target has too many headers")
        normalized_headers: dict[str, str] = {}
        for name, value in self.required_headers.items():
            if not isinstance(name, str):
                raise ArtifactStorageError("presigned target header name is invalid")
            normalized = name.lower().strip()
            if (
                len(normalized.encode()) > _MAX_HEADER_NAME_BYTES
                or not _HEADER_NAME.fullmatch(normalized)
                or normalized in _DENIED_HEADERS
                or normalized.startswith(
                    (
                        "proxy-",
                        "sec-",
                        "x-amz-copy-source",
                        "x-goog-copy-source",
                        "x-ms-copy-source",
                        "x-forwarded-",
                    )
                )
            ):
                raise ArtifactStorageError("presigned target header is not allowed")
            if (
                not isinstance(value, str)
                or len(value.encode()) > _MAX_HEADER_VALUE_BYTES
                or _has_control_characters(value)
            ):
                raise ArtifactStorageError("presigned target header value is invalid")
            normalized_headers[normalized] = value
        if not isinstance(self.expires_at, str):
            raise ArtifactStorageError("expires_at must be an RFC 3339 timestamp")
        try:
            expires = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ArtifactStorageError(
                "expires_at must be an RFC 3339 timestamp"
            ) from exc
        now = datetime.now(timezone.utc)
        try:
            max_ttl = int(
                os.environ.get(
                    "DYN_GENERATION_ARTIFACT_MAX_PRESIGNED_TTL_SECONDS",
                    str(_DEFAULT_MAX_PRESIGNED_TTL_SECONDS),
                )
            )
        except ValueError as exc:
            raise ArtifactStorageError(
                "presigned target lifetime limit is invalid"
            ) from exc
        if expires.tzinfo is None or expires <= now:
            raise ArtifactStorageError("presigned target has expired")
        if max_ttl <= 0 or (expires - now).total_seconds() > max_ttl:
            raise ArtifactStorageError(
                "presigned target lifetime exceeds operator limit"
            )
        if normalized_headers.get("if-none-match") != "*":
            raise ArtifactStorageError("presigned target must require If-None-Match: *")
        object.__setattr__(self, "url", url)
        if len(object_id.encode()) > _MAX_OBJECT_ID_BYTES or _has_control_characters(
            object_id
        ):
            raise ArtifactStorageError("presigned target object_id is invalid")
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(
            self, "required_headers", MappingProxyType(normalized_headers)
        )

    def __repr__(self) -> str:
        return (
            "PresignedHttpPutTarget(url=<redacted>, max_bytes="
            f"{self.max_bytes}, object_id={self.object_id!r}, "
            "required_headers=<redacted>, expires_at=<redacted>)"
        )


ArtifactTarget = PresignedHttpPutTarget


@dataclass(frozen=True)
class ArtifactReceipt:
    actual_bytes: int
    sha256: str
    object_id: str
    provider_identity: str | None = None


def _insecure_http_allowed(parsed) -> bool:
    enabled = os.environ.get("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "").lower()
    allowed_authorities = {
        value.strip().lower()
        for value in os.environ.get(
            "DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", ""
        ).split(",")
        if value.strip()
    }
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError:
        return False
    authority = host if port in (None, 80) else f"{host}:{port}"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        loopback = host == "localhost"
    else:
        loopback = address.is_loopback
    return (
        enabled in {"1", "true", "yes"}
        and parsed.scheme == "http"
        and loopback
        and authority in allowed_authorities
    )


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _presigned_host_allowed(parsed) -> bool:
    host = (parsed.hostname or "").lower()
    if host in {"localhost"} or host.endswith((".localhost", ".local")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            return False
    try:
        port = parsed.port
    except ValueError:
        return False
    authority = host if port in (None, 443) else f"{host}:{port}"
    allowed = {
        item.strip().lower()
        for item in os.environ.get("DYN_GENERATION_ARTIFACT_PRESIGNED_HOSTS", "").split(
            ","
        )
        if item.strip()
    }
    return authority in allowed


def target_from_settings(settings: Mapping[str, Any]) -> ArtifactTarget:
    delivery = settings.get("delivery")
    if not isinstance(delivery, dict) or delivery.get("mode") != "object_store":
        raise ArtifactStorageError("generation artifact delivery must use object_store")
    target = delivery.get("target")
    if not isinstance(target, dict):
        raise ArtifactStorageError("generation artifact target is required")
    kind = target.get("kind")
    if kind == "presigned_http_put":
        unexpected = set(target) - {
            "kind",
            "url",
            "expires_at",
            "max_bytes",
            "required_headers",
            "object_id",
        }
        if unexpected:
            raise ArtifactStorageError("presigned target has unsupported fields")
        url = target.get("url")
        expires_at = target.get("expires_at")
        max_bytes = target.get("max_bytes")
        object_id = target.get("object_id")
        required_headers = target.get("required_headers") or {}
        if (
            not isinstance(url, str)
            or not isinstance(max_bytes, int)
            or not isinstance(object_id, str)
        ):
            raise ArtifactStorageError("presigned artifact target is invalid")
        return PresignedHttpPutTarget(
            url=url,
            expires_at=expires_at,
            max_bytes=max_bytes,
            required_headers=cast(Mapping[str, str], required_headers),
            object_id=object_id,
        )
    raise ArtifactStorageError("generation artifact target kind is unsupported")


async def _put_presigned(data: bytes, target: PresignedHttpPutTarget) -> None:
    if len(data) > target.max_bytes:
        raise ArtifactStorageError("artifact exceeds presigned target max_bytes")
    expires_at = target.expires_at
    if not isinstance(expires_at, str):
        raise ArtifactStorageError("expires_at must be an RFC 3339 timestamp")
    expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    if expires <= datetime.now(timezone.utc):
        raise ArtifactStorageError("presigned target has expired")
    headers = dict(target.required_headers)
    filesystem = _ExactHttpPutFileSystem(
        asynchronous=True,
        encoded=True,
        client_kwargs={
            "timeout": aiohttp.ClientTimeout(
                connect=_HTTP_CONNECT_TIMEOUT_SECONDS,
                total=_HTTP_TOTAL_TIMEOUT_SECONDS,
            )
        },
    )
    operation_error: Exception | asyncio.CancelledError | None = None
    try:
        await filesystem._pipe_file(
            target.url, data, headers=headers, allow_redirects=False
        )
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
        operation_error = exc
    try:
        await filesystem.aclose()
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001
        if operation_error is None:
            operation_error = exc
    if operation_error is not None:
        if isinstance(operation_error, (ArtifactStorageError, asyncio.CancelledError)):
            raise operation_error
        raise ArtifactStorageError("presigned artifact PUT failed") from None


async def put_artifact(data: bytes, target: PresignedHttpPutTarget) -> ArtifactReceipt:
    """Write immutable bytes to exactly one authorized object destination."""
    await _put_presigned(data, target)
    return ArtifactReceipt(
        actual_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        object_id=target.object_id,
    )
