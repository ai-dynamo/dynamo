# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-object fsspec writers for generation artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from fsspec.asyn import sync_wrapper
from fsspec.core import url_to_fs
from fsspec.implementations.http import HTTPFileSystem

_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_HTTP_CONNECT_TIMEOUT_SECONDS = 10
_HTTP_TOTAL_TIMEOUT_SECONDS = 60
_DEFAULT_MAX_PRESIGNED_TTL_SECONDS = 3600
_DEFAULT_MANAGED_TIMEOUT_SECONDS = 60
_MAX_URL_BYTES = 8192
_MAX_OBJECT_ID_BYTES = 512
_MAX_PROFILE_CONFIG_BYTES = 1 << 20
_MAX_HEADER_COUNT = 16
_MAX_HEADER_VALUE_BYTES = 4096
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_ALLOWED_HEADERS = frozenset(
    {
        "content-type",
        "content-md5",
        "if-none-match",
        "x-amz-checksum-sha256",
        "x-amz-server-side-encryption",
        "x-amz-server-side-encryption-aws-kms-key-id",
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
        async with session.put(
            self.encode_url(url), data=value, headers=headers, **kwargs
        ) as response:
            if not 200 <= response.status < 300:
                raise ArtifactStorageError("presigned artifact PUT was not accepted")

    pipe_file = sync_wrapper(_pipe_file)


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
            if normalized not in _ALLOWED_HEADERS:
                raise ArtifactStorageError(
                    f"presigned target header {name!r} is not allowed"
                )
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


@dataclass(frozen=True)
class ManagedFsspecTarget:
    profile: str
    object_key: str

    def __post_init__(self) -> None:
        profile = self.profile.strip() if isinstance(self.profile, str) else ""
        key = self.object_key.strip() if isinstance(self.object_key, str) else ""
        if not profile or not _PROFILE_NAME.fullmatch(profile):
            raise ArtifactStorageError("managed target profile is required")
        _validate_object_key(key)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "object_key", key)


ArtifactTarget = PresignedHttpPutTarget | ManagedFsspecTarget


@dataclass(frozen=True)
class ArtifactReceipt:
    actual_bytes: int
    sha256: str
    object_id: str
    provider_identity: str | None = None


def _insecure_http_allowed(parsed) -> bool:
    enabled = os.environ.get("DYN_GENERATION_ARTIFACT_ALLOW_INSECURE_HTTP", "").lower()
    allowed_hosts = {
        value.strip()
        for value in os.environ.get(
            "DYN_GENERATION_ARTIFACT_INSECURE_HTTP_HOSTS", ""
        ).split(",")
        if value.strip()
    }
    return (
        enabled in {"1", "true", "yes"}
        and parsed.scheme == "http"
        and parsed.hostname in allowed_hosts
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


def _validate_object_key(key: str) -> None:
    if len(key.encode()) > 1024 or _has_control_characters(key):
        raise ArtifactStorageError("managed target object_key is invalid")
    if not key or "\\" in key or key.startswith("/"):
        raise ArtifactStorageError("managed target object_key must be relative")
    parts = key.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactStorageError("managed target object_key is not normalized")
    path = PurePosixPath(key)
    if path.is_absolute() or str(path) != key:
        raise ArtifactStorageError("managed target object_key is invalid")


def _profiles() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("DYN_GENERATION_ARTIFACT_STORAGE_PROFILES", "{}")
    if len(raw.encode()) > _MAX_PROFILE_CONFIG_BYTES:
        raise ArtifactStorageError("generation artifact storage profiles are too large")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ArtifactStorageError(
            "generation artifact storage profiles are invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ArtifactStorageError(
            "generation artifact storage profiles must be an object"
        )
    profiles: dict[str, dict[str, Any]] = {}
    for name, config in value.items():
        if not isinstance(name, str) or not isinstance(config, dict):
            raise ArtifactStorageError("generation artifact storage profile is invalid")
        if set(config) - {"url", "storage_options", "allowed_prefixes", "create_only"}:
            raise ArtifactStorageError(
                "generation artifact storage profile has unknown fields"
            )
        url = config.get("url")
        options = config.get("storage_options", {})
        allowed_prefixes = config.get("allowed_prefixes")
        create_only = config.get("create_only")
        if (
            not isinstance(url, str)
            or not url
            or not isinstance(options, dict)
            or not isinstance(allowed_prefixes, list)
            or not allowed_prefixes
            or any(not isinstance(prefix, str) for prefix in allowed_prefixes)
            or create_only is not True
        ):
            raise ArtifactStorageError("generation artifact storage profile is invalid")
        normalized_prefixes = []
        for prefix in allowed_prefixes:
            normalized = prefix.strip().rstrip("/")
            _validate_object_key(normalized)
            normalized_prefixes.append(normalized)
        profiles[name] = {
            "url": url,
            "storage_options": dict(options),
            "allowed_prefixes": tuple(normalized_prefixes),
        }
    return profiles


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
        return PresignedHttpPutTarget(
            url=target.get("url"),
            expires_at=target.get("expires_at"),
            max_bytes=target.get("max_bytes"),
            required_headers=target.get("required_headers") or {},
            object_id=target.get("object_id"),
        )
    if kind == "managed_fsspec":
        if set(target) != {"kind", "profile", "object_key"}:
            raise ArtifactStorageError("managed target has unsupported fields")
        return ManagedFsspecTarget(
            profile=target.get("profile"), object_key=target.get("object_key")
        )
    raise ArtifactStorageError("generation artifact target kind is unsupported")


async def _put_presigned(data: bytes, target: PresignedHttpPutTarget) -> None:
    if len(data) > target.max_bytes:
        raise ArtifactStorageError("artifact exceeds presigned target max_bytes")
    expires = datetime.fromisoformat(target.expires_at.replace("Z", "+00:00"))
    if expires <= datetime.now(timezone.utc):
        raise ArtifactStorageError("presigned target has expired")
    headers = dict(target.required_headers)
    filesystem = _ExactHttpPutFileSystem(
        encoded=True,
        client_kwargs={
            "timeout": aiohttp.ClientTimeout(
                connect=_HTTP_CONNECT_TIMEOUT_SECONDS,
                total=_HTTP_TOTAL_TIMEOUT_SECONDS,
            )
        },
    )
    try:
        await asyncio.to_thread(
            filesystem.pipe_file,
            target.url,
            data,
            headers=headers,
            allow_redirects=False,
        )
    except ArtifactStorageError:
        raise
    except Exception:  # noqa: BLE001 - provider exceptions are not standardized
        raise ArtifactStorageError("presigned artifact PUT failed") from None


async def _put_managed(data: bytes, target: ManagedFsspecTarget) -> None:
    enabled = os.environ.get(
        "DYN_GENERATION_ARTIFACT_ENABLE_MANAGED_FSSPEC", ""
    ).lower()
    if enabled not in {"1", "true", "yes"}:
        raise ArtifactStorageError("managed fsspec artifact delivery is not enabled")
    try:
        max_bytes = int(
            os.environ.get("DYN_GENERATION_ARTIFACT_MAX_BYTES", str(_DEFAULT_MAX_BYTES))
        )
    except ValueError as exc:
        raise ArtifactStorageError("generation artifact byte limit is invalid") from exc
    if len(data) > max_bytes:
        raise ArtifactStorageError("artifact exceeds managed storage byte limit")
    profile = _profiles().get(target.profile)
    if profile is None:
        raise ArtifactStorageError("generation artifact storage profile is unknown")
    if not any(
        target.object_key == prefix or target.object_key.startswith(f"{prefix}/")
        for prefix in profile["allowed_prefixes"]
    ):
        raise ArtifactStorageError(
            "managed target object_key is outside the profile prefix"
        )
    try:
        timeout = int(
            os.environ.get(
                "DYN_GENERATION_ARTIFACT_MANAGED_TIMEOUT_SECONDS",
                str(_DEFAULT_MANAGED_TIMEOUT_SECONDS),
            )
        )
    except ValueError as exc:
        raise ArtifactStorageError("managed artifact timeout is invalid") from exc
    if timeout <= 0:
        raise ArtifactStorageError("managed artifact timeout is invalid")
    try:
        storage_options = dict(profile["storage_options"])
        config_kwargs = dict(storage_options.get("config_kwargs") or {})
        config_kwargs.setdefault("connect_timeout", min(timeout, 10))
        config_kwargs.setdefault("read_timeout", timeout)
        config_kwargs.setdefault("retries", {"max_attempts": 2, "mode": "standard"})
        storage_options["config_kwargs"] = config_kwargs
        filesystem, root = url_to_fs(
            profile["url"],
            asynchronous=True,
            skip_instance_cache=True,
            **storage_options,
        )
        protocols = filesystem.protocol
        if isinstance(protocols, str):
            protocols = (protocols,)
        if not filesystem.async_impl or not set(protocols).intersection({"s3", "s3a"}):
            raise ArtifactStorageError(
                "managed artifact profile must use the verified async S3 backend"
            )
        path = "/".join(part for part in (root.rstrip("/"), target.object_key) if part)
        session = await filesystem.set_session()
        try:
            async with asyncio.timeout(timeout):
                await filesystem._pipe_file(path, data, mode="create")
        finally:
            await session.__aexit__(None, None, None)
    except ArtifactStorageError:
        raise
    except Exception:  # noqa: BLE001 - fsspec implementations vary by provider
        raise ArtifactStorageError("managed artifact write failed") from None


async def put_artifact(data: bytes, target: ArtifactTarget) -> ArtifactReceipt:
    """Write immutable bytes to exactly one authorized object destination."""
    if isinstance(target, PresignedHttpPutTarget):
        await _put_presigned(data, target)
        object_id = target.object_id
    else:
        await _put_managed(data, target)
        object_id = f"{target.profile}:{target.object_key}"
    return ArtifactReceipt(
        actual_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        object_id=object_id,
    )
