# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded opaque blob transport; acknowledgements are not verified completion."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import math
import re
import secrets
import socket
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal, cast
from urllib.parse import urlsplit

import aiohttp
import opendal
from aiohttp.abc import AbstractResolver
from yarl import URL

if sys.version_info >= (3, 11):
    from asyncio import timeout as _timeout
else:
    from async_timeout import timeout as _timeout

MAX_SINGLE_PUT_BYTES = 64 * 1024 * 1024
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_INVALID_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_INVALID_URL_CHAR = re.compile(r"""[^A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]""")
_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_EMBEDDED_IPV4_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("::/96"),
)
_DENIED_HEADERS = frozenset(
    {
        "authorization",
        "connection",
        "cookie",
        "destination",
        "expect",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "forwarded",
        "x-http-method-override",
        "x-method-override",
        "x-original-url",
        "x-rewrite-url",
    }
)


class ArtifactStorageError(RuntimeError):
    """Credential-safe failure; an ambiguous write requires reconciliation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        outcome: Literal["rejected", "ambiguous"] = "rejected",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.outcome = outcome


def _invalid(message: str) -> ArtifactStorageError:
    return ArtifactStorageError("invalid_target", message)


def _bounded_text(value: str, limit: int, message: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > limit
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise _invalid(message)


def _byte_limit(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= MAX_SINGLE_PUT_BYTES
    ):
        raise _invalid("byte limit is outside the supported single-PUT range")


def _expiry(value: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64 or not _RFC3339.fullmatch(value):
        raise _invalid("expires_at must be an RFC 3339 timestamp")
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, suffix = normalized.split(".", 1)
        fraction = suffix.split("+", 1)[0].split("-", 1)[0]
        normalized = prefix + "." + fraction[:6].ljust(6, "0") + suffix[len(fraction) :]
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise _invalid("expires_at must be an RFC 3339 timestamp") from None
    if parsed.tzinfo is None:
        raise _invalid("expires_at must include a timezone")
    return parsed


def _headers(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping) or len(values) > 32:
        raise _invalid("required headers must be a bounded mapping")
    result: dict[str, str] = {}
    for name, value in values.items():
        if (
            not isinstance(name, str)
            or len(name) > 256
            or not _HEADER_NAME.fullmatch(name)
        ):
            raise _invalid("required header name is invalid")
        normalized = name.lower()
        if normalized in result:
            raise _invalid("duplicate required header name")
        if normalized in _DENIED_HEADERS or normalized.startswith(
            (
                "proxy-",
                "sec-",
                "x-forwarded-",
                "x-amz-copy-source",
                "x-goog-copy-source",
                "x-ms-copy-source",
            )
        ):
            raise _invalid("required header is not allowed")
        _bounded_text(value, 4096, "required header value is invalid")
        if not value.isascii():
            raise _invalid("required header value must be ASCII")
        result[normalized] = value
    if result.get("if-none-match") != "*":
        raise _invalid("presigned target must require If-None-Match: *")
    return MappingProxyType(result)


@dataclass(frozen=True)
class PresignedHttpPutTarget:
    """A caller-issued, exact-object write capability. Never log its fields."""

    url: str = field(repr=False)
    required_headers: Mapping[str, str] = field(repr=False)
    expires_at: str = field(repr=False)
    max_bytes: int
    object_id: str
    verification_ref: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _bounded_text(self.url, 8192, "presigned URL is invalid")
        if (
            not self.url.isascii()
            or self.url != self.url.strip()
            or _INVALID_ESCAPE.search(self.url)
            or _INVALID_URL_CHAR.search(self.url)
        ):
            raise _invalid(
                "presigned URL must preserve its encoded ASCII representation"
            )
        try:
            parsed = urlsplit(self.url)
            valid_port = parsed.port is None or 0 < parsed.port <= 65535
        except ValueError:
            raise _invalid("presigned URL authority is invalid") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "#" in self.url
            or not valid_port
        ):
            raise _invalid("presigned URL authority is invalid")
        _byte_limit(self.max_bytes)
        _bounded_text(self.object_id, 512, "object_id is invalid")
        if self.verification_ref is not None:
            _bounded_text(
                self.verification_ref, 512, "verification reference is invalid"
            )
        _expiry(self.expires_at)
        object.__setattr__(self, "required_headers", _headers(self.required_headers))


def _object_key(value: str) -> None:
    _bounded_text(value, 1024, "object key is invalid")
    if "\\" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        raise _invalid("object key must be normalized and relative")


@dataclass(frozen=True)
class ManagedStoreTarget:
    profile: str
    object_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str) or not _PROFILE_NAME.fullmatch(
            self.profile
        ):
            raise _invalid("managed profile name is invalid")
        _object_key(self.object_key)


@dataclass(frozen=True)
class StoragePolicy:
    """Trusted operator policy. Hosts are exact authorities, including custom ports."""

    allowed_hosts: frozenset[str] = frozenset()
    allowed_networks: tuple[str, ...] = ()
    allow_http_hosts: frozenset[str] = frozenset()
    max_bytes: int = MAX_SINGLE_PUT_BYTES
    timeout_seconds: float = 60
    max_presigned_ttl_seconds: int = 3600

    def __post_init__(self) -> None:
        _byte_limit(self.max_bytes)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise _invalid("storage timeout must be positive and finite")
        if self.max_presigned_ttl_seconds <= 0:
            raise _invalid("presigned lifetime limit must be positive")
        try:
            for network in self.allowed_networks:
                ipaddress.ip_network(network)
        except ValueError:
            raise _invalid("allowed network is invalid") from None
        object.__setattr__(self, "allowed_hosts", frozenset(self.allowed_hosts))
        object.__setattr__(self, "allow_http_hosts", frozenset(self.allow_http_hosts))
        object.__setattr__(self, "allowed_networks", tuple(self.allowed_networks))

    def check_address(self, value: str) -> None:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            raise _invalid("destination address is invalid") from None
        if isinstance(address, ipaddress.IPv6Address):
            if address.ipv4_mapped:
                address = address.ipv4_mapped
            elif not address.is_loopback and any(
                address in prefix for prefix in _EMBEDDED_IPV4_PREFIXES
            ):
                address = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        forbidden = (
            address.is_link_local or address.is_multicast or address.is_unspecified
        )
        explicitly_allowed = any(
            address in ipaddress.ip_network(item) for item in self.allowed_networks
        )
        if forbidden or not (address.is_global or explicitly_allowed):
            raise ArtifactStorageError(
                "destination_denied", "destination address is not allowed"
            )


@dataclass(frozen=True)
class ManagedProfile:
    """Trusted, pre-authorized OpenDAL operator; never constructed from a request."""

    operator: opendal.AsyncOperator = field(repr=False)
    allowed_prefixes: tuple[str, ...]
    max_bytes: int = MAX_SINGLE_PUT_BYTES

    def __post_init__(self) -> None:
        _byte_limit(self.max_bytes)
        if not self.allowed_prefixes:
            raise _invalid("managed profile requires allowed prefixes")
        for prefix in self.allowed_prefixes:
            _object_key(prefix)
        object.__setattr__(self, "allowed_prefixes", tuple(self.allowed_prefixes))


@dataclass(frozen=True)
class PutReceipt:
    """Local digest and transport acknowledgement, never a verified ready result."""

    actual_bytes: int
    sha256: str
    object_id: str
    provider_identity: str | None = None
    outcome: Literal["acknowledged"] = "acknowledged"


class _PolicyResolver(AbstractResolver):
    def __init__(self, policy: StoragePolicy) -> None:
        self.policy = policy
        self.delegate = aiohttp.ThreadedResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        records = await self.delegate.resolve(
            host, port, cast(socket.AddressFamily, family)
        )
        if not records:
            raise ArtifactStorageError(
                "destination_denied", "destination has no allowed address"
            )
        for record in records:
            self.policy.check_address(record["host"])
        return records

    async def close(self) -> None:
        await self.delegate.close()


def validate_presigned_target(
    target: PresignedHttpPutTarget, policy: StoragePolicy
) -> float:
    """Validate the grant and destination policy before inference allocates data."""
    parsed = urlsplit(target.url)
    host = parsed.hostname or ""
    default_port = 443 if parsed.scheme == "https" else 80
    authority_host = f"[{host}]" if ":" in host else host
    authority = (
        authority_host
        if parsed.port in (None, default_port)
        else f"{authority_host}:{parsed.port}"
    )
    if authority not in policy.allowed_hosts:
        raise ArtifactStorageError(
            "destination_denied", "presigned host is not allowed"
        )
    if parsed.scheme != "https" and authority not in policy.allow_http_hosts:
        raise ArtifactStorageError("destination_denied", "presigned URL must use HTTPS")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        policy.check_address(str(address))
    remaining = (
        _expiry(target.expires_at) - datetime.now(timezone.utc)
    ).total_seconds()
    if remaining <= 0:
        raise ArtifactStorageError("expired", "presigned target has expired")
    if remaining > policy.max_presigned_ttl_seconds:
        raise _invalid("presigned lifetime exceeds operator limit")
    return min(policy.timeout_seconds, remaining)


def _validate_presigned(
    data: bytes, target: PresignedHttpPutTarget, policy: StoragePolicy
) -> float:
    timeout = validate_presigned_target(target, policy)
    length = target.required_headers.get("content-length")
    if length is not None and (
        len(length) > 20 or not length.isdigit() or int(length) != len(data)
    ):
        raise _invalid("signed content length does not match the exact bytes")
    if len(data) > target.max_bytes:
        raise ArtifactStorageError("too_large", "blob exceeds target byte limit")
    return timeout


async def _put_presigned(
    data: bytes, target: PresignedHttpPutTarget, policy: StoragePolicy
) -> str | None:
    timeout = _validate_presigned(data, target, policy)
    resolver = _PolicyResolver(policy)
    try:
        connector = aiohttp.TCPConnector(
            resolver=resolver, use_dns_cache=False, limit=1
        )
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=timeout, connect=min(timeout, 10)),
            trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),
            auto_decompress=False,
            skip_auto_headers={"User-Agent", "Accept-Encoding", "Content-Type"},
        ) as session:
            headers = {"content-length": str(len(data)), **target.required_headers}
            async with session.put(
                URL(target.url, encoded=True),
                data=data,
                headers=headers,
                allow_redirects=False,
            ) as response:
                if not 200 <= response.status < 300:
                    ambiguous = response.status >= 500 or response.status in {409, 412}
                    raise ArtifactStorageError(
                        "commit_unknown" if ambiguous else "put_rejected",
                        "presigned PUT requires reconciliation"
                        if ambiguous
                        else "presigned PUT was rejected",
                        outcome="ambiguous" if ambiguous else "rejected",
                    )
                return response.headers.get("x-amz-version-id")
    except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError, OSError):
        raise ArtifactStorageError(
            "commit_unknown", "presigned PUT outcome is unknown", outcome="ambiguous"
        ) from None
    finally:
        await resolver.close()


def validate_managed_target(
    target: ManagedStoreTarget, profiles: Mapping[str, ManagedProfile]
) -> ManagedProfile:
    """Validate authorization and conditional-write support before inference."""
    profile = profiles.get(target.profile)
    if profile is None:
        raise ArtifactStorageError(
            "destination_denied", "managed profile is not authorized"
        )
    if not any(
        target.object_key == prefix or target.object_key.startswith(prefix + "/")
        for prefix in profile.allowed_prefixes
    ):
        raise ArtifactStorageError(
            "destination_denied", "object key is outside the allowed prefix"
        )
    if not profile.operator.capability().write_with_if_not_exists:
        raise ArtifactStorageError("unsupported", "provider lacks conditional creation")
    return profile


async def _put_managed(
    data: bytes,
    target: ManagedStoreTarget,
    policy: StoragePolicy,
    profiles: Mapping[str, ManagedProfile],
) -> None:
    profile = validate_managed_target(target, profiles)
    if len(data) > profile.max_bytes:
        raise ArtifactStorageError("too_large", "blob exceeds profile byte limit")
    try:
        async with _timeout(policy.timeout_seconds):
            await profile.operator.write(target.object_key, data, if_not_exists=True)
    except (
        opendal.exceptions.PermissionDenied,
        opendal.exceptions.Unsupported,
        opendal.exceptions.ConfigInvalid,
    ):
        raise ArtifactStorageError("put_rejected", "managed PUT was rejected") from None
    except (
        opendal.exceptions.Error,
        opendal.exceptions.AlreadyExists,
        opendal.exceptions.ConditionNotMatch,
        opendal.exceptions.Conflict,
        opendal.exceptions.IsADirectory,
        opendal.exceptions.IsSameFile,
        opendal.exceptions.NotADirectory,
        opendal.exceptions.NotFound,
        opendal.exceptions.RangeNotSatisfied,
        opendal.exceptions.RateLimited,
        opendal.exceptions.Unexpected,
        TimeoutError,
        asyncio.TimeoutError,
    ):
        raise ArtifactStorageError(
            "commit_unknown",
            "managed PUT outcome requires reconciliation",
            outcome="ambiguous",
        ) from None


async def put_artifact(
    data: bytes,
    target: PresignedHttpPutTarget | ManagedStoreTarget,
    *,
    policy: StoragePolicy,
    profiles: Mapping[str, ManagedProfile] | None = None,
) -> PutReceipt:
    """Deliver opaque bytes once. The caller owns retry, authorization and verification."""
    if not isinstance(data, bytes):
        raise _invalid("blob source must be immutable bytes")
    if len(data) > policy.max_bytes:
        raise ArtifactStorageError("too_large", "blob exceeds operator byte limit")
    if isinstance(target, PresignedHttpPutTarget):
        provider_identity = await _put_presigned(data, target, policy)
        object_id = target.object_id
    elif isinstance(target, ManagedStoreTarget):
        await _put_managed(data, target, policy, profiles or {})
        provider_identity = None
        object_id = f"obj_{secrets.token_hex(16)}"
    else:
        raise _invalid("unsupported blob target")
    return PutReceipt(
        len(data), hashlib.sha256(data).hexdigest(), object_id, provider_identity
    )
