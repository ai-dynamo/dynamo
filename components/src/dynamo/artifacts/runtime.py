# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator-configured delivery for a single worker and security domain."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import ipaddress
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

from dynamo.artifacts.operations import ArtifactOperationStore, ArtifactService
from dynamo.artifacts.storage import (
    ManagedProfile,
    ManagedStoreTarget,
    PresignedHttpPutTarget,
    StoragePolicy,
    put_artifact,
    validate_presigned_target,
)

_SAFE_URL_KEY = re.compile(r"[A-Za-z0-9._~/-]+\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


class ArtifactDependencyError(RuntimeError):
    """An optional artifact-delivery dependency is unavailable."""


class ArtifactAuthorityClosed(ValueError):
    """The delivery authority is closing and cannot admit new work."""


class ArtifactRuntime:
    """Profiles and verifier registrations come exclusively from trusted config.

    A dedicated database and profiles belong to one worker/security domain.
    This implementation does not provide cross-worker controller takeover.
    """

    def __init__(self, config, capability_key):
        self._authority_file = None
        self._closing = False
        self._admission_lock = asyncio.Lock()
        try:
            self._initialize(config, capability_key)
        except ArtifactDependencyError:
            if self._authority_file is not None:
                self._authority_file.close()
            raise
        except Exception:  # noqa: BLE001 -- sanitize configuration failures
            if self._authority_file is not None:
                self._authority_file.close()
            raise ValueError("artifact authority configuration is invalid") from None

    def _initialize(self, config, capability_key):
        try:
            import opendal
        except ModuleNotFoundError as exc:
            if exc.name == "opendal":
                raise ArtifactDependencyError(
                    "artifact delivery requires ai-dynamo[artifact-delivery]"
                ) from None
            raise

        self._object_id_key = hmac.digest(
            capability_key, b"dynamo-artifact-object-id-v1", "sha256"
        )
        self.owner = config["owner"]
        if not isinstance(self.owner, str) or not 1 <= len(self.owner) <= 128:
            raise ValueError("invalid artifact authority owner")
        self.policy = StoragePolicy(
            **{
                **config.get("storage_policy", {}),
                "allowed_hosts": frozenset(
                    config.get("storage_policy", {}).get("allowed_hosts", [])
                ),
                "allow_http_hosts": frozenset(
                    config.get("storage_policy", {}).get("allow_http_hosts", [])
                ),
            }
        )
        self.profiles = {}
        for name, profile in config.get("profiles", {}).items():
            if profile.get("immutable") is not True:
                raise ValueError("artifact profile requires certified immutability")
            prefixes = profile["allowed_prefixes"]
            if (
                not isinstance(prefixes, Sequence)
                or isinstance(prefixes, (str, bytes, bytearray))
                or not all(isinstance(prefix, str) for prefix in prefixes)
            ):
                raise ValueError("artifact profile requires a prefix sequence")
            operator = opendal.AsyncOperator(profile["service"], **profile["options"])
            self.profiles[name] = ManagedProfile(operator, tuple(prefixes))
        self.verifiers = self._configured_verifiers(config.get("verifiers", {}))
        database = Path(config["operation_db"]).resolve()
        database.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(
            str(database) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        self._authority_file = os.fdopen(lock_fd, "r+b")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.store = ArtifactOperationStore(
            database,
            terminal_ttl_seconds=config.get("terminal_ttl_seconds", 86400),
        )
        self.store.recover()
        self.service = ArtifactService(
            self.store,
            self._transfer,
            self._verify,
            max_source_bytes=config.get("max_source_bytes", 256 << 20),
            max_pending_operations=config.get("max_pending_operations", 128),
        )

    def _validate_verifier_prefix(self, prefix):
        if (
            not isinstance(prefix, str)
            or len(prefix) > 8192
            or not prefix.isascii()
            or any(ord(character) < 33 or ord(character) == 127 for character in prefix)
            or "?" in prefix
            or "#" in prefix
        ):
            raise ValueError("invalid verifier URL prefix")
        parsed = urlsplit(prefix)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path.endswith("/")
            or not _SAFE_URL_KEY.fullmatch(parsed.path)
            or any(
                segment in {"", ".", ".."} for segment in parsed.path.split("/")[1:-1]
            )
        ):
            raise ValueError("invalid verifier URL prefix")
        host = parsed.hostname
        port = parsed.port
        default_port = 443 if parsed.scheme == "https" else 80
        authority_host = f"[{host}]" if ":" in host else host
        authority = (
            authority_host
            if port in (None, default_port)
            else f"{authority_host}:{port}"
        )
        if authority not in self.policy.allowed_hosts:
            raise ValueError("verifier host is not authorized")
        if parsed.scheme != "https" and authority not in self.policy.allow_http_hosts:
            raise ValueError("insecure verifier URL prefix")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            self.policy.check_address(str(address))

    def _configured_verifiers(self, registrations):
        from dynamo.artifacts.storage import validate_managed_target

        if not isinstance(registrations, dict):
            raise TypeError("invalid verifier registrations")
        configured = {}
        for name, entry in registrations.items():
            if not isinstance(name, str) or not 1 <= len(name) <= 512:
                raise ValueError("invalid verifier reference")
            if not isinstance(entry, dict):
                raise TypeError("invalid verifier registration")
            profile = entry["profile"]
            if "put_url_sha256" in entry:
                if set(entry) != {"profile", "object_key", "put_url_sha256"}:
                    raise ValueError("invalid exact-URL verifier registration")
                if not isinstance(
                    entry["put_url_sha256"], str
                ) or not _SHA256_HEX.fullmatch(entry["put_url_sha256"]):
                    raise ValueError("invalid exact-URL verifier digest")
                key = entry["object_key"]
            else:
                if set(entry) != {"profile", "object_key_prefix", "url_prefix"}:
                    raise ValueError("invalid dynamic verifier registration")
                key = entry["object_key_prefix"]
                self._validate_verifier_prefix(entry["url_prefix"])
            validate_managed_target(ManagedStoreTarget(profile, key), self.profiles)
            configured[name] = dict(entry)
        return configured

    def _target(self, settings):
        settings = dict(settings)
        kind = settings.pop("kind", None)
        if kind == "managed_store":
            return ManagedStoreTarget(**settings)
        if kind == "presigned_http_put":
            return PresignedHttpPutTarget(**settings)
        raise ValueError("unsupported artifact destination")

    def _binding(self, settings):
        if settings["kind"] == "managed_store":
            return settings["profile"], settings["object_key"]
        registration = self.verifiers.get(settings.get("verification_ref"))
        if registration is None:
            raise ValueError("presigned upload requires a verifier registration")
        if "put_url_sha256" in registration:
            digest = hashlib.sha256(settings["url"].encode()).hexdigest()
            if not hmac.compare_digest(registration["put_url_sha256"], digest):
                raise ValueError("verifier registration does not bind this PUT grant")
            return registration["profile"], registration["object_key"]
        prefix = urlsplit(registration["url_prefix"])
        target = urlsplit(settings["url"])
        if (
            target.scheme != prefix.scheme
            or target.netloc != prefix.netloc
            or not target.path.startswith(prefix.path)
        ):
            raise ValueError("verifier registration does not bind this PUT grant")
        suffix = target.path[len(prefix.path) :]
        if not suffix or not _SAFE_URL_KEY.fullmatch(suffix):
            raise ValueError("verifier registration does not bind this PUT grant")
        key = f"{registration['object_key_prefix']}/{suffix}"
        ManagedStoreTarget(registration["profile"], key)
        return registration["profile"], key

    def _managed_object_id(self, profile, key):
        return (
            "obj_"
            + hmac.new(
                self._object_id_key,
                f"{profile}\0{key}".encode(),
                hashlib.sha256,
            ).hexdigest()
        )

    def validate(self, settings, mode):
        if self._closing:
            raise ArtifactAuthorityClosed("artifact authority is closed")
        if mode != "object_store":
            raise ValueError(
                "relay requests require an independently configured relay authority"
            )
        target = self._target(settings)
        if isinstance(target, PresignedHttpPutTarget):
            validate_presigned_target(target, self.policy)
        profile, key = self._binding(settings)
        if profile not in self.profiles:
            raise ValueError("artifact storage profile is not authorized")
        # Validate confined object identity before inference, including delegated
        # targets whose verifier owns a separate managed read profile.
        confined = ManagedStoreTarget(profile, key)
        from dynamo.artifacts.storage import validate_managed_target

        validate_managed_target(confined, self.profiles)

    async def _transfer(self, data, settings):
        receipt = await put_artifact(
            data, self._target(settings), policy=self.policy, profiles=self.profiles
        )
        return {
            key: value
            for key, value in asdict(receipt).items()
            if key in {"actual_bytes", "sha256", "object_id", "provider_identity"}
        }

    async def _verify(self, data, settings, receipt):
        profile, key = self._binding(settings)
        operator = self.profiles[profile].operator
        version = receipt.get("provider_identity")
        version_options = {"version": version} if version else {}
        try:
            metadata = await operator.stat(key, **version_options)
            if metadata.content_length != len(data):
                raise ValueError("stored object length differs")
            stored = await operator.read(
                key, offset=0, size=len(data), **version_options
            )
        except Exception:  # noqa: BLE001 -- sanitize provider errors
            raise ValueError("stored object verification unavailable") from None
        if len(stored) != len(data):
            raise ValueError("stored object exceeded verification bound")
        # The certified immutable profile binds this exact object key; provider
        # versions may supply a stronger receipt, but ETag is not a digest.
        identity = (
            settings["object_id"]
            if settings["kind"] == "presigned_http_put"
            else self._managed_object_id(profile, key)
        )
        return {
            "actual_bytes": len(stored),
            "sha256": hashlib.sha256(stored).hexdigest(),
            "object_id": identity,
            "provider_identity": receipt.get("provider_identity"),
        }

    async def submit(self, data, contents, destination, *, mode="object_store"):
        if self._closing:
            raise ArtifactAuthorityClosed("artifact authority is closed")
        async with self._admission_lock:
            self.validate(destination, mode)
            return await self.service.submit(data, self.owner, contents, destination)

    def status(self, operation_id):
        return self.store.get(operation_id, self.owner)

    async def close(self):
        """Settle owned transfers before releasing this database's authority."""
        self._closing = True
        # Let already-admitted submissions finish creating durable operations.
        async with self._admission_lock:
            pass
        try:
            await self.service.drain()
        finally:
            if self._authority_file is not None:
                self._authority_file.close()
                self._authority_file = None


@cache
def configured_artifact_runtime():
    path = os.environ.get("DYNAMO_ARTIFACT_CONFIG")
    if not path:
        return None
    try:
        config = json.loads(Path(path).read_text())
        key = os.environ["DYNAMO_ARTIFACT_CAPABILITY_KEY"].encode()
        return ArtifactRuntime(config, key)
    except ArtifactDependencyError:
        raise
    except Exception:  # noqa: BLE001 -- sanitize authority configuration failures
        raise ValueError("artifact authority configuration is invalid") from None


async def initialize_configured_artifact_runtime():
    """Fail fast on config and database recovery before accepting requests."""
    return await asyncio.to_thread(configured_artifact_runtime)


async def close_configured_artifact_runtime():
    """Close an already-created authority without initializing one at shutdown."""
    if configured_artifact_runtime.cache_info().currsize:
        authority = configured_artifact_runtime()
        if authority is not None:
            await authority.close()
