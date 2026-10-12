# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded opaque-byte relay over an operator-owned Mooncake Store client.

The caller configures and authenticates access to the internal store. Descriptors
are integrity metadata, not public authorization or a durable-storage promise.
The relay owns its injected client; close it after consumers acknowledge receipt.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

MAX_RELAY_BYTES = 64 << 20
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class RelayError(RuntimeError):
    """A bounded public error containing no native provider diagnostics."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class NativeByteStore(Protocol):
    """The bounded native Store API; neither tensors nor inference concepts."""

    def put(self, key: str, value: bytes) -> int:
        ...

    def get_size(self, key: str) -> int:
        ...

    def register_buffer(self, address: int, size: int) -> int:
        ...

    def get_into(self, key: str, address: int, size: int) -> int:
        ...

    def unregister_buffer(self, address: int) -> int:
        ...

    def remove(self, key: str, force: bool = False) -> int:
        ...

    def close(self) -> int:
        ...


@dataclass(frozen=True)
class RelayDescriptor:
    key: str
    byte_count: int
    sha256: str


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise RelayError(code, message)


def _native(call: Callable[[], int], operation: str) -> int:
    try:
        result = call()
    except (RuntimeError, OSError, TypeError, ValueError):
        raise RelayError("relay_io_failed", f"relay {operation} failed") from None
    _require(type(result) is int, "relay_io_failed", f"relay {operation} failed")
    return result


class MooncakeByteRelay:
    """Serialize bounded native calls using unique keys under an operation namespace."""

    def __init__(
        self,
        store: NativeByteStore,
        *,
        namespace: str,
        max_bytes: int = MAX_RELAY_BYTES,
    ) -> None:
        _require(
            isinstance(namespace, str)
            and len(namespace) <= 512
            and all(_SEGMENT.fullmatch(part) for part in namespace.split("/")),
            "invalid_namespace",
            "relay namespace is invalid",
        )
        _require(
            type(max_bytes) is int and 0 < max_bytes <= MAX_RELAY_BYTES,
            "invalid_limit",
            "relay byte limit is invalid",
        )
        self._store = store
        self._namespace = namespace
        self._max_bytes = max_bytes
        self._lock = threading.RLock()
        self._closed = False
        self._quarantined_buffer: ctypes.Array | None = None

    def _available(self) -> None:
        _require(
            not self._closed and self._quarantined_buffer is None,
            "relay_unavailable",
            "relay client is unavailable",
        )

    def _descriptor(self, descriptor: RelayDescriptor) -> None:
        _require(
            isinstance(descriptor, RelayDescriptor),
            "invalid_descriptor",
            "relay descriptor is invalid",
        )
        prefix = self._namespace + "/"
        _require(
            isinstance(descriptor.key, str)
            and descriptor.key.startswith(prefix)
            and re.fullmatch(r"[0-9a-f]{32}", descriptor.key[len(prefix) :]) is not None
            and type(descriptor.byte_count) is int
            and 0 < descriptor.byte_count <= self._max_bytes
            and isinstance(descriptor.sha256, str)
            and _DIGEST.fullmatch(descriptor.sha256) is not None,
            "invalid_descriptor",
            "relay descriptor scope, size or digest is invalid",
        )

    def publish(self, data: bytes) -> RelayDescriptor:
        """Publish owned bytes and verify them; a native PUT acknowledgement alone is insufficient."""
        with self._lock:
            self._available()
            _require(
                type(data) is bytes and 0 < len(data) <= self._max_bytes,
                "invalid_payload",
                "relay requires nonempty bounded immutable bytes",
            )
            descriptor = RelayDescriptor(
                f"{self._namespace}/{uuid.uuid4().hex}",
                len(data),
                hashlib.sha256(data).hexdigest(),
            )
            result = _native(lambda: self._store.put(descriptor.key, data), "publish")
            _require(result == 0, "relay_publish_failed", "relay publication failed")
            self.read(descriptor)
            return descriptor

    def read(self, descriptor: RelayDescriptor) -> bytes:
        """Read into a descriptor-sized buffer and verify length and digest before return."""
        with self._lock:
            self._available()
            self._descriptor(descriptor)
            size = _native(lambda: self._store.get_size(descriptor.key), "stat")
            _require(size >= 0, "relay_unavailable", "relay object is unavailable")
            _require(
                size == descriptor.byte_count,
                "relay_length_mismatch",
                "relay object length differs",
            )
            buffer = ctypes.create_string_buffer(descriptor.byte_count)
            address = ctypes.addressof(buffer)
            result = _native(
                lambda: self._store.register_buffer(address, descriptor.byte_count),
                "register",
            )
            _require(result == 0, "relay_io_failed", "relay buffer registration failed")
            self._quarantined_buffer = buffer
            try:
                count = _native(
                    lambda: self._store.get_into(
                        descriptor.key, address, descriptor.byte_count
                    ),
                    "read",
                )
                _require(
                    count == descriptor.byte_count,
                    "relay_length_mismatch",
                    "relay received length differs",
                )
                data = buffer.raw
                _require(
                    hmac.compare_digest(
                        hashlib.sha256(data).hexdigest(), descriptor.sha256
                    ),
                    "relay_digest_mismatch",
                    "relay object digest differs",
                )
            finally:
                result = _native(
                    lambda: self._store.unregister_buffer(address), "unregister"
                )
                _require(
                    result == 0, "relay_io_failed", "relay buffer unregistration failed"
                )
                self._quarantined_buffer = None
            return data

    def remove(self, descriptor: RelayDescriptor, *, force: bool = False) -> None:
        """Remove an acknowledged object; force also retires its remaining read lease."""
        with self._lock:
            self._available()
            self._descriptor(descriptor)
            _require(
                type(force) is bool, "invalid_cleanup", "relay cleanup mode is invalid"
            )
            result = _native(
                lambda: self._store.remove(descriptor.key, force=force), "remove"
            )
            _require(result == 0, "relay_remove_failed", "relay object removal failed")

    def close(self) -> None:
        """Close the owned native client before releasing any quarantined receive buffer."""
        with self._lock:
            if self._closed:
                return
            result = _native(self._store.close, "close")
            _require(result == 0, "relay_io_failed", "relay client close failed")
            self._closed = True
            self._quarantined_buffer = None
