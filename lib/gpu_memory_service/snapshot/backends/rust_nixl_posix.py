# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ctypes wrapper for the optional in-process Rust NIXL POSIX helper.

The helper deliberately avoids importing ``nixl._api`` so the GMS loader can use
NIXL POSIX file -> pinned-host staging without paying the Python binding import
cost on restore cold start.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from collections.abc import Mapping
from typing import Optional

_ERROR_LEN = 4096
_LIB: Optional[ctypes.CDLL] = None


class RustNixlPosixContextStats(ctypes.Structure):
    _fields_ = [
        ("dlopen_s", ctypes.c_double),
        ("create_agent_backend_s", ctypes.c_double),
        ("total_s", ctypes.c_double),
    ]


class RustNixlPosixReadStats(ctypes.Structure):
    _fields_ = [
        ("register_file_s", ctypes.c_double),
        ("register_host_s", ctypes.c_double),
        ("create_req_s", ctypes.c_double),
        ("transfer_s", ctypes.c_double),
        ("cleanup_s", ctypes.c_double),
        ("total_s", ctypes.c_double),
    ]


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    lib_path = Path(__file__).with_name("libgms_nixl_posix.so")
    lib = ctypes.CDLL(str(lib_path))
    lib.gms_nixl_posix_context_create.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(RustNixlPosixContextStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.gms_nixl_posix_context_create.restype = ctypes.c_int
    lib.gms_nixl_posix_context_destroy.argtypes = [ctypes.c_void_p]
    lib.gms_nixl_posix_context_destroy.restype = None
    lib.gms_nixl_posix_read.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.POINTER(RustNixlPosixReadStats),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    lib.gms_nixl_posix_read.restype = ctypes.c_int
    _LIB = lib
    return lib


def is_available() -> bool:
    try:
        _load_lib()
    except OSError:
        return False
    return True


def _error_text(error: ctypes.Array[ctypes.c_char]) -> str:
    value = error.value.decode("utf-8", errors="replace")
    return value or "unknown Rust NIXL POSIX helper error"


class RustNixlPosixContext:
    """Owns one NIXL POSIX agent/backend created by the Rust helper."""

    def __init__(
        self,
        agent_name: str,
        *,
        backend_params: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._lib = _load_lib()
        self._ctx = ctypes.c_void_p()
        self.stats = RustNixlPosixContextStats()
        error = ctypes.create_string_buffer(_ERROR_LEN)
        encoded_params = [
            (str(key).encode("utf-8"), str(value).encode("utf-8"))
            for key, value in dict(backend_params or {}).items()
        ]
        key_array = (ctypes.c_char_p * len(encoded_params))(
            *(key for key, _value in encoded_params)
        )
        value_array = (ctypes.c_char_p * len(encoded_params))(
            *(value for _key, value in encoded_params)
        )
        rc = self._lib.gms_nixl_posix_context_create(
            agent_name.encode("utf-8"),
            key_array,
            value_array,
            len(encoded_params),
            ctypes.byref(self._ctx),
            ctypes.byref(self.stats),
            error,
            _ERROR_LEN,
        )
        if rc != 0:
            raise RuntimeError(_error_text(error))
        self._agent_name = agent_name.encode("utf-8")

    def read(
        self,
        *,
        fd: int,
        file_offset: int,
        host_ptr: int,
        size: int,
    ) -> RustNixlPosixReadStats:
        if self._ctx.value is None:
            raise RuntimeError("Rust NIXL POSIX context is closed")
        stats = RustNixlPosixReadStats()
        error = ctypes.create_string_buffer(_ERROR_LEN)
        rc = self._lib.gms_nixl_posix_read(
            self._ctx,
            self._agent_name,
            fd,
            file_offset,
            host_ptr,
            size,
            ctypes.byref(stats),
            error,
            _ERROR_LEN,
        )
        if rc != 0:
            raise RuntimeError(_error_text(error))
        return stats

    def close(self) -> None:
        if self._ctx.value is None:
            return
        ctx = self._ctx
        self._ctx = ctypes.c_void_p()
        self._lib.gms_nixl_posix_context_destroy(ctx)

    def __enter__(self) -> "RustNixlPosixContext":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
