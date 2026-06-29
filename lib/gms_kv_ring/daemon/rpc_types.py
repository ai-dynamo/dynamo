# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts shared by daemon RPC handlers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeAlias

if TYPE_CHECKING:
    from gms_kv_ring.daemon.server import Daemon
Message: TypeAlias = dict[str, Any]
Response: TypeAlias = dict[str, Any]
Handler: TypeAlias = Callable[["Daemon", Message], Response]


def required_str(msg: Message, key: str) -> str:
    return str(msg[key])


def required_int(msg: Message, key: str) -> int:
    return int(msg[key])


def optional_int(msg: Message, key: str, default: int = 0) -> int:
    return int(msg.get(key, default))


def required_digest(msg: Message, key: str = "content_hash") -> bytes:
    return bytes.fromhex(required_str(msg, key))


def error_response(exc: Exception) -> Response:
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def dispatch_table(
    daemon: "Daemon", msg: Message, handlers: dict[str, Handler]
) -> Response:
    op = msg.get("op")
    handler = handlers.get(op) if isinstance(op, str) else None
    if handler is None:
        return {"ok": False, "error": f"unknown op {op!r}"}
    try:
        return handler(daemon, msg)
    except Exception as exc:
        return error_response(exc)
