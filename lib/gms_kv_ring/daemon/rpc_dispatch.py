# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composition root for daemon RPC domains."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gms_kv_ring.daemon.rpc_content import HANDLERS as CONTENT_HANDLERS
from gms_kv_ring.daemon.rpc_lifecycle import HANDLERS as LIFECYCLE_HANDLERS
from gms_kv_ring.daemon.rpc_transfer import HANDLERS as TRANSFER_HANDLERS
from gms_kv_ring.daemon.rpc_types import Message, Response, dispatch_table

if TYPE_CHECKING:
    from gms_kv_ring.daemon.server import Daemon

HANDLERS = {**LIFECYCLE_HANDLERS, **CONTENT_HANDLERS, **TRANSFER_HANDLERS}


def dispatch(daemon: "Daemon", msg: Message) -> Response:
    return dispatch_table(daemon, msg, HANDLERS)
