# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UDS transport: grpc-over-unix-socket.

Use case: same-Pod sidecar plugin. Trust boundary is the Pod itself —
TLS adds CPU overhead with no security benefit. Socket file permission
(``0660`` + shared GID) is the access control mechanism.

**v11**: ``unix://`` endpoint with mTLS config in YAML is a config-time
error (see ``transport/config.py``).
"""

from __future__ import annotations

import grpc

from dynamo.planner.plugins.transport._grpc_base import _GrpcTransportBase, grpc_channel_options


# Linux UDS path limit (``sun_path``)
_LINUX_UDS_PATH_MAX = 108


class UdsTransport(_GrpcTransportBase):
    """gRPC over Unix Domain Socket — same-Pod sidecar plugin transport."""

    def __init__(
        self,
        plugin_id: str,
        endpoint: str,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not endpoint.startswith("unix://"):
            raise ValueError(
                f"UdsTransport endpoint must start with 'unix://', got {endpoint!r}"
            )
        path = endpoint[len("unix://") :]
        if not path:
            raise ValueError(f"UdsTransport endpoint missing path: {endpoint!r}")
        # Linux socket file path length check (``struct sockaddr_un.sun_path``)
        if len(path) > _LINUX_UDS_PATH_MAX:
            raise ValueError(
                f"UdsTransport endpoint path too long ({len(path)} > "
                f"{_LINUX_UDS_PATH_MAX}): {path!r}"
            )
        self._path = path
        super().__init__(plugin_id, endpoint, timeout_seconds)

    def _build_channel(self) -> grpc.aio.Channel:
        # gRPC unix scheme: "unix:/abs/path" (single slash; the "unix://" in
        # endpoint is the dynamo plugin URL convention)
        target = f"unix:{self._path}"
        return grpc.aio.insecure_channel(target, options=grpc_channel_options())


__all__ = ["UdsTransport"]
