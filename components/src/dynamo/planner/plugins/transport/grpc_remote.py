# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC over TCP transport — cross-Pod plugin.

PR #1 ships plaintext gRPC only, gated behind ``allow_insecure=True``
(DEV ONLY — startup logs WARNING). mTLS support (cert-manager / Secret
mount) lands in a follow-up PR.
"""

from __future__ import annotations

import logging

import grpc

from dynamo.planner.plugins.transport._grpc_base import _GrpcTransportBase, grpc_channel_options

log = logging.getLogger(__name__)


class GrpcTransport(_GrpcTransportBase):
    """gRPC over TCP for cross-Pod plugins. Plaintext only in PR #1."""

    def __init__(
        self,
        plugin_id: str,
        endpoint: str,
        timeout_seconds: float = 5.0,
        *,
        allow_insecure: bool = False,
    ) -> None:
        if not endpoint.startswith("grpc://"):
            raise ValueError(
                f"GrpcTransport endpoint must start with 'grpc://', got {endpoint!r}"
            )
        target = endpoint[len("grpc://") :]
        if not target:
            raise ValueError(f"GrpcTransport endpoint missing host:port: {endpoint!r}")
        if not allow_insecure:
            raise ValueError(
                f"GrpcTransport(plugin_id={plugin_id!r}): plaintext gRPC requires "
                f"allow_insecure=True (DEV ONLY — startup logs WARNING). mTLS "
                f"support lands in a follow-up PR."
            )
        log.warning(
            "GrpcTransport(plugin_id=%s, endpoint=%s): allow_insecure=True; "
            "channel will be plaintext. DEV ONLY — never use in production.",
            plugin_id,
            endpoint,
        )
        self._target = target
        super().__init__(plugin_id, endpoint, timeout_seconds)

    def _build_channel(self) -> grpc.aio.Channel:
        return grpc.aio.insecure_channel(self._target, options=grpc_channel_options())


__all__ = ["GrpcTransport"]
