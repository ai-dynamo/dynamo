# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC over TCP transport — cross-Pod plugin.

**Default mTLS-required**: ``GrpcTransport`` constructed without
``mtls_config`` raises unless ``allow_insecure=True`` is explicitly passed.
This forces deliberate security choices and eliminates "forgot to enable
TLS" production incidents.

mTLS credential loading reuses dynamo platform cert-manager convention
(``MtlsConfig.from_k8s_secret_mount``); see ``_mtls.py``.
"""

from __future__ import annotations

import logging

import grpc

from dynamo.planner.plugins.transport._grpc_base import _GrpcTransportBase, grpc_channel_options
from dynamo.planner.plugins.transport._mtls import MtlsConfig

log = logging.getLogger(__name__)


class GrpcTransport(_GrpcTransportBase):
    """gRPC over TCP for cross-Pod plugins. mTLS required by default."""

    def __init__(
        self,
        plugin_id: str,
        endpoint: str,
        mtls_config: MtlsConfig | None = None,
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
        if mtls_config is None and not allow_insecure:
            raise ValueError(
                f"GrpcTransport(plugin_id={plugin_id!r}): mtls_config is None and "
                f"allow_insecure=False; refusing to create insecure channel. "
                f"Either provide MtlsConfig (production) or set allow_insecure=True "
                f"explicitly (dev only — startup logs WARNING)."
            )
        if mtls_config is None and allow_insecure:
            log.warning(
                "GrpcTransport(plugin_id=%s, endpoint=%s): allow_insecure=True; "
                "channel will be plaintext. DEV ONLY — never use in production.",
                plugin_id,
                endpoint,
            )
        self._target = target
        self._mtls = mtls_config
        super().__init__(plugin_id, endpoint, timeout_seconds)

    def _build_channel(self) -> grpc.aio.Channel:
        if self._mtls is not None:
            creds = grpc.ssl_channel_credentials(
                root_certificates=self._mtls.ca_bundle,
                private_key=self._mtls.client_key,
                certificate_chain=self._mtls.client_cert,
            )
            return grpc.aio.secure_channel(self._target, creds, options=grpc_channel_options())
        # allow_insecure path; warning already logged in __init__
        return grpc.aio.insecure_channel(self._target, options=grpc_channel_options())


__all__ = ["GrpcTransport"]
