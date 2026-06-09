# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-node transport for GMS daemons (Phase 3 of cross-node design).

NIXL-based peer-to-peer KV transport. Each daemon owns a NIXL agent
with the UCX backend, listens for inbound transfers, and pushes
outbound transfers to peer daemons.

Single-host loopback uses UCX over SHM transport; production
deployments use UCX over InfiniBand or RoCE.

See `docs/CROSS_NODE_DESIGN.md` and `docs/CROSS_NODE_IMPLEMENTATION.md`."""

from gms_kv_ring.daemon.transport.nixl_transport import (
    NixlTransport,
    PeerHandle,
    TransportClosed,
    TransportNotAvailable,
)

__all__ = [
    "NixlTransport",
    "PeerHandle",
    "TransportClosed",
    "TransportNotAvailable",
]
