# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional cross-node transport implementations."""

__all__ = [
    "NixlTransport",
    "PeerHandle",
    "TransportClosed",
    "TransportNotAvailable",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module("gms_kv_ring.daemon.transport.nixl_transport")
    return getattr(module, name)
