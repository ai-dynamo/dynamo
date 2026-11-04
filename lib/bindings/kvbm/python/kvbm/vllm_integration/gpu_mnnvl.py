# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in FABRIC VMM storage; importing this module does not load CUDA or NVML.

Only allocation handles/addresses use host memory. KV bytes are mapped DEVICE
memory. Each PyTorch CUDA-array-interface storage retains its allocation owner;
the owner must outlive every local view and all native transfer leases.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass


def _gpu_uuid(value):
    if not isinstance(value, str):
        raise ValueError("A physical GPU UUID is required")
    return "GPU-" + str(uuid.UUID(value.removeprefix("GPU-")))


@dataclass(frozen=True)
class FabricIdentity:
    gpu_uuid: str
    cluster_uuid: str
    clique_id: int
    health_mask: int = 0

    def __post_init__(self):
        if self.gpu_uuid != _gpu_uuid(self.gpu_uuid):
            raise ValueError("Canonical physical GPU UUID required")
        if (
            not isinstance(self.cluster_uuid, str)
            or len(self.cluster_uuid) != 32
            or any(c not in "0123456789abcdef" for c in self.cluster_uuid)
            or int(self.cluster_uuid, 16) == 0
            or type(self.clique_id) is not int
            or not 0 <= self.clique_id <= 0xFFFFFFFF
        ):
            raise ValueError("Completed nonzero MNNVL cluster and u32 clique required")

    @property
    def domain(self):
        return f"mnnvl:{self.cluster_uuid}:{self.clique_id}"


def effective_domain(layout_domain, fabric_domain):
    """Bind the existing exact router domain to layout, transport and fabric."""
    if (
        not isinstance(layout_domain, str)
        or not layout_domain
        or layout_domain != layout_domain.strip()
        or "\x00" in layout_domain
    ):
        raise ValueError("Explicit unpadded certified layout domain required")
    parts = fabric_domain.split(":") if isinstance(fabric_domain, str) else []
    if (
        len(parts) != 3
        or parts[0] != "mnnvl"
        or not parts[2].isascii()
        or not parts[2].isdecimal()
    ):
        raise ValueError("Canonical observed MNNVL fabric domain required")
    identity = FabricIdentity(
        "GPU-00000000-0000-0000-0000-000000000001", parts[1], int(parts[2])
    )
    if identity.domain != fabric_domain:
        raise ValueError("Canonical observed MNNVL fabric domain required")
    body = {
        "schema": 1,
        "layout_domain": layout_domain,
        "gpu_transport": "mnnvl",
        "fabric_domain": fabric_domain,
    }
    return (
        "mnnvl-sha256:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
