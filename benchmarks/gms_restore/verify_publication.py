# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate all rank publications against one plan, without initializing CUDA."""

import argparse
import json
from pathlib import Path

from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import (
    ListAllocationsRequest,
    ListAllocationsResponse,
)

p = argparse.ArgumentParser()
p.add_argument("plan", type=Path)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
for r in plan["ranks"]:
    socket = f"{r['socket_dir']}/gms_{r['socket_device']}_weights.sock"
    session = _GMSClientSession(
        socket, RequestedLockType.RO, connect_timeout=10, admission_timeout=10
    )
    try:
        if session.identity[1] != r["destination_uuid"]:
            raise RuntimeError("wrong rank GPU")
        records = session._call(
            ListAllocationsRequest(), ListAllocationsResponse
        ).allocations
        expected = sorted(
            (x["allocation_id"], x["aligned_size"]) for x in r["allocations"]
        )
        if sorted((x.allocation_id, x.aligned_size) for x in records) != expected:
            raise RuntimeError("wrong capture allocations")
    finally:
        session.close()
print(
    json.dumps(
        {
            "capture_id": plan["capture_id"],
            "published": True,
            "ranks": len(plan["ranks"]),
        }
    )
)
