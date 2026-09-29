# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare every restored GMS allocation byte against its original artifact."""

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

from cuda.bindings import driver as cuda
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.device import get_device_uuid
from gpu_memory_service.v1.protocol import (
    ListAllocationsRequest,
    ListAllocationsResponse,
)
from gpu_memory_service.v1.snapshot.weight_artifact import _map_export, _release_mapping

p = argparse.ArgumentParser()
p.add_argument("--artifact", required=True, type=Path)
p.add_argument("--socket", default="/gms/gms_0_weights.sock")
a = p.parse_args()
started = time.monotonic()
manifest = json.loads((a.artifact / "manifest.json").read_text())
init_vmm(VMMDeviceType.CUDA)
vmm = get_vmm()
vmm.ensure_initialized()
vmm.runtime_set_device(0)
session = _GMSClientSession(
    a.socket, RequestedLockType.RO, connect_timeout=5, admission_timeout=5
)
try:
    assert session.identity[1] == get_device_uuid(0)
    records = session._call(
        ListAllocationsRequest(), ListAllocationsResponse
    ).allocations
    assert sorted((r.allocation_id, r.aligned_size) for r in records) == sorted(
        (r["allocation_id"], r["aligned_size"]) for r in manifest["allocations"]
    )
    results = []
    chunk = 16 * 1024**2
    buffer = (ctypes.c_ubyte * chunk)()
    for record in records:
        saved = next(
            r
            for r in manifest["allocations"]
            if r["allocation_id"] == record.allocation_id
        )
        mapping = _map_export(
            session,
            record,
            vmm,
            0,
            int(vmm.get_allocation_granularity(0)),
            GrantedLockType.RO,
        )
        expected = hashlib.sha256()
        actual = hashlib.sha256()
        try:
            with (a.artifact / saved["shard"]).open("rb") as f:
                f.seek(saved["offset"])
                for offset in range(0, record.aligned_size, chunk):
                    size = min(chunk, record.aligned_size - offset)
                    data = f.read(size)
                    assert len(data) == size
                    expected.update(data)
                    (rc,) = cuda.cuMemcpyDtoH(
                        ctypes.addressof(buffer), mapping[0].base + offset, size
                    )
                    assert int(rc) == 0
                    actual.update(memoryview(buffer).cast("B")[:size])
        finally:
            _release_mapping(vmm, mapping)
        assert expected.digest() == actual.digest(), record.allocation_id
        results.append(
            {
                "allocation_id": record.allocation_id,
                "bytes": record.aligned_size,
                "sha256": actual.hexdigest(),
            }
        )
    print(
        json.dumps(
            {
                "gpu_uuid": session.identity[1],
                "all_bytes_match": True,
                "allocations": results,
                "verification_s": time.monotonic() - started,
            }
        ),
        flush=True,
    )
finally:
    session.close()
