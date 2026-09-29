# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join immutable capture rank records with named DRA results, never GPU order.

The capture document must carry artifact manifest SHA256 and exact allocation
IDs/sizes, plus layout and per-rank source UUID, captured ordinal and socket device.
The output's cuda_device_map is passed to the experimental Snapshot agent through
SNAPSHOT_CUDA_DEVICE_MAP. The very same rank records configure the GMS processes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID


def gpu_uuid(value):
    if not isinstance(value, str) or not value.startswith("GPU-"):
        raise ValueError("expected GPU UUID")
    if str(UUID(value[4:])) != value[4:]:
        raise ValueError("expected canonical GPU UUID")
    return value


def resolve(capture, claim, slices):
    ranks = capture["ranks"]
    if not ranks or sorted(r["rank"] for r in ranks) != list(range(len(ranks))):
        raise ValueError("capture must contain each logical rank exactly once")
    if not capture["capture_id"] or not capture["layout"]:
        raise ValueError("missing capture identity or parallelism layout")
    devices = {}
    for item in slices["items"]:
        spec = item["spec"]
        if spec["driver"] != "gpu.nvidia.com":
            continue
        for device in spec.get("devices", []):
            key = (spec["driver"], spec["pool"]["name"], device["name"])
            value = device["attributes"]["uuid"]["string"]
            if key in devices and devices[key] != value:
                raise ValueError("conflicting ResourceSlice UUID")
            devices[key] = gpu_uuid(value)
    allocated = {}
    for result in claim["status"]["allocation"]["devices"]["results"]:
        request = result["request"]
        if request in allocated:
            raise ValueError("rank request must select exactly one GPU")
        allocated[request] = devices[
            (result["driver"], result["pool"], result["device"])
        ]
    if set(allocated) != {f"tp-{r['rank']}" for r in ranks}:
        raise ValueError("named DRA requests do not match captured ranks")
    resolved = []
    for rank in sorted(ranks, key=lambda r: r["rank"]):
        gpu_uuid(rank["source_uuid"])
        manifest = Path(rank["artifact"]) / "manifest.json"
        data = manifest.read_bytes()
        if hashlib.sha256(data).hexdigest() != rank["manifest_sha256"]:
            raise ValueError("artifact is not from this capture")
        allocations = json.loads(data)["allocations"]
        expected = sorted((x["allocation_id"], x["aligned_size"]) for x in allocations)
        if expected != sorted(
            (x["allocation_id"], x["aligned_size"]) for x in rank["allocations"]
        ):
            raise ValueError("allocation IDs/sizes differ from captured engine")
        resolved.append(
            {
                **rank,
                "request": f"tp-{rank['rank']}",
                "destination_uuid": allocated[f"tp-{rank['rank']}"],
                "local_device": 0,
            }
        )
    for key in ("source_uuid", "destination_uuid", "captured_ordinal", "socket_device"):
        if len({r[key] for r in resolved}) != len(ranks):
            raise ValueError(f"duplicate {key}")
    return {
        "capture_id": capture["capture_id"],
        "layout": capture["layout"],
        "claim_uid": claim["metadata"]["uid"],
        "ranks": resolved,
        "cuda_device_map": ",".join(
            r["source_uuid"] + "=" + r["destination_uuid"] for r in resolved
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("capture", "claim", "slices"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    print(
        json.dumps(
            resolve(
                *(
                    json.loads(getattr(args, n).read_text())
                    for n in ("capture", "claim", "slices")
                )
            ),
            indent=2,
        )
    )
