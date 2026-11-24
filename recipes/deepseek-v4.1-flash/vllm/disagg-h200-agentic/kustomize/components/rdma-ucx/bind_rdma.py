# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bind UCX to this pod's four assigned IB devices, then exec unchanged argv."""

import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"Duplicate allocation metadata key: {key!r}")
        result[key] = value
    return result


def allocated_verbs(environ):
    addresses = environ.get("PCIDEVICE_RDMA_IB", "").split(",")
    if (
        len(addresses) != 4
        or len(set(addresses)) != 4
        or not all(
            re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[01][0-9a-f]\.[0-7]", a)
            for a in addresses
        )
    ):
        raise RuntimeError("Expected four unique PCI BDFs in PCIDEVICE_RDMA_IB")
    try:
        info = json.loads(
            environ["PCIDEVICE_RDMA_IB_INFO"], object_pairs_hook=unique_object
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Missing/malformed PCIDEVICE_RDMA_IB_INFO") from exc
    if not isinstance(info, dict) or set(info) != set(addresses):
        raise RuntimeError("Allocation BDF list and INFO keys disagree")
    selected = {}
    for address in addresses:
        entry = info[address]
        try:
            device_id = entry["generic"]["deviceID"]
            path = entry["rdma"]["uverbs"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Incomplete allocation metadata: {address}") from exc
        if device_id != address:
            raise RuntimeError(f"Allocation deviceID mismatch: {address}")
        if not isinstance(path, str) or not re.fullmatch(
            r"/dev/infiniband/uverbs[0-9]+", path
        ):
            raise RuntimeError(f"Invalid allocated uverbs path: {path!r}")
        if path in selected:
            raise RuntimeError(f"Duplicate allocated uverbs path: {path}")
        selected[path] = address
    return selected


def discover(
    dev_root=Path("/dev/infiniband"), sys_root=Path("/sys/class"), environ=None
):
    allocated = allocated_verbs(os.environ if environ is None else environ)
    exposed = sorted(dev_root.glob("uverbs*"))
    selected = {dev_root / Path(path).name: bdf for path, bdf in allocated.items()}
    if not set(selected).issubset(exposed):
        raise RuntimeError("Allocated uverbs device is not visible")
    rows = []
    for dev, address in sorted(selected.items()):
        if not dev.is_char_device():
            raise RuntimeError(f"Not a character device: {dev}")
        verbs = sys_root / "infiniband_verbs" / dev.name
        number = dev.stat().st_rdev
        device_number = f"{os.major(number)}:{os.minor(number)}"
        if (verbs / "dev").read_text().strip() != device_number:
            raise RuntimeError(f"Character device/sysfs number mismatch: {dev}")
        verbs_path = verbs.resolve(strict=True)
        if verbs_path.name != dev.name or verbs_path.parent.name != "infiniband_verbs":
            raise RuntimeError(f"Unexpected verbs sysfs path: {verbs_path}")
        pci = verbs_path.parent.parent
        if pci.name != address:
            raise RuntimeError(f"Allocated PCI/sysfs mismatch: {address}: {pci}")
        candidates = set()
        ibdev = verbs / "ibdev"
        if ibdev.is_symlink():
            candidates.add(ibdev.resolve(strict=True).name)
        elif ibdev.is_file():
            candidates.add(ibdev.read_text().strip())
        candidates.update(p.name for p in (verbs / "device/infiniband").glob("*"))
        if len(candidates) != 1:
            raise RuntimeError(f"Ambiguous/missing IB mapping: {dev}: {candidates}")
        name = candidates.pop()
        if not re.fullmatch(r"mlx5_[0-9]+", name):
            raise RuntimeError(f"Unexpected IB device name: {name!r}")
        ib = sys_root / "infiniband" / name
        if (ib / "device").resolve(strict=True) != pci:
            raise RuntimeError(f"Verbs/IB PCI mapping mismatch: {dev}: {name}")
        active = []
        for port in sorted((ib / "ports").glob("*")):
            state = (port / "state").read_text().strip()
            link = (port / "link_layer").read_text().strip()
            physical = (port / "phys_state").read_text().strip()
            if state.startswith("4:") and link == "InfiniBand":
                if not port.name.isdecimal() or not physical.startswith("5:"):
                    raise RuntimeError(f"Invalid active IB port: {port}: {physical}")
                active.append(port.name)
        if len(active) != 1:
            raise RuntimeError(f"Expected one active IB port for {name}, got {active}")
        rows.append(
            {
                "uverbs": str(dev),
                "allocated_pci": address,
                "device_number": device_number,
                "ib_device": name,
                "port": active[0],
                "pci_path": str((ib / "device").resolve(strict=True)),
                "verbs_sysfs_path": str(verbs.resolve(strict=True)),
            }
        )
    if len({row["ib_device"] for row in rows}) != 4:
        raise RuntimeError("Assigned verbs devices do not map to four distinct IB NICs")
    return rows


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        raise RuntimeError("Usage: bind_rdma.py -- <unchanged worker command and args>")
    command = sys.argv[2:]
    rows = discover()
    visible = sorted(str(p) for p in Path("/dev/infiniband").glob("uverbs*"))
    target = ",".join(sorted(f"{r['ib_device']}:{r['port']}" for r in rows))
    inherited = os.environ.get("UCX_NET_DEVICES")
    if inherited and inherited != target:
        raise RuntimeError(
            f"Conflicting UCX_NET_DEVICES: {inherited!r}; assigned {target}"
        )
    os.environ["UCX_NET_DEVICES"] = target
    print(
        json.dumps(
            {
                "event": "DH01_ASSIGNED_RDMA_BINDING",
                "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "pod": os.environ.get("POD_NAME"),
                "node": os.environ.get("NODE_NAME"),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "argv": command,
                "assigned": rows,
                "allocated_verbs_count": len(rows),
                "allocated_pci_bdfs": sorted(r["allocated_pci"] for r in rows),
                "visible_verbs_count": len(visible),
                "visible_uverbs": visible,
                "UCX_NET_DEVICES": target,
                "scope": "Assigned NIC binding; not a transport-integrity pass.",
            }
        ),
        flush=True,
    )
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
