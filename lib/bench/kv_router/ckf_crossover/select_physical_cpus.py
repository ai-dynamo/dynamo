#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path


def expand_cpu_list(value):
    cpus = []
    for part in value.strip().split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            cpus.extend(range(start, end + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    status = Path("/proc/self/status").read_text().splitlines()
    allowed_text = next(
        line.split(":", 1)[1].strip()
        for line in status
        if line.startswith("Cpus_allowed_list:")
    )
    allowed = expand_cpu_list(allowed_text)
    selected = {}
    nodes = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        package = int((topology / "physical_package_id").read_text())
        core = int((topology / "core_id").read_text())
        selected.setdefault((package, core), cpu)
        node_links = list(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
        nodes.update(int(link.name[4:]) for link in node_links)
    cpus = list(selected.values())
    if args.limit is not None:
        cpus = cpus[: args.limit]
    result = {
        "allowed": allowed_text,
        "physical_cpus": ",".join(str(cpu) for cpu in cpus),
        "physical_core_count": len(cpus),
        "numa_nodes": ",".join(str(node) for node in sorted(nodes)),
    }
    if args.shell:
        print(result["physical_cpus"])
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
