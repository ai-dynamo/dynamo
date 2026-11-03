# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Once a second, append one JSON line of node telemetry: busy and total jiffies per NUMA
domain, CPU jiffies and RSS summed over the processes of each tracked name, NIC bytes and
packets, and TCP retransmits.

usage: sampler.py OUT.jsonl NIC NAME [NAME ...]   (NAME is a process comm, e.g. global-frontend)
"""

import json
import os
import sys
import time

DOMAINS = {"numa0": range(0, 72), "numa1": range(72, 144)}


def cpu_jiffies():
    busy = {name: 0 for name in DOMAINS}
    total = {name: 0 for name in DOMAINS}
    with open("/proc/stat") as stat:
        for line in stat:
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            fields = line.split()
            cpu = int(fields[0][3:])
            values = [int(value) for value in fields[1:9]]
            idle = values[3] + values[4]
            for name, cpus in DOMAINS.items():
                if cpu in cpus:
                    busy[name] += sum(values) - idle
                    total[name] += sum(values)
    return {name: [busy[name], total[name]] for name in DOMAINS}


def processes(names):
    cpu = {name: 0 for name in names}
    rss = {name: 0 for name in names}
    count = {name: 0 for name in names}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as stat:
                text = stat.read()
            comm = text[text.index("(") + 1 : text.rindex(")")]
            if comm not in cpu:
                continue
            fields = text[text.rindex(")") + 2 :].split()
            cpu[comm] += int(fields[11]) + int(fields[12])
            rss[comm] += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
            count[comm] += 1
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            continue
    return cpu, rss, count


def nic(device):
    base = f"/sys/class/net/{device}/statistics"
    values = {}
    for key in ("rx_bytes", "tx_bytes", "rx_packets", "tx_packets"):
        try:
            with open(f"{base}/{key}") as handle:
                values[key] = int(handle.read())
        except FileNotFoundError:
            values[key] = None
    return values


def retransmits():
    with open("/proc/net/snmp") as snmp:
        lines = [line.split() for line in snmp if line.startswith("Tcp:")]
    header, values = lines[0], lines[1]
    return int(values[header.index("RetransSegs")])


def main():
    out, device, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    with open(out, "a", buffering=1) as handle:
        while True:
            cpu, rss, count = processes(names)
            row = {
                "t": time.time(),
                "cpu": cpu_jiffies(),
                "procs": cpu,
                "rss": rss,
                "nprocs": count,
                "nic": nic(device),
                "retrans": retransmits(),
            }
            handle.write(json.dumps(row) + "\n")
            time.sleep(1.0 - time.time() % 1.0)


if __name__ == "__main__":
    main()
