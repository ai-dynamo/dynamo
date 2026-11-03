# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-second rates of the TCP loss, queueing and memory counters each node's netstat-start
and netstat-end snapshots differ by, plus NIC drops and the end-of-window socket memory.

usage: netdiff.py RESULTS_DIR
"""

import json
import sys
from pathlib import Path

KEYS = (
    "TcpRetransSegs TcpOutSegs TcpInSegs TcpExtTCPTimeouts TcpExtTCPLossProbes TcpExtTCPSackRecovery "
    "TcpExtTCPFastRetrans TcpExtTCPSlowStartRetrans TcpExtTCPSynRetrans TcpExtListenOverflows "
    "TcpExtListenDrops TcpExtTCPBacklogDrop TcpExtTCPRcvQDrop TcpExtPruneCalled TcpExtRcvPruned "
    "TcpExtTCPOFODrop TcpExtTCPMemoryPressures TcpExtTCPZeroWindowDrop TcpExtTCPAbortOnMemory "
    "TcpExtTCPRcvCollapsed TcpExtTCPToZeroWindowAdv TcpExtTCPDSACKRecv TcpExtTCPSpuriousRTOs "
    "TcpExtTCPLostRetransmit TcpExtTCPAbortOnTimeout"
).split()


def load(path):
    counters, nic, at, sockstat, section, sysctl = {}, {}, None, "", "", []
    for line in path.read_text().splitlines():
        if line.startswith("# at"):
            at = float(line.split()[2])
            continue
        if line.startswith("#kernel"):
            continue
        if line.startswith("#"):
            section = line.strip()
            continue
        parts = line.split()
        if not section and len(parts) >= 2 and parts[1].isdigit():
            counters[parts[0]] = int(parts[1])
        elif section == "# sockstat" and line.startswith("TCP:"):
            sockstat = line.strip()
        elif section == "# nic" and len(parts) == 2:
            nic[parts[0]] = int(parts[1])
        elif section == "# sysctl":
            sysctl.append(line.strip())
    return at, counters, nic, sockstat, sysctl


def main():
    directory = Path(sys.argv[1])
    roles = json.loads((directory / "roles.json").read_text())
    for key in ("frontend", "mocks", "client_c", "client_d"):
        host = roles[key]
        t0, a, nic0, _, sysctl = load(directory / host / "netstat-start.txt")
        t1, b, nic1, sockstat, _ = load(directory / host / "netstat-end.txt")
        seconds = t1 - t0
        rates = {
            name.replace("TcpExt", ""): (b.get(name, 0) - a.get(name, 0)) / seconds
            for name in KEYS
        }
        print(f"{key} {host} ({seconds:.0f} s) {sockstat}")
        print(
            "   ",
            "  ".join(
                f"{name}={value:,.0f}/s" for name, value in rates.items() if value
            ),
        )
        drops = {
            name: nic1[name] - nic0[name]
            for name in nic1
            if nic1[name] != nic0.get(name)
        }
        print("    nic", drops or "no drops")
        if key == "frontend":
            print("    " + " | ".join(sysctl))


if __name__ == "__main__":
    main()
