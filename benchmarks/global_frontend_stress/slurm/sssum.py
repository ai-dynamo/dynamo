# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize an `ss -tin` sample: per local port, sockets, and how many are receive-window,
send-buffer or app limited, with retransmit, RTO and cwnd figures.

usage: sssum.py ss-end.txt
"""

import re
import statistics
import sys
from collections import defaultdict


def main():
    lines = open(sys.argv[1]).read().splitlines()
    groups = defaultdict(list)
    for head, info in zip(lines[0::2], lines[1::2]):
        fields = head.split()
        if len(fields) < 4:
            continue
        local, peer = fields[2], fields[3]
        port = local.rsplit(":", 1)[1]
        role = (
            f"server :{port}"
            if int(port) < 32768
            else f"client -> :{peer.rsplit(':', 1)[1]}"
        )
        groups[role].append((int(fields[0]), int(fields[1]), info))
    for role, sockets in sorted(groups.items(), key=lambda item: -len(item[1])):

        def has(word):
            return sum(1 for *_, info in sockets if word in info)

        def values(name):
            out = []
            for *_, info in sockets:
                match = re.search(rf"\b{name}:([\d.]+)", info)
                if match:
                    out.append(float(match.group(1)))
            return out

        retrans = [
            int(m.group(1))
            for *_, info in sockets
            for m in [re.search(r"retrans:\d+/(\d+)", info)]
            if m
        ]
        recvq = [r for r, _, _ in sockets]
        sendq = [s for _, s, _ in sockets]
        print(
            f"{role}: {len(sockets)} sockets; rwnd_limited {has('rwnd_limited')} sndbuf_limited {has('sndbuf_limited')} "
            f"app_limited {has('app_limited')}; with retrans {len(retrans)} (sum {sum(retrans)}); "
            f"rto median {statistics.median(values('rto') or [0])}; cwnd median {statistics.median(values('cwnd') or [0])}; "
            f"Recv-Q>0 {sum(1 for r in recvq if r)} (max {max(recvq)}); Send-Q>0 {sum(1 for s in sendq if s)} (max {max(sendq)})"
        )


if __name__ == "__main__":
    main()
