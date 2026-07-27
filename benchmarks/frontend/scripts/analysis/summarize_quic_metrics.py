#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate steady-window QUIC response indicators from Prometheus snapshots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_snapshot(paths: list[Path]) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            metric = fields[0].split("{", 1)[0]
            if not metric.startswith("dynamo_quic_response_"):
                continue
            try:
                values[metric] += float(fields[1])
            except ValueError:
                continue
    return dict(values)


def delta(before: dict[str, float], after: dict[str, float], metric: str) -> float:
    return after.get(metric, 0.0) - before.get(metric, 0.0)


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", nargs="+", required=True, type=Path)
    parser.add_argument("--after", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    before = read_snapshot(args.before)
    after = read_snapshot(args.after)
    payloads = delta(
        before, after, "dynamo_quic_response_payloads_total"
    )
    datagrams = delta(
        before, after, "dynamo_quic_response_udp_tx_datagrams_total"
    )
    stream_frames = delta(
        before, after, "dynamo_quic_response_stream_frames_tx_total"
    )
    transmit_ios = delta(
        before, after, "dynamo_quic_response_udp_tx_ios_total"
    )
    connection_blocked = delta(
        before,
        after,
        "dynamo_quic_response_connection_blocked_frames_tx_total",
    )
    stream_blocked = delta(
        before, after, "dynamo_quic_response_stream_blocked_frames_tx_total"
    )
    reconnects = delta(
        before, after, "dynamo_quic_response_reconnects_total"
    )
    connection_failures = delta(
        before, after, "dynamo_quic_response_connection_failures_total"
    )

    summary = {
        "window_deltas": {
            "response_payloads": payloads,
            "udp_tx_datagrams": datagrams,
            "quic_stream_frames": stream_frames,
            "udp_tx_ios": transmit_ios,
            "connection_flow_control_blocked": connection_blocked,
            "stream_flow_control_blocked": stream_blocked,
            "reconnects": reconnects,
            "connection_failures": connection_failures,
        },
        "response_payloads_per_udp_tx_datagram": ratio(payloads, datagrams),
        "quic_stream_frames_per_udp_tx_datagram": ratio(
            stream_frames, datagrams
        ),
        "udp_tx_datagrams_per_udp_tx_io": ratio(datagrams, transmit_ios),
        "passes_packet_density_gate": (
            datagrams > 0 and payloads / datagrams >= 2.0
        ),
        "requires_high_window_comparison": (
            connection_blocked > 0 or stream_blocked > 0
        ),
        "passes_connection_stability_gate": (
            reconnects == 0 and connection_failures == 0
        ),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
