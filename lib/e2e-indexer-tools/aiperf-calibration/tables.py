#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render calibration summaries as a markdown table. Usage: tables.py RESULTS_DIR [PREFIX...]"""

from __future__ import annotations

import glob
import json
import os
import sys


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    root = sys.argv[1]
    prefixes = sys.argv[2:] or [""]
    rows = []
    for p in sorted(
        glob.glob(
            os.path.join(root, "*", os.environ.get("SUMMARY", "summary.local.txt"))
        )
    ):
        name = os.path.basename(os.path.dirname(p))
        if not any(name.startswith(x) for x in prefixes):
            continue
        s = json.loads(open(p).read().strip().splitlines()[-1])
        rows.append(s)

    def key(s):
        name = s["name"]
        tail = name.rsplit("-", 1)[-1]
        num = int(tail[1:]) if tail[:1] in "rc" and tail[1:].isdigit() else 0
        return (name.rsplit("-", 1)[0], num)

    rows.sort(key=key)
    print(
        "| trial | inst | offered/inst | achieved req/s | send ok | AIPerf cores (window) | CPU ms/req incl. drain | record drain s | c2s p99 ms | lat p50/p99 ms | ISL(bytes/4)/OSL | stub cores |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        per = s.get("per_instance", [])
        off = s.get("offered_per_inst")
        try:
            ok = s["total_rps"] >= 0.95 * float(off) * s["instances"]
        except (TypeError, ValueError, KeyError):
            ok = None
        c2s = max((p.get("c2s_p99_ms") or 0) for p in per) if per else None
        lat50 = max((p.get("lat_p50_ms") or 0) for p in per) if per else None
        lat99 = max((p.get("lat_p99_ms") or 0) for p in per) if per else None
        isl = per[0].get("isl_avg") if per else None
        osl = per[0].get("osl_avg") if per else None
        print(
            f"| {s['name']} | {s.get('instances')} | {off} | {fmt(s.get('total_rps'))} | "
            f"{'-' if ok is None else ('yes' if ok else 'NO')} | {fmt(s.get('total_aiperf_cores'), 2)} | "
            f"{fmt(s.get('cpu_ms_per_req_incl_drain'), 1)} | {fmt(s.get('max_rp_drain_s'), 0)} | {fmt(c2s, 1)} | {fmt(lat50, 1)}/{fmt(lat99, 1)} | "
            f"{fmt(isl, 0)}/{fmt(osl, 0)} | {fmt(s.get('stub_cores'), 2)} |"
        )


if __name__ == "__main__":
    main()
