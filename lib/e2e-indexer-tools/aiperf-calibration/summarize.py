#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize one calibration trial directory written by run_trial.sh (stdlib only)."""

from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys


def pick(d: dict, key: str, stat: str = "avg"):
    v = d.get(key)
    if isinstance(v, dict):
        return v.get(stat)
    return v


def main() -> None:
    out = sys.argv[1]
    meta = json.load(open(os.path.join(out, "meta.json")))
    samples = [
        json.loads(line)
        for line in open(os.path.join(out, "cpu.jsonl"))
        if line.strip()
    ]
    inst = []
    for path in sorted(
        glob.glob(
            os.path.join(out, "i*", "art", "**", "profile_export_aiperf.json"),
            recursive=True,
        )
    ):
        d = json.load(open(path))
        lab = "aiperf" + path.split(os.sep + "i", 1)[1].split(os.sep, 1)[0]
        start = dt.datetime.fromisoformat(d["start_time"]).timestamp()
        end = dt.datetime.fromisoformat(d["end_time"]).timestamp()
        inst.append((lab, d, start, end))
    if not inst:
        print(
            json.dumps(
                {
                    "name": meta["name"],
                    "error": "no profile_export_aiperf.json",
                    "rc": meta["rc"],
                }
            )
        )
        return
    w0 = max(s for _, _, s, _ in inst)
    w1 = min(e for _, _, _, e in inst)
    if w1 - w0 > 20:
        w0, w1 = w0 + 5, w1 - 2
    win = [s for s in samples if w0 <= s["t"] <= w1]
    n = max(1, len(win))

    def avg(key: str) -> float:
        return sum(s.get(key, 0.0) for s in win) / n

    def by_name(key: str) -> dict[str, float]:
        agg: dict[str, float] = {}
        for s in win:
            for k, v in s.get(key + ".by_name", {}).items():
                agg[k] = agg.get(k, 0.0) + v / n
        return {
            k: round(v, 2)
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1])
            if v >= 0.05
        }

    res = {
        "name": meta["name"],
        "rc": meta["rc"],
        "offered_per_inst": meta.get("offered"),
        "instances": len(inst),
        "window_s": round(w1 - w0, 1),
        "window_samples": len(win),
        "stub_cores": round(avg("stub"), 2),
        "stub_procs": len(meta["stub_cpus"].split(",")),
        "node_busy_cores": round(avg("node_busy_cores"), 2),
    }
    tot_rps = 0.0
    tot_cores = 0.0
    tot_work = 0.0
    tot_req = 0
    per = []
    for lab, d, start, end in inst:
        rps = pick(d, "request_throughput") or 0.0
        cores = avg(lab)
        # Whole-lifetime CPU (startup + dataset build + drain), from the sampler.
        life = sum(s.get(lab, 0.0) for s in samples)
        # Post-run record backlog: seconds after the profile end while record processors stay busy.
        drain = 0.0
        for smp in samples:
            if smp["t"] <= end:
                continue
            rp = sum(
                v
                for k, v in smp.get(lab + ".by_name", {}).items()
                if "record_processor" in k
            )
            if rp >= 0.5:
                drain = smp["t"] - end
        # CPU-seconds from profile start through the record drain, per completed request.
        work = 0.0
        prev_t = None
        for smp in samples:
            if prev_t is not None and start <= smp["t"] <= end + drain + 1.5:
                work += smp.get(lab, 0.0) * (smp["t"] - prev_t)
            prev_t = smp["t"]
        nreq = pick(d, "request_count") or 0
        tot_rps += rps
        tot_cores += cores
        tot_work += work
        tot_req += nreq
        per.append(
            {
                "inst": lab,
                "rps": round(rps, 1),
                "req": pick(d, "request_count"),
                "err_pct": pick(d, "request_error_rate"),
                "dur_s": round(end - start, 1),
                "cores": round(cores, 2),
                "cpu_ms_per_req": round(cores / rps * 1000, 3) if rps else None,
                "life_cpu_s": round(life, 1),
                "rp_drain_s": round(drain, 1),
                "cpu_ms_per_req_incl_drain": round(work / nreq * 1000, 3)
                if nreq
                else None,
                "c2s_p50_ms": pick(d, "credit_to_start_latency", "p50"),
                "c2s_p99_ms": pick(d, "credit_to_start_latency", "p99"),
                "lat_p50_ms": pick(d, "request_latency", "p50"),
                "lat_p99_ms": pick(d, "request_latency", "p99"),
                "ttft_p99_ms": pick(d, "time_to_first_token", "p99"),
                "isl_avg": pick(d, "input_sequence_length"),
                "osl_avg": pick(d, "output_sequence_length"),
                "eff_conc": pick(d, "effective_concurrency"),
                "cancelled": d.get("was_cancelled"),
                "by_name": by_name(lab),
            }
        )
    res["total_rps"] = round(tot_rps, 1)
    res["total_aiperf_cores"] = round(tot_cores, 2)
    res["cpu_ms_per_req"] = round(tot_cores / tot_rps * 1000, 3) if tot_rps else None
    res["cpu_ms_per_req_incl_drain"] = (
        round(tot_work / tot_req * 1000, 3) if tot_req else None
    )
    res["max_rp_drain_s"] = max((p["rp_drain_s"] for p in per), default=None)
    res["per_instance"] = per
    print(json.dumps(res))


if __name__ == "__main__":
    main()
