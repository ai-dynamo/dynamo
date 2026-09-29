# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate completed PVC trials and quantify contention by measured phase."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from summarize import json_events, seconds, summarize

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
a = p.parse_args()
rows = []
for file in sorted(a.root.glob("*/timing.json")):
    case = file.parent
    if not (case / "agent.txt").exists():
        continue
    summary = summarize(case)
    timing = json.loads(file.read_text())
    ph = summary["agent"]["phases"]
    init = []
    loads = []
    opens = 0
    direct_paths = set()
    for rank, elapsed in enumerate(summary["rank_preload_s"]):
        log = (case / f"gms-{rank}.txt").read_text()
        socket = next(
            json.loads(x)["elapsed_s"]
            for x in log.splitlines()
            if x.startswith('{"event": "sockets"')
        )
        init.append(socket)
        loads.append(elapsed - socket)
        opens += log.count('"o_direct": true')
        assert '"o_direct": false' not in log
        direct_paths.update(
            event["path"]
            for event in json_events(log)
            if event.get("event") == "artifact_open"
        )
    entries = []
    for line in (case / "agent.txt").read_text().splitlines():
        if "{" not in line:
            continue
        try:
            entries.append((line, json.loads(line[line.index("{") :])))
        except ValueError:
            pass
    start = max(
        i
        for i, (line, data) in enumerate(entries)
        if "=== Starting external restore ===" in line
    )
    phases = [d for l, d in entries[start:] if "Native PageBroker phase" in l]
    prep = sum(
        json.loads(d["report"])["native_prepare_seconds"]
        for d in phases
        if d["operation"] == "PREPARE"
    )
    transfers = [
        d["duration"]
        for d in phases
        if d["operation"] == "TRANSFER" and json.loads(d["report"]).get("bytes", 0) > 0
    ]
    row = {
        "case": case.name,
        "backend": timing["backend"],
        "workers": timing["workers"],
        "deployment_mode": timing.get("deployment_mode", "pod"),
        "runtime_discovery": timing.get("runtime_discovery", False),
        "dgd_shared_memory_size": timing.get("dgd_shared_memory_size") or "default",
        "chunk_mib": timing.get("chunk_mib", 16) or 0,
        "gms_cpu_request": timing.get("gms_cpu_request", 1),
        "gms_cpu_limit": timing.get("gms_cpu_limit", 8),
        "numa": timing["numa"],
        "qualified_pvc_mount": timing.get("qualified_pvc_mount", False),
        "isolated_pvc_transport": timing.get("isolated_pvc_transport", False),
        "overlap": timing.get("overlap", False),
        "early_trigger": timing.get("early_trigger", False),
        "agent_revision": timing.get("agent_revision", "006a3823"),
        "create_to_trigger_s": timing["trigger_epoch"] - timing["create_epoch"],
        "create_to_agent_s": datetime.fromisoformat(
            entries[start][0].split("\t")[0].replace("Z", "+00:00")
        ).timestamp()
        - timing["create_epoch"],
        "pod_s": timing["pod_create_to_ready_s"],
        "preload_span_s": summary["preload_span_s"],
        "server_init_mean_s": statistics.mean(init),
        "load_mean_s": statistics.mean(loads),
        "criu_s": seconds(ph["criu_restore"]),
        "cuda_phase_s": seconds(ph["cuda_restore"]),
        "native_prepare_sum_s": prep,
        "residual_transfer_mean_s": statistics.mean(transfers),
        "agent_s": summary["agent_restore_s"],
        "direct_file_opens": opens,
        "unique_direct_files": len(direct_paths),
    }
    assert opens > 0
    rows.append(row)
(a.root / "comparison.json").write_text(json.dumps(rows, indent=2))
print(
    "| Case | GMS all-rank span | CRIU | CUDA phase | Native restore calls | Pod to ready |"
)
print("|---|---:|---:|---:|---:|---:|")
for r in rows:
    print(
        "| "
        + r["case"]
        + " | "
        + " | ".join(
            f"{r[k]:.3f}"
            for k in [
                "preload_span_s",
                "criu_s",
                "cuda_phase_s",
                "native_prepare_sum_s",
                "pod_s",
            ]
        )
        + " |"
    )


def group_key(r):
    return (
        r["backend"],
        r["workers"],
        r["deployment_mode"],
        r["runtime_discovery"],
        r["dgd_shared_memory_size"],
        r["chunk_mib"],
        r["gms_cpu_request"],
        r["gms_cpu_limit"],
        r["numa"],
        r["qualified_pvc_mount"],
        r["isolated_pvc_transport"],
        r["overlap"],
        r["early_trigger"],
        r["agent_revision"],
    )


for key in sorted({group_key(r) for r in rows}):
    group = [r for r in rows if group_key(r) == key]
    print(
        key,
        "n=",
        len(group),
        {
            k: round(statistics.mean(r[k] for r in group), 3)
            for k in group[0]
            if k.endswith("_s")
        },
    )
