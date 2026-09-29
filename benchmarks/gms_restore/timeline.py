# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render measured GMS and PageBroker phase intervals as a standalone Gantt chart."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from summarize import json_events, seconds

p = argparse.ArgumentParser()
p.add_argument("case", type=Path)
a = p.parse_args()
root = a.case
t = json.loads((root / "timing.json").read_text())
pub = json.loads((root / "publications.json").read_text())
zero = t["create_epoch"]
origin_name = (
    "DGD creation request"
    if t.get("deployment_mode") == "dgd"
    else "pod creation request"
)
entries = []
for line in (root / "agent.txt").read_text().splitlines():
    if "{" not in line:
        continue
    try:
        entries.append(
            (
                datetime.fromisoformat(
                    line.split("\t")[0].replace("Z", "+00:00")
                ).timestamp(),
                line,
                json.loads(line[line.index("{") :]),
            )
        )
    except (ValueError, json.JSONDecodeError):
        pass
start = max(x[0] for x in entries if "=== Starting external restore ===" in x[1])
entries = [x for x in entries if x[0] >= start]
summary = next(x for x in entries if "Restore timing summary" in x[1])
end = summary[0]
colors = {
    "init": "#4e79a7",
    "load": "#59a14f",
    "gate": "#bab0ac",
    "criu": "#b07aa1",
    "prepare": "#f28e2b",
    "transfer": "#76b7b2",
    "complete": "#e15759",
    "agent": "#d9d9d9",
}
rows = []


def row(label, segments):
    rows.append((label, segments))


startup_origin = zero
if t.get("deployment_mode") == "dgd":
    observed = [
        json.loads(line)["observed_epoch"]
        for line in (root / "virtual-watch.jsonl").read_text().splitlines()
        if "observed_epoch" in json.loads(line)
    ]
    if observed:
        startup_origin = min(observed)
        row("DGD → child Pod observed", [(zero, startup_origin, "gate")])
row("Pod startup", [(startup_origin, min(x["started_epoch"] for x in pub), "agent")])
watch_path = root / "host-watch.jsonl"
if watch_path.exists():
    for line in watch_path.read_text().splitlines():
        event = json.loads(line)
        main = next(
            (
                c
                for c in event.get("containers") or []
                if c["name"] == "main"
                and c.get("containerID")
                and "running" in c.get("state", {})
            ),
            None,
        )
        if main:
            container_start = datetime.fromisoformat(
                main["state"]["running"]["startedAt"].replace("Z", "+00:00")
            ).timestamp()
            row(
                "Main start → running status observed",
                [(container_start, event["observed_epoch"], "gate")],
            )
            break
for r in pub:
    events = list(json_events((root / f"gms-{r['rank']}.txt").read_text()))
    sockets = next((x["elapsed_s"] for x in events if x.get("event") == "sockets"), 0)
    s = r["started_epoch"]
    row(
        f"GMS rank {r['rank']}",
        [(s, s + sockets, "init"), (s + sockets, r["published_epoch"], "load")],
    )
last = max(x["published_epoch"] for x in pub)
gate_end = float((root / "gate.txt").read_text())
row("Publication validation", [(last, gate_end, "gate")])
row("Request → restore handler", [(t["trigger_epoch"], start, "gate")])
row("Engine restore operation", [(start, end, "agent")])
# CRIU completion lacks a dedicated return timestamp. Place its measured duration
# immediately before Native restore order, which follows the call in nsrestore.
criu_end = next(x[0] for x in entries if "Native restore order" in x[1])
duration = seconds(summary[2]["restore"]["phases"]["criu_restore"])
row("CRIU (position approximate)", [(criu_end - duration, criu_end, "criu")])
pids = {}
for at, line, data in entries:
    if "Native PageBroker phase" not in line:
        continue
    op = data["operation"].lower()
    pids.setdefault(data["pid"], []).append((at - data["duration"], at, op))
for pid, segments in pids.items():
    row(f"CUDA process {pid}", segments)
for line in (root / "main.txt").read_text().splitlines():
    if line.startswith("GMS_WAKE_GATE "):
        wake = json.loads(line.removeprefix("GMS_WAKE_GATE "))
        row(
            "Engine waits for published weights",
            [(wake["entered_epoch"], wake["passed_epoch"], "gate")],
        )
row("Wake / generation / Ready observed", [(end, t["ready_epoch"], "gate")])
fig, ax = plt.subplots(figsize=(15, 9), layout="constrained")
for y, (label, segments) in enumerate(rows):
    for s, e, c in segments:
        ax.broken_barh([(s - zero, e - s)], (y - 0.34, 0.68), facecolors=colors[c])
ax.set_yticks(range(len(rows)), [x[0] for x in rows])
ax.invert_yaxis()
ax.set_xlim(0, t["ready_epoch"] - zero + 0.3)
ax.set_xlabel(f"Seconds since {origin_name}")
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)
ax.axvline(t["trigger_epoch"] - zero, color="#555", linestyle="--", linewidth=0.8)
ax.axvline(last - zero, color="#59a14f", linestyle=":", linewidth=1)
fig.suptitle(
    f"GLM TP8 restore — {root.name}\n{t.get('storage', 'RAM-staged weights; PVC engine checkpoint')} | {t['ready_epoch'] - zero:.2f} s to coherent readiness",
    fontsize=15,
)
ax.legend(
    handles=[
        Patch(color=colors[k], label=v)
        for k, v in [
            ("init", "GMS initialization"),
            ("load", "Weight load + commit"),
            ("criu", "CRIU"),
            ("prepare", "CUDA PREPARE"),
            ("transfer", "Residual GPU transfer"),
            ("complete", "CUDA COMPLETE"),
            ("gate", "Orchestration / wake"),
        ]
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    fontsize=8,
)
fig.savefig(root / "timeline.svg")
fig.savefig(root / "timeline.png", dpi=150)
svg = (
    "\n".join(
        line.rstrip() for line in (root / "timeline.svg").read_text().splitlines()
    )
    + "\n"
)
(root / "timeline.svg").write_text(svg)
(root / "timeline-data.json").write_text(
    json.dumps(
        {
            "origin_epoch": zero,
            "rows": rows,
            "note": "CRIU placement approximate; durations measured. CUDA phases are broker request intervals, not exclusively driver call time.",
        },
        indent=2,
    )
)
svg = svg[svg.index("<svg") :]
ordering = (
    "GMS loads overlap CRIU and CUDA restoration. The captured engine waits for verified publication immediately before resuming weight use."
    if t.get("overlap")
    else "GMS loads run concurrently with each other and finish before the engine restore trigger."
)
(root / "timeline.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>Restore timeline</title><style>body{margin:12px;font:14px system-ui}svg{width:100%;height:auto}</style>'
    + svg
    + "<p>"
    + ordering
    + " The Snapshot agent daemon is already running before pod creation; the handler row is per-request dispatch, not daemon startup. Main-container status observation uses a remote API watch and includes observation latency; startedAt has one-second precision. PREPARE/TRANSFER/COMPLETE show measured broker request intervals. CRIU placement is approximate; its duration is measured. The final interval includes any publication wait, wake, generation and readiness observation.</p>"
)
