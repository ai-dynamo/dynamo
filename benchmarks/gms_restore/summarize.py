# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive timing boundaries from recorded publication clocks and agent logs."""

import argparse
import json
import re
from pathlib import Path


def json_events(text):
    """Read JSON log records, including adjacent prints from loader threads."""
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        remaining = line.strip()
        while remaining.startswith("{"):
            event, end = decoder.raw_decode(remaining)
            yield event
            remaining = remaining[end:].lstrip()


def seconds(value):
    units = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1, "m": 60, "h": 3600}
    parts = re.findall(r"([0-9.]+)(ns|µs|us|ms|s|m|h)", value)
    if "".join(n + u for n, u in parts) != value:
        raise ValueError(value)
    return sum(float(n) * units[u] for n, u in parts)


def summarize(path):
    timing = json.loads((path / "timing.json").read_text())
    pub = json.loads((path / "publications.json").read_text())
    first = min(x["started_epoch"] for x in pub)
    last = max(x["published_epoch"] for x in pub)
    lines = (path / "agent.txt").read_text().splitlines()
    entry = next(
        x
        for x in reversed(lines)
        if "Restore timing summary" in x
        and timing.get("pod_name", "gms-v1-glm-restore-0928") in x
    )
    agent = json.loads(entry[entry.index("{") :])["restore"]
    plan = json.loads((path / "plan.json").read_text())
    result = {
        "case": path.name,
        "capture_id": plan["capture_id"],
        "weights_bytes": sum(
            x["aligned_size"] for r in plan["ranks"] for x in r["allocations"]
        ),
        "preload_span_s": last - first,
        "rank_preload_s": [x["elapsed_s"] for x in pub],
        "first_gms_start_to_ready_s": timing["ready_epoch"] - first,
        "pod_create_to_ready_s": timing["pod_create_to_ready_s"],
        "verification_orchestration_gap_s": timing["verification_orchestration_gap_s"],
        "agent_restore_s": seconds(agent["duration"]),
        "agent": agent,
        "inference_recorded": (path / "inference.json").exists(),
    }
    (path / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    for file in sorted(a.root.rglob("timing.json")):
        print(json.dumps(summarize(file.parent)))
