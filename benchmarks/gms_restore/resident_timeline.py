# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render cold/resident GMS restore evidence, preserving measured clock boundaries."""

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from summarize import json_events, seconds

COLORS = {
    "orchestration": "#a9adb8",
    "discovery": "#c09549",
    "init": "#6384b3",
    "context": "#8064ad",
    "allocation": "#db9e45",
    "buffers": "#7eacd0",
    "read": "#399f68",
    "publication": "#298287",
    "criu": "#b175ab",
    "prepare": "#e38a35",
    "transfer": "#68bdb5",
    "complete": "#d65859",
    "wait": "#c6b8a0",
    "inference": "#607d8b",
    "idle": "#e4e8ee",
    "operation": "#e4e8ee",
    "observation": "#d6dce5",
    "dispatch": "#a1bbc3",
}
LABELS = {
    "orchestration": "Orchestration",
    "discovery": "Container discovery",
    "init": "Imports / server",
    "context": "CUDA context",
    "allocation": "Allocate / map",
    "buffers": "Pinned buffers",
    "read": "Payload load window",
    "publication": "Publish / validate",
    "criu": "CRIU",
    "prepare": "CUDA PREPARE",
    "transfer": "Residual transfer",
    "complete": "CUDA COMPLETE",
    "wait": "Wait for weights",
    "inference": "Wake / inference / readiness",
    "idle": "Prestarted and waiting",
    "operation": "Snapshot operation",
    "observation": "API status observation",
    "dispatch": "Dispatch / open payload",
}
NOTES = [
    "t0 is the client DGD creation request. Ready means workload Pod readiness after coherent generation; it is not DGD Ready.",
    "Resident GMS initialization occurs before t0. The prewarm panel shows its cost separately; payload reads and weight allocations begin only after the load trigger.",
    "Main container start uses CRI nanoseconds where available. Kubernetes startedAt fallback has one-second precision. API-watch receipt is an observation, not the container start or agent input time.",
    "Agent startup milestones use at_unix_nano from the node. Cross-machine clocks and remote watch receipt introduce uncertainty; durations within one process are preferable.",
    "CRIU placement ends at Native restore order and is approximate; its measured duration is exact to the logged precision. CUDA bars are PageBroker request intervals, not just driver-call time.",
    "Per-rank spans are not aggregate throughput: concurrent rank starts redistribute shared bandwidth. Compare earliest read across all ranks to the latest transfer-routine completion, using the same total payload bytes.",
    "Active start means Python worker startup for a cold rank, and trigger observation for an already-online resident rank; these intentionally measure the initialization moved outside the request timer.",
    "The payload load window begins at the first O_DIRECT read and ends when every transfer lane returns. Cold lanes close/free their pinned slots and FDs before returning; resident lanes drain copies and retain their rings. Thus 448 GiB/window is an effective load rate, not raw storage bandwidth or an exact last-GPU-copy timestamp. Allocation/import and pinned-buffer setup are distinct; setup can overlap early reads.",
    "The final wake/inference/readiness segment includes correctness generation and readiness observation. Native GMS wake logs have only coarse wall-clock timestamps, so their durations are reported without invented exact placement.",
]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def iso_epoch(value):
    # Preserve all nine fractional digits before converting to plotting seconds.
    match = re.fullmatch(r"(.+?)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)", value)
    if not match:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
    base, fraction, zone = match.groups()
    ns = (
        int(datetime.fromisoformat(base + zone.replace("Z", "+00:00")).timestamp())
        * 10**9
    )
    ns += int((fraction or "").ljust(9, "0")[:9] or "0")
    return ns / 10**9


def cri_epoch(value):
    if isinstance(value, str) and "T" in value:
        return iso_epoch(value)
    return int(value or 0) / 10**9


def make_segment(start, end, kind, detail=""):
    if start is None or end is None or end < start:
        return None
    return {"start": start, "end": end, "kind": kind, "detail": detail or LABELS[kind]}


def make_row(label, *segments):
    return {
        "label": label,
        "segments": [segment for segment in segments if segment is not None],
    }


def agent_entries(path, zero, ready, pod):
    output = []
    for line in lines(path):
        if "{" not in line:
            continue
        try:
            data = json.loads(line[line.index("{") :])
            at = data.get("at_unix_nano", 0) / 10**9 or iso_epoch(line.split("\t")[0])
        except (ValueError, TypeError):
            continue
        if not zero - 1 <= at <= ready + 1:
            continue
        if data.get("pod") and data["pod"].split("/")[-1] != pod:
            continue
        output.append((at, line, data))
    return output


def watch_info(path):
    values = [
        value
        for line in lines(path)
        if line.strip()
        for value in [json.loads(line)]
        if "observed_epoch" in value
    ]
    starts = []
    observed = []
    for value in values:
        for container in value.get("containers") or []:
            if container["name"] != "main" or not container.get("containerID"):
                continue
            running = container.get("state", {}).get("running")
            if running:
                starts.append(iso_epoch(running["startedAt"]))
                observed.append(value["observed_epoch"])
    return {
        "pod_observed": min(
            (value["observed_epoch"] for value in values), default=None
        ),
        "main_started": min(starts, default=None),
        "main_running_observed": min(observed, default=None),
    }


def phase_pair(events, first, last, kind, detail=None):
    starts = [
        event["epoch"]
        for event in events
        if event.get("event") == first and "epoch" in event
    ]
    ends = [
        event["epoch"]
        for event in events
        if event.get("event") == last and "epoch" in event
    ]
    return (
        make_segment(min(starts), max(ends), kind, detail or "")
        if starts and ends
        else None
    )


def gms_rank(case, publication, zero):
    rank = publication["rank"]
    events = list(json_events("\n".join(lines(case / f"gms-{rank}.txt"))))
    by_name = {}
    for event in events:
        if "epoch" in event:
            by_name.setdefault(event.get("event"), []).append(event["epoch"])

    def first(name, fallback=None):
        return min(by_name.get(name, []), default=fallback)

    def last(name, fallback=None):
        return max(by_name.get(name, []), default=fallback)

    started, published = publication["started_epoch"], publication["published_epoch"]
    daemon = publication.get("daemon_started_epoch", started)
    online = publication.get("online_epoch")
    detailed = bool(by_name.get("first_read_start"))
    sockets = first("sockets")
    if sockets is None:
        sockets = next(
            (daemon + e["elapsed_s"] for e in events if e.get("event") == "sockets"),
            daemon,
        )
    import_start, allocation_end = (
        first("allocation_import_start"),
        last("allocation_import_complete"),
    )
    read_start, transfer_end = first("first_read_start"), last("transfer_complete")
    load_start = first("load_weights_start", started)
    leader = [
        make_row(
            "Python / backend imports",
            phase_pair(events, "imports_start", "imports_complete", "init"),
        ),
        make_row(
            "Driver cuInit",
            phase_pair(events, "cuinit_start", "cuinit_complete", "context"),
        ),
        make_row(
            "GMS server setup",
            make_segment(first("server_init_start"), sockets, "init"),
        ),
        make_row(
            "Loader primary context",
            phase_pair(
                events, "loader_context_start", "loader_context_ready", "context"
            ),
        ),
        make_row(
            "Pinned rings (all lanes)",
            phase_pair(events, "lane_buffers_start", "lane_ready", "buffers"),
        ),
        make_row(
            "V1 metadata / session",
            make_segment(load_start, import_start, "orchestration"),
        ),
        make_row(
            "Exact IDs: allocate / import",
            make_segment(import_start, allocation_end, "allocation"),
        ),
        make_row(
            "First read → transfer routine done",
            make_segment(read_start, transfer_end, "read"),
        ),
        make_row(
            "Unmap / commit / publication",
            make_segment(transfer_end, published, "publication"),
        ),
    ]
    if detailed:
        overview = make_row(
            f"GMS rank {rank}",
            make_segment(started, import_start, "init"),
            make_segment(import_start, allocation_end, "allocation"),
            make_segment(
                allocation_end,
                read_start,
                "dispatch" if online is not None else "buffers",
            ),
            make_segment(read_start, transfer_end, "read"),
            make_segment(transfer_end, published, "publication"),
        )
    else:
        overview = make_row(
            f"GMS rank {rank}",
            make_segment(started, sockets, "init"),
            make_segment(
                sockets,
                published,
                "read",
                "Legacy load interval; no detailed read timestamps",
            ),
        )
    prewarm = (
        make_row(
            f"GMS rank {rank}",
            make_segment(daemon, first("loader_context_start", online), "init"),
            phase_pair(
                events, "loader_context_start", "loader_context_ready", "context"
            ),
            phase_pair(events, "lane_buffers_start", "lane_ready", "buffers"),
            make_segment(online, zero, "idle"),
        )
        if online is not None
        else None
    )
    metrics = {
        "rank": rank,
        "detailed": detailed,
        "load_started_offset_s": started - zero,
        "publication_offset_s": published - zero,
        "daemon_started_offset_s": daemon - zero,
        "transfer_complete_offset_s": None
        if transfer_end is None
        else transfer_end - zero,
        "transfer_bytes": max(
            (
                event.get("bytes", 0)
                for event in events
                if event.get("event") == "transfer_complete"
            ),
            default=0,
        ),
    }
    for key, begin, end in [
        ("prewarm_s", daemon, online),
        ("cuinit_s", first("cuinit_start"), last("cuinit_complete")),
        (
            "loader_context_s",
            first("loader_context_start"),
            last("loader_context_ready"),
        ),
        ("imports_s", first("imports_start"), last("imports_complete")),
        ("pin_rings_s", first("lane_buffers_start"), last("lane_ready")),
        ("allocation_import_s", import_start, allocation_end),
        ("metadata_s", load_start, import_start),
        ("transfer_s", read_start, transfer_end),
        ("transfer_to_publication_s", transfer_end, published),
        ("active_start_to_first_read_s", started, read_start),
        ("dgd_to_first_read_s", zero, read_start),
    ]:
        metrics[key] = None if begin is None or end is None else end - begin
    metrics["direct_files"] = sorted(
        {
            e["path"]
            for e in events
            if e.get("event") == "artifact_open" and e.get("o_direct")
        }
    )
    return {
        "rank": rank,
        "overview": overview,
        "prewarm": prewarm,
        "detail": leader,
        "metrics": metrics,
    }


def experiment_group(case, timing, ranks):
    resident = any(rank["prewarm"] for rank in ranks)
    group = "resident" if resident else "cold"
    preinstalled = bool(timing.get("preinstalled_cuda_host_path"))
    network = bool(timing.get("runtime_network_discovery")) or "network" in case.name
    if preinstalled:
        group += "_preinstalled"
    if network:
        group += "_network"
    if (
        not preinstalled
        and not network
        and not re.fullmatch(r"daemonset-(?:cold|resident)-[123]", case.name)
    ):
        group += "_followup_control"
    return group


def aggregate(rows):
    numeric = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key.endswith(("_s", "GBps", "GiBps")) and isinstance(value, (int, float))
        }
    )
    result = {
        "n": len(rows),
        "cases": [row["case"] for row in rows],
        "mean": {},
        "range": {},
    }
    for key in numeric:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        result["mean"][key] = statistics.mean(values)
        result["range"][key] = [min(values), max(values)]
    return result


def parse_case(case):
    timing = read_json(case / "timing.json")
    publications = read_json(case / "publications.json")
    zero, ready = timing["create_epoch"], timing["ready_epoch"]
    entries = agent_entries(case / "agent.txt", zero, ready, timing.get("pod_name"))
    handler = next(
        at for at, line, _ in entries if "=== Starting external restore ===" in line
    )
    operation_end, _, summary = next(
        item for item in entries if "Restore timing summary" in item[1]
    )
    restore = summary["restore"]
    ranks = [
        gms_rank(case, publication, zero)
        for publication in sorted(publications, key=lambda p: p["rank"])
    ]
    host, virtual = (
        watch_info(case / "host-watch.jsonl"),
        watch_info(case / "virtual-watch.jsonl"),
    )
    cri_records = read_json(case / "cri-starts.json", [])
    main_cri = [
        r
        for r in cri_records
        if r.get("name", r.get("metadata", {}).get("name")) == "main"
        and cri_epoch(r.get("startedAt", 0)) > 0
    ]
    main_start = min(
        (cri_epoch(r["startedAt"]) for r in main_cri), default=host["main_started"]
    )
    precision = (
        "CRI nanoseconds" if main_cri else "Kubernetes startedAt (1 s precision)"
    )
    milestones = [
        (at, data)
        for at, line, data in entries
        if "Restore startup milestone" in line and at <= handler
    ]
    hot = [
        make_row(
            "DGD request → child Pod observed",
            make_segment(zero, virtual["pod_observed"], "orchestration"),
        )
    ]
    first_queue = min(
        (at for at, data in milestones if data.get("stage") == "queue_add"),
        default=None,
    )
    first_process = min(
        (at for at, data in milestones if data.get("stage") == "queue_process"),
        default=None,
    )
    if first_queue:
        hot.append(
            make_row(
                "Request → agent informer event",
                make_segment(zero, first_queue, "orchestration"),
            )
        )
        hot.append(
            make_row(
                "Informer event → queue processing",
                make_segment(first_queue, first_process, "orchestration"),
            )
        )
    descriptions = {
        "preflight": "Preflight",
        "finalizer": "Pod finalizer API update",
        "snapshot_get": "Snapshot API/cache lookup",
        "content_get": "Content API/cache lookup",
        "target_validation": "Restore-target validation",
        "artifact_validation": "Checkpoint artifact validation",
        "compatibility": "GPU / driver compatibility",
        "container_resolution": "Find running main container",
        "status_apply": "Restore status API update",
    }
    paired = []
    for key, label in descriptions.items():
        segments, pending = [], None
        for at, data in milestones:
            stage = data.get("stage")
            if stage == key + "_start":
                pending = at
            elif stage == key + "_done":
                begin = (
                    pending
                    if pending is not None
                    else at - data.get("elapsed_seconds", 0)
                )
                segment = make_segment(
                    begin,
                    at,
                    "discovery" if key == "container_resolution" else "orchestration",
                    json.dumps(data, sort_keys=True),
                )
                segments.append(segment)
                paired.append((key, begin, at, data))
                pending = None
        if segments:
            hot.append(make_row(label, *segments))
    event_done = max(
        (at for at, data in milestones if data.get("stage") == "restore_event_done"),
        default=None,
    )
    resolution_done = max(
        (
            at
            for at, data in milestones
            if data.get("stage") == "container_resolution_done"
        ),
        default=None,
    )
    if resolution_done:
        hot.append(
            make_row(
                "Container found → handler",
                make_segment(resolution_done, handler, "orchestration"),
            )
        )
    if event_done:
        hot.append(
            make_row(
                "Event emitted → handler",
                make_segment(event_done, handler, "orchestration"),
            )
        )
    hot.extend(
        [
            make_row(
                f"Main start → handler ({precision})",
                make_segment(main_start, handler, "discovery"),
            ),
            make_row(
                "Main start → host status observed",
                make_segment(main_start, host["main_running_observed"], "observation"),
            ),
            make_row(
                "Main start → virtual status observed",
                make_segment(
                    main_start, virtual["main_running_observed"], "observation"
                ),
            ),
        ]
    )
    main_rows = [
        make_row(
            "Request → restore handler", make_segment(zero, handler, "orchestration")
        )
    ]
    if main_start is not None:
        main_rows.append(
            make_row(
                f"Main start → handler ({precision})",
                make_segment(main_start, handler, "discovery"),
            )
        )
    main_rows += [r["overview"] for r in ranks]
    last_publication = max(p["published_epoch"] for p in publications)
    gate_lines = lines(case / "gate.txt")
    gate = float(gate_lines[0]) if gate_lines else None
    main_rows.append(
        make_row(
            "Verify all eight artifacts",
            make_segment(last_publication, gate, "publication"),
        )
    )
    main_rows.append(
        make_row(
            "Snapshot restore operation",
            make_segment(handler, operation_end, "operation"),
        )
    )
    criu_end = next(
        (at for at, line, _ in entries if "Native restore order" in line), None
    )
    criu_duration = seconds(restore["phases"]["criu_restore"])
    main_rows.append(
        make_row(
            "CRIU (placement approximate)",
            make_segment(
                None if criu_end is None else criu_end - criu_duration, criu_end, "criu"
            ),
        )
    )
    pids = {}
    for at, line, data in entries:
        if "Native PageBroker phase" in line:
            operation = data["operation"].lower()
            if operation in COLORS:
                pids.setdefault(data["pid"], []).append(
                    make_segment(at - data["duration"], at, operation)
                )
    main_rows += [
        make_row(f"CUDA process {pid}", *segments) for pid, segments in pids.items()
    ]
    wake = next(
        (
            e
            for e in (
                json.loads(line.removeprefix("GMS_WAKE_GATE "))
                for line in lines(case / "main.txt")
                if line.startswith("GMS_WAKE_GATE ")
            )
            if "entered_epoch" in e and zero <= e["entered_epoch"] <= ready
        ),
        None,
    )
    if wake:
        main_rows.extend(
            [
                make_row(
                    "Restore return → app gate",
                    make_segment(operation_end, wake["entered_epoch"], "orchestration"),
                ),
                make_row(
                    "Engine waits for weights",
                    make_segment(wake["entered_epoch"], wake["passed_epoch"], "wait"),
                ),
                make_row(
                    "Wake / generation / Ready observed",
                    make_segment(wake["passed_epoch"], ready, "inference"),
                ),
            ]
        )
    else:
        main_rows.append(
            make_row(
                "Wake / generation / Ready observed",
                make_segment(operation_end, ready, "inference"),
            )
        )
    rank_metrics = [r["metrics"] for r in ranks]
    metrics = {
        "case": case.name,
        "group": experiment_group(case, timing, ranks),
        "cohort": "primary"
        if re.fullmatch(r"daemonset-(?:cold|resident)-[123]", case.name)
        else "followup",
        "preinstalled_cuda_host_path": timing.get("preinstalled_cuda_host_path"),
        "dgd_to_ready_s": ready - zero,
        "dgd_to_handler_s": handler - zero,
        "main_start_offset_s": None if main_start is None else main_start - zero,
        "main_start_precision": precision,
        "main_start_to_handler_s": None if main_start is None else handler - main_start,
        "main_running_host_observed_offset_s": None
        if host["main_running_observed"] is None
        else host["main_running_observed"] - zero,
        "restore_operation_s": seconds(restore["duration"]),
        "criu_s": criu_duration,
        "cuda_s": seconds(restore["phases"]["cuda_restore"]),
        "all_published_offset_s": last_publication - zero,
        "first_read_offset_s": min(
            (
                r["dgd_to_first_read_s"]
                for r in rank_metrics
                if r["dgd_to_first_read_s"] is not None
            ),
            default=None,
        ),
        "wake_gate_wait_s": None
        if not wake
        else wake["passed_epoch"] - wake["entered_epoch"],
        "gate_passed_to_ready_s": None if not wake else ready - wake["passed_epoch"],
        "gms_wake_logged_durations_s": [
            float(m)
            for m in re.findall(
                r"GMS V1 wake complete.*?total_elapsed=([\d.]+)s",
                "\n".join(lines(case / "main.txt")),
            )
        ],
        "rank_metrics": rank_metrics,
        "startup_stages": [
            {
                "stage": key,
                "start_offset_s": begin - zero,
                "end_offset_s": end - zero,
                "duration_s": end - begin,
                "metadata": data,
            }
            for key, begin, end, data in paired
        ],
        "startup_milestones": [
            {"offset_s": at - zero, **data} for at, data in milestones
        ],
        "unique_direct_payload_files": len(
            {path for rank in rank_metrics for path in rank["direct_files"]}
        ),
    }
    read_starts = [
        rank["dgd_to_first_read_s"]
        for rank in rank_metrics
        if rank["dgd_to_first_read_s"] is not None
    ]
    transfer_ends = [
        rank["transfer_complete_offset_s"]
        for rank in rank_metrics
        if rank["transfer_complete_offset_s"] is not None
    ]
    complete = len(read_starts) == len(transfer_ends) == len(rank_metrics)
    payload_bytes = sum(rank["transfer_bytes"] for rank in rank_metrics)
    span = max(transfer_ends) - min(read_starts) if complete else None
    metrics.update(
        {
            "payload_transfer_bytes": payload_bytes,
            "all_transfer_complete_offset_s": max(transfer_ends) if complete else None,
            "aggregate_read_to_transfer_complete_s": span,
            "effective_payload_GBps": payload_bytes / span / 1e9
            if span and payload_bytes
            else None,
            "effective_payload_GiBps": payload_bytes / span / 1024**3
            if span and payload_bytes
            else None,
        }
    )
    for key in [
        "prewarm_s",
        "cuinit_s",
        "loader_context_s",
        "imports_s",
        "pin_rings_s",
        "allocation_import_s",
        "metadata_s",
        "transfer_s",
        "transfer_to_publication_s",
        "active_start_to_first_read_s",
    ]:
        values = [r[key] for r in rank_metrics if r.get(key) is not None]
        metrics["rank_mean_" + key] = statistics.mean(values) if values else None
    return {
        "case": case.name,
        "origin_epoch": zero,
        "ready_epoch": ready,
        "handler_epoch": handler,
        "metrics": metrics,
        "rows": main_rows,
        "hot_path": hot,
        "prewarm": [r["prewarm"] for r in ranks if r["prewarm"]],
        "ranks": ranks,
        "notes": NOTES,
    }


def draw(path, data, key, suffix, xmin=None, xmax=None):
    rows = [row for row in data[key] if row["segments"]]
    if not rows:
        return
    zero = data["origin_epoch"]
    fig, ax = plt.subplots(
        figsize=(16, max(4, 0.34 * len(rows) + 2.7)), layout="constrained"
    )
    for index, row in enumerate(rows):
        for segment in row["segments"]:
            start, end = segment["start"] - zero, segment["end"] - zero
            ax.broken_barh(
                [(start, max(end - start, 0.003))],
                (index - 0.33, 0.66),
                facecolors=COLORS[segment["kind"]],
            )
    ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.axvline(0, color="#273447", linewidth=0.8)
    if key != "prewarm":
        ax.axvline(
            data["handler_epoch"] - zero, color="#997333", linestyle="--", linewidth=0.8
        )
    if key == "rows":
        ax.axvline(
            data["metrics"]["all_published_offset_s"],
            color=COLORS["read"],
            linestyle=":",
            linewidth=0.8,
        )
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlabel("Seconds relative to DGD creation request (t0)")
    title = {
        "rows": "Restore timeline",
        "hot_path": "Request → restore handler detail",
        "prewarm": "GMS prewarm outside the restore timer",
    }[key]
    fig.suptitle(
        f"{data['case']} — {title}\n{data['metrics']['dgd_to_ready_s']:.3f} s to coherent readiness",
        fontsize=13,
    )
    kinds = list(dict.fromkeys(s["kind"] for row in rows for s in row["segments"]))
    ax.legend(
        handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in kinds],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=4,
        fontsize=8,
    )
    fig.savefig(path / f"{suffix}.svg")
    fig.savefig(path / f"{suffix}.png", dpi=150)
    plt.close(fig)
    svgpath = path / f"{suffix}.svg"
    svgpath.write_text(
        "\n".join(line.rstrip() for line in svgpath.read_text().splitlines()) + "\n"
    )


HTML = r"""<!doctype html><meta charset="utf-8"><title>Resident GMS restore timelines</title>
<style>body{margin:20px;font:14px system-ui;background:#f6f8fb;color:#253247}h1{font-size:22px}button,select{font:inherit;padding:7px;margin:4px;border:1px solid #cad1dd;border-radius:5px;background:white}button.active{background:#273e60;color:white}.card{background:white;border:1px solid #dde3eb;border-radius:8px;padding:14px;margin:14px 0;overflow:auto}.metrics{display:flex;flex-wrap:wrap;gap:25px}.metrics strong{font-size:23px;display:block}svg{min-width:850px;width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:7px;border-bottom:1px solid #e6eaf0}.legend span{display:inline-block;margin:5px 12px 5px 0}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px}p{line-height:1.5}.muted{color:#657184}#tip{position:fixed;background:#172437;color:white;padding:9px;max-width:470px;border-radius:5px;pointer-events:none;display:none;white-space:pre-wrap;z-index:5}</style>
<h1>GMS DaemonSet: prestarted initialization, timed PVC/O_DIRECT payload load</h1>
<div><label>Trial <select id="case"></select></label><button data-mode="rows">Full restore</button><button data-mode="hot_path">Request → handler</button><button data-mode="prewarm">Prewarm before t0</button><button data-mode="rank">Rank detail</button><label id="ranklabel">Rank <select id="rank"></select></label></div>
<div id="metrics" class="card metrics"></div><div class="card"><div id="caption"></div><div id="chart"></div><div id="legend" class="legend"></div></div><div class="card"><h3>Trial comparison</h3><div id="comparison"></div></div><details class="card"><summary>Measurement boundaries and caveats</summary><div id="notes"></div></details><div id="tip"></div>
<script>const DATA=__DATA__,COLORS=__COLORS__,LABELS=__LABELS__;let mode='rows';const selector=document.querySelector('#case'),rankSelector=document.querySelector('#rank');
DATA.forEach((d,i)=>selector.add(new Option(d.case,i)));for(let i=0;i<8;i++)rankSelector.add(new Option(i,i));
const fmt=v=>v==null?'—':v.toFixed(3)+' s';const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){const d=DATA[selector.value],m=d.metrics,z=d.origin_epoch;document.querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));document.querySelector('#ranklabel').style.display=mode==='rank'?'inline':'none';document.querySelector('#metrics').innerHTML=[['DGD → Ready',m.dgd_to_ready_s],['DGD → handler',m.dgd_to_handler_s],['Main start → handler',m.main_start_to_handler_s],['First payload read',m.first_read_offset_s],['Wait for weights',m.wake_gate_wait_s]].map(([k,v])=>`<div>${k}<strong>${fmt(v)}</strong></div>`).join('');
let rows=mode==='rank'?d.ranks[rankSelector.value].detail:d[mode];rows=rows.filter(r=>r.segments.length);let min=0,max=d.ready_epoch-z;
if(mode==='prewarm'||mode==='rank'){min=Math.min(0,...rows.flatMap(r=>r.segments.map(s=>s.start-z)));max=Math.max(0,...rows.flatMap(r=>r.segments.map(s=>s.end-z)));}if(mode==='hot_path')max=Math.max(d.handler_epoch-z,...rows.flatMap(r=>r.segments.map(s=>s.end-z)))+.2;
if(max-min<.1)max=min+.1;const width=1350,left=370,right=30,top=32,rowH=29,height=top+rows.length*rowH+55,x=t=>left+(t-min)/(max-min)*(width-left-right);let svg=`<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;for(let i=0;i<=10;i++){let t=min+(max-min)*i/10;svg+=`<line x1="${x(t)}" y1="22" x2="${x(t)}" y2="${height-42}" stroke="#e7ebf1"/><text x="${x(t)}" y="${height-22}" text-anchor="middle" fill="#68768a" font-size="12">${t.toFixed(2)}</text>`;}
rows.forEach((r,i)=>{const y=top+i*rowH;svg+=`<text x="${left-12}" y="${y+15}" text-anchor="end" font-size="12" fill="#253247">${esc(r.label)}</text>`;r.segments.forEach(s=>{let start=s.start-z,end=s.end-z,detail=`${r.label}\n${LABELS[s.kind]}\n${start.toFixed(6)} → ${end.toFixed(6)} s\nDuration ${(end-start).toFixed(6)} s\n${s.detail}`;svg+=`<rect class="bar" data-tip="${esc(detail)}" x="${x(start)}" y="${y}" width="${Math.max(.8,x(end)-x(start))}" height="21" fill="${COLORS[s.kind]}" rx="2"/>`;});});svg+=`<line x1="${x(0)}" y1="20" x2="${x(0)}" y2="${height-42}" stroke="#243c60" stroke-dasharray="4 3"/></svg>`;document.querySelector('#chart').innerHTML=svg;
const used=[...new Set(rows.flatMap(r=>r.segments.map(s=>s.kind)))];document.querySelector('#legend').innerHTML=used.map(k=>`<span><i class="swatch" style="background:${COLORS[k]}"></i>${LABELS[k]}</span>`).join('');document.querySelector('#caption').innerHTML=`<strong>${esc(d.case)}</strong><p class="muted">${mode==='prewarm'?'Negative times are initialization outside the restore timer. The pale interval is the already-warm service waiting for the workload request.':mode==='hot_path'?'Same-clock agent stages split dispatch overhead. API status observation can arrive after the restore handler has already begun. Main start source: '+esc(m.main_start_precision):mode==='rank'?'Detailed phase spans can overlap: lanes initialize and begin reads independently. Negative times belong to resident prewarm.':'All bars share t0 = DGD creation request. Hover a bar for exact offsets and its measurement boundary.'}</p>`;
const tip=document.querySelector('#tip');document.querySelectorAll('.bar').forEach(el=>{el.onmousemove=e=>{tip.textContent=el.dataset.tip;tip.style.display='block';tip.style.left=Math.min(e.clientX+12,window.innerWidth-490)+'px';tip.style.top=Math.min(e.clientY+12,window.innerHeight-150)+'px';};el.onmouseleave=()=>tip.style.display='none';});}
document.querySelectorAll('button').forEach(b=>b.onclick=()=>{mode=b.dataset.mode;render();});selector.onchange=render;rankSelector.onchange=render;document.querySelector('#notes').innerHTML=DATA[0].notes.map(n=>'<p>'+esc(n)+'</p>').join('');const cols=[['Trial','case'],['Cohort','group'],['Ready','dgd_to_ready_s'],['Handler','dgd_to_handler_s'],['First read','first_read_offset_s'],['All published','all_published_offset_s'],['CRIU','criu_s'],['CUDA','cuda_s'],['Weight wait','wake_gate_wait_s']];document.querySelector('#comparison').innerHTML='<table><tr>'+cols.map(c=>'<th>'+c[0]+'</th>').join('')+'</tr>'+DATA.map(d=>'<tr>'+cols.map(([title,key])=>'<td>'+(key==='case'?esc(d.case):key==='group'?esc(d.metrics.group):fmt(d.metrics[key]))+'</td>').join('')+'</tr>').join('')+'</table>';render();</script>"""


def write_html(path, cases):
    payload = json.dumps(cases).replace("</", "<\\/")
    path.write_text(
        HTML.replace("__DATA__", payload)
        .replace("__COLORS__", json.dumps(COLORS))
        .replace("__LABELS__", json.dumps(LABELS))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Results directory, or a single completed trial"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pattern", default="daemonset-*")
    args = parser.parse_args()
    paths = (
        [args.root]
        if (args.root / "timing.json").exists()
        else sorted(args.root.glob(args.pattern))
    )
    cases, skipped = [], []
    for path in paths:
        if not path.is_dir() or not (path / "timing.json").exists():
            continue
        try:
            case = parse_case(path)
        except (ValueError, KeyError, TypeError, StopIteration) as error:
            skipped.append(
                {"case": path.name, "reason": str(error) or type(error).__name__}
            )
            continue
        cases.append(case)
        (path / "resident-timeline.json").write_text(json.dumps(case, indent=2) + "\n")
        draw(
            path,
            case,
            "rows",
            "resident-timeline",
            xmin=0,
            xmax=case["ready_epoch"] - case["origin_epoch"] + 0.3,
        )
        draw(path, case, "hot_path", "resident-hot-path", xmin=0)
        draw(path, case, "prewarm", "resident-prewarm", xmax=0)
        write_html(path / "resident-timeline.html", [case])
    if not cases:
        raise SystemExit("No completed cases: " + json.dumps(skipped))
    groups = {}
    for group in sorted({case["metrics"]["group"] for case in cases}):
        groups[group] = aggregate(
            [case["metrics"] for case in cases if case["metrics"]["group"] == group]
        )
    eligible = {
        group: value for group, value in groups.items() if group.startswith("resident")
    }
    if eligible:
        fastest = min(
            eligible, key=lambda group: eligible[group]["mean"]["dgd_to_ready_s"]
        )
        ordered = sorted(
            [case for case in cases if case["metrics"]["group"] == fastest],
            key=lambda c: c["metrics"]["dgd_to_ready_s"],
        )
        representative = ordered[len(ordered) // 2]
        cases = [representative] + [
            case for case in cases if case is not representative
        ]
    controls = [
        case["metrics"]
        for case in cases
        if case["case"] in {"daemonset-resident-3", "daemonset-resident-4"}
    ]
    preinstalled = [
        case["metrics"]
        for case in cases
        if case["case"] in {"daemonset-preinstalled-1", "daemonset-preinstalled-2"}
    ]
    secondary = {}
    if preinstalled:
        secondary["preinstalled_cuda"] = {
            "note": "Separate followup: adjacent controls are resident-3 and resident-4; preinstalled cases must not be pooled into the primary cold/resident comparison.",
            "controls": aggregate(controls),
            "preinstalled": aggregate(preinstalled),
        }
    output = args.output or (
        args.root / "daemonset-study"
        if not (args.root / "timing.json").exists()
        else args.root
    )
    output.mkdir(parents=True, exist_ok=True)
    write_html(output / "timelines.html", cases)
    (output / "comparison.json").write_text(
        json.dumps(
            {
                "notes": NOTES,
                "groups": groups,
                "secondary_comparisons": secondary,
                "cases": [case["metrics"] for case in cases],
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "groups": groups,
                "skipped": skipped,
                "html": str(output / "timelines.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
