# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare completed DGD restore trials with and without CRI identity discovery."""

import argparse
import json
import statistics
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from summarize import json_events, seconds

SEQUENCE = [
    "control-1",
    "runtime-1",
    "runtime-2",
    "control-2",
    "control-3",
    "runtime-3",
    "control-4",
    "runtime-4",
]
REQUIRED = [
    "timing.json",
    "publications.json",
    "agent.txt",
    "main.txt",
    "inference.json",
    "cpu-main-after.txt",
    "resident-agent-before.json",
    "plan.json",
    "ownership.json",
    "operator-created-pod.json",
    "ready-pod.json",
    "dgd-created.json",
    "dgd-manifest.json",
    "host-watch.jsonl",
    "virtual-watch.jsonl",
] + [f"gms-{rank}.txt" for rank in range(8)]
NOTES = [
    (
        "t0 is the client DGD creation request; the main endpoint is observed workload Pod Ready "
        "after coherent generation, distinct from the DGD Ready condition."
    ),
    (
        "Pod creationTimestamp and container startedAt have one-second precision. "
        "Pod creation offsets may be negative by rounding; Pod-to-Ready has that uncertainty."
    ),
    "Watch receipt times include remote observation latency and cross-host clock uncertainty.",
    "Agent restore handler entry is per-request dispatch, not daemon startup.",
    "CRIU/CUDA durations are agent phase measurements; CUDA includes PageBroker work.",
    (
        "Main-series adjacent pairs follow B A / A B / A B. Deltas are runtime minus control; "
        "negative values mean faster with identity discovery."
    ),
    (
        "The first A/B pair used the operator default 8Gi /dev/shm and is qualification only. "
        "Main-series groups use explicit 32Gi, classified from the rendered DGD and actual Pod mount."
    ),
    (
        "This small same-node series uses preallocated DRA, cached images and PVC O_DIRECT; "
        "NFS server cache state is uncontrolled."
    ),
]


def read_json(path):
    return json.loads(path.read_text())


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def agent_entries(path, pod_name, start, end):
    entries = []
    for line in path.read_text().splitlines():
        if "{" not in line:
            continue
        try:
            at = epoch(line.split("\t")[0])
            data = json.loads(line[line.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        if start <= at <= end and data.get("pod", "").split("/")[-1] == pod_name:
            entries.append((at, line, data))
    return entries


def validate_ownership(path, timing):
    chain = read_json(path / "ownership.json")
    pod = read_json(path / "operator-created-pod.json")["metadata"]
    ready = read_json(path / "ready-pod.json")["metadata"]
    dgd = read_json(path / "dgd-created.json")["metadata"]
    require(chain[0]["kind"] == "Pod", "ownership chain must begin at Pod")
    require(chain[-1]["kind"] == "DynamoGraphDeployment", "ownership must end at DGD")
    require(chain[0]["uid"] == pod["uid"] == ready["uid"], "Pod incarnation changed")
    require(chain[0]["name"] == pod["name"] == timing["pod_name"], "Pod name mismatch")
    require(chain[-1]["uid"] == dgd["uid"], "DGD owner UID mismatch")
    require(chain[-1]["name"] == timing["dgd_name"], "DGD owner name mismatch")
    for child, parent in pairwise(chain):
        owners = [x for x in child["ownerReferences"] if x.get("controller")]
        require(len(owners) == 1, "ownership chain requires one controlling owner")
        require(child["namespace"] == parent["namespace"], "owner namespace mismatch")
        for key in ["apiVersion", "kind", "name", "uid"]:
            require(owners[0][key] == parent[key], f"owner {key} mismatch")
    return pod["uid"], chain


def analyze_case(path):
    timing = read_json(path / "timing.json")
    require(timing["deployment_mode"] == "dgd", "requires DGD trials")
    zero, ready = timing["create_epoch"], timing["ready_epoch"]
    group = "runtime" if timing["runtime_discovery"] else "control"
    require(f"-{group}-" in path.name, "case group and configuration disagree")
    pod_uid, chain = validate_ownership(path, timing)
    dgd = read_json(path / "dgd-manifest.json")
    components = dgd["spec"]["components"]
    require(len(components) == 1, "expected one DGD engine component")
    requested_shm = components[0].get("sharedMemorySize", "8Gi")
    rendered = read_json(path / "operator-created-pod.json")
    main = next(c for c in rendered["spec"]["containers"] if c["name"] == "main")
    shm_mount = next(m for m in main["volumeMounts"] if m["mountPath"] == "/dev/shm")
    actual_shm = next(
        v for v in rendered["spec"]["volumes"] if v["name"] == shm_mount["name"]
    )["emptyDir"]["sizeLimit"]
    require(
        requested_shm == actual_shm,
        "DGD requested and actual /dev/shm capacity disagree",
    )
    require(actual_shm in {"8Gi", "32Gi"}, "unexpected shared memory cohort")
    cohort = "main_32Gi" if actual_shm == "32Gi" else "qualification_8Gi"
    publications = read_json(path / "publications.json")
    require(
        sorted(p["rank"] for p in publications) == list(range(8)),
        "expected eight ranks",
    )
    first = min(p["started_epoch"] for p in publications)
    published = max(p["published_epoch"] for p in publications)
    entries = agent_entries(path / "agent.txt", timing["pod_name"], zero, ready)
    starts = [e for e in entries if "=== Starting external restore ===" in e[1]]
    summaries = [e for e in entries if "Restore timing summary" in e[1]]
    require(len(starts) == len(summaries) == 1, "expected one restore operation")
    handler, operation_end = starts[0][0], summaries[0][0]
    restore = summaries[0][2]["restore"]
    phases = restore["phases"]
    wake = [
        json.loads(line.removeprefix("GMS_WAKE_GATE "))
        for line in (path / "main.txt").read_text().splitlines()
        if line.startswith("GMS_WAKE_GATE ")
    ]
    wake = [w for w in wake if zero <= w["entered_epoch"] <= w["passed_epoch"] <= ready]
    require(len(wake) == 1, "expected one engine publication gate interval")
    wake = wake[0]
    require("berlin" in timing["restored_text"].lower(), "readiness inference failed")
    inference = read_json(path / "inference.json")
    require("rayleigh" in inference["text"].lower(), "second inference failed")
    resident = read_json(path / "resident-agent-before.json")["status"][
        "containerStatuses"
    ]
    require(
        all(c["ready"] for c in resident), "resident service not Ready before trial"
    )
    service_starts = {
        c["name"]: epoch(c["state"]["running"]["startedAt"]) - zero for c in resident
    }
    require(
        {"agent", "pagebroker"}.issubset(service_starts), "missing resident service"
    )
    require(
        all(v < 0 for v in service_starts.values()),
        "resident service started after timer",
    )
    direct_paths, initialization = set(), {}
    for rank in range(8):
        events = list(json_events((path / f"gms-{rank}.txt").read_text()))
        initialization[rank] = next(
            e["elapsed_s"] for e in events if e.get("event") == "sockets"
        )
        for event in events:
            if event.get("event") == "artifact_open":
                require(event["o_direct"] is True, "payload read did not use O_DIRECT")
                direct_paths.add(event["path"])
    require(len(direct_paths) == 112, "expected 112 unique O_DIRECT payload files")
    plan = read_json(path / "plan.json")
    discovery = [
        {
            "offset_s": at - zero,
            **{
                k: data.get(k)
                for k in ["pod_uid", "identity_discovery", "container_id"]
            },
        }
        for at, line, data in entries
        if "Resolved restore container via node runtime" in line
    ]
    runtime_proof = any(
        d["identity_discovery"] is True and d["pod_uid"] == pod_uid for d in discovery
    )
    row = {
        "case": path.name,
        "group": group,
        "cohort": cohort,
        "shared_memory_size": actual_shm,
        "pod_name": timing["pod_name"],
        "pod_uid": pod_uid,
        "timer_origin": timing["timer_origin"],
        "start_epoch": zero,
        "agent_revision": timing["agent_revision"],
        "dgd_to_ready_s": timing["dgd_create_to_ready_s"],
        "pod_to_ready_s": timing["pod_create_to_ready_s"],
        "dgd_to_pod_create_s": timing["dgd_create_to_pod_create_s"],
        "dgd_create_return_s": timing["dgd_create_return_epoch"] - zero,
        "dgd_to_handler_s": handler - zero,
        "pod_to_handler_s": handler - timing["generated_pod_create_epoch"],
        "restore_operation_s": seconds(restore["duration"]),
        "dgd_to_operation_end_s": operation_end - zero,
        "criu_s": seconds(phases["criu_restore"]),
        "cuda_s": seconds(phases["cuda_restore"]),
        "gms_span_s": published - first,
        "dgd_to_first_gms_s": first - zero,
        "dgd_to_all_published_s": published - zero,
        "gms_init_mean_s": statistics.mean(initialization.values()),
        "gms_rank_load_mean_s": statistics.mean(
            p["elapsed_s"] - initialization[p["rank"]] for p in publications
        ),
        "dgd_to_wake_gate_entered_s": wake["entered_epoch"] - zero,
        "dgd_to_wake_gate_passed_s": wake["passed_epoch"] - zero,
        "wake_gate_wait_s": wake["passed_epoch"] - wake["entered_epoch"],
        "publication_to_gate_passed_s": wake["passed_epoch"] - published,
        "gate_passed_to_ready_s": ready - wake["passed_epoch"],
        "resident_service_start_offsets_s": service_starts,
        "resident_services_ready_before_timer": True,
        "unique_direct_payload_files": len(direct_paths),
        "weights_bytes": sum(
            a["aligned_size"] for rank in plan["ranks"] for a in rank["allocations"]
        ),
        "ownership_chain": chain,
        "ownership_validated": True,
        "inference_correct": True,
        "runtime_discovery_observed": runtime_proof,
        "runtime_discovery_logs": discovery,
    }
    for kind in ["host", "virtual"]:
        events = [
            e
            for e in read_json_lines(path / f"{kind}-watch.jsonl")
            if "observed_epoch" in e
        ]
        if kind == "virtual":
            events = [e for e in events if e.get("uid") == pod_uid]
        events = [e for e in events if e["observed_epoch"] >= zero]
        if not events:
            continue
        row[f"dgd_to_{kind}_pod_observed_s"] = (
            min(e["observed_epoch"] for e in events) - zero
        )
        for event in sorted(events, key=lambda e: e["observed_epoch"]):
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
                container_start = epoch(main["state"]["running"]["startedAt"])
                row[f"dgd_to_{kind}_main_start_s"] = container_start - zero
                row[f"dgd_to_{kind}_main_status_observed_s"] = (
                    event["observed_epoch"] - zero
                )
                row[f"{kind}_main_start_to_handler_s"] = handler - container_start
                row[f"{kind}_main_status_observed_to_handler_s"] = (
                    handler - event["observed_epoch"]
                )
                break
    dgd_watch = path / "dgd-watch.jsonl"
    if dgd_watch.exists():
        for event in read_json_lines(dgd_watch):
            metadata = event.get("metadata") or {}
            if metadata.get("uid") != chain[-1]["uid"]:
                continue
            status = event.get("status") or {}
            conditions = status.get("conditions") or []
            if any(c["type"] == "Ready" and c["status"] == "True" for c in conditions):
                row["dgd_to_dgd_ready_observed_s"] = event["observed_epoch"] - zero
                row["workload_ready_to_dgd_ready_observed_s"] = (
                    event["observed_epoch"] - ready
                )
                break
    return row


def build_comparison(root):
    rows, incomplete = [], []
    for suffix in SEQUENCE:
        path = root / ("discovery-dgd-" + suffix)
        missing = [name for name in REQUIRED if not (path / name).exists()]
        if missing:
            incomplete.append({"case": path.name, "missing": missing})
            continue
        rows.append(analyze_case(path))
    groups = {}
    for group in ["control", "runtime"]:
        cases = [r for r in rows if r["group"] == group and r["cohort"] == "main_32Gi"]
        if not cases:
            continue
        metrics = set.intersection(
            *[
                {
                    k
                    for k, v in r.items()
                    if k.endswith("_s") and isinstance(v, (int, float))
                }
                for r in cases
            ]
        )
        groups[group] = {
            "n": len(cases),
            "cases": [r["case"] for r in cases],
            "metrics": {
                k: {
                    "mean": statistics.mean(r[k] for r in cases),
                    "range": [min(r[k] for r in cases), max(r[k] for r in cases)],
                }
                for k in sorted(metrics)
            },
            "runtime_discovery_proof_count": sum(
                r["runtime_discovery_observed"] for r in cases
            ),
        }
    by_name = {r["case"]: r for r in rows}
    pairs = []
    for index in range(1, 5):
        control = by_name.get(f"discovery-dgd-control-{index}")
        runtime = by_name.get(f"discovery-dgd-runtime-{index}")
        if control and runtime:
            require(
                control["cohort"] == runtime["cohort"],
                "paired shared-memory capacity mismatch",
            )
            metrics = {
                k
                for k, v in control.items()
                if k.endswith("_s") and isinstance(v, (int, float)) and k in runtime
            }
            pairs.append(
                {
                    "pair": index,
                    "cohort": control["cohort"],
                    "order": "B A" if index == 2 else "A B",
                    "control": control["case"],
                    "runtime": runtime["case"],
                    "runtime_minus_control_s": {
                        k: runtime[k] - control[k] for k in sorted(metrics)
                    },
                }
            )
    return {
        "notes": NOTES,
        "expected_sequence": SEQUENCE,
        "complete": len(rows) == len(SEQUENCE),
        "incomplete_cases": incomplete,
        "cases": rows,
        "groups": groups,
        "qualification_cases": [
            r["case"] for r in rows if r["cohort"] == "qualification_8Gi"
        ],
        "pairs": pairs,
    }


def markdown(report):
    lines = [
        "# DGD container discovery comparison",
        "",
        f"Completed trials: {len(report['cases'])}/{len(SEQUENCE)}. Main group aggregates use 32Gi cases only.",
        "",
    ]
    columns = [
        ("dgd_to_ready_s", "DGD request → workload Ready"),
        ("pod_to_ready_s", "Pod → Ready*"),
        ("dgd_to_handler_s", "DGD → handler"),
        ("criu_s", "CRIU"),
        ("cuda_s", "CUDA"),
        ("gms_span_s", "GMS span"),
        ("dgd_to_all_published_s", "All published"),
        ("wake_gate_wait_s", "Gate wait"),
    ]
    lines.extend(
        [
            "Seconds; group values are mean (min–max).",
            "",
            "| Group | n | " + " | ".join(label for _, label in columns) + " |",
            "|---|---:|" + "---:|" * len(columns),
        ]
    )
    for group, data in report["groups"].items():
        cells = []
        for key, _ in columns:
            metric = data["metrics"][key]
            cells.append(
                f"{metric['mean']:.3f} ({metric['range'][0]:.3f}–{metric['range'][1]:.3f})"
            )
        lines.append(f"| {group} | {data['n']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "| Case / cohort | "
            + " | ".join(label for _, label in columns)
            + " | Runtime lookup proven |",
            "|---|" + "---:|" * len(columns) + "---|",
        ]
    )
    for row in report["cases"]:
        lines.append(
            "| "
            + row["case"]
            + " / "
            + row["cohort"]
            + " | "
            + " | ".join(f"{row[key]:.3f}" for key, _ in columns)
            + f" | {row['runtime_discovery_observed']} |"
        )
    lines.extend(
        [
            "",
            "| Pair / cohort | Order | Δ request → workload Ready | Δ handler entry | Δ GMS span | Δ gate wait |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair in report["pairs"]:
        delta = pair["runtime_minus_control_s"]
        lines.append(
            f"| {pair['pair']} / {pair['cohort']} | {pair['order']} | "
            + " | ".join(
                f"{delta[key]:+.3f}"
                for key in [
                    "dgd_to_ready_s",
                    "dgd_to_handler_s",
                    "gms_span_s",
                    "wake_gate_wait_s",
                ]
            )
            + " |"
        )
    startup_columns = [
        ("dgd_to_virtual_pod_observed_s", "Pod first observed"),
        ("dgd_to_pod_create_s", "Pod created*"),
        ("dgd_to_first_gms_s", "First GMS"),
        ("dgd_to_wake_gate_entered_s", "Wake gate entered"),
        ("dgd_to_wake_gate_passed_s", "Wake gate passed"),
        ("dgd_to_dgd_ready_observed_s", "DGD Ready observed"),
    ]
    lines.extend(
        [
            "",
            "Event offsets from DGD creation request; Pod first-observed time includes API-watch latency.",
            "",
            "| Case | " + " | ".join(label for _, label in startup_columns) + " |",
            "|---|" + "---:|" * len(startup_columns),
        ]
    )
    for row in report["cases"]:
        lines.append(
            "| "
            + row["case"]
            + " | "
            + " | ".join(
                f"{row[key]:.3f}" if key in row else "unobserved"
                for key, _ in startup_columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Every included case validates the DGD ownership chain, two inference prompts, eight rank publications, 112 unique O_DIRECT payload files, and resident agent/PageBroker readiness with starts before t0.",
            "",
        ]
    )
    lines.extend("- " + note for note in report["notes"])
    if report["incomplete_cases"]:
        lines.extend(
            ["", "Incomplete cases are skipped until collection finishes:", ""]
        )
        lines.extend("- " + item["case"] for item in report["incomplete_cases"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="default-config results directory")
    parser.add_argument("--output", type=Path, help="defaults to ROOT/discovery-study")
    args = parser.parse_args()
    report = build_comparison(args.root)
    output = args.output or args.root / "discovery-study"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "comparison.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
