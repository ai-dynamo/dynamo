# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot native PageBroker→GMS evidence with independent service ownership.

Usage: python pagebroker_timeline.py RESULTS_CASE [RESULTS_CASE ...]
No missing native timestamps are inferred from Python loader spans. Per-case JSON,
SVG, PNG and interactive HTML are written beside the input evidence; --output-dir
places all generated files elsewhere (useful for parser qualification).
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from resident_timeline import agent_entries, cri_epoch, iso_epoch, watch_info
from resident_timeline import parse_case as parse_python_evidence
from summarize import seconds

COLORS = {
    "init": "#7a85b5",
    "idle": "#dce3ea",
    "orchestration": "#8293a4",
    "discovery": "#bf9950",
    "allocation": "#db9e45",
    "queue": "#bd6462",
    "read_copy": "#339e73",
    "python_read": "#4b9c75",
    "commit": "#287f91",
    "criu": "#a975b4",
    "prepare": "#de8b45",
    "transfer": "#63b8b1",
    "complete": "#d25c67",
    "inference": "#5f7793",
    "gate": "#aeaa7d",
}
LABELS = {
    "init": "Service initialization (before timer)",
    "idle": "Prestarted service waiting",
    "orchestration": "DGD / controller dispatch",
    "discovery": "Main container → handler",
    "allocation": "GMS RW lease / allocate / import",
    "queue": "Waiting for shared per-GPU ring",
    "read_copy": "PVC O_DIRECT + GPU copy (combined wall interval)",
    "python_read": "Python O_DIRECT loader transfer window",
    "commit": "Drain / validate / publish",
    "criu": "CRIU",
    "prepare": "CUDA PREPARE request",
    "transfer": "Engine residual transfer request",
    "complete": "CUDA COMPLETE request",
    "inference": "Wake / coherent generation / workload Ready",
    "gate": "Engine waits for matching weights",
}
NOTES = [
    "DGD owns the engine workload. A separate node DaemonSet owns the one-GPU GMS V1 servers; a resident PageBroker service owns payload I/O, imported VMM mappings and transfer rings. Sharing t0 does not place GMS or PageBroker inside DGD.",
    "t0 is the DGD creation request. The artifact is discovered only after DGD creation, then the independent coordinator dispatches PageBroker. Those metadata steps remain inside the measured latency. Ready is the workload Pod after coherent generation, not necessarily the DGD Ready condition.",
    "Prestarted-service readiness is shown before t0. Only measured startup/online intervals are drawn; an absent timestamp is not treated as zero cost.",
    "Native read_copy bars bracket the unchanged PageBroker TransferBuffers call, which overlaps O_DIRECT reads and host-to-device copies and drains before returning. Separate disk-read and DMA wall intervals cannot be recovered from these logs.",
    "Aggregate payload rate uses all completed payload bytes divided by earliest read_copy_start to latest read_copy_complete, across all ranks. It is an effective load rate, not raw PVC bandwidth. Summed queue or storage-service time can overlap across ranks and must not be added to wall time.",
    "Engine CUDA bars are client request intervals. Their TRANSFER duration includes ring queueing; the embedded PageBroker report records queue_wait_seconds separately. CRIU placement uses the Native restore order marker and is approximate; its logged duration is preserved.",
    "CRI main startedAt retains nanosecond precision. Kubernetes startedAt fallback has one-second precision. Watch receipt is a separate observation and may lag actual container start.",
    "Generation/capture filtering prevents older resident-service operations from entering this trial. Publication success still requires captured allocation IDs/sizes, UUID, server nonce and manifest validation; a Gantt chart alone is not a correctness proof.",
    "Cross-machine timestamps may have clock error. Same-process durations are preferred; sample-size and cohort comparisons are reported separately without pooling different CPU/backend settings.",
]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def records(path):
    """Accept JSONL, kubectl prefixes and adjacent JSON records; ignore plain logs."""
    decoder = json.JSONDecoder()
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        offset = line.find("{")
        while offset >= 0:
            try:
                value, end = decoder.raw_decode(line[offset:])
            except ValueError:
                break
            if isinstance(value, dict):
                yield value
            offset = line.find("{", offset + end)


def segment(start, end, kind, detail=""):
    if start is None or end is None or end < start:
        return None
    return {"start": start, "end": end, "kind": kind, "detail": detail or LABELS[kind]}


def row(label, *segments):
    return {"label": label, "segments": [s for s in segments if s is not None]}


def relative_rows(rows, zero):
    return [
        {
            "label": r["label"],
            "segments": [
                dict(s, start=s["start"] - zero, end=s["end"] - zero)
                for s in r["segments"]
            ],
        }
        for r in rows
        if r["segments"]
    ]


def offset(value, zero):
    return None if value is None else value - zero


def terminal_reports(publications):
    for value in publications:
        report = value.get("report", value.get("pagebroker_report", value))
        if isinstance(report, str):
            try:
                report = json.loads(report)
            except ValueError:
                continue
        if isinstance(report, dict) and report.get("event") == "gms_weights_loaded":
            yield report


def native_phases(path, capture, generation, zero, end):
    output, seen = [], set()
    for value in records(path):
        if value.get("event") != "gms_load_phase" or not value.get("epoch_ns"):
            continue
        if capture and value.get("capture_id") != capture:
            continue
        if generation and value.get("generation") != generation:
            continue
        at = value["epoch_ns"] / 1e9
        if not zero - 1 <= at <= end + 1:
            continue
        fingerprint = json.dumps(value, sort_keys=True)
        if fingerprint not in seen:
            output.append(dict(value, epoch=at))
            seen.add(fingerprint)
    return sorted(output, key=lambda e: e["epoch_ns"])


def pair_phases(events, first, last, kind, shard=None):
    pending, output = None, []
    for event in events:
        if shard is not None and event.get("shard") != shard:
            continue
        if event["phase"] == first:
            pending = event
        elif event["phase"] == last and pending is not None:
            detail = json.dumps({"start": pending, "end": event}, sort_keys=True)
            item = segment(pending["epoch"], event["epoch"], kind, detail)
            if item:
                output.append(item)
            pending = None
    return output


def ownership(case, timing, ranks, zero):
    pod = read_json(case / "resident-gms-before.json", {})
    metadata = pod.get("metadata", {})
    owners = metadata.get("ownerReferences", [])
    engine_pod = read_json(case / "operator-created-pod.json", {})
    engine_containers = [
        c["name"] for c in engine_pod.get("spec", {}).get("containers", [])
    ]
    state = read_json(case / "resident-gms-ready.json", {})
    online = state.get("online", [])
    ready_epoch = timing.get("resident_gms_ready_verified_epoch")
    expected = set(ranks)
    records_by_rank = {item.get("rank"): item for item in online}
    online_complete = (
        bool(expected)
        and expected == set(records_by_rank)
        and len(online) == len(expected)
    )
    generic_service = "service_generation" in state
    service_generation = timing.get(
        "resident_gms_service_generation", timing.get("resident_gms_generation")
    )
    generation_key = "service_generation" if generic_service else "generation"
    server_generation_matches = (
        bool(service_generation)
        and state.get(generation_key) == service_generation
        and bool(online_complete)
        and all(
            records_by_rank[r].get(generation_key) == service_generation
            for r in expected
        )
    )
    online_before = bool(
        online_complete
        and all(
            records_by_rank[r].get("online_epoch", math.inf) < zero for r in expected
        )
    )
    pod_ready = next(
        (
            x
            for x in pod.get("status", {}).get("conditions", [])
            if x.get("type") == "Ready"
        ),
        {},
    )
    return {
        "gms_pod": metadata.get("name", timing.get("resident_gms_pod")),
        "gms_pod_uid": metadata.get("uid"),
        "gms_owners": [
            {k: owner.get(k) for k in ("kind", "name", "uid")} for owner in owners
        ],
        "gms_owned_by_daemonset": any(x.get("kind") == "DaemonSet" for x in owners),
        "gms_not_owned_by_dgd": bool(owners)
        and all(x.get("kind") != "DynamoGraphDeployment" for x in owners),
        "engine_containers": engine_containers,
        "no_gms_container_in_engine_pod": bool(engine_containers)
        and all(not c.startswith("gms") for c in engine_containers),
        "gms_ready_condition_before_request": pod_ready.get("status") == "True"
        and iso_epoch(pod_ready["lastTransitionTime"]) < zero
        if pod_ready.get("lastTransitionTime")
        else None,
        "gms_ready_verified_offset_s": offset(ready_epoch, zero),
        "all_rank_servers_online_before_request": online_before,
        "identity_format": "generic service / DGD load"
        if generic_service
        else "legacy capture-bound service",
        "service_generation": service_generation,
        "load_generation": timing.get("resident_gms_generation"),
        "online_service_generation_matches": server_generation_matches,
        "online_rank_uuids_match_plan": bool(online_complete)
        and all(
            records_by_rank[r].get("uuid") == ranks[r].get("destination_uuid")
            for r in expected
        ),
        "no_capture_selected_at_service_ready": not state.get("capture_id")
        and not state.get("generation")
        and bool(online_complete)
        and all(not e.get("capture_id") and not e.get("generation") for e in online)
        if generic_service
        else None,
        "all_rank_servers_pristine_before_request": bool(online_complete)
        and all(
            e.get("payload_bytes_read") == 0 and e.get("weight_allocations") == 0
            for e in online
        ),
        "no_loader_context_or_ring_before_request": bool(online_complete)
        and all(
            e.get("cuda_context_current") is False
            and e.get("cuda_primary_context_active") is False
            and e.get("loader_lanes_ready") == 0
            and e.get("pinned_bytes") == 0
            for e in online
        )
        if generic_service
        else None,
        "preload_payload_bytes": sum(x.get("payload_bytes_read", 0) for x in online)
        if online
        else None,
        "pagebroker_ready_verified_offset_s": offset(
            timing.get("resident_services_ready_verified_epoch"), zero
        ),
        "online": online,
    }


def parse_python_case(case, timing):
    """Preserve the existing loader measurement boundary for the explicit control."""
    source = parse_python_evidence(case)
    measured = source["metrics"]
    zero = source["origin_epoch"]
    plan = read_json(case / "plan.json", {})
    ranks = {r["rank"]: r for r in plan.get("ranks", [])}
    evidence = ownership(case, timing, ranks, zero)
    kinds = {
        "read": "python_read",
        "publication": "commit",
        "context": "init",
        "buffers": "init",
        "dispatch": "orchestration",
        "wait": "gate",
        "operation": "idle",
        "observation": "orchestration",
    }

    def adapt(rows):
        rows = relative_rows(rows, zero)
        for r in rows:
            for s in r["segments"]:
                s["kind"] = kinds.get(s["kind"], s["kind"])
        return rows

    metrics = {
        "dgd_to_workload_ready_s": measured["dgd_to_ready_s"],
        "dgd_to_handler_s": measured["dgd_to_handler_s"],
        "main_start_offset_s": measured["main_start_offset_s"],
        "main_start_to_handler_s": measured["main_start_to_handler_s"],
        "main_start_precision": measured["main_start_precision"],
        "criu_s": measured["criu_s"],
        "cuda_s": measured["cuda_s"],
        "first_read_offset_s": measured["first_read_offset_s"],
        "last_copy_offset_s": measured["all_transfer_complete_offset_s"],
        "all_committed_offset_s": measured["all_published_offset_s"],
        "payload_window_s": measured["aggregate_read_to_transfer_complete_s"],
        "completed_bytes": measured["payload_transfer_bytes"],
        "payload_effective_GB_per_s": measured["effective_payload_GBps"],
        "payload_effective_GiB_per_s": measured["effective_payload_GiBps"],
        "gms_queue_seconds_sum": None,
        "native_residual_queue_seconds_sum": None,
        "wake_gate_wait_s": measured["wake_gate_wait_s"],
        "rank_metrics": measured["rank_metrics"],
    }
    files = [path for rank in measured["rank_metrics"] for path in rank["direct_files"]]
    return {
        "case": case.name,
        "source_directory": str(case.resolve()),
        "capture_id": timing.get("capture_id"),
        "generation": timing.get("resident_gms_generation"),
        "zero_epoch": zero,
        "ready_offset_s": measured["dgd_to_ready_s"],
        "mode": "gms_python_loader",
        "artifact_selection": "preselected capture; optimistic end-to-end control",
        "settings": timing,
        "ownership": evidence,
        "metrics": metrics,
        "validation": {
            "measurement_source": "existing resident Python loader parser",
            "unique_o_direct_paths": len(set(files)),
            "inference_recorded": (case / "inference.json").exists(),
        },
        "warnings": [
            "This Python control preselected the capture before DGD creation. Its end-to-end latency is optimistic and is not a fair comparison with post-DGD artifact discovery."
        ],
        "rows": adapt(source["rows"]),
        "hot_path_rows": adapt(source["hot_path"]),
        "prewarm_rows": adapt(source["prewarm"]),
        "rank_rows": {str(r["rank"]): adapt(r["detail"]) for r in source["ranks"]},
        "notes": [
            "This is the explicit Python-loader control. Its independent GMS DaemonSet fuses server and Python loader in each rank container; PageBroker only restores engine residual state. The control is never labeled as native PageBroker payload transfer."
        ]
        + source["notes"],
    }


def parse_case(case, pagebroker_log=None):
    timing = read_json(case / "timing.json")
    if not timing or not timing.get("ready_epoch"):
        raise ValueError(f"{case}: incomplete timing.json (workload Ready missing)")
    if (
        timing.get("weight_transfer_owner") == "gms_python_loader"
        or timing.get("pagebroker_gms") is False
    ):
        return parse_python_case(case, timing)
    zero, ready = timing["create_epoch"], timing["ready_epoch"]
    plan = read_json(case / "plan.json", {})
    ranks = {r["rank"]: r for r in plan.get("ranks", [])}
    publications = read_json(case / "publications.json", [])
    capture = timing.get("capture_id", plan.get("capture_id"))
    generation = timing.get("resident_gms_generation")
    if not generation:
        generations = {p.get("generation") for p in publications} - {None}
        generation = next(iter(generations)) if len(generations) == 1 else None
    events = native_phases(
        pagebroker_log or case / "pagebroker.txt", capture, generation, zero, ready
    )
    warnings = []
    if not generation:
        warnings.append(
            "No unique generation in timing/publications; native event filtering uses capture and time window only."
        )
    if not events:
        warnings.append(
            "No native PageBroker GMS phase records; native transfer timings are unavailable."
        )
    entries = agent_entries(case / "agent.txt", zero, ready, timing.get("pod_name"))
    handler = next(
        (at for at, line, _ in entries if "=== Starting external restore ===" in line),
        None,
    )
    summary = next(
        (
            (at, data["restore"])
            for at, line, data in entries
            if "Restore timing summary" in line
        ),
        (None, {}),
    )
    restore_end, restore = summary
    host, virtual = (
        watch_info(case / "host-watch.jsonl"),
        watch_info(case / "virtual-watch.jsonl"),
    )
    cri = read_json(case / "cri-starts.json", [])
    main_starts = [
        cri_epoch(r["startedAt"])
        for r in cri
        if r.get("name", r.get("metadata", {}).get("name")) == "main"
        and r.get("startedAt")
    ]
    main_start = min(main_starts, default=host["main_started"])
    precision = (
        "CRI nanoseconds" if main_starts else "Kubernetes startedAt (1 s precision)"
    )
    rows = [
        row(
            "DGD request → workload Pod observed",
            segment(zero, virtual["pod_observed"], "orchestration"),
        ),
        row(
            "Snapshot agent: request → restore handler",
            segment(zero, handler, "orchestration"),
        ),
        row(
            "DGD engine: main started → handler",
            segment(main_start, handler, "discovery", precision),
        ),
    ]
    dgd_created = timing.get("dgd_create_return_epoch")
    dgd_uid = read_json(case / "dgd-created.json", {}).get("metadata", {}).get("uid")
    selection = read_json(
        case / "snapshot-selection.json", timing.get("snapshot_selection") or {}
    )
    selection_start = selection.get("discovery_started_epoch")
    selection_end = selection.get("discovery_completed_epoch")
    rows.insert(
        0,
        row(
            "DGD create request → API response",
            segment(zero, dgd_created, "orchestration"),
        ),
    )
    coordinator = [
        e
        for e in records(case / "coordinator.txt")
        if "epoch" in e
        and zero <= e["epoch"] <= ready
        and (not generation or e.get("generation", generation) == generation)
    ]
    trigger = read_json(
        case / "resident-gms-trigger.json",
        timing.get("resident_gms_load_trigger") or {},
    )
    trigger_response = trigger.get("response", {})
    discovery_start = min(
        (
            e.get("manifest_discovery_start_epoch", e["epoch"])
            for e in coordinator
            if e.get("event") == "manifest_discovery_start"
        ),
        default=trigger_response.get("manifest_discovery_start_epoch"),
    )
    discovery_end = max(
        (
            e.get("manifest_discovery_complete_epoch", e["epoch"])
            for e in coordinator
            if e.get("event") == "manifest_discovery_complete"
        ),
        default=trigger_response.get("manifest_discovery_complete_epoch"),
    )
    request_starts = [
        e["epoch"] for e in coordinator if e.get("event") == "pagebroker_request_start"
    ]
    request_first = min(request_starts, default=None)
    request_last = max(request_starts, default=None)
    rows.extend(
        [
            row(
                "Created DGD → Snapshot/Content API selection",
                segment(dgd_created, selection_start, "orchestration"),
            ),
            row(
                "Snapshot / Content API lookups (select artifact)",
                segment(
                    selection_start,
                    selection_end,
                    "orchestration",
                    json.dumps(selection, sort_keys=True),
                ),
            ),
            row(
                "Artifact selected → coordinator receives request",
                segment(selection_end, discovery_start, "orchestration"),
            ),
            row(
                "Independent coordinator: manifest discovery",
                segment(discovery_start, discovery_end, "orchestration"),
            ),
            row(
                "Manifest discovered → PageBroker requests dispatched",
                segment(discovery_end, request_last, "orchestration"),
            ),
        ]
    )
    hot_rows = list(rows)
    for item in cri:
        name = item.get("name", item.get("metadata", {}).get("name", "unknown"))
        if name != "main" and item.get("startedAt") and item.get("finishedAt"):
            hot_rows.append(
                row(
                    f"DGD init: {name}",
                    segment(
                        cri_epoch(item["startedAt"]),
                        cri_epoch(item["finishedAt"]),
                        "orchestration",
                        "CRI nanoseconds",
                    ),
                )
            )
    for key in (
        "preflight",
        "finalizer",
        "snapshot_get",
        "content_get",
        "artifact_validation",
        "status_apply",
        "container_resolution",
        "network_resolution",
    ):
        pending, spans = None, []
        for at, line, data in entries:
            if "Restore startup milestone" not in line or (handler and at > handler):
                continue
            stage = data.get("stage")
            if stage == key + "_start" or (
                key == "network_resolution" and stage == "runtime_network_poll_start"
            ):
                pending = at
            if stage == key + "_done":
                spans.append(
                    segment(
                        pending
                        if pending is not None
                        else at - data.get("elapsed_seconds", 0),
                        at,
                        "orchestration",
                        json.dumps(data, sort_keys=True),
                    )
                )
                pending = None
        hot_rows.append(row(f"Snapshot: {key}", *spans))
    evidence = ownership(case, timing, ranks, zero)
    prewarm = []
    for online in sorted(evidence["online"], key=lambda e: e["rank"]):
        rank, at = online["rank"], online.get("online_epoch")
        prewarm.append(
            row(
                f"Independent GMS DaemonSet: rank {rank}",
                segment(online.get("daemon_started_epoch"), at, "init"),
                segment(at, zero, "idle"),
            )
        )
    service_ready = timing.get("resident_services_ready_verified_epoch")
    prewarm.insert(
        0,
        row(
            "Independent Snapshot / PageBroker: verified Ready",
            segment(
                service_ready,
                zero,
                "idle",
                "Measured readiness verification → DGD request; initialization began earlier.",
            ),
        ),
    )
    rank_rows, rank_metrics, starts, ends, payloads = {}, [], [], [], []
    for rank in sorted({event["rank"] for event in events}):
        selected = [e for e in events if e["rank"] == rank]
        spans = pair_phases(
            selected, "lease_start", "allocation_import_complete", "allocation"
        )
        detail_rows = [
            row(f"PB → GMS rank {rank}: lease / allocation / import", *spans)
        ]
        shards = sorted({e["shard"] for e in selected if "shard" in e})
        for shard in shards:
            queue = pair_phases(
                selected, "ring_queue_start", "ring_acquired", "queue", shard
            )
            transfer = pair_phases(
                selected, "read_copy_start", "read_copy_complete", "read_copy", shard
            )
            spans += queue + transfer
            detail_rows.append(row(f"PB rank {rank}: {shard}", *queue, *transfer))
        commits = pair_phases(selected, "copies_drained", "committed", "commit")
        spans += commits
        detail_rows.append(row(f"PB → GMS rank {rank}: validate / commit", *commits))
        rows.append(row(f"Independent PageBroker → GMS V1 rank {rank}", *spans))
        rank_rows[str(rank)] = relative_rows(detail_rows, zero)
        reads = [e for e in selected if e["phase"] == "read_copy_start"]
        copies = [e for e in selected if e["phase"] == "read_copy_complete"]
        committed = [e for e in selected if e["phase"] == "committed"]
        starts += reads
        ends += copies
        payloads += [(rank, e.get("shard")) for e in reads]
        rank_metrics.append(
            {
                "rank": rank,
                "first_read_offset_s": offset(
                    min((e["epoch"] for e in reads), default=None), zero
                ),
                "last_copy_offset_s": offset(
                    max((e["epoch"] for e in copies), default=None), zero
                ),
                "committed_offset_s": offset(
                    max((e["epoch"] for e in committed), default=None), zero
                ),
                "completed_bytes": sum(e.get("bytes", 0) for e in copies),
                "shards_started": len(reads),
                "shards_completed": len(copies),
                "queue_seconds_sum": sum(
                    e.get("queue_seconds", 0)
                    for e in selected
                    if e["phase"] == "ring_acquired"
                ),
            }
        )
    phases = restore.get("phases", {})
    criu_duration = (
        seconds(phases["criu_restore"]) if phases.get("criu_restore") else None
    )
    criu_end = next(
        (at for at, line, _ in entries if "Native restore order" in line), None
    )
    rows.append(
        row(
            "DGD engine: CRIU (placement approximate)",
            segment(
                criu_end - criu_duration
                if criu_end and criu_duration is not None
                else None,
                criu_end,
                "criu",
            ),
        )
    )
    pids, native_queue = defaultdict(list), 0
    for at, line, data in entries:
        if "Native PageBroker phase" not in line:
            continue
        operation = data["operation"].lower()
        if operation not in {"prepare", "transfer", "complete"}:
            continue
        pids[data["pid"]].append(
            segment(
                at - data["duration"], at, operation, json.dumps(data, sort_keys=True)
            )
        )
        if operation == "transfer":
            try:
                native_queue += json.loads(data.get("report", "{} ")).get(
                    "queue_wait_seconds", 0
                )
            except ValueError:
                warnings.append(
                    "Unparseable native TRANSFER report; queue sum may be incomplete."
                )
    rows += [
        row(f"DGD engine CUDA PID {pid} (resident PageBroker)", *spans)
        for pid, spans in pids.items()
    ]
    cuda_end = max(
        (
            s["end"]
            for spans in pids.values()
            for s in spans
            if s and s["kind"] == "complete"
        ),
        default=None,
    )
    rows.append(
        row(
            "Snapshot: post-CUDA hooks / cleanup → restore return",
            segment(
                cuda_end,
                restore_end,
                "orchestration",
                "Measured interval includes post-CUDA work, nsrestore return, namespace unmounts and process validation. The deployed b33a9962 helper calls the cuInterpose restore coordinator after the CUDA timer stops; current logs do not separate coordinator work from cleanup or establish a GMS wait.",
            ),
        )
    )
    wake = next(
        (
            e
            for e in records(case / "main.txt")
            if "entered_epoch" in e and zero <= e["entered_epoch"] <= ready
        ),
        None,
    )
    if wake:
        rows.append(
            row(
                "DGD engine: restore return → application gate",
                segment(restore_end, wake["entered_epoch"], "orchestration"),
            )
        )
        rows.append(
            row(
                "DGD engine: waits for matching GMS publications",
                segment(wake["entered_epoch"], wake.get("passed_epoch"), "gate"),
            )
        )
    rows.append(
        row(
            "DGD engine: wake / generation / workload Ready",
            segment(
                wake.get("passed_epoch") if wake else restore_end, ready, "inference"
            ),
        )
    )
    first_read = min((e["epoch"] for e in starts), default=None)
    last_copy = max((e["epoch"] for e in ends), default=None)
    transfer_span = (
        last_copy - first_read
        if first_read is not None and last_copy is not None
        else None
    )
    completed_bytes = sum(e.get("bytes", 0) for e in ends)
    expected_bytes = sum(
        a["aligned_size"] for r in ranks.values() for a in r.get("allocations", [])
    )
    expected_payloads = {
        (rank, a["shard"])
        for rank, r in ranks.items()
        for a in r.get("allocations", [])
    }
    actual_payloads = set(payloads)
    reports = list(terminal_reports(publications))
    end_payloads = [(e["rank"], e.get("shard")) for e in ends]
    reports_by_rank = {r.get("rank"): r for r in reports}
    online_by_rank = {r.get("rank"): r for r in evidence["online"]}
    reports_match = (
        bool(ranks)
        and len(reports) == len(ranks)
        and set(reports_by_rank) == set(ranks)
    )
    reports_match = reports_match and all(
        reports_by_rank[rank].get("capture_id") == capture
        and reports_by_rank[rank].get("generation") == generation
        and reports_by_rank[rank].get("destination_uuid")
        == spec.get("destination_uuid")
        and reports_by_rank[rank].get("manifest_sha256") == spec.get("manifest_sha256")
        and bool(reports_by_rank[rank].get("server_nonce"))
        and reports_by_rank[rank].get("server_nonce")
        == online_by_rank.get(rank, {}).get("weights_server_nonce")
        for rank, spec in ranks.items()
    )
    metrics = {
        "dgd_to_workload_ready_s": ready - zero,
        "dgd_to_handler_s": offset(handler, zero),
        "dgd_create_return_offset_s": offset(dgd_created, zero),
        "snapshot_api_selection_start_offset_s": offset(selection_start, zero),
        "snapshot_api_selection_complete_offset_s": offset(selection_end, zero),
        "snapshot_api_selection_s": selection_end - selection_start
        if selection_start is not None and selection_end is not None
        else None,
        "manifest_discovery_start_offset_s": offset(discovery_start, zero),
        "manifest_discovery_complete_offset_s": offset(discovery_end, zero),
        "manifest_discovery_s": discovery_end - discovery_start
        if discovery_start is not None and discovery_end is not None
        else None,
        "first_pagebroker_request_offset_s": offset(request_first, zero),
        "last_pagebroker_request_offset_s": offset(request_last, zero),
        "main_start_offset_s": offset(main_start, zero),
        "main_start_to_handler_s": handler - main_start
        if handler and main_start
        else None,
        "main_start_precision": precision,
        "criu_s": criu_duration,
        "cuda_s": seconds(phases["cuda_restore"])
        if phases.get("cuda_restore")
        else None,
        "cuda_complete_offset_s": offset(cuda_end, zero),
        "restore_return_offset_s": offset(restore_end, zero),
        "cuda_complete_to_restore_return_s": restore_end - cuda_end
        if restore_end is not None and cuda_end is not None
        else None,
        "restore_return_to_application_gate_s": wake["entered_epoch"] - restore_end
        if wake and restore_end is not None
        else None,
        "first_read_offset_s": offset(first_read, zero),
        "last_copy_offset_s": offset(last_copy, zero),
        "all_committed_offset_s": max(
            (
                r["committed_offset_s"]
                for r in rank_metrics
                if r["committed_offset_s"] is not None
            ),
            default=None,
        ),
        "payload_window_s": transfer_span,
        "completed_bytes": completed_bytes,
        "expected_bytes": expected_bytes,
        "payload_effective_GB_per_s": completed_bytes / transfer_span / 1e9
        if transfer_span
        else None,
        "payload_effective_GiB_per_s": completed_bytes / transfer_span / 2**30
        if transfer_span
        else None,
        "gms_queue_seconds_sum": sum(r["queue_seconds_sum"] for r in rank_metrics),
        "native_residual_queue_seconds_sum": native_queue,
        "wake_gate_wait_s": wake.get("passed_epoch", wake["entered_epoch"])
        - wake["entered_epoch"]
        if wake
        else None,
        "rank_metrics": rank_metrics,
    }
    validation = {
        "native_records_present": bool(events),
        "artifact_discovery_after_dgd_created": discovery_start >= dgd_created
        if discovery_start is not None and dgd_created is not None
        else None,
        "pagebroker_dispatch_after_manifest_discovery": request_first >= discovery_end
        if request_first is not None and discovery_end is not None
        else None,
        "dgd_uid": dgd_uid,
        "load_generation_matches_created_dgd_uid": generation == dgd_uid
        if dgd_uid
        else None,
        "snapshot_selection_matches_created_dgd_uid": selection.get("dgd_uid")
        == dgd_uid
        if dgd_uid and selection
        else None,
        "load_request_matches_created_dgd_uid": trigger_response.get("dgd_uid")
        == dgd_uid
        and trigger_response.get("generation") == generation
        if dgd_uid and trigger_response
        else None,
        "load_request_uses_prestarted_service_generation": trigger_response.get(
            "service_generation"
        )
        == evidence["service_generation"]
        if trigger_response
        else None,
        "snapshot_api_selection_after_dgd_created": selection_start >= dgd_created
        if selection_start is not None and dgd_created is not None
        else None,
        "manifest_discovery_after_snapshot_selection": discovery_start >= selection_end
        if discovery_start is not None and selection_end is not None
        else None,
        "all_payloads_o_direct": bool(starts)
        and all(e.get("o_direct") is True for e in starts),
        "unique_rank_shard_payloads": len(actual_payloads),
        "no_duplicate_payload_starts": len(payloads) == len(actual_payloads),
        "completed_payloads_match_starts": len(end_payloads) == len(set(end_payloads))
        and set(end_payloads) == actual_payloads,
        "payload_set_matches_plan": bool(expected_payloads)
        and actual_payloads == expected_payloads,
        "completed_bytes_match_plan": bool(expected_bytes)
        and completed_bytes == expected_bytes,
        "each_rank_completed_bytes_match_plan": bool(ranks)
        and len(rank_metrics) == len(ranks)
        and all(
            r["completed_bytes"]
            == sum(
                a["aligned_size"]
                for a in ranks.get(r["rank"], {}).get("allocations", [])
            )
            for r in rank_metrics
        ),
        "committed_rank_set_matches_plan": bool(ranks)
        and {r["rank"] for r in rank_metrics if r["committed_offset_s"] is not None}
        == set(ranks),
        "all_reads_after_request": bool(starts)
        and min(e["epoch"] for e in starts) >= zero,
        "native_rank_uuid_matches_plan": bool(events)
        and all(
            e.get("destination_uuid")
            == ranks.get(e["rank"], {}).get("destination_uuid")
            for e in events
        ),
        "terminal_report_provenance_matches_plan": reports_match,
        "terminal_reports": reports,
        "inference_recorded": (case / "inference.json").exists(),
        "inference_has_text": bool(read_json(case / "inference.json", {}).get("text")),
    }
    for name in (
        "gms_owned_by_daemonset",
        "no_gms_container_in_engine_pod",
        "all_rank_servers_online_before_request",
    ):
        if not evidence[name]:
            warnings.append(
                f"Ownership / prestart evidence incomplete or false: {name}."
            )
    return {
        "case": case.name,
        "source_directory": str(case.resolve()),
        "capture_id": capture,
        "generation": generation,
        "service_generation": evidence["service_generation"],
        "zero_epoch": zero,
        "ready_offset_s": ready - zero,
        "mode": "pagebroker",
        "artifact_selection": "post-DGD manifest discovery"
        if discovery_start is not None
        else "unverified; discovery timing missing",
        "settings": {
            key: timing.get(key)
            for key in (
                "backend",
                "gms_backend",
                "pagebroker_gms",
                "weight_transfer_owner",
                "pagebroker_cpu_request",
                "pagebroker_cpu_limit",
                "gms_cpu_request",
                "gms_cpu_limit",
                "workers",
                "chunk_mib",
                "agent_revision",
            )
        },
        "ownership": evidence,
        "metrics": metrics,
        "validation": validation,
        "warnings": warnings,
        "rows": relative_rows(rows, zero),
        "hot_path_rows": relative_rows(hot_rows, zero),
        "prewarm_rows": relative_rows(prewarm, zero),
        "rank_rows": rank_rows,
        "notes": NOTES,
    }


def draw(path, data, key="rows"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = data[key]
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(17, 1.9 + 0.38 * len(rows)))
    used = set()
    for index, r in enumerate(rows):
        for s in r["segments"]:
            axis.barh(
                index,
                max(s["end"] - s["start"], 0.001),
                left=s["start"],
                height=0.68,
                color=COLORS[s["kind"]],
            )
            used.add(s["kind"])
    axis.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=9)
    axis.invert_yaxis()
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_xlabel(
        "Seconds relative to DGD request; GMS and PageBroker are independent resident services"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    if key != "prewarm_rows":
        axis.set_xlim(0, max(data["ready_offset_s"], 0.001))
    axis.set_title(
        f"{data['case']} · workload Ready +{data['ready_offset_s']:.3f} s", loc="left"
    )
    figure.legend(
        handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in COLORS if k in used],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)


HTML = """<!doctype html><meta charset="utf-8"><title>PageBroker → GMS restore timeline</title>
<style>body{font:15px system-ui;margin:26px;background:#f3f5f8;color:#243447}h1{font-size:26px}.card{background:white;border:1px solid #dce2e8;border-radius:10px;padding:18px;margin:14px 0}.owners{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.muted{color:#586878}button,select{padding:8px;margin-right:6px}svg{width:100%;min-width:1050px}#plot{overflow:auto}.legend{display:flex;flex-wrap:wrap;gap:14px}.swatch{display:inline-block;width:13px;height:13px;margin-right:5px}.metrics{display:flex;gap:24px;flex-wrap:wrap}.metrics strong{display:block;font-size:23px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #eee;text-align:right}td:first-child,th:first-child{text-align:left}</style>
<h1 id="heading">Independent PageBroker → GMS V1 restore</h1><div class="card owners"><div><b>Dynamo operator → DGD → engine Pod</b><p>CRIU restores captured process state. CUDA restores residual state. Engine reconnects to the matching GMS allocations.</p></div><div><b>Independent GMS DaemonSet</b><p id="gms-owner-description">One GPU per server container; servers are online before t0. GMS owns the saved allocation IDs and publishes only after native transfer commits.</p></div><div><b>Independent resident PageBroker</b><p id="pb-owner-description">One native GPU engine owns PVC O_DIRECT reads, pinned rings, imported VMM mappings and GPU copies. No Python loader CUDA context in this path.</p></div></div>
<div class="card"><select id="case"></select><button data-view="rows">Restore</button><button data-view="hot_path_rows">Startup detail</button><button data-view="prewarm_rows">Before t0</button><button data-view="rank_rows">Shards</button><select id="rank"></select><p class="muted" id="boundary"></p><div class="metrics" id="metrics"></div></div><div class="card"><div id="plot"></div><div id="legend" class="legend"></div></div><div class="card"><b>Evidence checks</b><pre id="proof"></pre></div><div class="card"><b>Trial comparison (settings are separate cohorts)</b><div id="comparison"></div></div><details class="card"><summary>Measurement boundaries</summary><div id="notes"></div></details>
<script>const DATA=__DATA__,COLORS=__COLORS__,LABELS=__LABELS__;let view='rows';const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>typeof x==='number'?x.toFixed(3):'unavailable';const select=document.querySelector('#case'),rank=document.querySelector('#rank');select.innerHTML=DATA.map((d,i)=>`<option value="${i}">${esc(d.case)}</option>`).join('');function render(){const d=DATA[+select.value],m=d.metrics,isPython=d.mode==='gms_python_loader';document.querySelector('#heading').textContent=isPython?'Independent GMS Python-loader control':'Independent PageBroker → GMS V1 restore';document.querySelector('#gms-owner-description').textContent=isPython?'Separate DaemonSet: each one-GPU rank container fuses the V1 server and Python loader. Startup is before t0.':'Separate DaemonSet: one GPU per GMS V1 server container, online before t0. GMS owns allocation IDs and publishes after native transfer commits.';document.querySelector('#pb-owner-description').textContent=isPython?'PageBroker restores engine residual state only. Python rank loaders own the weight transfer in this control.':'One native GPU engine owns PVC O_DIRECT reads, pinned rings, imported VMM mappings and GPU copies. No Python loader CUDA context. Artifact discovery follows DGD creation.';const prior=rank.value;rank.innerHTML=Object.keys(d.rank_rows).map(r=>`<option value="${r}">Rank ${r}</option>`).join('');if([...rank.options].some(o=>o.value===prior))rank.value=prior;rank.style.display=view==='rank_rows'?'inline':'none';const rows=view==='rank_rows'?(d.rank_rows[rank.value]||[]):d[view];const segs=rows.flatMap(r=>r.segments);let lo=Math.min(0,...segs.map(s=>s.start)),hi=Math.max(view==='prewarm_rows'?0:d.ready_offset_s,...segs.map(s=>s.end));if(hi<=lo)hi=lo+1;const left=395,w=1150,top=42,rh=28,H=top+rh*rows.length+45,x=t=>left+(t-lo)/(hi-lo)*(w-left-20);let svg=`<svg viewBox="0 0 ${w} ${H}" role="img" aria-label="Restore Gantt"><rect width="${w}" height="${H}" fill="white"/>`;for(let i=0;i<=10;i++){const t=lo+(hi-lo)*i/10;svg+=`<line x1="${x(t)}" x2="${x(t)}" y1="28" y2="${H-28}" stroke="#e2e7ed"/><text x="${x(t)}" y="20" text-anchor="middle" font-size="11">${t.toFixed(1)}s</text>`;}rows.forEach((r,i)=>{const y=top+i*rh;svg+=`<text x="${left-10}" y="${y+14}" text-anchor="end" font-size="11">${esc(r.label)}</text>`;r.segments.forEach(s=>{svg+=`<rect x="${x(s.start)}" y="${y}" width="${Math.max(1,x(s.end)-x(s.start))}" height="19" fill="${COLORS[s.kind]}"><title>${esc(r.label)}: ${fmt(s.start)} → ${fmt(s.end)} s (${fmt(s.end-s.start)} s)\n${esc(s.detail)}</title></rect>`;});});svg+=`<line x1="${x(0)}" x2="${x(0)}" y1="26" y2="${H-25}" stroke="#243447" stroke-dasharray="4,3"/><text x="${left}" y="${H-5}" font-size="11">Seconds relative to DGD request · independent services share a clock axis</text></svg>`;document.querySelector('#plot').innerHTML=svg;document.querySelector('#legend').innerHTML=[...new Set(segs.map(s=>s.kind))].map(k=>`<span><i class="swatch" style="background:${COLORS[k]}"></i>${esc(LABELS[k])}</span>`).join('');document.querySelector('#metrics').innerHTML=[['Workload Ready',m.dgd_to_workload_ready_s,'s'],['Restore handler',m.dgd_to_handler_s,'s'],['Payload window',m.payload_window_s,'s'],['Aggregate payload',m.payload_effective_GB_per_s,'GB/s'],['CRIU',m.criu_s,'s'],['CUDA',m.cuda_s,'s']].map(([k,v,u])=>`<div>${k}<strong>${fmt(v)} ${u}</strong></div>`).join('');document.querySelector('#boundary').textContent=view==='prewarm_rows'?'Negative time is measured initialization outside the restore timer; pale bars are already-online waiting.':'Hover for exact measured boundaries. Queue and read+copy are distinct; disk and CUDA subphases are not invented.';const proof={ownership:{...d.ownership,online:undefined},validation:{...d.validation,terminal_reports:undefined},warnings:d.warnings};document.querySelector('#proof').textContent=JSON.stringify(proof,null,2);document.querySelector('#notes').innerHTML=d.notes.map(n=>`<p>${esc(n)}</p>`).join('');}document.querySelectorAll('button').forEach(b=>b.onclick=()=>{view=b.dataset.view;render();});select.onchange=render;rank.onchange=render;const cols=[['Trial','case'],['Payload owner','mode'],['Ready','dgd_to_workload_ready_s'],['Handler','dgd_to_handler_s'],['Payload window','payload_window_s'],['GB/s','payload_effective_GB_per_s'],['CRIU','criu_s'],['CUDA','cuda_s'],['Native ring wait (sum)','native_residual_queue_seconds_sum']];document.querySelector('#comparison').innerHTML='<table><tr>'+cols.map(([k])=>`<th>${esc(k)}</th>`).join('')+'</tr>'+DATA.map(d=>'<tr>'+cols.map(([_,k])=>`<td>${k==='case'?esc(d.case):k==='mode'?esc(d.mode):fmt(d.metrics[k])}</td>`).join('')+'</tr>').join('')+'</table>';render();</script>"""


def write_html(path, cases):
    payload = json.dumps(cases).replace("<", "\\u003c")
    path.write_text(
        HTML.replace("__DATA__", payload)
        .replace("__COLORS__", json.dumps(COLORS))
        .replace("__LABELS__", json.dumps(LABELS))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--pagebroker-log", type=Path, help="Override log for a single case"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.pagebroker_log and len(args.cases) != 1:
        parser.error("--pagebroker-log requires exactly one case")
    results = []
    for case in args.cases:
        try:
            data = parse_case(case, args.pagebroker_log)
        except (ValueError, OSError, KeyError) as error:
            print(json.dumps({"case": str(case), "skipped": str(error)}))
            continue
        results.append(data)
        destination = args.output_dir / case.name if args.output_dir else case
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "pagebroker-timeline.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
        write_html(destination / "pagebroker-timeline.html", [data])
        if not args.no_plots:
            for key, stem in (
                ("rows", "pagebroker-timeline"),
                ("hot_path_rows", "pagebroker-hot-path"),
                ("prewarm_rows", "pagebroker-prewarm"),
            ):
                draw(destination / stem, data, key)
        print(
            json.dumps(
                {
                    "case": case.name,
                    "metrics": data["metrics"],
                    "warnings": data["warnings"],
                }
            )
        )
    if results:
        study = args.output_dir or args.cases[0].parent / "pagebroker-study"
        study.mkdir(parents=True, exist_ok=True)
        write_html(study / "timelines.html", results)
        (study / "timeline-comparison.json").write_text(
            json.dumps(
                [
                    {
                        k: d[k]
                        for k in (
                            "case",
                            "settings",
                            "metrics",
                            "validation",
                            "warnings",
                        )
                    }
                    for d in results
                ],
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
