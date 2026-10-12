# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process a completed capture session into an investigation bundle.

Run outside the serving frontend:

    python -m dynamo.profiling.bundle <session-dir> --otel-jsonl <collector.jsonl>

A request is selected when its `worker.operation.<phase>` span ran on a captured
target during that target's recording interval, or when the plan named its
trace ID. Worker spans are assigned through router decisions
(`kv_router.route_request` worker_id = `handle_payload` instance_id) and joined
to captured processes by the PID in `dynamo.process.epoch`. Each request gets
its own Perfetto timeline: its lifecycle spans plus the native events of its
assigned workers that overlap its operation spans. Engine work inside an
operation span is shared batch work and is not attributed per kernel.
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import ijson


def attrs(span: dict) -> dict:
    return {
        a["key"]: next(iter(a["value"].values()), None)
        for a in span.get("attributes", [])
    }


def start_end(span: dict) -> tuple[int, int]:
    return int(span["startTimeUnixNano"]), int(span["endTimeUnixNano"])


def load_traces(paths: list[Path]) -> dict[str, dict[str, dict]]:
    traces: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for line in path.open():
            for resource in json.loads(line).get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        traces[span["traceId"]][span["spanId"]] = span
    return traces


def recording_intervals(manifest: dict) -> dict[str, tuple[int, int]]:
    """Observed native bounds per target; Kineto reports them per rank file."""
    intervals = {}
    for item in manifest["artifact_validation"]:
        observed = [
            b
            for b in [item.get("recording_bounds")]
            + [a.get("recording_bounds") for a in item.get("artifacts", [])]
            if b and b.get("status") == "observed"
        ]
        if observed:
            intervals[item["target_id"]] = (
                min(b["start_unix_ns"] for b in observed),
                max(b["end_unix_ns"] for b in observed),
            )
    return intervals


def router_targets(manifest: dict, traces: dict) -> dict[str, str]:
    """Map router worker IDs to captured targets through the worker PID.

    Worker spans under `handle_payload` carry `<pid>:<uuid>` process epochs, and
    the coordinator recorded each target's control-process PID. Span host
    attributes are not host identities, so a PID shared by targets stays unmapped.
    """
    targets_by_pid: dict[str, set[str]] = defaultdict(set)
    for target, clock in manifest["clock"]["workers"].items():
        if clock.get("start", {}).get("pid"):
            targets_by_pid[str(clock["start"]["pid"])].add(target)
    mapping: dict[str, set[str]] = defaultdict(set)
    for spans in traces.values():
        payloads = {
            s["spanId"]: attrs(s).get("instance_id")
            for s in spans.values()
            if s["name"] == "handle_payload"
        }
        for span in spans.values():
            if span.get("parentSpanId") in payloads:
                pid = str(attrs(span).get("dynamo.process.epoch", "")).split(":")[0]
                mapping[payloads[span["parentSpanId"]]] |= targets_by_pid.get(
                    pid, set()
                )
    return {worker: min(t) for worker, t in mapping.items() if len(t) == 1}


def router_decision(span: dict | None, spans: dict) -> tuple[str | None, str | None]:
    """(worker ID, phase) from the nearest ancestor router decision."""
    while span is not None:
        if span["name"] == "kv_router.route_request":
            decision = attrs(span)
            return decision.get("worker_id"), str(decision.get("phase", "")).lower()
        span = spans.get(span.get("parentSpanId"))
    return None, None


def router_worker(span: dict | None, spans: dict) -> str | None:
    return router_decision(span, spans)[0]


def index_requests(manifest, traces, workers, intervals) -> list[dict]:
    triggers = set(manifest["plan"]["trace_ids"])
    entries = []
    for trace_id, spans in traces.items():
        phases = []
        for span in spans.values():
            if span["name"].startswith("worker.operation."):
                worker_id = router_worker(span, spans)
                target = workers.get(worker_id)
                start, end = start_end(span)
                low, high = intervals.get(target, (0, 0))
                phases.append(
                    {
                        "phase": span["name"].rsplit(".", 1)[1],
                        "router_worker_id": worker_id,
                        "target_id": target,
                        "operation_span_id": span["spanId"],
                        "start_unix_ns": start,
                        "end_unix_ns": end,
                        "recording_overlap_ns": max(
                            0, min(end, high) - max(start, low)
                        ),
                    }
                )
        if trace_id in triggers or any(p["recording_overlap_ns"] for p in phases):
            request_ids = {
                attrs(s).get("dynamo.request.id") for s in spans.values()
            } - {None}
            entries.append(
                {
                    "trace_id": trace_id,
                    "request_id": min(request_ids, default=None),
                    "trigger": trace_id in triggers,
                    "phases": sorted(phases, key=lambda p: p["start_unix_ns"]),
                }
            )
    return sorted(
        entries,
        key=lambda e: min(start_end(s)[0] for s in traces[e["trace_id"]].values()),
    )


def kineto_events(path: Path):
    """Yield Kineto events with raw worker-clock nanosecond start/end."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as stream:
        header = stream.read(1 << 20).decode(errors="replace")
        match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
        if not match or not 0 <= match.start() < header.find('"traceEvents"'):
            raise ValueError(f"{path}: baseTimeNanoseconds must precede traceEvents")
        base = int(match.group(1))
        stream.seek(0)
        for event in ijson.items(stream, "traceEvents.item"):
            if "ts" in event:
                event["start_ns"] = base + round(event["ts"] * 1000)
                event["end_ns"] = event["start_ns"] + round(event.get("dur", 0) * 1000)
            yield event


# Native readers by collector; Nsight SQLite conversion is a later addition.
NATIVE_READERS = {"torch_profiler": kineto_events}


class Timeline:
    """Streaming Chrome/Perfetto JSON writer on the coordinator clock."""

    def __init__(self, path: Path, origin_ns: int):
        self.stream = gzip.open(path, "wt", compresslevel=1)
        self.stream.write('{"displayTimeUnit":"ms","traceEvents":[')
        self.origin, self.first, self.pids, self.tids, self.count = (
            origin_ns,
            True,
            {},
            {},
            0,
        )
        self.groups: dict[
            str, int
        ] = {}  # lane -> flow order (frontend, prefill, decode)

    def emit(self, event: dict) -> None:
        self.stream.write(
            ("" if self.first else ",\n") + json.dumps(event, default=float)
        )
        self.first, self.count = False, self.count + 1

    def pid(self, key, name: str, metadata=(), sort: int | None = None) -> int:
        if key not in self.pids:
            self.pids[key] = pid = len(self.pids) + 1
            sort = 1_000_000 + pid if sort is None else sort
            self.emit(
                {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}
            )
            self.emit(
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": pid,
                    "args": {"sort_index": sort},
                }
            )
            for event in metadata:
                self.emit({**event, "pid": pid})
        return self.pids[key]

    def tid(self, pid: int, name: str) -> int:
        if (pid, name) not in self.tids:
            self.tids[pid, name] = len(self.tids) + 1
            self.emit(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": self.tids[pid, name],
                    "args": {"name": name},
                }
            )
        return self.tids[pid, name]

    def interval(self, name, cat, start, end, pid, tid, args=None) -> None:
        self.emit(
            {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": pid,
                "tid": tid,
                "ts": (start - self.origin) / 1000,
                "dur": max(0, end - start) / 1000,
                "args": args or {},
            }
        )

    def close(self) -> int:
        self.stream.write("]}\n")
        self.stream.close()
        return self.count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_lifecycle(paths: list[Path], selected: set[str], output: Path) -> None:
    """Keep collector export batches, filtered to selected (uncut) traces."""
    with output.open("w") as lifecycle:
        for path in paths:
            for line in path.open():
                batch = json.loads(line)
                for resource in batch.get("resourceSpans", []):
                    for scope in resource.get("scopeSpans", []):
                        scope["spans"] = [
                            s
                            for s in scope.get("spans", [])
                            if s["traceId"] in selected
                        ]
                    resource["scopeSpans"] = [
                        s for s in resource.get("scopeSpans", []) if s["spans"]
                    ]
                batch["resourceSpans"] = [
                    r for r in batch.get("resourceSpans", []) if r["scopeSpans"]
                ]
                if batch["resourceSpans"]:
                    lifecycle.write(json.dumps(batch, separators=(",", ":")) + "\n")


def build(session: Path, otel_paths: list[Path], output: Path) -> dict:
    manifest = json.loads((session / "manifest.json").read_text())
    if manifest["state"] not in ("completed", "failed"):
        raise SystemExit(f"Session is still {manifest['state']}")
    traces = load_traces(otel_paths)
    intervals = recording_intervals(manifest)
    offsets = {"frontend": 0} | {
        target: clock["start"]["offset_ns"]
        for target, clock in manifest["clock"]["workers"].items()
        if clock.get("start", {}).get("alignment") == "estimated"
    }
    workers = router_targets(manifest, traces)
    index = index_requests(manifest, traces, workers, intervals)
    controls = sorted(
        (
            {"target_id": p["target_id"], "event": key[:-6], "unix_ns": p[key]}
            for p in manifest["profiles"]
            for key in (
                "start_requested_at_ns",
                "started_at_ns",
                "stop_requested_at_ns",
                "stopped_at_ns",
            )
            if key in p
        ),
        key=lambda event: event["unix_ns"],
    )

    for folder in ("lifecycle", "control", "timelines"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(session / "manifest.json", output / "manifest.json")
    write_lifecycle(
        otel_paths,
        {e["trace_id"] for e in index},
        output / "lifecycle/traces.otlp.jsonl",
    )
    (output / "control/events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in controls)
    )

    # One timeline per request, sharing one origin so files line up.
    origin = min(
        [
            start_end(s)[0]
            - offsets.get(
                workers.get(router_worker(s, traces[e["trace_id"]])) or "frontend", 0
            )
            for e in index
            for s in traces[e["trace_id"]].values()
        ]
        + [low - offsets.get(t, 0) for t, (low, _) in intervals.items()]
        + [e["unix_ns"] for e in controls]
    )
    timelines, windows = {}, defaultdict(list)
    for entry in index:
        spans = traces[entry["trace_id"]]
        name = entry["request_id"] or entry["trace_id"]
        entry["timeline"] = f"timelines/{name}.json.gz"
        timeline = timelines[entry["trace_id"]] = Timeline(
            output / entry["timeline"], origin
        )
        # Deployment flow: frontend, prefill and decode groups, each with its
        # OTel spans first and that worker's native tracks directly below.
        lanes = {}
        for span in spans.values():
            worker_id, phase = router_decision(span, spans)
            lanes[span["spanId"]] = (
                ("frontend", 0)
                if worker_id is None
                else (
                    workers.get(worker_id)
                    or f"{phase} worker {worker_id} (not captured)",
                    {"prefill": 1, "decode": 2}.get(phase, 3),
                )
            )
        timeline.groups = dict(set(lanes.values()))
        for lane, group in sorted(
            timeline.groups.items(), key=lambda item: (item[1], item[0])
        ):
            timeline.pid(("otel", lane), f"{lane} · OTel spans", sort=group * 1000)
        for span in sorted(spans.values(), key=lambda s: start_end(s)[0]):
            worker_id = router_worker(span, spans)
            lane = lanes[span["spanId"]][0]
            start, end = start_end(span)
            pid = timeline.pid(("otel", lane), f"{lane} · OTel spans")
            timeline.interval(
                span["name"],
                "otel",
                start - offsets.get(lane, 0),
                end - offsets.get(lane, 0),
                pid,
                timeline.tid(pid, "request"),
                {
                    "span_id": span["spanId"],
                    "raw_start_unix_ns": start,
                    "router_worker_id": worker_id,
                },
            )
        roots = [s for s in spans.values() if s["name"] == "http-request"]
        if "frontend" in intervals and roots:
            low, high = intervals["frontend"]
            start, end = start_end(roots[0])
            if start < high and end > low:
                windows["frontend"].append((max(start, low), min(end, high), timeline))
        for phase in entry["phases"]:
            target = phase["target_id"]
            if phase["recording_overlap_ns"]:
                low, high = intervals[target]
                windows[target].append(
                    (
                        max(phase["start_unix_ns"], low),
                        min(phase["end_unix_ns"], high),
                        timeline,
                    )
                )
                pid = timeline.pid(("otel", target), f"{target} · OTel spans")
                timeline.interval(
                    f"{target} recording",
                    "capture",
                    low - offsets[target],
                    high - offsets[target],
                    pid,
                    timeline.tid(pid, "native recording"),
                )
        for event in controls:
            timeline.emit(
                {
                    "ph": "i",
                    "s": "p",
                    "cat": "control",
                    "name": f"{event['target_id']} {event['event']}",
                    "pid": timeline.pid(("control",), "Control"),
                    "tid": 1,
                    "ts": (event["unix_ns"] - origin) / 1000,
                }
            )

    # Read each native trace once; route events to every overlapping request.
    native = {}
    for item in manifest["artifact_validation"]:
        target = item["target_id"]
        collector = next(
            (p["collector"] for p in manifest["profiles"] if p["target_id"] == target),
            None,
        )
        reader = NATIVE_READERS.get(collector)
        if reader is None or item["status"] != "verified":
            native[target] = {
                "status": "pending" if reader is None else "omitted",
                "reason": f"no timeline reader for {collector}"
                if reader is None
                else f"artifacts {item['status']}",
            }
            continue
        shift, routed, read = offsets.get(target, 0), 0, 0
        for artifact in item.get("artifacts", []):
            names, labels, threads = {}, {}, defaultdict(list)
            for event in reader(Path(artifact["location"])):
                read += 1
                key = (target, artifact.get("rank_id"), event.get("pid"))
                if event.get("ph") == "M":
                    if event.get("name") == "process_name":
                        names[key] = event["args"].get("name")
                    elif event.get("name") == "process_labels":
                        labels[key] = event["args"].get("labels")
                    elif event.get("name", "").startswith("thread_"):
                        threads[key].append(event)
                    continue
                if "start_ns" not in event:
                    continue
                start, end = event.pop("start_ns"), event.pop("end_ns")
                for low, high, timeline in windows[target]:
                    if start < high and max(end, start + 1) > low:
                        label = f"{target} · {names.get(key, 'native')} ({labels.get(key, event.get('pid'))})"
                        group = (
                            timeline.groups.get(target, 3) * 1000
                            + 100
                            + len(timeline.pids)
                        )
                        timeline.emit(
                            {
                                **event,
                                "pid": timeline.pid(key, label, threads[key], group),
                                "ts": (start - shift - origin) / 1000,
                            }
                        )
                        routed += 1
        native[target] = {
            "status": "included",
            "events_read": read,
            "events_written": routed,
            "clock_offset_ns": shift,
        }
    for entry in index:
        entry["timeline_events"] = timelines[entry["trace_id"]].close()
    (output / "lifecycle/trace-index.json").write_text(
        json.dumps(
            {
                "router_targets": workers,
                "recording_intervals": intervals,
                "requests": index,
            },
            indent=2,
        )
        + "\n"
    )

    triggers = set(manifest["plan"]["trace_ids"])
    summary = {
        "schema_version": 1,
        "session_id": manifest["session_id"],
        "method": __doc__.split("\n\n", 2)[2].strip(),
        "timeline_origin_unix_ns": origin,
        "files": {
            str(p.relative_to(output)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "bundle.json"
        },
        "native_profiles": [
            {
                k: a[k]
                for k in ("location", "rank_id", "sha256_uncompressed", "bytes")
                if k in a
            }
            | {"target_id": item["target_id"]}
            for item in manifest["artifact_validation"]
            for a in item.get("artifacts", [])
        ],
        "coverage": {
            "profiler": manifest["artifact_status"],
            "lifecycle": {
                "status": "observed" if index else "missing",
                "requests": len(index),
                "trigger_traces_found": f"{len(triggers & set(traces))}/{len(triggers)}",
                "uncaptured_router_worker_ids": sorted(
                    {p["router_worker_id"] for e in index for p in e["phases"]}
                    - set(workers)
                    - {None}
                ),
                "completeness": "observed_spans_only",
            },
            "native_timeline": native,
        },
    }
    (output / "bundle.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "session", type=Path, help="session directory containing manifest.json"
    )
    parser.add_argument(
        "--otel-jsonl",
        type=Path,
        action="append",
        required=True,
        help="collector OTLP export-batch JSONL; repeat for multiple files",
    )
    parser.add_argument(
        "--output", type=Path, help="bundle directory (default: <session>/bundle)"
    )
    args = parser.parse_args()
    summary = build(
        args.session, args.otel_jsonl, args.output or args.session / "bundle"
    )
    print(json.dumps(summary["coverage"], indent=2))


if __name__ == "__main__":
    main()
