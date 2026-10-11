# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process a completed capture session into an investigation bundle.

Run outside the serving frontend:

    python -m dynamo.profiling.bundle <session-dir> --otel-jsonl <collector.jsonl>

Lifecycle traces are selected by target identity and time overlap: a request is
included when its worker operation ran on a captured target during that target's
recording interval, or when the plan named its trace ID. Worker spans are
assigned to targets through router decisions (`kv_router.route_request`
worker_id = `handle_payload` instance_id), joined to the captured process by the
host/PID recorded in the manifest. Native events and lifecycle spans share one
timeline aligned to the coordinator clock; engine work inside an operation span
is shared batch work, not attributed per kernel.
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

FRONTEND = "frontend"


def attrs(span: dict) -> dict:
    return {
        a["key"]: next(iter(a["value"].values()), None)
        for a in span.get("attributes", [])
    }


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
        bounds = [item.get("recording_bounds")] + [
            a.get("recording_bounds") for a in item.get("artifacts", [])
        ]
        observed = [b for b in bounds if b and b.get("status") == "observed"]
        if observed:
            intervals[item["target_id"]] = (
                min(b["start_unix_ns"] for b in observed),
                max(b["end_unix_ns"] for b in observed),
            )
    return intervals


def clock_offsets(manifest: dict) -> dict[str, int]:
    offsets = {FRONTEND: 0}
    for target, clock in manifest["clock"]["workers"].items():
        if clock.get("start", {}).get("alignment") == "estimated":
            offsets[target] = clock["start"]["offset_ns"]
    return offsets


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
                epoch = str(attrs(span).get("dynamo.process.epoch", ""))
                mapping[payloads[span["parentSpanId"]]] |= targets_by_pid.get(
                    epoch.split(":")[0], set()
                )
    return {worker: min(t) for worker, t in mapping.items() if len(t) == 1}


def router_decision(span: dict | None, spans: dict) -> dict | None:
    while span is not None:
        if span["name"] == "kv_router.route_request":
            return attrs(span)
        span = spans.get(span.get("parentSpanId"))
    return None


def index_requests(manifest, traces, workers, intervals) -> list[dict]:
    triggers = set(manifest["plan"]["trace_ids"])
    entries = []
    for trace_id, spans in traces.items():
        phases = []
        for span in spans.values():
            if not span["name"].startswith("worker.operation."):
                continue
            decision = router_decision(spans.get(span.get("parentSpanId")), spans) or {}
            worker_id = decision.get("worker_id")
            target = workers.get(worker_id)
            start, end = int(span["startTimeUnixNano"]), int(span["endTimeUnixNano"])
            low, high = intervals.get(target, (0, 0))
            phases.append(
                {
                    "phase": span["name"].rsplit(".", 1)[1],
                    "router_worker_id": worker_id,
                    "target_id": target,
                    "operation_span_id": span["spanId"],
                    "start_unix_ns": start,
                    "end_unix_ns": end,
                    "recording_overlap_ns": max(0, min(end, high) - max(start, low)),
                }
            )
        if trace_id in triggers or any(p["recording_overlap_ns"] for p in phases):
            request_ids = {
                attrs(s).get("dynamo.request.id") for s in spans.values()
            } - {None}
            entries.append(
                {
                    "trace_id": trace_id,
                    "request_id": min(request_ids) if request_ids else None,
                    "trigger": trace_id in triggers,
                    "phases": sorted(phases, key=lambda p: p["start_unix_ns"]),
                }
            )
    return sorted(
        entries,
        key=lambda e: (e["phases"] or [{"start_unix_ns": 0}])[0]["start_unix_ns"],
    )


def kineto_events(path: Path):
    """Yield Kineto events with raw worker-clock nanosecond timestamps."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as stream:
        header = stream.read(1 << 20).decode(errors="replace")
        match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
        if not match or not 0 <= match.start() < header.find('"traceEvents"'):
            raise ValueError(f"{path}: baseTimeNanoseconds must precede traceEvents")
        base = int(match.group(1))
        stream.seek(0)
        for event in ijson.items(stream, "traceEvents.item"):
            if "ts" in event:
                event["ts_ns"] = base + round(event["ts"] * 1000)
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

    def emit(self, event: dict) -> None:
        self.stream.write(
            ("" if self.first else ",\n") + json.dumps(event, default=float)
        )
        self.first = False
        self.count += 1

    def pid(self, key, name: str) -> int:
        if key not in self.pids:
            self.pids[key] = len(self.pids) + 1
            self.emit(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": self.pids[key],
                    "args": {"name": name},
                }
            )
            self.emit(
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": self.pids[key],
                    "args": {"sort_index": self.pids[key]},
                }
            )
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

    def ts(self, unix_ns: int) -> float:
        return (unix_ns - self.origin) / 1000

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


def build(session: Path, otel_paths: list[Path], output: Path) -> dict:
    manifest = json.loads((session / "manifest.json").read_text())
    if manifest["state"] not in ("completed", "failed"):
        raise SystemExit(f"Session is still {manifest['state']}")
    traces = load_traces(otel_paths)
    intervals = recording_intervals(manifest)
    offsets = clock_offsets(manifest)
    workers = router_targets(manifest, traces)
    index = index_requests(manifest, traces, workers, intervals)
    selected = {entry["trace_id"] for entry in index}

    output.mkdir(parents=True, exist_ok=True)
    for folder in ("lifecycle", "control"):
        (output / folder).mkdir(exist_ok=True)
    shutil.copy2(session / "manifest.json", output / "manifest.json")
    # Keep the collector's OTLP export-batch representation, filtered to the
    # selected traces; traces are never cut at recording boundaries.
    with (output / "lifecycle/traces.otlp.jsonl").open("w") as lifecycle:
        for path in otel_paths:
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
    controls = []
    for profile in manifest["profiles"]:
        for key in (
            "start_requested_at_ns",
            "started_at_ns",
            "stop_requested_at_ns",
            "stopped_at_ns",
        ):
            if key in profile:
                controls.append(
                    {
                        "target_id": profile["target_id"],
                        "event": key[:-6],
                        "unix_ns": profile[key],
                    }
                )
    controls.sort(key=lambda event: event["unix_ns"])
    (output / "control/events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in controls)
    )

    # Shared timeline: lifecycle spans per process lane, then native events.
    spans_by_trace = {t: traces[t] for t in selected}
    starts = [
        int(s["startTimeUnixNano"])
        for spans in spans_by_trace.values()
        for s in spans.values()
    ]
    origin = min(
        starts
        + [low for low, _ in intervals.values()]
        + [e["unix_ns"] for e in controls]
    )
    timeline = Timeline(output / "timeline.json.gz", origin)
    for entry in index:
        spans = spans_by_trace[entry["trace_id"]]
        label = f"req {(entry['request_id'] or entry['trace_id'])[:8]}" + (
            " (trigger)" if entry["trigger"] else ""
        )
        for span in sorted(spans.values(), key=lambda s: int(s["startTimeUnixNano"])):
            decision = router_decision(spans.get(span.get("parentSpanId")), spans)
            worker_id = decision.get("worker_id") if decision else None
            lane = (
                FRONTEND
                if decision is None
                else workers.get(worker_id, f"worker {worker_id}")
            )
            shift = offsets.get(lane, 0)
            start, end = int(span["startTimeUnixNano"]), int(span["endTimeUnixNano"])
            pid = timeline.pid(("otel", lane), f"OTel {lane}")
            timeline.emit(
                {
                    "ph": "X",
                    "cat": "otel",
                    "name": span["name"],
                    "pid": pid,
                    "tid": timeline.tid(pid, label),
                    "ts": timeline.ts(start - shift),
                    "dur": max(0, end - start) / 1000,
                    "args": {
                        "trace_id": entry["trace_id"],
                        "span_id": span["spanId"],
                        "raw_start_unix_ns": start,
                        "clock_offset_ns": shift,
                        "router_worker_id": worker_id,
                    },
                }
            )
    for target, (low, high) in intervals.items():
        shift = offsets.get(target, 0)
        pid = timeline.pid(("capture", target), f"Recording {target}")
        timeline.emit(
            {
                "ph": "X",
                "cat": "capture",
                "name": f"{target} recording",
                "pid": pid,
                "tid": 1,
                "ts": timeline.ts(low - shift),
                "dur": (high - low) / 1000,
                "args": {"clock_offset_ns": shift},
            }
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
                "ts": timeline.ts(event["unix_ns"]),
            }
        )

    native = {}
    for item in manifest["artifact_validation"]:
        target = item["target_id"]
        collector = next(
            (
                p["collector"]
                for p in manifest["profiles"]
                if p.get("target_id") == target
            ),
            None,
        )
        reader = NATIVE_READERS.get(collector)
        if reader is None:
            native[target] = {
                "status": "pending",
                "reason": f"no timeline reader for {collector}",
            }
            continue
        if item["status"] != "verified":
            native[target] = {
                "status": "omitted",
                "reason": f"artifacts {item['status']}",
            }
            continue
        shift, count = offsets.get(target, 0), 0
        for artifact in item.get("artifacts", []):
            for event in reader(Path(artifact["location"])):
                key = (target, artifact.get("rank_id"), event.get("pid"))
                if event.get("ph") == "M":
                    if event.get("name") == "process_name":
                        timeline.pid(
                            key,
                            f"{target} {artifact.get('rank_id')} · {event['args'].get('name')} {event.get('pid')}",
                        )
                    elif event.get("name", "").startswith("thread_"):
                        event["pid"] = timeline.pid(
                            key,
                            f"{target} {artifact.get('rank_id')} · pid {event.get('pid')}",
                        )
                        timeline.emit(event)
                    continue
                if "ts_ns" not in event:
                    continue
                event["pid"] = timeline.pid(
                    key, f"{target} {artifact.get('rank_id')} · pid {event.get('pid')}"
                )
                event["ts"] = timeline.ts(event.pop("ts_ns") - shift)
                timeline.emit(event)
                count += 1
        native[target] = {
            "status": "included",
            "events": count,
            "clock_offset_ns": shift,
        }
    events = timeline.close()

    triggers = set(manifest["plan"]["trace_ids"])
    found_triggers = sorted(triggers & set(traces))
    summary = {
        "schema_version": 1,
        "session_id": manifest["session_id"],
        "method": __doc__.split("\n\n", 2)[2].strip(),
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "bundle.json"
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
                "trigger_traces_found": f"{len(found_triggers)}/{len(triggers)}",
                "unmapped_router_worker_ids": sorted(
                    {p["router_worker_id"] for e in index for p in e["phases"]}
                    - set(workers)
                    - {None}
                ),
                "completeness": "observed_spans_only",
            },
            "native_timeline": native,
        },
        "timeline_events": events,
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
