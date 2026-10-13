# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process a completed capture session into an investigation bundle.

Run outside the serving frontend:

    python -m dynamo.profiling.bundle <session-dir> --otel-jsonl <collector.jsonl>

writes <session>/bundle/ and <session>/bundle.tar.gz; --include-profiles also
packs the native Kineto/Nsight files, which are otherwise referenced by hash.

A request is selected when its `worker.operation.<phase>` span ran on a captured
target during that target's recording interval, or when the plan named its
trace ID. Worker spans are assigned through router decisions
(`kv_router.route_request` worker_id = `handle_payload` instance_id) and joined
to captured processes by the PID in `dynamo.process.epoch`. Each request gets
its own Perfetto timeline: its lifecycle span tree plus the native events of its
assigned workers that overlap its operation spans. The frontend recording
window and control events are drawn; its detailed host trace is listed only. Engine work inside an
operation span is shared batch work and is not attributed per kernel.
"""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import tarfile
from collections import defaultdict
from contextlib import closing
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


def nsys_events(path: Path):
    """Yield Nsight SQLite export events as Chrome events with raw host-clock ns.

    The native `.nsys-rep` is kept as is; its automatic SQLite export is read.
    Python function ranges arrive as NVTX ranges in the PyTorch domain.
    """
    if path.suffix != ".sqlite":
        return
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
        # Event timestamps are relative to the session start anchored here.
        (epoch,) = db.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        domains = (
            dict(
                db.execute(
                    "SELECT domainId, text FROM NVTX_EVENTS WHERE eventType = 75"
                )
            )
            if "NVTX_EVENTS" in tables
            else {}
        )
        for pid, name in db.execute("SELECT pid, name FROM PROCESSES"):
            yield {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "args": {"name": name},
            }
            yield {
                "ph": "M",
                "name": "process_labels",
                "pid": pid,
                "args": {"labels": "CPU"},
            }
        if "ThreadNames" in tables:
            for name_id, tid in db.execute("SELECT nameId, globalTid FROM ThreadNames"):
                yield {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": (tid >> 24) & 0xFFFFFF,
                    "tid": tid & 0xFFFFFF,
                    "args": {"name": strings.get(name_id, "")},
                }
        devices = set()
        queries = {
            "nvtx": "SELECT start, end, COALESCE(text, textId), globalTid, domainId FROM NVTX_EVENTS "
            "WHERE eventType IN (59, 60) AND end IS NOT NULL",
            "osrt": "SELECT start, end, nameId, globalTid, NULL FROM OSRT_API",
            "cuda_runtime": "SELECT start, end, nameId, globalTid, NULL FROM CUPTI_ACTIVITY_KIND_RUNTIME",
            "cuda_driver": "SELECT start, end, nameId, globalTid, NULL FROM CUPTI_ACTIVITY_KIND_DRIVER",
            "kernel": "SELECT start, end, demangledName, deviceId, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL",
            "gpu_memcpy": "SELECT start, end, NULL, deviceId, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY",
            "gpu_memset": "SELECT start, end, NULL, deviceId, streamId FROM CUPTI_ACTIVITY_KIND_MEMSET",
        }
        for cat, query in queries.items():
            if query.split(" FROM ")[1].split()[0] not in tables:
                continue
            gpu = cat in ("kernel", "gpu_memcpy", "gpu_memset")
            for start, end, name, lane, extra in db.execute(query):
                if gpu:
                    pid, tid = f"GPU {lane}", extra
                    if lane not in devices:
                        devices.add(lane)
                        yield {
                            "ph": "M",
                            "name": "process_name",
                            "pid": pid,
                            "args": {"name": f"CUDA device {lane}"},
                        }
                        yield {
                            "ph": "M",
                            "name": "process_labels",
                            "pid": pid,
                            "args": {"labels": pid},
                        }
                else:
                    pid, tid = (lane >> 24) & 0xFFFFFF, lane & 0xFFFFFF
                kind = (
                    {"PyTorch": "pytorch", "vLLM": "vllm"}.get(domains.get(extra), cat)
                    if cat == "nvtx"
                    else cat
                )
                yield {
                    "ph": "X",
                    "cat": kind,
                    "name": strings.get(name, name) if name is not None else cat,
                    "pid": pid,
                    "tid": tid,
                    "dur": (end - start) / 1000,
                    "start_ns": epoch + start,
                    "end_ns": epoch + end,
                }


# Native readers by collector.
NATIVE_READERS = {"torch_profiler": kineto_events, "nsys": nsys_events}


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
        # One OTel span tree per request: worker spans stay nested under the
        # frontend router decisions, each shifted onto its worker's clock.
        lanes = {}
        for span in spans.values():
            # A router decision span itself runs on the frontend.
            parent = spans.get(span.get("parentSpanId"))
            worker_id, phase = router_decision(parent, spans)
            lanes[span["spanId"]] = (
                ("frontend", 0)
                if worker_id is None
                else (
                    workers.get(worker_id)
                    or f"{phase} worker {worker_id} (not captured)",
                    {"prefill": 1, "decode": 2}.get(phase, 3),
                )
            )
        # Native tracks follow in flow order: frontend, prefill, decode.
        timeline.groups = dict(set(lanes.values()))
        otel = timeline.pid(("otel",), "Request · OTel spans", sort=0)
        recording = timeline.pid(("recording",), "Recording windows", sort=1)
        for span in sorted(
            spans.values(), key=lambda s: (start_end(s)[0], -start_end(s)[1])
        ):
            lane = lanes[span["spanId"]][0]
            start, end = start_end(span)
            timeline.interval(
                span["name"],
                "otel",
                start - offsets.get(lane, 0),
                end - offsets.get(lane, 0),
                otel,
                timeline.tid(otel, "spans"),
                {
                    "span_id": span["spanId"],
                    "process": lane,
                    "raw_start_unix_ns": start,
                    "router_worker_id": router_worker(
                        spans.get(span.get("parentSpanId")), spans
                    ),
                },
            )
        # The frontend recording window and control events stay visible; only
        # its detailed host trace is left off the timeline (see below).
        recorded = {}
        roots = [s for s in spans.values() if s["name"] == "http-request"]
        if "frontend" in intervals and roots:
            start, end = start_end(roots[0])
            low, high = intervals["frontend"]
            if start < high and end > low:
                recorded["frontend"] = (low, high)
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
                recorded[target] = (low, high)
        for target, (low, high) in recorded.items():
            shift = offsets.get(target, 0)
            timeline.interval(
                f"{target} recording",
                "capture",
                low - shift,
                high - shift,
                recording,
                timeline.tid(recording, target),
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
        if target == "frontend":
            # The detailed frontend host trace is captured and listed but not
            # drawn until it carries CPU samples or per-request ranges.
            native[target] = {
                "status": "excluded",
                "reason": "frontend host trace detail",
            }
            continue
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
                # Native data exists only inside recording; clip events that
                # began before or ended after it and keep their original bounds.
                recorded_low, recorded_high = intervals[target]
                if start < recorded_low or end > recorded_high:
                    event["args"] = {
                        **event.get("args", {}),
                        "original_start_unix_ns": start,
                        "original_end_unix_ns": end,
                    }
                    start, end = max(start, recorded_low), min(end, recorded_high)
                    event["dur"] = max(0, end - start) / 1000
                for low, high, timeline in windows[target]:
                    if start < high and max(end, start + 1) > low:
                        # Kineto keeps profiler bookkeeping in unnamed pseudo-processes.
                        label = (
                            f"{target} · {names[key]} ({labels.get(key, event.get('pid'))})"
                            if key in names
                            else f"{target} · Kineto profiler ({event.get('pid')})"
                        )
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


def archive(output: Path, summary: dict, include_profiles: bool) -> Path:
    """Package the bundle directory, optionally with the native profiles."""
    path = output.parent / f"{output.name}.tar.gz"
    with tarfile.open(path, "w:gz", compresslevel=1) as tar:
        tar.add(output, arcname=output.name)
        if include_profiles:
            for profile in summary["native_profiles"]:
                source = Path(profile["location"])
                tar.add(
                    source,
                    arcname=f"{output.name}/profiles/{profile['target_id']}/{source.name}",
                )
    path.with_name(path.name + ".sha256").write_text(f"{sha256(path)}  {path.name}\n")
    return path


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
    parser.add_argument(
        "--include-profiles",
        action="store_true",
        help="copy native profiles into the archive instead of referencing them",
    )
    args = parser.parse_args()
    output = args.output or args.session / "bundle"
    summary = build(args.session, args.otel_jsonl, output)
    path = archive(output, summary, args.include_profiles)
    print(json.dumps(summary["coverage"], indent=2))
    print(f"archive: {path} ({path.stat().st_size} bytes, sha256 {sha256(path)})")


if __name__ == "__main__":
    main()
