# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process a completed capture session into an investigation bundle.

Run outside the serving frontend:

    python -m dynamo.profiling.bundle <session-dir> --otel-jsonl <collector.jsonl>

writes <session>/bundle/ and <session>/bundle.tar.gz (with a .sha256 sidecar)
carrying the raw native Kineto/Nsight traces under profiles/<target>/. Kineto
sessions get timeline.json.gz, an overall Perfetto timeline of the capture
window. Nsight sessions get reports/: the native reports, one of them (the frontend's)
with every request's span tree, the recording windows and control events added
as NVTX rows at the top of the multi-report view (Nsight Systems' nsys_writer,
or ImportNvtxt as a fallback), plus the same ranges as an NVTXT text file;
--perfetto also writes the Perfetto timeline, which is the fallback when
neither tool is available. --per-request also writes one Perfetto timeline per
request. Run on an extracted bundle (with --output) to regenerate any view from
the archive alone.

A request is selected when its `worker.operation[.<phase>]` span (no phase suffix
in aggregated serving) ran on a captured
target during that target's recording interval, or when the plan named its
trace ID. Worker spans are assigned through router decisions
(`kv_router.route_request` worker_id = `handle_payload` instance_id) and joined
to captured targets by the worker instance ID their `profiling_info` reported
(older manifests: the PID in `dynamo.process.epoch`). The overall
timeline draws every selected request's lifecycle span tree and each captured
worker's native events over its recording window; per-request timelines keep
only the native events overlapping that request's operation spans. The frontend
recording window and control events are drawn; its detailed host trace is listed
only. Engine work inside an operation span is shared batch work and is not
attributed per kernel.
"""

import argparse
import io
import json
import os
import shutil
import struct
import tarfile
import tempfile
import zlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from . import nvtxt
from .inspection import inspect_rank, inspect_target, kineto_result, sha256_file
from .native import NATIVE_READERS, draw_shard, process_meta


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


def cpus() -> int:
    # The CPUs this job may use (Slurm affinity), not the whole node.
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def pool(tasks: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=max(1, min(cpus(), tasks)))


class ParallelGzip(io.RawIOBase):
    """Single-member gzip written by compressing 8 MiB blocks on threads.

    Like pigz, each block is raw deflate ended with a full flush, so the
    concatenation is one valid deflate stream; zlib releases the GIL while
    compressing. The CRC is computed in order over the uncompressed data.
    """

    BLOCK = 8 << 20

    def __init__(self, path: Path, level: int = 1):
        self.file = path.open("wb")
        self.file.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff")
        self.level, self.crc, self.size = level, 0, 0
        self.pending, self.buffer = deque(), bytearray()
        self.workers = max(1, cpus())
        self.threads = ThreadPoolExecutor(max_workers=self.workers)

    def writable(self) -> bool:
        return True

    def deflate(self, block: bytes, mode: int) -> bytes:
        compressor = zlib.compressobj(self.level, zlib.DEFLATED, -15)
        return compressor.compress(block) + compressor.flush(mode)

    def submit(self, block: bytes, mode: int) -> None:
        self.crc = zlib.crc32(block, self.crc)
        self.size += len(block)
        self.pending.append(self.threads.submit(self.deflate, block, mode))
        while len(self.pending) > 2 * self.workers:
            self.file.write(self.pending.popleft().result())

    def write(self, data) -> int:
        self.buffer += data
        while len(self.buffer) >= self.BLOCK:
            self.submit(bytes(self.buffer[: self.BLOCK]), zlib.Z_FULL_FLUSH)
            del self.buffer[: self.BLOCK]
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self.submit(bytes(self.buffer), zlib.Z_FINISH)
            while self.pending:
                self.file.write(self.pending.popleft().result())
            self.file.write(struct.pack("<II", self.crc, self.size & 0xFFFFFFFF))
            self.file.close()
            self.threads.shutdown()
        super().close()


def verify_native(manifest: dict) -> dict[str, dict]:
    """Inspect every native file in parallel; the capture only checked presence."""
    profiles = {p["target_id"]: p for p in manifest["profiles"]}
    kineto, others = [], []
    for item in manifest["artifact_validation"]:
        collector = profiles.get(item["target_id"], {}).get("collector")
        ranked = collector == "torch_profiler" and item.get("artifacts")
        (kineto if ranked else others).append(item)
    ranks = [artifact for item in kineto for artifact in item["artifacts"]]
    evidence = [profiles.get(item["target_id"], {}) for item in others]
    with pool(len(ranks) + len(others)) as executor:
        inspected = iter(executor.map(inspect_rank, ranks))
        targets = executor.map(inspect_target, others, evidence)
        results = {item["target_id"]: r for item, r in zip(others, targets)}
        for item in kineto:
            results[item["target_id"]] = kineto_result(
                item, [next(inspected) for _ in item["artifacts"]]
            )
    return results


def recording_intervals(items: list[dict]) -> dict[str, tuple[int, int]]:
    """Observed native bounds per target; Kineto reports them per rank file."""
    intervals = {}
    for item in items:
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
    """Map router worker IDs to captured targets.

    Workers report their runtime connection ID, which the router records as the
    worker ID (`handle_payload` instance_id), so the join holds across hosts.
    Manifests without it fall back to the PID in the `<pid>:<uuid>` process
    epochs of worker spans; PIDs repeat across hosts, so a PID shared by targets
    stays unmapped.
    """
    by_instance: dict[str, str] = {}
    targets_by_pid: dict[str, set[str]] = defaultdict(set)
    for target, clock in manifest["clock"]["workers"].items():
        start = clock.get("start", {})
        if start.get("worker_instance_id") is not None:
            by_instance[str(start["worker_instance_id"])] = target
        elif start.get("pid"):
            targets_by_pid[str(start["pid"])].add(target)
    mapping: dict[str, set[str]] = defaultdict(set)
    for spans in traces.values():
        payloads = {
            s["spanId"]: attrs(s).get("instance_id")
            for s in spans.values()
            if s["name"] == "handle_payload"
        }
        for worker in payloads.values():
            if str(worker) in by_instance:
                mapping[worker].add(by_instance[str(worker)])
        for span in spans.values():
            worker = payloads.get(span.get("parentSpanId"))
            if worker is not None and str(worker) not in by_instance:
                pid = str(attrs(span).get("dynamo.process.epoch", "")).split(":")[0]
                mapping[worker] |= targets_by_pid.get(pid, set())
    return {worker: min(t) for worker, t in mapping.items() if len(t) == 1}


def router_decision(span: dict | None, spans: dict) -> tuple[str | None, str | None]:
    """(worker ID, phase) from the nearest ancestor router decision."""
    while span is not None:
        if span["name"] == "kv_router.route_request":
            decision = attrs(span)
            return decision.get("worker_id"), str(decision.get("phase", "")).lower()
        span = spans.get(span.get("parentSpanId"))
    return None, None


def index_requests(manifest, traces, workers, intervals) -> list[dict]:
    triggers = set(manifest["plan"]["trace_ids"])
    entries = []
    for trace_id, spans in traces.items():
        phases = []
        for span in spans.values():
            name = span["name"]
            if name == "worker.operation" or name.startswith("worker.operation."):
                worker_id = router_decision(span, spans)[0]
                target = workers.get(worker_id)
                start, end = start_end(span)
                low, high = intervals.get(target, (0, 0))
                phases.append(
                    {
                        # Aggregated workers emit an unsuffixed operation span.
                        "phase": name.removeprefix("worker.operation").lstrip(".")
                        or "aggregated",
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
            values = [attrs(s) for s in spans.values()]
            ids = {
                key: min({a.get(key) for a in values} - {None}, default=None)
                for key in ("dynamo.request.id", "dynamo.session.id")
            }
            entries.append(
                {
                    "trace_id": trace_id,
                    "request_id": ids["dynamo.request.id"],
                    "session_id": ids["dynamo.session.id"],
                    "trigger": trace_id in triggers,
                    "phases": sorted(phases, key=lambda p: p["start_unix_ns"]),
                }
            )
    return sorted(
        entries,
        key=lambda e: min(start_end(s)[0] for s in traces[e["trace_id"]].values()),
    )


class Timeline:
    """Streaming Chrome/Perfetto JSON writer on the coordinator clock."""

    def __init__(self, path: Path, origin_ns: int):
        self.stream = io.TextIOWrapper(
            io.BufferedWriter(ParallelGzip(path), 1 << 20), encoding="utf-8"
        )
        self.stream.write('{"displayTimeUnit":"ms","traceEvents":[')
        self.origin, self.count, self.pids, self.tids = origin_ns, 0, {}, {}

    def emit(self, event: dict) -> None:
        separator = ",\n" if self.count else ""
        self.stream.write(separator + json.dumps(event, default=float))
        self.count += 1

    def pid(self, key, name: str, sort: int) -> int:
        if key not in self.pids:
            self.pids[key] = len(self.pids) + 1
            for event in process_meta(self.pids[key], name, "sort_index", sort):
                self.emit(event)
        return self.pids[key]

    def tid(self, pid: int, name: str) -> int:
        if (pid, name) not in self.tids:
            tid = self.tids[pid, name] = len(self.tids) + 1
            meta = {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid}
            self.emit(meta | {"args": {"name": name}})
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

    def extend(self, shard: str, count: int) -> None:
        """Append a pool-drawn shard of comma-joined events."""
        if count:
            self.stream.write(",\n" if self.count else "")
            with open(shard) as source:
                shutil.copyfileobj(source, self.stream, 1 << 20)
            self.count += count

    def close(self) -> int:
        self.stream.write("]}\n")
        self.stream.close()
        return self.count


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


def request_lanes(spans: dict, workers: dict) -> dict:
    """Lane, flow group and router worker ID per span; routing is on the frontend."""
    lanes = {}
    for span in spans.values():
        worker_id, phase = router_decision(spans.get(span.get("parentSpanId")), spans)
        lanes[span["spanId"]] = (
            ("frontend", 0, None)
            if worker_id is None
            else (
                workers.get(worker_id) or f"{phase} worker {worker_id} (not captured)",
                {"prefill": 1, "decode": 2}.get(phase, 3),
                worker_id,
            )
        )
    return lanes


def draw_request(timeline, spans: dict, lanes: dict, offsets: dict, track: str) -> None:
    """One request's span tree on its own track; worker spans on worker clocks."""
    pid = timeline.pid(("otel",), "Requests · OTel spans", sort=0)
    tid = timeline.tid(pid, track)
    for span in sorted(
        spans.values(), key=lambda s: (start_end(s)[0], -start_end(s)[1])
    ):
        lane, _, worker_id = lanes[span["spanId"]]
        start, end = start_end(span)
        timeline.interval(
            span["name"],
            "otel",
            start - offsets.get(lane, 0),
            end - offsets.get(lane, 0),
            pid,
            tid,
            {
                "span_id": span["spanId"],
                "process": lane,
                "raw_start_unix_ns": start,
                "router_worker_id": worker_id,
            }
            | (
                {"session_id": session}
                if (session := attrs(span).get("dynamo.session.id"))
                else {}
            ),
        )


def draw_window(timeline, recorded: dict, offsets: dict, controls: list) -> None:
    """Recording windows per target and control events."""
    recording = timeline.pid(("recording",), "Recording windows", sort=1)
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
                "pid": timeline.pid(("control",), "Control", sort=10**9),
                "tid": 1,
                "ts": (event["unix_ns"] - timeline.origin) / 1000,
            }
        )


def request_label(entry: dict) -> str:
    route = " → ".join(
        p["target_id"] or f"{p['phase']} (not captured)" for p in entry["phases"]
    )
    label = f"req {(entry['request_id'] or entry['trace_id'])[:8]} · {route}"
    if entry.get("session_id"):
        label = f"session {entry['session_id'][:8]} · {label}"
    return label + (" (trigger)" if entry["trigger"] else "")


def by_session(index: list[dict]) -> list[dict]:
    """Requests grouped by client session (in first-seen order), then by start."""

    def group(entry: dict) -> str:
        return entry.get("session_id") or entry["trace_id"]

    first: dict = {}
    for entry in index:
        first.setdefault(group(entry), len(first))
    return sorted(index, key=lambda entry: first[group(entry)])


def nsight_reports(
    output, checked, index, traces, lanes, offsets, intervals, controls, tools
) -> dict:
    """Package the Nsight reports for the multi-report view, with the request
    flow imported as NVTX into one of them, as in the Perfetto view.

    Every selected request's span tree goes to the frontend report (or, without
    one, to each worker report it ran on): one thread per request with nested
    push/pop ranges, a thread per recording window, and control-event markers.
    Times move from each span's lane clock to the report's clock with the
    measured offsets and are clipped to the report's session. Other reports are
    copied unchanged into reports/.
    """
    (output / "reports").mkdir(parents=True, exist_ok=True)
    usable = {}
    reports = {}
    for item in checked:
        paths = {
            Path(a["location"]).suffix: Path(a["location"])
            for a in item.get("artifacts", [])
        }
        if item["status"] == "verified" and {".nsys-rep", ".sqlite"} <= set(paths):
            usable[item["target_id"]] = paths
        else:
            reports[item["target_id"]] = {
                "status": "omitted",
                "reason": f"artifacts {item['status']}",
            }
    hosts = ["frontend"] if "frontend" in usable else list(usable)
    for target, paths in usable.items():
        report = f"reports/{target}.nsys-rep"
        copy = output / report
        if target not in hosts:
            shutil.copy2(paths[".nsys-rep"], copy)
            reports[target] = {"status": "native", "report": report}
            continue
        utc, clock, stop = nvtxt.report_clock(paths[".sqlite"])
        low, high = clock, stop if stop is not None else 2**62
        shift = offsets.get(target, 0)

        def local(unix_ns: int, lane: str = "frontend") -> int:
            # Lane clock -> coordinator clock -> this host -> report clock.
            return unix_ns - offsets.get(lane, 0) + shift - utc + clock

        def clipped(begin: int, end: int, lane: str, message: str, color: int):
            start, end = local(begin, lane), local(end, lane)
            if end < low or start > high:
                return None
            if start < low:
                message += f" (began {(low - start) / 1e9:.1f} s before the report)"
            if end > high:
                message += f" (ended {(end - high) / 1e9:.1f} s after the report)"
            return max(start, low), min(end, high), message, color

        rows, dropped = [("Capture control", [])], 0  # numbered from CONTROL_THREAD
        for window, (begin, end) in sorted(intervals.items()):
            # Recording bounds are on each target's own clock.
            span = clipped(begin, end, window, f"{window} recording", nvtxt.RECORDING)
            if span:
                rows.append((f"Recording · {window}", [span]))
        marks = [
            (time, nvtxt.CONTROL_THREAD, f"{event['target_id']} {event['event']}")
            for event in controls
            if low <= (time := local(event["unix_ns"])) <= high
        ]
        for entry in by_session(index):
            if target != "frontend" and target not in {
                p["target_id"] for p in entry["phases"]
            }:
                continue
            spans = {}
            for span in traces[entry["trace_id"]].values():
                lane = lanes[entry["trace_id"]][span["spanId"]][0]
                begin, end = start_end(span)
                message = f"{span['name']} · {lane}"
                if span["name"] in ("http-request", "request.lifecycle"):
                    message += f" · request {entry['request_id'] or entry['trace_id']}"
                    if entry.get("session_id"):
                        message += f" · session {entry['session_id']}"
                    message += f" · trace {entry['trace_id']}"
                    if client := attrs(span).get("x_request_id"):
                        message += f" · x-request-id {client}"
                kept = clipped(begin, end, lane, message, nvtxt.color(span["name"]))
                dropped += kept is None
                if kept:
                    spans[span["spanId"]] = (*kept, span.get("parentSpanId"))
            label = request_label(entry)
            rows += [
                (f"{label} (overlap {number})" if number else label, nested)
                for number, nested in enumerate(nvtxt.tree_threads(spans))
            ]
        threads = dict(enumerate(rows, nvtxt.CONTROL_THREAD))
        text = output / "reports" / f"{target}.nvtxt"
        text.write_text(nvtxt.render(threads, marks))
        entry = {
            "nvtxt": f"reports/{target}.nvtxt",
            "ranges": sum(len(spans) for _, spans in rows),
            "marks": len(marks),
            "spans_outside_report": dropped,
        }
        errors = []
        for method, tool in zip(("nsys_writer", "ImportNvtxt"), tools):
            if tool is None:
                continue
            try:
                imported = (
                    nvtxt.write(tool, paths[".nsys-rep"], copy, threads, marks, clock)
                    if method == "nsys_writer"
                    else nvtxt.merge(tool, paths[".nsys-rep"], text, copy)
                )
            except Exception as exc:  # writer/backend or importer failure
                errors.append(f"{method}: {exc}")
                continue
            reports[target] = (
                entry
                | {
                    "status": "merged",
                    "method": method,
                    "report": report,
                    "events_imported": imported,
                    "clock_offset_ns": shift,
                }
                | ({"fallback_from": errors} if errors else {})
            )
            break
        else:
            # Without request rows the native report still opens with the rest.
            shutil.copy2(paths[".nsys-rep"], copy)
            reason = "; ".join(errors) or "neither nsys_writer nor ImportNvtxt found"
            reports[target] = entry | {
                "status": "failed" if errors else "nvtxt_only",
                "reason": reason,
            }
    return reports


def build(
    session: Path,
    otel_paths: list[Path],
    output: Path,
    per_request: bool = False,
    perfetto: bool = False,
    nsys_root: str | None = None,
) -> dict:
    manifest = json.loads((session / "manifest.json").read_text())
    if (session / "profiles").is_dir():
        # An extracted bundle: read its packed raw traces, not the original paths.
        for item in manifest["artifact_validation"]:
            folder = session / "profiles" / item["target_id"]
            for artifact in item.get("artifacts", []):
                packed = folder / Path(artifact["location"]).name
                if packed.exists():
                    artifact["location"] = str(packed)
    if manifest["state"] not in ("completed", "failed"):
        raise SystemExit(f"Session is still {manifest['state']}")
    # Each build starts from an empty output, which must not hold its inputs.
    inputs = [session, *otel_paths] + [
        Path(artifact["location"])
        for item in manifest["artifact_validation"]
        for artifact in item.get("artifacts", [])
    ]
    if any(path.resolve().is_relative_to(output.resolve()) for path in inputs):
        raise SystemExit(f"Output {output} contains the session or its inputs")
    if output.exists():
        if (output / "bundle.json").is_file():
            shutil.rmtree(output)
        elif not output.is_dir() or any(output.iterdir()):
            raise SystemExit(f"Output {output} exists and is not an earlier bundle")
    # Full verification happens here, off the capture path; its results give
    # the native recording bounds (Kineto bounds need a full trace parse).
    verification = verify_native(manifest)
    checked = []
    for item in manifest["artifact_validation"]:
        result = verification[item["target_id"]]
        merged = {**item, **result, "target_id": item["target_id"]}
        bounds = result.get("recording_bounds") or {}
        if bounds.get("status") != "observed" and item.get("recording_bounds"):
            merged["recording_bounds"] = item["recording_bounds"]
        checked.append(merged)
    traces = load_traces(otel_paths)
    intervals = recording_intervals(checked)
    offsets = {"frontend": 0} | {
        target: clock["start"]["offset_ns"]
        for target, clock in manifest["clock"]["workers"].items()
        if clock.get("start", {}).get("alignment") == "estimated"
    }
    workers = router_targets(manifest, traces)
    index = index_requests(manifest, traces, workers, intervals)
    stages = ("start_requested", "started", "stop_requested", "stopped")
    controls = sorted(
        (
            {"target_id": p["target_id"], "event": e, "unix_ns": p[f"{e}_at_ns"]}
            for p in manifest["profiles"]
            for e in stages
            if f"{e}_at_ns" in p
        ),
        key=lambda event: event["unix_ns"],
    )

    for folder in ("lifecycle", "control") + (("timelines",) if per_request else ()):
        (output / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(session / "manifest.json", output / "manifest.json")
    selected = {e["trace_id"] for e in index}
    write_lifecycle(otel_paths, selected, output / "lifecycle/traces.otlp.jsonl")
    (output / "control/events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in controls)
    )

    # All timelines share one origin so files line up; spans use their drawn lane.
    lanes = {
        e["trace_id"]: request_lanes(traces[e["trace_id"]], workers) for e in index
    }
    origin = min(
        [
            start_end(s)[0] - offsets.get(lanes[e["trace_id"]][s["spanId"]][0], 0)
            for e in index
            for s in traces[e["trace_id"]].values()
        ]
        + [low - offsets.get(t, 0) for t, (low, _) in intervals.items()]
        + [e["unix_ns"] for e in controls],
        default=manifest.get("accepted_at_ns", 0),
    )
    # Nsight sessions default to their native reports with the request spans
    # imported as NVTX; Kineto (and Nsight on request) get the Perfetto timeline.
    collectors = {p["target_id"]: p["collector"] for p in manifest["profiles"]}
    nsight = bool(collectors) and set(collectors.values()) == {"nsys"}
    reports, tools = {}, (None, None)
    if nsight:
        root = nvtxt.nsys_root(nsys_root)
        tools = (nvtxt.find_writer(root), nvtxt.find_importer(root))
        reports = nsight_reports(
            output, checked, index, traces, lanes, offsets, intervals, controls, tools
        )
    draw = not nsight or perfetto or per_request or tools == (None, None)
    # The window timeline shows every request and every captured worker for
    # its whole recording; per-request timelines are optional.
    window = Timeline(output / "timeline.json.gz", origin) if draw else None
    groups: dict[str, int] = {}  # lane -> flow order (frontend, prefill, decode)
    requests = []  # per-request views: (entry, label, recorded, windows by target)
    for entry in by_session(index) if draw else []:
        spans, request = traces[entry["trace_id"]], lanes[entry["trace_id"]]
        label = request_label(entry)
        groups.update(lane[:2] for lane in request.values())
        draw_request(window, spans, request, offsets, label)
        if not per_request:
            continue
        recorded, spans_windows = {}, defaultdict(list)
        roots = [s for s in spans.values() if s["name"] == "http-request"]
        if "frontend" in intervals and roots:
            start, end = start_end(roots[0])
            if start < intervals["frontend"][1] and end > intervals["frontend"][0]:
                recorded["frontend"] = intervals["frontend"]
        for phase in entry["phases"]:
            target = phase["target_id"]
            if phase["recording_overlap_ns"]:
                low, high = recorded[target] = intervals[target]
                spans_windows[target].append(
                    (max(phase["start_unix_ns"], low), min(phase["end_unix_ns"], high))
                )
        requests.append((entry, label, recorded, spans_windows))
    if draw:
        draw_window(window, intervals, offsets, controls)

    # Native files to draw: one pool task per Kineto rank trace or Nsight export.
    native, files = {}, []
    for item in checked:
        target = item["target_id"]
        collector = collectors.get(target)
        if not draw:
            native[target] = {"status": "nsight_report", "report": reports[target]}
            continue
        # The detailed frontend host trace is captured and listed but not
        # drawn until it carries CPU samples or per-request ranges.
        status, reason = (
            ("excluded", "frontend host trace detail")
            if target == "frontend"
            else ("pending", f"no timeline reader for {collector}")
            if collector not in NATIVE_READERS
            else ("omitted", f"artifacts {item['status']}")
            if item["status"] != "verified"
            else ("omitted", "recording bounds unknown")
            if target not in intervals
            else ("included", None)
        )
        if reason:
            native[target] = {"status": status, "reason": reason}
            continue
        native[target] = {
            "status": status,
            "events_read": 0,
            "events_written": 0,
            "clock_offset_ns": offsets.get(target, 0),
        }
        files += [(target, collector, a["location"]) for a in item["artifacts"]]

    # Per-request views go in batches so open timelines stay bounded; the
    # window view is drawn with the first batch.
    numbered = list(enumerate(requests, 1))
    batches = [numbered[i : i + 200] for i in range(0, len(numbered), 200)] or [[]]
    with tempfile.TemporaryDirectory(prefix="dynamo-bundle-") as scratch, pool(
        len(files)
    ) as executor:
        for number, batch in enumerate(batches if draw else []):
            timelines = {0: window} if number == 0 else {}
            for view, (entry, label, recorded, _) in batch:
                name = entry["request_id"] or entry["trace_id"]
                entry["timeline"] = f"timelines/{name}.json.gz"
                timelines[view] = Timeline(output / entry["timeline"], origin)
                spans = traces[entry["trace_id"]]
                draw_request(
                    timelines[view], spans, lanes[entry["trace_id"]], offsets, label
                )
                draw_window(timelines[view], recorded, offsets, controls)
            tasks = []
            for shard, (target, collector, location) in enumerate(files):
                views = [(*intervals[target], 0)] if number == 0 else []
                views += [
                    (low, high, view)
                    for view, (*_, spans_windows) in batch
                    for low, high in spans_windows.get(target, [])
                ]
                if views:
                    tasks.append(
                        {
                            "collector": collector,
                            "location": location,
                            "target": target,
                            "recorded": intervals[target],
                            "shift": offsets.get(target, 0),
                            "origin": origin,
                            "views": views,
                            "pid_base": 10_000 + 1000 * shard,
                            "sort": groups.get(target, 3) * 10**6
                            + 1000 * (shard + 1),
                            "directory": scratch,
                            "prefix": f"{number}-{shard}",
                        }
                    )
            # A target's files are appended together once all have converted,
            # or not at all if any failed.
            remaining, done = Counter(t["target"] for t in tasks), defaultdict(list)
            for task, result in zip(tasks, executor.map(draw_shard, tasks)):
                target = task["target"]
                done[target].append(result)
                remaining[target] -= 1
                if remaining[target] or native[target]["status"] != "included":
                    continue
                results = done.pop(target)
                failed = [r["error"] for r in results if "error" in r]
                if failed:
                    native[target] = {"status": "failed", "reason": "; ".join(failed)}
                    continue
                stats = native[target]
                for result in results:
                    stats["events_read"] += result["read"] if number == 0 else 0
                    stats["events_written"] += result["written"]
                    for view, (shard, count) in result["shards"].items():
                        timelines[view].extend(shard, count)
                        os.remove(shard)
            for view, (entry, *_) in batch:
                entry["timeline_events"] = timelines[view].close()
    window_events = window.close() if draw else None
    trace_index = {"router_targets": workers, "recording_intervals": intervals}
    (output / "lifecycle/trace-index.json").write_text(
        json.dumps(trace_index | {"requests": index}, indent=2) + "\n"
    )

    triggers = set(manifest["plan"]["trace_ids"])
    fields = ("location", "rank_id", "sha256", "sha256_uncompressed", "bytes")
    summary = {
        "schema_version": 1,
        "session_id": manifest["session_id"],
        "method": __doc__.split("\n\n")[-1].strip(),
        "timeline_origin_unix_ns": origin,
        "timeline": {"file": "timeline.json.gz", "events": window_events}
        if draw
        else None,
        "nsight_reports": reports,
        "per_request_timelines": per_request,
        "files": {
            str(p.relative_to(output)): {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "bundle.json"
        },
        "native_profiles": [
            {k: a[k] for k in fields if k in a}
            | {
                "target_id": item["target_id"],
                # A file gone since capture fails its target, not the archive.
                "archived": Path(a["location"]).is_file(),
            }
            for item in checked
            for a in item.get("artifacts", [])
        ],
        "artifact_verification": verification,
        "coverage": {
            "capture": manifest["artifact_status"],
            "profiler": "verified"
            if all(v["status"] == "verified" for v in verification.values())
            else "partial",
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


def archive(output: Path, summary: dict) -> tuple[Path, str]:
    """Package the bundle with the raw native Kineto/Nsight traces as one tar.gz."""
    path = output.parent / f"{output.name}.tar.gz"
    with ParallelGzip(path) as stream, tarfile.open(fileobj=stream, mode="w|") as tar:
        tar.add(output, arcname=output.name)
        for profile in summary["native_profiles"]:
            if not profile.get("archived", True):
                continue
            source = Path(profile["location"])
            tar.add(
                source,
                arcname=f"{output.name}/profiles/{profile['target_id']}/{source.name}",
            )
    digest = sha256_file(path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return path, digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("session", type=Path, help="session directory containing manifest.json")
    add(
        "--otel-jsonl",
        type=Path,
        action="append",
        help="collector OTLP export-batch JSONL; repeat for multiple files "
        "(default for an extracted bundle: lifecycle/traces.otlp.jsonl)",
    )
    add("--output", type=Path, help="bundle directory (default: <session>/bundle)")
    add(
        "--per-request",
        action="store_true",
        help="also write one timeline per request under timelines/",
    )
    add(
        "--perfetto",
        action="store_true",
        help="Nsight sessions: also convert native events to the Perfetto timeline "
        "(the default for Nsight is reports with request spans imported as NVTX)",
    )
    add(
        "--nsys-root",
        help="Nsight Systems install root providing nsys_writer and ImportNvtxt "
        "(default: $NSYS_HOME, or the install of nsys on PATH)",
    )
    args = parser.parse_args()
    otel = args.otel_jsonl or [args.session / "lifecycle/traces.otlp.jsonl"]
    if not all(path.is_file() for path in otel):
        parser.error(
            "--otel-jsonl is required unless the session is an extracted bundle"
        )
    output = args.output or args.session / "bundle"
    summary = build(
        args.session, otel, output, args.per_request, args.perfetto, args.nsys_root
    )
    path, digest = archive(output, summary)
    print(json.dumps(summary["coverage"], indent=2))
    print(f"archive: {path} ({path.stat().st_size} bytes, sha256 {digest})")


if __name__ == "__main__":
    main()
