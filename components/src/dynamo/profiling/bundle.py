# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-process a completed capture session into an investigation bundle.

Run outside the serving frontend:

    python -m dynamo.profiling.bundle <session-dir> --otel-jsonl <collector.jsonl>

writes <session>/bundle/ and <session>/bundle.tar.gz (with a .sha256 sidecar)
carrying the raw native Kineto/Nsight traces under profiles/<target>/. Kineto
sessions get timeline.json.gz, an overall Perfetto timeline of the capture
window. Nsight sessions get reports/<target>.nsys-rep: each native report with
the request spans, recording window and control events imported as NVTX (via
an NVTXT file and Nsight Systems' ImportNvtxt), for the multi-report view;
--perfetto also writes the Perfetto timeline, which is the fallback when
ImportNvtxt is unavailable. --per-request also writes one Perfetto timeline per
request. Run on an extracted bundle (with --output) to regenerate any view from
the archive alone.

A request is selected when its `worker.operation.<phase>` span ran on a captured
target during that target's recording interval, or when the plan named its
trace ID. Worker spans are assigned through router decisions
(`kv_router.route_request` worker_id = `handle_payload` instance_id) and joined
to captured processes by the PID in `dynamo.process.epoch`. The overall
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
import subprocess
import tarfile
import tempfile
import zlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from . import nvtxt
from .inspection import inspect_rank, inspect_target, kineto_result, sha256_file
from .native import NATIVE_READERS, draw_shard


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
    items = manifest["artifact_validation"]
    kineto = {
        i["target_id"]
        for i in items
        if profiles.get(i["target_id"], {}).get("collector") == "torch_profiler"
        and i.get("artifacts")
    }
    ranks = [(i, a) for i in items if i["target_id"] in kineto for a in i["artifacts"]]
    others = [i for i in items if i["target_id"] not in kineto]
    with pool(len(ranks) + len(others)) as executor:
        rank_results = executor.map(inspect_rank, [a for _, a in ranks])
        other_results = executor.map(
            inspect_target, others, [profiles.get(i["target_id"], {}) for i in others]
        )
        results = {i["target_id"]: r for i, r in zip(others, other_results)}
        per_target = defaultdict(list)
        for (item, _), result in zip(ranks, rank_results):
            per_target[item["target_id"]].append(result)
    for item in items:
        if item["target_id"] in kineto:
            results[item["target_id"]] = kineto_result(
                item, per_target[item["target_id"]]
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


class Timeline:
    """Streaming Chrome/Perfetto JSON writer on the coordinator clock."""

    def __init__(self, path: Path, origin_ns: int):
        self.stream = io.TextIOWrapper(
            io.BufferedWriter(ParallelGzip(path), 1 << 20), encoding="utf-8"
        )
        self.stream.write('{"displayTimeUnit":"ms","traceEvents":[')
        self.origin, self.first, self.count = origin_ns, True, 0
        self.pids: dict = {}
        self.tids: dict = {}

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

    def extend(self, shard: str, count: int) -> None:
        """Append a pool-drawn shard of comma-joined events."""
        if count:
            self.stream.write("" if self.first else ",\n")
            with open(shard) as source:
                shutil.copyfileobj(source, self.stream, 1 << 20)
            self.first, self.count = False, self.count + count

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
    """Process lane and flow group per span; router decisions run on the frontend."""
    lanes = {}
    for span in spans.values():
        worker_id, phase = router_decision(spans.get(span.get("parentSpanId")), spans)
        lanes[span["spanId"]] = (
            ("frontend", 0)
            if worker_id is None
            else (
                workers.get(worker_id) or f"{phase} worker {worker_id} (not captured)",
                {"prefill": 1, "decode": 2}.get(phase, 3),
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
        lane = lanes[span["spanId"]][0]
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
                "router_worker_id": router_worker(
                    spans.get(span.get("parentSpanId")), spans
                ),
            },
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
    return label + (" (trigger)" if entry["trigger"] else "")


def nsight_reports(
    output, checked, index, traces, lanes, offsets, intervals, controls, importer
) -> dict:
    """Write each Nsight target's request spans as NVTXT and merge them into a
    copy of its report (reports/<target>.nsys-rep) for the multi-report view.

    The frontend report gets every selected request; a worker report gets the
    requests that ran on it. Span times move from their lane's clock to the
    report host's clock with the measured offsets.
    """
    (output / "reports").mkdir(parents=True, exist_ok=True)
    reports = {}
    for item in checked:
        target = item["target_id"]
        paths = {
            Path(a["location"]).suffix: Path(a["location"])
            for a in item.get("artifacts", [])
        }
        if item["status"] != "verified" or not {".nsys-rep", ".sqlite"} <= set(paths):
            reports[target] = {
                "status": "omitted",
                "reason": f"artifacts {item['status']}",
            }
            continue
        utc, clock = nvtxt.report_clock(paths[".sqlite"])
        shift = offsets.get(target, 0)

        def local(unix_ns: int, lane: str = "frontend") -> int:
            # Lane clock -> coordinator clock -> this host -> report clock.
            return unix_ns - offsets.get(lane, 0) + shift - utc + clock

        threads = {nvtxt.CONTROL_THREAD: "Capture control"}
        ranges, marks = [], []
        if target in intervals:
            low, high = intervals[target]  # already on this report's clock
            ranges.append(
                (
                    low - utc + clock,
                    high - utc + clock,
                    nvtxt.CONTROL_THREAD,
                    f"{target} recording",
                )
            )
        for event in controls:
            if event["target_id"] == target:
                marks.append(
                    (
                        local(event["unix_ns"]),
                        nvtxt.CONTROL_THREAD,
                        f"{target} {event['event']}",
                    )
                )
        for number, entry in enumerate(index, 2):
            if target != "frontend" and target not in {
                p["target_id"] for p in entry["phases"]
            }:
                continue
            threads[number] = request_label(entry)
            ids = f"request {entry['request_id'] or entry['trace_id']}"
            for span in traces[entry["trace_id"]].values():
                lane = lanes[entry["trace_id"]][span["spanId"]][0]
                begin, end = start_end(span)
                values = attrs(span)
                message = f"{span['name']} · {lane}"
                if span["name"] in ("http-request", "request.lifecycle"):
                    message += f" · {ids} · trace {entry['trace_id']}"
                    if values.get("x_request_id"):
                        message += f" · x-request-id {values['x_request_id']}"
                ranges.append((local(begin, lane), local(end, lane), number, message))
        text = output / "reports" / f"{target}.nvtxt"
        text.write_text(nvtxt.render(threads, ranges, marks))
        entry = {
            "nvtxt": f"reports/{target}.nvtxt",
            "ranges": len(ranges),
            "marks": len(marks),
        }
        if importer is None:
            reports[target] = entry | {
                "status": "nvtxt_only",
                "reason": "ImportNvtxt not found",
            }
            continue
        merged = output / "reports" / f"{target}.nsys-rep"
        try:
            imported = nvtxt.merge(importer, paths[".nsys-rep"], text, merged)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            reports[target] = entry | {"status": "failed", "reason": str(exc)}
            continue
        reports[target] = entry | {
            "status": "merged",
            "report": f"reports/{target}.nsys-rep",
            "events_imported": imported,
            "clock_offset_ns": shift,
        }
    return reports


def build(
    session: Path,
    otel_paths: list[Path],
    output: Path,
    per_request: bool = False,
    perfetto: bool = False,
    import_nvtxt: str | None = None,
) -> dict:
    manifest = json.loads((session / "manifest.json").read_text())
    if (session / "profiles").is_dir():
        # An extracted bundle: read its packed raw traces, not the original paths.
        for item in manifest["artifact_validation"]:
            for artifact in item.get("artifacts", []):
                packed = (
                    session
                    / "profiles"
                    / item["target_id"]
                    / Path(artifact["location"]).name
                )
                if packed.exists():
                    artifact["location"] = str(packed)
    if manifest["state"] not in ("completed", "failed"):
        raise SystemExit(f"Session is still {manifest['state']}")
    # Full verification happens here, off the capture path; its results give
    # the native recording bounds (Kineto bounds need a full trace parse).
    verification = verify_native(manifest)
    checked = []
    for item in manifest["artifact_validation"]:
        result = verification[item["target_id"]]
        merged = {**item, **result, "target_id": item["target_id"]}
        if (result.get("recording_bounds") or {}).get(
            "status"
        ) != "observed" and item.get("recording_bounds"):
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

    for folder in ("lifecycle", "control") + (("timelines",) if per_request else ()):
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
    reports, importer = {}, None
    if nsight:
        importer = nvtxt.find_importer(import_nvtxt)
        reports = nsight_reports(
            output,
            checked,
            index,
            traces,
            lanes,
            offsets,
            intervals,
            controls,
            importer,
        )
    draw = not nsight or perfetto or per_request or importer is None
    # The window timeline shows every request and every captured worker for
    # its whole recording; per-request timelines are optional.
    window = Timeline(output / "timeline.json.gz", origin) if draw else None
    groups: dict[str, int] = {}  # lane -> flow order (frontend, prefill, decode)
    requests = []  # per-request views: (entry, label, recorded, windows by target)
    for entry in index if draw else []:
        spans, request = traces[entry["trace_id"]], lanes[entry["trace_id"]]
        label = request_label(entry)
        groups.update(dict(set(request.values())))
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
        elif target == "frontend":
            # The detailed frontend host trace is captured and listed but not
            # drawn until it carries CPU samples or per-request ranges.
            native[target] = {
                "status": "excluded",
                "reason": "frontend host trace detail",
            }
        elif collector not in NATIVE_READERS:
            native[target] = {
                "status": "pending",
                "reason": f"no timeline reader for {collector}",
            }
        elif item["status"] != "verified":
            native[target] = {
                "status": "omitted",
                "reason": f"artifacts {item['status']}",
            }
        elif target not in intervals:
            native[target] = {"status": "omitted", "reason": "recording bounds unknown"}
        else:
            native[target] = {
                "status": "included",
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
            {
                k: a[k]
                for k in (
                    "location",
                    "rank_id",
                    "sha256",
                    "sha256_uncompressed",
                    "bytes",
                )
                if k in a
            }
            | {"target_id": item["target_id"]}
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
    parser.add_argument(
        "session", type=Path, help="session directory containing manifest.json"
    )
    parser.add_argument(
        "--otel-jsonl",
        type=Path,
        action="append",
        help="collector OTLP export-batch JSONL; repeat for multiple files "
        "(default for an extracted bundle: lifecycle/traces.otlp.jsonl)",
    )
    parser.add_argument(
        "--output", type=Path, help="bundle directory (default: <session>/bundle)"
    )
    parser.add_argument(
        "--per-request",
        action="store_true",
        help="also write one timeline per request under timelines/",
    )
    parser.add_argument(
        "--perfetto",
        action="store_true",
        help="Nsight sessions: also convert native events to the Perfetto timeline "
        "(the default for Nsight is reports with request spans imported as NVTX)",
    )
    parser.add_argument(
        "--import-nvtxt",
        help="path to Nsight Systems' ImportNvtxt (default: $NSYS_IMPORT_NVTXT, PATH, "
        "or next to nsys)",
    )
    args = parser.parse_args()
    otel = args.otel_jsonl or [args.session / "lifecycle/traces.otlp.jsonl"]
    if not all(path.is_file() for path in otel):
        parser.error(
            "--otel-jsonl is required unless the session is an extracted bundle"
        )
    output = args.output or args.session / "bundle"
    summary = build(
        args.session, otel, output, args.per_request, args.perfetto, args.import_nvtxt
    )
    path, digest = archive(output, summary)
    print(json.dumps(summary["coverage"], indent=2))
    print(f"archive: {path} ({path.stat().st_size} bytes, sha256 {digest})")


if __name__ == "__main__":
    main()
