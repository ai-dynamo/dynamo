# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native trace readers and per-file timeline shards for the bundle command.

Each Kineto rank trace or Nsight SQLite export is drawn by one pool process
into plain JSON shards (one per timeline view), which the bundle concatenates.
Kept importable so a process pool works when the bundle runs as ``__main__``.
"""

import gzip
import json
import re
import sqlite3
from bisect import bisect_left
from contextlib import ExitStack, closing
from pathlib import Path

import ijson


def kineto_events(path: Path):
    """Yield Kineto events with raw worker-clock nanosecond start/end."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as stream:
        header = stream.read(1 << 20).decode(errors="replace")
        match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
        if not match or not 0 <= match.start() < header.find('"traceEvents"'):
            raise ValueError(f"{path}: baseTimeNanoseconds must precede traceEvents")
        base = int(match.group(1))
        stream.seek(0)
        # Kineto ts/dur are microseconds relative to baseTimeNanoseconds: small
        # enough that floats keep nanosecond resolution.
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if "ts" in event:
                event["start_ns"] = base + round(event["ts"] * 1000)
                event["end_ns"] = event["start_ns"] + round(event.get("dur", 0) * 1000)
            yield event


def process_meta(pid, name: str, labels: str) -> list[dict]:
    return [
        {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}},
        {"ph": "M", "name": "process_labels", "pid": pid, "args": {"labels": labels}},
    ]


# Event table -> (category, SELECT columns start, end, name, lane, extra).
NSYS_TABLES = {
    "NVTX_EVENTS": (
        "nvtx",
        "start, end, COALESCE(text, textId), globalTid, domainId",
        " WHERE eventType IN (59, 60) AND end IS NOT NULL",
    ),
    "OSRT_API": ("osrt", "start, end, nameId, globalTid, NULL", ""),
    "CUPTI_ACTIVITY_KIND_RUNTIME": (
        "cuda_runtime",
        "start, end, nameId, globalTid, NULL",
        "",
    ),
    "CUPTI_ACTIVITY_KIND_DRIVER": (
        "cuda_driver",
        "start, end, nameId, globalTid, NULL",
        "",
    ),
    "CUPTI_ACTIVITY_KIND_KERNEL": (
        "kernel",
        "start, end, demangledName, deviceId, streamId",
        "",
    ),
    "CUPTI_ACTIVITY_KIND_MEMCPY": (
        "gpu_memcpy",
        "start, end, NULL, deviceId, streamId",
        "",
    ),
    "CUPTI_ACTIVITY_KIND_MEMSET": (
        "gpu_memset",
        "start, end, NULL, deviceId, streamId",
        "",
    ),
}
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
NVTX_DOMAINS = {"PyTorch": "pytorch", "vLLM": "vllm"}


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
            yield from process_meta(pid, name, "CPU")
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
        for table, (cat, columns, where) in NSYS_TABLES.items():
            if table not in tables:
                continue
            gpu = cat in GPU_CATEGORIES
            for start, end, name, lane, extra in db.execute(
                f"SELECT {columns} FROM {table}{where}"
            ):
                if gpu:
                    pid, tid = f"GPU {lane}", extra
                    if lane not in devices:
                        devices.add(lane)
                        yield from process_meta(pid, f"CUDA device {lane}", pid)
                else:
                    pid, tid = (lane >> 24) & 0xFFFFFF, lane & 0xFFFFFF
                yield {
                    "ph": "X",
                    "cat": NVTX_DOMAINS.get(domains.get(extra), cat)
                    if cat == "nvtx"
                    else cat,
                    "name": strings.get(name, name) if name is not None else cat,
                    "pid": pid,
                    "tid": tid,
                    "dur": (end - start) / 1000,
                    "start_ns": epoch + start,
                    "end_ns": epoch + end,
                }


NATIVE_READERS = {"torch_profiler": kineto_events, "nsys": nsys_events}


def draw_shard(task: dict) -> dict:
    """Draw one native file into per-view shards of comma-joined JSON events.

    task: collector, location, target, recorded (low, high), shift, origin,
    sort (process sort-index base), pid_base, views [(low, high, view)], and
    directory. Processes get pids from pid_base so shards never collide.
    """
    low_bound, high_bound = task["recorded"]
    shift, origin = task["shift"], task["origin"]
    views = sorted(task["views"])
    lows = [view[0] for view in views]
    longest = max((high - low for low, high, _ in views), default=0)
    directory, prefix = Path(task["directory"]), task["prefix"]
    names, labels, threads, pids = {}, {}, {}, {}
    files, counts, emitted = {}, {}, set()
    read = written = 0
    with ExitStack() as stack:

        def write(view, line: str) -> None:
            if view not in files:
                path = directory / f"{prefix}.{view}.json"
                files[view] = stack.enter_context(path.open("w"))
                counts[view] = 0
            files[view].write(line if not counts[view] else ",\n" + line)
            counts[view] += 1

        for event in NATIVE_READERS[task["collector"]](Path(task["location"])):
            read += 1
            key = event.get("pid")
            if event.get("ph") == "M":
                if event.get("name") == "process_name":
                    names[key] = event["args"].get("name")
                elif event.get("name") == "process_labels":
                    labels[key] = event["args"].get("labels")
                elif event.get("name", "").startswith("thread_"):
                    threads.setdefault(key, []).append(event)
                continue
            if "start_ns" not in event:
                continue
            start, end = event.pop("start_ns"), event.pop("end_ns")
            if end < low_bound or start > high_bound:
                continue
            # Native data exists only inside recording; clip events that began
            # before or ended after it and keep their original bounds.
            if start < low_bound or end > high_bound:
                event["args"] = {
                    **event.get("args", {}),
                    "original_start_unix_ns": start,
                    "original_end_unix_ns": end,
                }
                start, end = max(start, low_bound), min(end, high_bound)
                event["dur"] = max(0, end - start) / 1000
            # Views whose window overlaps [start, end); windows are sorted by low.
            touch = max(end, start + 1)
            hits = []
            index = bisect_left(lows, touch)
            while index > 0:
                index -= 1
                low, high, view = views[index]
                if low < start - longest:
                    break
                if high > start:
                    hits.append(view)
            if not hits:
                continue
            if key not in pids:
                pids[key] = task["pid_base"] + len(pids)
            pid = pids[key]
            event["pid"], event["ts"] = pid, (start - shift - origin) / 1000
            line = json.dumps(event)
            for view in hits:
                if (view, key) not in emitted:
                    emitted.add((view, key))
                    # Kineto keeps profiler bookkeeping in unnamed pseudo-processes.
                    name = (
                        f"{task['target']} · {names[key]} ({labels.get(key, key)})"
                        if key in names
                        else f"{task['target']} · Kineto profiler ({key})"
                    )
                    meta = [
                        {
                            "ph": "M",
                            "name": "process_name",
                            "pid": pid,
                            "args": {"name": name},
                        },
                        {
                            "ph": "M",
                            "name": "process_sort_index",
                            "pid": pid,
                            "args": {
                                "sort_index": task["sort"] + pid - task["pid_base"]
                            },
                        },
                    ] + [{**thread, "pid": pid} for thread in threads.get(key, [])]
                    for item in meta:
                        write(view, json.dumps(item))
                write(view, line)
            written += len(hits)
    shards = {
        view: (str(directory / f"{prefix}.{view}.json"), count)
        for view, count in counts.items()
    }
    return {"read": read, "written": written, "shards": shards}
