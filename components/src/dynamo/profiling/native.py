# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native trace readers and per-file timeline shards for the bundle command.

Each Kineto rank trace or Nsight SQLite export is drawn by one pool process
into plain JSON shards (one per timeline view), which the bundle concatenates.
Kept importable so a process pool works when the bundle runs as ``__main__``.
"""

import codecs
import gzip
import json
import re
from bisect import bisect_left
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import ijson

from .nvtxt import read_only


class Layout(ValueError):
    """The trace is not in Kineto's own writer layout; use the token parser."""


# Kineto writes each traceEvents item as a block from a "  {" line to a "  },"
# line; nested objects are indented further and JSON strings hold no newlines.
BLOCK_END = "\n  },\n"
ARRAY_END = "\n  }\n  ]"


def kineto_blocks(stream, chunk: int = 1 << 22):
    """Yield the header dict, then each traceEvents item as JSON text.

    Splitting Kineto's fixed layout and decoding blocks with the C json module
    is several times faster than token streaming. Raises Layout on any
    deviation so callers can fall back to ijson.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    text, marker = "", '"traceEvents": ['
    while marker not in text:
        data = stream.read(chunk)
        if not data or len(text) > (16 << 20):
            raise Layout("no traceEvents array near the start")
        text += decoder.decode(data)
    head, text = text.split(marker, 1)
    try:
        yield json.loads(head + '"traceEvents": []}')
    except ValueError as exc:
        raise Layout(f"header: {exc}") from exc
    while True:
        data = stream.read(chunk)
        text += decoder.decode(data, final=not data)
        *blocks, text = text.split(BLOCK_END)
        for block in blocks:
            if not block.lstrip().startswith("{"):
                raise Layout("unexpected text between events")
            yield block + "\n  }"
        if not data:
            break
    last, end, trailer = text.rpartition(ARRAY_END)
    if end:
        if not last.lstrip().startswith("{"):
            raise Layout("unexpected text before the last event")
        yield last + "\n  }"
    elif text.strip().startswith("]"):
        trailer = text.strip()[1:]  # an empty traceEvents array
    else:
        raise Layout("traceEvents does not end in the writer layout")
    try:
        json.loads('{"traceEvents": []' + trailer)
    except ValueError as exc:
        raise Layout(f"trailer: {exc}") from exc


def decoded(blocks):
    for block in blocks:
        try:
            yield json.loads(block)
        except ValueError as exc:  # let the token parser decide
            raise Layout(str(exc)) from exc


def open_trace(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def kineto_events(path: Path, fast: bool = True):
    """Yield Kineto events with raw worker-clock nanosecond start/end.

    With fast=True a layout deviation raises Layout (possibly after yielding
    events); the caller restarts with fast=False, the ijson token parser.
    """
    # ts/dur are microseconds near 1e12: float spacing there is ~0.1 ns.
    with open_trace(path) as stream:
        if fast:
            blocks = kineto_blocks(stream)
            base = next(blocks).get("baseTimeNanoseconds")
            if type(base) is not int:
                raise Layout("no integer baseTimeNanoseconds")
            events = decoded(blocks)
        else:
            header = stream.read(1 << 20).decode(errors="replace")
            match = re.search(r'"baseTimeNanoseconds"\s*:\s*(\d+)', header)
            if not match or not 0 <= match.start() < header.find('"traceEvents"'):
                raise ValueError(
                    f"{path}: baseTimeNanoseconds must precede traceEvents"
                )
            base = int(match.group(1))
            stream.seek(0)
            events = ijson.items(stream, "traceEvents.item", use_float=True)
        for event in events:
            if "ts" in event:
                event["start_ns"] = base + round(event["ts"] * 1000)
                event["end_ns"] = event["start_ns"] + round(event.get("dur", 0) * 1000)
            yield event


def process_meta(pid, name: str, key: str, value) -> list[dict]:
    return [
        {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}},
        {"ph": "M", "name": f"process_{key}", "pid": pid, "args": {key: value}},
    ]


# Event table -> (category, SELECT columns start, end, name, lane, extra, WHERE).
NVTX = "start, end, COALESCE(text, textId), globalTid, domainId"
HOST = "start, end, nameId, globalTid, NULL"
DEVICE = "start, end, {}, deviceId, streamId"
NSYS_TABLES = {
    "NVTX_EVENTS": ("nvtx", NVTX, " WHERE eventType IN (59, 60) AND end IS NOT NULL"),
    "OSRT_API": ("osrt", HOST, ""),
    "CUPTI_ACTIVITY_KIND_RUNTIME": ("cuda_runtime", HOST, ""),
    "CUPTI_ACTIVITY_KIND_DRIVER": ("cuda_driver", HOST, ""),
    "CUPTI_ACTIVITY_KIND_KERNEL": ("kernel", DEVICE.format("demangledName"), ""),
    "CUPTI_ACTIVITY_KIND_MEMCPY": ("gpu_memcpy", DEVICE.format("NULL"), ""),
    "CUPTI_ACTIVITY_KIND_MEMSET": ("gpu_memset", DEVICE.format("NULL"), ""),
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
    with read_only(path) as (db, tables):
        # Event timestamps are relative to the session start anchored here.
        (epoch,) = db.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
        strings = dict(db.execute("SELECT id, value FROM StringIds"))
        query = "SELECT domainId, text FROM NVTX_EVENTS WHERE eventType = 75"
        domains = dict(db.execute(query)) if "NVTX_EVENTS" in tables else {}
        for pid, name in db.execute("SELECT pid, name FROM PROCESSES"):
            yield from process_meta(pid, name, "labels", "CPU")
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
                        yield from process_meta(
                            pid, f"CUDA device {lane}", "labels", pid
                        )
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

    Kineto traces use the fast block reader; on a layout deviation the shards
    are rewritten from scratch with the ijson reader. A trace that still
    cannot be read fails its target rather than the bundle.
    """
    try:
        try:
            return draw_views(task, NATIVE_READERS[task["collector"]])
        except Layout:
            return draw_views(task, partial(kineto_events, fast=False))
    except Exception as exc:
        return {
            "read": 0,
            "written": 0,
            "shards": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def draw_views(task: dict, reader) -> dict:
    """task: collector, location, target, recorded (low, high), shift, origin,
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

        for event in reader(Path(task["location"])):
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
            hits, index = [], bisect_left(lows, max(end, start + 1))
            while index > 0:
                index -= 1
                low, high, view = views[index]
                if low < start - longest:
                    break
                if high > start:
                    hits.append(view)
            if not hits:
                continue
            pid = pids.setdefault(key, task["pid_base"] + len(pids))
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
                    meta = process_meta(
                        pid, name, "sort_index", task["sort"] + pid - task["pid_base"]
                    ) + [{**thread, "pid": pid} for thread in threads.get(key, [])]
                    for item in meta:
                        write(view, json.dumps(item))
                write(view, line)
            written += len(hits)
    shards = {view: (str(files[view].name), count) for view, count in counts.items()}
    return {"read": read, "written": written, "shards": shards}
