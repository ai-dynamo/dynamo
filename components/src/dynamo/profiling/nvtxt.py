# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add OTel request spans to an Nsight report as NVTX ranges.

The Nsight bundle keeps the native reports and adds the request flow, as in the
Perfetto view, to one report for the multi-report view: a row per request with
its span tree as nested push/pop ranges, plus recording windows and control
events. Nsight Systems' Python writer (nsys_writer) merges them under a
machine-level scope, which the timeline places above every target's processes;
ImportNvtxt is the fallback (its ranges appear after the processes). The NVTXT
text form is always written as a portable record. Times are on the report's own
clock: its session start is TARGET_INFO_SESSION_START_TIME.utcEpochNs in Unix
time and ANALYSIS_DETAILS.startTime on the report clock. Ranges are clipped to
the report's session so its time window is unchanged.
"""

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from pathlib import Path

# Synthetic "Dynamo requests" process. Nsight orders processes by PID across a
# multi-report view, and host-wide reports use every PID from 1, so 0 keeps the
# requests above all native processes, as in the Perfetto view.
PROCESS_ID = 0
CONTROL_THREAD = 1  # recording window and control events

# ARGB colours by span kind (Tableau 10).
COLORS = (
    (("http-request", "request.", "frontend"), 0xFF4E79A7),  # frontend request: blue
    (("kv_router.",), 0xFFF28E2B),  # routing: orange
    (("handle_payload", "worker.admission"), 0xFFBAB0AC),  # hand-off: grey
    (("worker.operation.prefill",), 0xFF59A14F),  # prefill: green
    (("worker.operation.decode",), 0xFFB07AA1),  # decode: purple
    (("response.streaming",), 0xFF76B7B2),  # streaming: teal
)
RECORDING = 0xFFE15759  # recording windows: red
OTHER = 0xFF9C755F


def color(name: str) -> int:
    return next((c for prefixes, c in COLORS if name.startswith(prefixes)), OTHER)


def quoted(text: str) -> str:
    # NVTXT strings are quoted and cannot contain newlines.
    return '"' + str(text).replace('"', "'").replace("\n", " ") + '"'


@contextmanager
def read_only(path: Path):
    """A read-only connection to a SQLite export and its table names."""
    with closing(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)) as db:
        yield db, {row[0] for row in db.execute("SELECT name FROM sqlite_master")}


def report_clock(sqlite_path: Path) -> tuple[int, int, int | None]:
    """(Unix ns, report-clock ns) of the session start, and its report-clock end."""
    with read_only(sqlite_path) as (db, tables):
        (utc,) = db.execute(
            "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
        clock = stop = None
        columns = {
            row[1]
            for row in db.execute("PRAGMA table_info(TARGET_INFO_SESSION_START_TIME)")
        }
        if "systemClockNs" in columns:
            (clock,) = db.execute(
                "SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
            ).fetchone()
        if "ANALYSIS_DETAILS" in tables:
            clock, stop = db.execute(
                "SELECT startTime, stopTime FROM ANALYSIS_DETAILS"
            ).fetchone()
    if clock is None:
        raise ValueError(f"{sqlite_path.name}: no session clock anchor")
    return utc, clock, stop


TOLERANCE_NS = 1_000_000  # clock and export jitter allowed when nesting


def tree_threads(spans: dict) -> list[list]:
    """Lay out one request's span tree as nested push/pop threads.

    spans: {span_id: (start, end, message, color, parent_id)}. The first thread
    holds the tree depth-first with children ordered by start time, so ranges
    nest as parent and child spans. As in the Perfetto view, a span that lies
    within an earlier sibling's time range (within TOLERANCE_NS) is drawn inside
    that sibling. A child is clamped into its parent (spans from different
    hosts can overrun by clock-offset error); only a subtree that crosses an
    earlier sibling's boundary moves to another thread.
    """
    children: dict = {}
    roots = []
    for span_id, span in spans.items():
        (children.setdefault(span[4], []) if span[4] in spans else roots).append(
            span_id
        )

    def settle(ids: list) -> list:
        # Nest siblings contained in an earlier sibling under it, top-down so
        # spans adopted into a node are settled with its children.
        kept: list = []
        for span_id in sorted(
            ids, key=lambda i: (spans[i][0], -spans[i][1], spans[i][2])
        ):
            start, end = spans[span_id][:2]
            holder = next(
                (
                    k
                    for k in reversed(kept)
                    if spans[k][0] <= start and end <= spans[k][1] + TOLERANCE_NS
                ),
                None,
            )
            if holder is None:
                kept.append(span_id)
            else:
                children.setdefault(holder, []).append(span_id)
        for span_id in kept:
            if span_id in children:
                children[span_id] = settle(children[span_id])
        return kept

    roots = settle(roots)
    threads, free = [[]], [0]

    def ordered(ids, bounds):
        items = []
        for span_id in ids:
            start, end, message, argb, _ = spans[span_id]
            if bounds:
                # Strictly inside the parent (1 ns in from each edge when there
                # is room): clipping can give nested spans identical intervals,
                # which Nsight would draw in arbitrary order.
                low, high = bounds
                if high - low > 2:
                    low, high = low + 1, high - 1
                inside = min(max(start, low), high)
                inside_end = max(min(end, high), inside)
                # Mark real corrections (clock offset), not the ns-level inset.
                if abs(inside - start) > 1000 or abs(inside_end - end) > 1000:
                    message += " (clamped to parent)"
                start, end = inside, inside_end
            items.append((start, end, message, argb, span_id))
        return sorted(items, key=lambda item: (item[0], -item[1], item[2]))

    def spare(start: int, end: int) -> int:
        for number in range(1, len(threads)):
            if free[number] <= start:
                free[number] = end
                return number
        threads.append([])
        free.append(end)
        return len(threads) - 1

    def lay(items: list, thread: int, last: float) -> None:
        for start, end, message, argb, span_id in items:
            if start < last:  # overlaps an earlier sibling
                row = spare(start, end)
            else:
                row, last = thread, end
            threads[row].append((start, end, message, argb))
            lay(ordered(children.get(span_id, []), (start, end)), row, start)

    lay(ordered(roots, None), 0, float("-inf"))
    return [thread for thread in threads if thread]


def events(spans: list) -> list[tuple]:
    """Push/pop records for one thread's strictly nested spans, in order."""
    records, open_ends = [], []
    for start, end, message, argb in spans:
        while open_ends and open_ends[-1] <= start:
            records.append((open_ends.pop(), "pop", None, None))
        records.append((start, "push", message, argb))
        open_ends.append(end)
    while open_ends:
        records.append((open_ends.pop(), "pop", None, None))
    return records


def render(threads: dict[int, tuple[str, list]], marks: list) -> str:
    """NVTXT text: threads {tid: (name, nested spans)}, marks [(time, tid, message)],
    all in report-clock nanoseconds."""
    lines = [
        "# Dynamo request lifecycle spans (OTel), generated by dynamo.profiling.bundle",
        "TimeBase = Manual",
        f"ProcessId = {PROCESS_ID}",
        f"NameProcess, {PROCESS_ID}, {quoted('Dynamo requests')}",
    ]
    lines += [
        f"NameOsThread, {PROCESS_ID}, {tid}, {quoted(name)}"
        for tid, (name, _) in sorted(threads.items())
    ]
    lines += ["@RangePush, Time, ThreadId, Color, Message", "@RangePop, Time, ThreadId"]
    for tid, (_, spans) in sorted(threads.items()):
        for time, kind, message, argb in events(spans):
            lines.append(
                f"RangePush, {time}, {tid}, 0x{argb:08X}, {quoted(message)}"
                if kind == "push"
                else f"RangePop, {time}, {tid}"
            )
    lines.append("@Marker, Time, ThreadId, Message")
    lines += [
        f"Marker, {t}, {tid}, {quoted(message)}" for t, tid, message in sorted(marks)
    ]
    return "\n".join(lines) + "\n"


def nsys_root(explicit: str | None = None) -> Path | None:
    """Nsight Systems install root: --nsys-root, $NSYS_HOME, or nsys on PATH."""
    for candidate in (explicit, os.environ.get("NSYS_HOME")):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    nsys = shutil.which("nsys")
    # <install>/target-*/nsys or <install>/bin/nsys
    return Path(nsys).resolve().parents[1] if nsys else None


def find_importer(root: Path | None) -> str | None:
    found = sorted(root.glob("host-*/ImportNvtxt")) if root else []
    return str(found[0]) if found else shutil.which("ImportNvtxt")


def find_writer(root: Path | None) -> dict | None:
    """nsys_writer and its backend from the install, if the nvtx package is present.

    The writer runs in a fresh interpreter (see write): its backend needs the
    newer libstdc++ shipped with Nsight Systems, which cannot load once a process
    has the system's older one, as a bundle process may.
    """
    packages = sorted(root.glob("target-*/python/packages")) if root else []
    backends = sorted(root.glob("host-*/libNvtxwBackend.so")) if root else []
    if not packages or not backends or importlib.util.find_spec("nvtx") is None:
        return None
    return {"packages": str(packages[0]), "backend": str(backends[0])}


def write(
    writer: dict, report: Path, output: Path, threads: dict, marks: list, clock: int
) -> int:
    """Merge the ranges into a copy of the report with nsys_writer, in a subprocess;
    returns the number of events written."""
    spec = output.with_suffix(".writer.json")
    spec.write_text(
        json.dumps(
            {
                "report": str(report),
                "output": str(output),
                "clock": clock,
                "threads": {
                    str(tid): [name, spans] for tid, (name, spans) in threads.items()
                },
                "marks": marks,
            }
        )
    )
    env = os.environ | {
        "NSYS_WRITER_BACKEND": writer["backend"],
        "PYTHONPATH": os.pathsep.join(
            [writer["packages"], *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
    }
    try:
        command = [sys.executable, "-m", "dynamo.profiling.nvtxt", str(spec)]
        return run(command, output, r"wrote (\d+) events", env=env)
    finally:
        spec.unlink(missing_ok=True)


def run(command: list, output: Path, pattern: str, env=None, tool=None) -> int:
    """Run a report tool to completion; returns the event count it printed."""
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=600
    )
    count = re.search(pattern, result.stdout + result.stderr)
    if result.returncode or not count or not output.is_file():
        tail = result.stderr or result.stdout
        raise RuntimeError(
            f"{tool} failed ({result.returncode}): {tail[-500:]}"
            if tool
            else tail.strip()[-500:]
        )
    return int(count.group(1))


def write_in_process(spec: dict) -> int:
    """Each thread becomes a child scope of a machine-level "Dynamo requests"
    scope, so every request gets its own row above all targets' processes."""
    import nsys_writer
    from nvtx.writer import PredefinedScope

    clock, events = spec["clock"], 0
    with nsys_writer.Session(
        str(Path(spec["output"]).with_suffix("")),
        report_merge=spec["report"],
        time_base=nsys_writer.TimeBase.RELATIVE,
    ) as session:
        domain = session.get_domain("Dynamo requests")
        root = domain.get_scope(
            "Dynamo requests", parent=PredefinedScope.CURRENT_HW_MACHINE
        )
        for tid, (name, spans) in sorted(
            spec["threads"].items(), key=lambda item: int(item[0])
        ):
            scope = domain.get_scope(name, parent=root)
            with session.create_stream(name, domain=domain, scope=scope) as stream:
                for start, end, message, argb in spans:
                    stream.write_pushpop(
                        start - clock, end - clock, message=message, color=argb
                    )
                    events += 1
                if int(tid) == CONTROL_THREAD:
                    for time, _, message in sorted(spec["marks"]):
                        stream.write_mark(
                            time - clock, message=message, color=RECORDING
                        )
                        events += 1
    return events


def merge(importer: str, report: Path, nvtxt: Path, output: Path) -> int:
    """Import NVTXT into a copy of the report; returns the imported event count."""
    command = [importer, "--cmd", "merge", "-i", report, "-n", nvtxt, "-o", output]
    return run(command, output, r"(\d+) events imported", tool="ImportNvtxt")


if __name__ == "__main__":
    print(f"wrote {write_in_process(json.loads(Path(sys.argv[1]).read_text()))} events")
