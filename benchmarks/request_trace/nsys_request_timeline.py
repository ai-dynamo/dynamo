# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Show one request's Dynamo-level timeline across nsys reports and OTel spans.

A Dynamo process started with ``DYN_NVTX=1`` and captured by Nsight Systems
records request-scoped NVTX ranges and markers (contract v2) whose message is
the static stage name (for example ``kv_router.route_request:prefill``) and whose
64-bit payload (``NVTX_EVENTS.uint64Value``) is ``rid_key(rid)``. Each process also
emits identity marks ``dynamo.request rid=<rid>[ trace=..][ xrid=..][ engine_rid=..]``
carrying the same payload.

This tool takes one request key (request id, ``chatcmpl-<id>``, W3C trace id or
``x-request-id``), finds that request in one or more nsys reports (one per
process or per node), optionally joins the OTel spans of the same request from
Tempo or a saved JSON file, and prints:

1. how the key was resolved,
2. a chronological table across every captured process,
3. a TTFT breakdown with unattributed gaps,
4. the time window to zoom to in each nsys report.

``--summary`` instead prints per-stage aggregates across all requests in the reports.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import dataclasses
import glob
import hashlib
import json
import math
import re
import sqlite3
import statistics
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# eventType values in NVTX_EVENTS (nsys 2026.x): 34 mark, 59 push/pop, 60 start/end.
_EVT_MARK = 34
_EVT_PUSHPOP = 59
_EVT_STARTEND = 60

# Request-scoped NVTX contract (v2): names are static stages, the request is the
# 64-bit payload ``rid_key(rid)``, and identity marks map the key back to the rid.
_IDENTITY_RE = re.compile(
    r"^dynamo\.request rid=(?P<rid>[^ ]+)(?P<rest>( [^ =]+=[^ ]*)*)$"
)
_STAGE_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_FNV64_OFFSET = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_KEY_MASK = 0x7FFF_FFFF_FFFF_FFFF
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_TRACE_ID_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")

_CHATCMPL_PREFIX = "chatcmpl-"

_STAGE_HTTP = "http-request"
_STAGE_PREPROCESS = "request.preprocessing"
_STAGE_SELECT = "kv_router.select_worker"
_STAGE_SCHEDULE = "kv_router.schedule"
_STAGE_ROUTE = ("kv_router.route_request", "router.route_request")
_STAGE_FIRST_TOKEN = "frontend.first_token"
_STAGE_HANDLE = "handle_payload"

# Value bound and whitespace set of the runtime's NVTX formatter
# (`format_request_message` in lib/runtime/src/nvtx.rs). The set is Rust's
# `char::is_whitespace` (Unicode White_Space), which differs from `str.isspace`.
_MAX_VALUE_BYTES = 128
_RUST_WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000"
    + "".join(chr(c) for c in range(0x2000, 0x200B))
)
_LIFECYCLE_ROLE_PREFIXES = ("worker.operation.", "response.streaming.")
_WORKER_ROLES = ("prefill", "decode", "aggregated", "encode")
_WORKER_STAGES = (
    "worker.admission_gate.wait",
    "worker.handler.prepare",
    "worker.engine.first_output",
    "worker.egress.first_chunk",
)

_RID_ATTRS = ("request_id", "dynamo.request.id")
_XRID_ATTRS = ("x_request_id",)

_SQL_BASE = """
SELECT e.start, e.end, e.eventType, e.globalTid,
       COALESCE(e.text, {sid_value}) AS msg, e.uint64Value
FROM NVTX_EVENTS e
{sid_join}
WHERE {where}
"""
# Every request-scoped range/mark of one request (identity marks included).
_WHERE_BY_KEY = "e.eventType IN (34, 59, 60) AND e.uint64Value = ? ORDER BY e.start"
# Identity marks whose text contains a ``key=value`` needle (verified in Python).
_WHERE_IDENTITY = (
    "e.eventType = 34 AND e.uint64Value IS NOT NULL "
    "AND instr(COALESCE(e.text, {sid_value}), ?) > 0"
)
# Closed request-scoped ranges of every request, for --summary.
_WHERE_SUMMARY = (
    "e.eventType IN (59, 60) AND e.uint64Value IS NOT NULL AND e.end IS NOT NULL"
)


class TimelineError(Exception):
    """A user-facing error (bad input, unresolved or ambiguous key)."""


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclasses.dataclass
class NvtxMessage:
    """A parsed request-scoped NVTX message."""

    stage: str
    rid: str
    attrs: dict[str, str]

    @property
    def base(self) -> str:
        return split_stage(self.stage)[0]

    @property
    def qualifier(self) -> str | None:
        return split_stage(self.stage)[1]


@dataclasses.dataclass
class NvtxEvent:
    """One request-scoped NVTX row from one report."""

    msg: NvtxMessage
    report: str  # report path
    pid: int
    tid: int
    mark: bool
    start_ns: int | None  # absolute unix ns, None when the start is unknown
    end_ns: int | None  # absolute unix ns, None for marks/open ranges
    open: bool
    rel_start_s: float | None  # report-relative seconds (Nsight GUI timeline)
    rel_end_s: float | None
    thread_name: str | None = None


@dataclasses.dataclass
class ProcessInfo:
    """A captured process inside one report."""

    report: str
    pid: int
    name: str | None
    host: str | None
    role: str
    # Set when two processes would otherwise share a label (e.g. containers that all
    # run the server as the same pid on one host).
    suffix: str = ""

    @property
    def label(self) -> str:
        host = f"@{self.host}" if self.host else ""
        return f"{self.role}/pid{self.pid}{host}{self.suffix}"


@dataclasses.dataclass
class ReportScan:
    """Result of scanning one nsys report."""

    path: str
    host: str | None
    session_start_ns: int
    duration_ns: int
    events: list[NvtxEvent]
    processes: dict[int, ProcessInfo]
    # Merged identity-mark attrs (trace, xrid, engine_rid, ...) per pid, and the
    # request ids those marks name for the scanned key.
    identities: dict[int, dict[str, str]] = dataclasses.field(default_factory=dict)
    rids: set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class OtelSpan:
    """One OTel span, normalized from Tempo/OTLP JSON."""

    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    service: str | None
    start_ns: int
    end_ns: int | None
    attrs: dict[str, Any]

    @property
    def rid(self) -> str | None:
        for key in _RID_ATTRS:
            value = self.attrs.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    @property
    def stage(self) -> str:
        phase = self.attrs.get("phase")
        if isinstance(phase, str) and phase:
            return f"{self.name}:{phase.lower()}"
        return self.name


@dataclasses.dataclass
class Row:
    """One timeline row: an NVTX event, an OTel span, or a paired both."""

    stage: str
    source: str  # "nvtx" | "otel" | "both"
    process: str
    role: str | None
    mark: bool
    start_ns: int | None
    end_ns: int | None
    open: bool
    report: str | None = None
    pid: int | None = None
    rel_start_s: float | None = None
    rel_end_s: float | None = None
    delta_ns: int | None = None  # otel start - nvtx start
    otel_start_ns: int | None = None
    otel_end_ns: int | None = None
    attrs: dict[str, str] = dataclasses.field(default_factory=dict)

    @property
    def base(self) -> str:
        return split_stage(self.stage)[0]

    @property
    def qualifier(self) -> str | None:
        return split_stage(self.stage)[1]


@dataclasses.dataclass
class Resolution:
    """How the user's key was turned into a request id."""

    key: str
    kind: str  # rid | trace | xrid
    rid: str | None
    via: str
    trace_id: str | None = None
    xrid: str | None = None
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class BreakdownItem:
    """One line of the TTFT breakdown."""

    kind: str  # mark | segment | gap | note
    name: str
    start_ms: float | None
    dur_ms: float | None
    depth: int = 0
    note: str = ""


@dataclasses.dataclass
class Breakdown:
    """TTFT breakdown relative to the request's first frontend event."""

    t0_source: str
    ttft_ms: float | None
    attributed_ms: float
    unattributed_ms: float
    items: list[BreakdownItem]
    notes: list[str]


# --------------------------------------------------------------------------
# Message parsing
# --------------------------------------------------------------------------


def split_stage(stage: str) -> tuple[str, str | None]:
    """Split ``name:qualifier`` into its parts.

    Stage names never contain ``:`` except for the qualifier separator.
    """
    base, sep, qualifier = stage.partition(":")
    return base, (qualifier if sep and qualifier else None)


def rid_key(rid: str) -> int:
    """Request key of ``rid``: FNV-1a-64 of its UTF-8 bytes with the top bit cleared.

    This is the payload (``NVTX_EVENTS.uint64Value``) of every request-scoped NVTX
    range and mark of the request; it stays positive in SQLite's signed INTEGER.
    """
    h = _FNV64_OFFSET
    for byte in rid.encode("utf-8"):
        h = ((h ^ byte) * _FNV64_PRIME) & 0xFFFF_FFFF_FFFF_FFFF
    return h & _KEY_MASK


def parse_identity(text: str | None) -> NvtxMessage | None:
    """Parse a ``dynamo.request rid=<rid>[ key=value]...`` identity mark."""
    if not text:
        return None
    match = _IDENTITY_RE.match(text)
    if match is None:
        return None
    attrs: dict[str, str] = {}
    for token in match.group("rest").split(" "):
        if not token:
            continue
        key, _, value = token.partition("=")
        attrs[key] = value
    return NvtxMessage("dynamo.request", match.group("rid"), attrs)


# --------------------------------------------------------------------------
# nsys: inputs, export and scanning
# --------------------------------------------------------------------------


def expand_inputs(items: Sequence[str]) -> list[Path]:
    """Expand files, directories and globs into report paths (.sqlite/.nsys-rep)."""
    paths: list[Path] = []
    for item in items:
        matched = [Path(p) for p in sorted(glob.glob(item))] or [Path(item)]
        for candidate in matched:
            if candidate.is_dir():
                paths.extend(sorted(candidate.glob("*.nsys-rep")))
                paths.extend(sorted(candidate.glob("*.sqlite")))
            else:
                paths.append(candidate)
    return list(dict.fromkeys(paths))


def _cache_path_for(rep: Path) -> Path:
    sibling = rep.with_suffix(".sqlite")
    parent = rep.resolve().parent
    if parent.exists() and _is_writable(parent):
        return sibling
    digest = hashlib.sha256(str(rep.resolve()).encode()).hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "dynamo-nsys-request-timeline"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{rep.stem}-{digest}.sqlite"


def _is_writable(path: Path) -> bool:
    try:
        probe = path / ".nsys_request_timeline_probe"
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


def ensure_sqlite(
    path: Path,
    nsys: str,
    runner: Callable[[list[str]], None] | None = None,
) -> Path:
    """Return a SQLite export for ``path``, converting a ``.nsys-rep`` if needed.

    An existing export at least as new as the report is reused.
    """
    if not path.exists():
        raise TimelineError(f"input does not exist: {path}")
    if path.suffix != ".nsys-rep":
        return path
    out = _cache_path_for(path)
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    cmd = [
        nsys,
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
        "true",
        "--output",
        str(out),
        str(path),
    ]
    print(f"exporting {path} -> {out}", file=sys.stderr)
    try:
        if runner is not None:
            runner(cmd)
        else:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise TimelineError(
            f"cannot run '{nsys}' to export {path}; pass --nsys <path>"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise TimelineError(
            f"nsys export failed for {path}: {(exc.stderr or exc.stdout or '').strip()}"
        ) from exc
    return out


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _pid_of(global_tid: int | None) -> int:
    return ((global_tid or 0) >> 24) & 0xFFFFFF


def _tid_of(global_tid: int | None) -> int:
    return (global_tid or 0) & 0xFFFFFF


def open_report(path: Path) -> sqlite3.Connection:
    """Open a SQLite export read-only."""
    uri = f"file:{urllib.parse.quote(str(path.resolve()))}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise TimelineError(f"cannot open {path}: {exc}") from exc


def _check_nsys_export(conn: sqlite3.Connection, path: Path) -> tuple[int, int]:
    """Return (session start unix ns, duration ns) of an nsys SQLite export."""
    if not _has_table(conn, "TARGET_INFO_SESSION_START_TIME") or not _has_table(
        conn, "ANALYSIS_DETAILS"
    ):
        raise TimelineError(
            f"{path}: not an nsys SQLite export (missing session/analysis tables)"
        )
    utc0_row = conn.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchone()
    dur_row = conn.execute("SELECT duration FROM ANALYSIS_DETAILS").fetchone()
    if utc0_row is None or dur_row is None:
        raise TimelineError(f"{path}: export has no session time information")
    return int(utc0_row[0]), int(dur_row[0])


def _query_events(
    conn: sqlite3.Connection, where: str, params: tuple[Any, ...]
) -> list[tuple[Any, ...]]:
    if not _has_table(conn, "NVTX_EVENTS"):
        return []
    has_sid = _has_table(conn, "StringIds")
    sid_value = "s.value" if has_sid else "NULL"
    sql = _SQL_BASE.format(
        sid_value=sid_value,
        sid_join="LEFT JOIN StringIds s ON s.id = e.textId" if has_sid else "",
        where=where.format(sid_value=sid_value),
    )
    return conn.execute(sql, params).fetchall()


def scan_report(path: Path, key: int) -> ReportScan:
    """Scan one report for the request-scoped NVTX rows whose payload is ``key``.

    Runs a single query selecting every range and mark with ``uint64Value = key``
    (the stage is the event text) plus a handful of small metadata queries. The
    ``dynamo.request rid=`` identity marks among the rows are not events: their
    attrs (trace, xrid, engine_rid, ...) are merged per process into
    ``ReportScan.identities`` and stamped on that process's events.
    """
    conn = open_report(path)
    try:
        utc0, duration = _check_nsys_export(conn, path)
        rows = _query_events(conn, _WHERE_BY_KEY, (key,))

        host = _read_host(conn)
        identities: dict[int, dict[str, str]] = {}
        rids: set[str] = set()
        raw: list[tuple[Any, ...]] = []
        for row in rows:
            ident = parse_identity(row[4]) if row[2] == _EVT_MARK else None
            if ident is None:
                raw.append(row)
                continue
            rids.add(ident.rid)
            merged = identities.setdefault(_pid_of(row[3]), {})
            for name, value in ident.attrs.items():
                merged.setdefault(name, value)
        rid = min(rids) if rids else f"key:{key:016x}"
        events: list[NvtxEvent] = []
        for start, end, etype, gtid, msg, _payload in raw:
            if not msg:
                continue
            attrs = dict(identities.get(_pid_of(gtid), {}))
            events.append(
                _make_event(
                    str(path),
                    utc0,
                    duration,
                    NvtxMessage(msg, rid, attrs),
                    start,
                    end,
                    etype,
                    gtid,
                )
            )

        pids = {e.pid for e in events}
        proc_names = _read_process_names(conn)
        thread_names = _read_thread_names(conn, {(e.pid, e.tid) for e in events})
        for event in events:
            event.thread_name = thread_names.get((event.pid, event.tid))
        processes = {
            pid: ProcessInfo(
                report=str(path),
                pid=pid,
                name=proc_names.get(pid),
                host=host,
                role=infer_role([e.msg for e in events if e.pid == pid]),
            )
            for pid in sorted(pids)
        }
        return ReportScan(
            str(path), host, utc0, duration, events, processes, identities, rids
        )
    except sqlite3.Error as exc:
        raise TimelineError(f"{path}: sqlite error: {exc}") from exc
    finally:
        conn.close()


def find_identities(path: Path, attr: str, value: str) -> set[tuple[str, int]]:
    """(rid, key) pairs of identity marks in ``path`` whose ``attr`` equals ``value``."""
    conn = open_report(path)
    try:
        _check_nsys_export(conn, path)
        needle = f" {attr}={value}"
        found: set[tuple[str, int]] = set()
        for *_, msg, payload in _query_events(conn, _WHERE_IDENTITY, (needle,)):
            ident = parse_identity(msg)
            if ident is not None and ident.attrs.get(attr) == value:
                found.add((ident.rid, int(payload)))
        return found
    except sqlite3.Error as exc:
        raise TimelineError(f"{path}: sqlite error: {exc}") from exc
    finally:
        conn.close()


def scan_stage_durations(path: Path) -> tuple[dict[str, list[float]], int]:
    """Durations (ms) of closed request-scoped ranges per stage, and the open count."""
    conn = open_report(path)
    try:
        _, duration = _check_nsys_export(conn, path)
        out: dict[str, list[float]] = {}
        open_count = 0
        for start, end, _etype, _gtid, msg, _payload in _query_events(
            conn, _WHERE_SUMMARY, ()
        ):
            if not msg or not _STAGE_NAME_RE.match(msg):
                continue
            if start is None or start > duration or end > duration:
                open_count += 1  # unclosed ranges keep a raw un-rebased timestamp
                continue
            out.setdefault(msg, []).append((end - start) / 1e6)
        return out, open_count
    except sqlite3.Error as exc:
        raise TimelineError(f"{path}: sqlite error: {exc}") from exc
    finally:
        conn.close()


def _make_event(
    report: str,
    utc0: int,
    duration: int,
    msg: NvtxMessage,
    start: int,
    end: int | None,
    etype: int,
    gtid: int,
) -> NvtxEvent:
    mark = etype == _EVT_MARK
    # Unclosed ranges keep a raw (un-rebased) timestamp larger than the session.
    start_unknown = start is None or start > duration
    end_unknown = (not mark) and (end is None or end > duration)
    is_open = (not mark) and (start_unknown or end_unknown)
    start_ns = None if start_unknown else utc0 + start
    end_ns: int | None
    if mark or end_unknown or start_unknown:
        end_ns = None
    else:
        end_ns = utc0 + end  # type: ignore[operator]
    return NvtxEvent(
        msg=msg,
        report=report,
        pid=_pid_of(gtid),
        tid=_tid_of(gtid),
        mark=mark,
        start_ns=start_ns,
        end_ns=end_ns,
        open=is_open,
        rel_start_s=None if start_unknown else start / 1e9,
        rel_end_s=end / 1e9 if (end_ns is not None and end is not None) else None,
    )


def _read_host(conn: sqlite3.Connection) -> str | None:
    if not _has_table(conn, "TARGET_INFO_SYSTEM_ENV"):
        return None
    row = conn.execute(
        "SELECT value FROM TARGET_INFO_SYSTEM_ENV WHERE name = 'Hostname' LIMIT 1"
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _read_process_names(conn: sqlite3.Connection) -> dict[int, str]:
    if not _has_table(conn, "PROCESSES"):
        return {}
    return {
        int(pid): str(name)
        for pid, name in conn.execute("SELECT pid, name FROM PROCESSES")
        if pid is not None and name
    }


def _read_thread_names(
    conn: sqlite3.Connection, wanted: set[tuple[int, int]]
) -> dict[tuple[int, int], str]:
    if (
        not wanted
        or not _has_table(conn, "ThreadNames")
        or not _has_table(conn, "StringIds")
    ):
        return {}
    out: dict[tuple[int, int], str] = {}
    query = (
        "SELECT t.globalTid, s.value FROM ThreadNames t "
        "JOIN StringIds s ON s.id = t.nameId"
    )
    for gtid, name in conn.execute(query):
        key = (_pid_of(gtid), _tid_of(gtid))
        if key in wanted:
            out[key] = str(name)
    return out


def assign_worker_legs(scans: Sequence["ReportScan"]) -> None:
    """Name generic worker processes after the frontend leg that dispatched to them.

    A worker whose own stages do not reveal its role (no lifecycle or handler ranges,
    e.g. a Rust mock worker) is matched by time: its ``handle_payload`` starts inside the
    frontend's ``route_request:<phase>`` range for that leg. When several legs contain it
    (concurrent prefill/decode), the latest-starting leg wins. Then labels that would
    still collide get the report file name appended.
    """
    legs = sorted(
        (
            (e.start_ns, e.end_ns, e.msg.qualifier)
            for scan in scans
            for e in scan.events
            if e.msg.base in _STAGE_ROUTE and e.msg.qualifier and e.start_ns is not None
        ),
        key=lambda leg: leg[0],
    )
    for scan in scans:
        for proc in scan.processes.values():
            if proc.role not in ("worker", "unknown"):
                continue
            starts = [
                e.start_ns
                for e in scan.events
                if e.pid == proc.pid and e.msg.base == _STAGE_HANDLE and e.start_ns
            ]
            if not starts:
                continue
            first = min(starts)
            containing = [
                phase
                for start, end, phase in legs
                if start <= first and (end is None or first <= end)
            ]
            if containing:
                proc.role = containing[-1]

    by_label: dict[str, list[ProcessInfo]] = {}
    for scan in scans:
        for proc in scan.processes.values():
            by_label.setdefault(proc.label, []).append(proc)
    for procs in by_label.values():
        if len(procs) > 1:
            for proc in procs:
                proc.suffix = f" [{Path(proc.report).name}]"


def infer_role(msgs: Iterable[NvtxMessage]) -> str:
    """Guess a process role from the stages it emitted for this request."""
    stages = [m.base for m in msgs]
    qualifiers = {
        m.qualifier
        for m in msgs
        if m.qualifier and (m.base.startswith("worker.") or m.base == _STAGE_HANDLE)
    }
    # Lifecycle spans name the role as a dotted suffix: worker.operation.prefill,
    # response.streaming.decode.
    for m in msgs:
        for prefix in _LIFECYCLE_ROLE_PREFIXES:
            role = m.base[len(prefix) :] if m.base.startswith(prefix) else ""
            if role in _WORKER_ROLES:
                qualifiers.add(role)
    worker_like = any(s == _STAGE_HANDLE or s.startswith("worker.") for s in stages)
    frontend_like = any(
        s in (_STAGE_HTTP, _STAGE_FIRST_TOKEN)
        or s.startswith("kv_router.")
        or s == "router.route_request"
        or s.startswith("request.")
        for s in stages
    )
    if worker_like:
        roles = sorted(q for q in qualifiers if q)
        if len(roles) == 1:
            return roles[0]
        if len(roles) > 1:
            return "+".join(roles)
        return "worker"
    if frontend_like:
        return "frontend"
    return "unknown"


# --------------------------------------------------------------------------
# OTel: JSON parsing and Tempo access
# --------------------------------------------------------------------------

# (status, parsed JSON or None); injected in tests.
HttpGet = Callable[[str], "tuple[int, Any]"]


def default_fetch(url: str, timeout: float = 15.0) -> tuple[int, Any]:
    """GET ``url`` and parse JSON. HTTP errors return (status, None)."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise TimelineError(f"unsupported URL scheme: {url}")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TimelineError(f"cannot reach Tempo at {url}: {exc}") from exc
    try:
        return status, json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimelineError(f"Tempo returned invalid JSON for {url}: {exc}") from exc


def _decode_id(value: Any) -> str:
    """Decode an OTLP trace/span id that is either hex or base64 into lowercase hex."""
    if not isinstance(value, str) or not value:
        return ""
    if _HEX_RE.match(value) and len(value) in (16, 32):
        return value.lower()
    try:
        return base64.b64decode(value, validate=True).hex()
    except (binascii.Error, ValueError):
        return value.lower()


def _any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [_any_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv.get("key"): _any_value(kv.get("value"))
            for kv in value["kvlistValue"].get("values", [])
        }
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    if not isinstance(attrs, list):
        return {}
    return {a["key"]: _any_value(a.get("value")) for a in attrs if "key" in a}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_otel_json(doc: Any) -> list[OtelSpan]:
    """Parse Tempo v1 (``batches``), Tempo v2 (``trace.resourceSpans``) or OTLP JSON."""
    if doc is None:
        return []
    if isinstance(doc, list):
        spans: list[OtelSpan] = []
        for item in doc:
            spans.extend(parse_otel_json(item))
        return spans
    if not isinstance(doc, dict):
        return []
    if "trace" in doc and isinstance(doc["trace"], dict):
        doc = doc["trace"]
    resource_spans = doc.get("resourceSpans") or doc.get("batches") or []
    out: list[OtelSpan] = []
    for rs in resource_spans:
        res_attrs = _attrs_to_dict((rs.get("resource") or {}).get("attributes"))
        service = res_attrs.get("service.name")
        scope_list = rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
        for scope in scope_list:
            for span in scope.get("spans", []):
                start = _as_int(span.get("startTimeUnixNano"))
                if start is None:
                    continue
                attrs = _attrs_to_dict(span.get("attributes"))
                end = _as_int(span.get("endTimeUnixNano"))
                out.append(
                    OtelSpan(
                        trace_id=_decode_id(span.get("traceId")),
                        span_id=_decode_id(span.get("spanId")),
                        parent_span_id=_decode_id(span.get("parentSpanId")),
                        name=str(span.get("name", "")),
                        service=str(service) if service is not None else None,
                        start_ns=start,
                        end_ns=end if end and end >= start else None,
                        attrs=attrs,
                    )
                )
    return out


def _escape_traceql(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _traceql_attr(name: str) -> str:
    return (
        f"span.{name}"
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        else f'span."{name}"'
    )


class OtelSource:
    """Interface of an OTel span source."""

    description = "otel"

    def spans_for_trace(self, trace_id: str) -> list[OtelSpan] | None:
        raise NotImplementedError

    def trace_ids_for(
        self, attrs: Sequence[str], value: str, window_s: tuple[int, int] | None
    ) -> list[str]:
        raise NotImplementedError


class FileOtelSource(OtelSource):
    """Spans loaded from a saved Tempo response or OTLP JSON file."""

    def __init__(self, path: Path) -> None:
        self.description = f"file {path}"
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TimelineError(f"cannot read OTel JSON {path}: {exc}") from exc
        self.spans = parse_otel_json(doc)
        if not self.spans:
            raise TimelineError(f"{path}: no spans found (expected Tempo or OTLP JSON)")

    def spans_for_trace(self, trace_id: str) -> list[OtelSpan] | None:
        spans = [s for s in self.spans if s.trace_id == trace_id]
        return spans or None

    def trace_ids_for(
        self, attrs: Sequence[str], value: str, window_s: tuple[int, int] | None
    ) -> list[str]:
        found = {
            s.trace_id
            for s in self.spans
            if any(s.attrs.get(a) == value for a in attrs)
        }
        return sorted(found)


class TempoOtelSource(OtelSource):
    """Spans fetched from the Tempo HTTP API."""

    def __init__(self, url: str, fetch: HttpGet | None = None) -> None:
        self.base = url.rstrip("/")
        self.description = f"tempo {self.base}"
        self._fetch = fetch or default_fetch
        self.notes: list[str] = []

    def spans_for_trace(self, trace_id: str) -> list[OtelSpan] | None:
        for path in ("/api/v2/traces/", "/api/traces/"):
            status, doc = self._fetch(f"{self.base}{path}{trace_id}")
            if status == 404:
                continue
            if status != 200:
                raise TimelineError(
                    f"Tempo returned HTTP {status} for trace {trace_id}"
                )
            if isinstance(doc, dict) and doc.get("status") == "partial":
                self.notes.append(f"Tempo reports trace {trace_id} as partial")
            spans = parse_otel_json(doc)
            if spans:
                return spans
        return None

    def trace_ids_for(
        self, attrs: Sequence[str], value: str, window_s: tuple[int, int] | None
    ) -> list[str]:
        for attr in attrs:
            query = f'{{ {_traceql_attr(attr)} = "{_escape_traceql(value)}" }}'
            params: dict[str, Any] = {"q": query, "limit": 20}
            if window_s is not None:
                params["start"], params["end"] = window_s
            status, doc = self._fetch(
                f"{self.base}/api/search?{urllib.parse.urlencode(params)}"
            )
            if status != 200 or not isinstance(doc, dict):
                if status not in (200, 400, 404):
                    raise TimelineError(f"Tempo search returned HTTP {status}")
                continue
            ids = sorted(
                {
                    str(t["traceID"]).lower().rjust(32, "0")
                    for t in doc.get("traces", [])
                    if t.get("traceID")
                }
            )
            if ids:
                return ids
        return []


def rids_of_spans(spans: Iterable[OtelSpan]) -> set[str]:
    return {s.rid for s in spans if s.rid}


# --------------------------------------------------------------------------
# Key resolution
# --------------------------------------------------------------------------


def key_candidates(key: str) -> list[tuple[str, str]]:
    """Ordered (kind, value) interpretations of a user key."""
    stripped = key.strip()
    if not stripped:
        raise TimelineError("--request must not be empty")
    if stripped.startswith(_CHATCMPL_PREFIX):
        return [("rid", stripped[len(_CHATCMPL_PREFIX) :])]
    if _TRACE_ID_RE.match(stripped):
        return [("trace", stripped.lower()), ("xrid", stripped)]
    if _UUID_RE.match(stripped):
        return [("rid", stripped), ("xrid", stripped)]
    return [("xrid", stripped), ("rid", stripped)]


def sanitize_nvtx_value(value: str) -> str:
    """Render ``value`` as the runtime writes it into an NVTX ``key=value`` token.

    Mirrors ``format_request_message`` in lib/runtime/src/nvtx.rs: whitespace becomes
    ``_``, control characters are dropped, and the result is cut to 128 UTF-8 bytes on
    a character boundary.
    """
    out: list[str] = []
    size = 0
    for ch in value:
        if ch in _RUST_WHITESPACE:
            ch = "_"
        elif unicodedata.category(ch) == "Cc":
            continue
        width = len(ch.encode("utf-8"))
        if size + width > _MAX_VALUE_BYTES:
            break
        out.append(ch)
        size += width
    return "".join(out)


def _scan_key(paths: Sequence[Path], key: int) -> list[ReportScan]:
    return [scan_report(p, key) for p in paths]


def _scan_rid(paths: Sequence[Path], rid: str) -> list[ReportScan]:
    return _scan_key(paths, rid_key(rid))


def _identity_matches(
    paths: Sequence[Path], kind: str, value: str
) -> set[tuple[str, int]]:
    """(rid, key) of identity marks with ``trace=``/``xrid=`` equal to ``value``."""
    if kind == "xrid":
        # NVTX carries the sanitized client id; OTel lookups keep the raw one.
        value = sanitize_nvtx_value(value)
    found: set[tuple[str, int]] = set()
    for path in paths:
        found |= find_identities(path, kind, value)
    return found


def _ambiguity_error(key: str, kind: str, rids: Iterable[str]) -> TimelineError:
    listing = "\n  ".join(sorted(rids))
    return TimelineError(
        f"key '{key}' ({kind}) is ambiguous; it matches several request ids:\n  "
        f"{listing}\nre-run with --request <rid>"
    )


def resolve_key(
    key: str,
    paths: Sequence[Path],
    otel: OtelSource | None,
    window_s: Callable[[list[ReportScan]], tuple[int, int] | None] | None = None,
) -> tuple[Resolution, list[ReportScan]]:
    """Resolve ``key`` to one request id and return the scans for that id."""
    tried: list[str] = []
    for kind, value in key_candidates(key):
        tried.append(f"{kind}={value}")
        scans: list[ReportScan] = []
        rid: str | None = None
        if kind == "rid":
            scans = _scan_rid(paths, value)
            if any(s.events for s in scans):
                rid = value
        else:
            matches = _identity_matches(paths, kind, value)
            rids = {r for r, _ in matches}
            if len(rids) > 1:
                raise _ambiguity_error(key, kind, rids)
            if matches:
                rid, payload = next(iter(matches))
                scans = _scan_key(paths, payload)
        if rid is not None:
            via = (
                "NVTX payload key"
                if kind == "rid"
                else f"NVTX identity marks ({kind}=)"
            )
            res = Resolution(key, kind, rid, via)
            _fill_ids_from_events(res, scans)
            if kind == "trace":
                res.trace_id = value
            if kind == "xrid":
                res.xrid = value
            return res, scans
        if otel is not None:
            res_otel = _resolve_via_otel(key, kind, value, otel, scans, window_s)
            if res_otel is not None:
                if res_otel.rid:
                    scans = _scan_rid(paths, res_otel.rid)
                _fill_ids_from_events(res_otel, scans)
                return res_otel, scans
    raise TimelineError(
        f"request key '{key}' not found in any nsys report"
        + (" or in OTel" if otel is not None else "")
        + f" (tried: {', '.join(tried)}). Check that the processes ran with "
        "DYN_NVTX=1 and that the report covers the request."
    )


def _fill_ids_from_events(res: Resolution, scans: Sequence[ReportScan]) -> None:
    for scan in scans:
        for attrs in scan.identities.values():
            res.trace_id = res.trace_id or attrs.get("trace")
            res.xrid = res.xrid or attrs.get("xrid")


def _resolve_via_otel(
    key: str,
    kind: str,
    value: str,
    otel: OtelSource,
    scans: list[ReportScan],
    window_fn: Callable[[list[ReportScan]], tuple[int, int] | None] | None,
) -> Resolution | None:
    window = window_fn(scans) if window_fn else None
    if kind == "trace":
        trace_ids = [value]
    else:
        attrs = _RID_ATTRS if kind == "rid" else _XRID_ATTRS
        trace_ids = otel.trace_ids_for(attrs, value, window)
    rids: set[str] = set()
    found_traces: list[str] = []
    for trace_id in trace_ids:
        spans = otel.spans_for_trace(trace_id)
        if not spans:
            continue
        found_traces.append(trace_id)
        rids |= rids_of_spans(spans)
    if kind == "rid":
        rids = {value} if found_traces else set()
    if len(rids) > 1:
        raise _ambiguity_error(key, kind, rids)
    if not found_traces:
        return None
    if len(found_traces) > 1:
        raise TimelineError(
            f"key '{key}' matches several OTel traces: {', '.join(found_traces)}"
        )
    rid = next(iter(rids)) if rids else None
    res = Resolution(
        key, kind, rid, f"OTel ({otel.description})", trace_id=found_traces[0]
    )
    if kind == "xrid":
        res.xrid = value
    if rid is None:
        res.notes.append(
            "OTel spans carry no request_id attribute; NVTX join not possible"
        )
    return res


# --------------------------------------------------------------------------
# Join: build rows
# --------------------------------------------------------------------------


def _roles_compatible(event_role: str | None, span_role: str | None) -> bool:
    known = ("frontend", "prefill", "decode")
    return event_role not in known or span_role not in known or event_role == span_role


def _role_from_service(service: str | None) -> str | None:
    if not service:
        return None
    lowered = service.lower()
    for role in ("frontend", "prefill", "decode"):
        if role in lowered:
            return role
    return None


def pair_spans(
    events: Sequence[NvtxEvent],
    spans: Sequence[OtelSpan],
    event_role: Callable[[NvtxEvent], str | None] = lambda _e: None,
) -> list[tuple[NvtxEvent, OtelSpan]]:
    """Pair NVTX events with OTel spans by stage (incl. phase qualifier).

    Within a stage the nearest unpaired span by start time wins, which keeps
    occurrence order for retries and multiple workers. A second pass pairs the
    leftovers by base stage name so a missing phase on one side still joins.
    When both the event's process role and the span's service name reveal a role
    (prefill, decode, frontend) they must agree, so cross-host clock skew cannot
    pair the prefill worker's ``handle_payload`` with the decode worker's.
    """
    pairs: list[tuple[NvtxEvent, OtelSpan]] = []
    used_events: set[int] = set()
    used_spans: set[int] = set()

    def run(
        event_key: Callable[[NvtxEvent], str], span_key: Callable[[OtelSpan], str]
    ) -> None:
        by_key: dict[str, list[int]] = {}
        for idx, span in enumerate(spans):
            if idx not in used_spans:
                by_key.setdefault(span_key(span), []).append(idx)
        order = sorted(
            (i for i in range(len(events)) if i not in used_events),
            key=lambda i: (events[i].start_ns is None, events[i].start_ns or 0),
        )
        for ei in order:
            event = events[ei]
            role = event_role(event)
            candidates = [
                i
                for i in by_key.get(event_key(event), [])
                if i not in used_spans
                and _roles_compatible(role, _role_from_service(spans[i].service))
            ]
            if not candidates:
                continue
            if event.start_ns is None:
                best = candidates[0]
            else:
                ref = event.start_ns
                best = min(candidates, key=lambda i: abs(spans[i].start_ns - ref))
            used_events.add(ei)
            used_spans.add(best)
            pairs.append((event, spans[best]))

    run(lambda e: e.msg.stage, lambda s: s.stage)
    run(lambda e: e.msg.base, lambda s: s.name)
    return pairs


def build_rows(scans: Sequence[ReportScan], spans: Sequence[OtelSpan]) -> list[Row]:
    """Merge NVTX events and OTel spans into chronologically sorted rows."""
    events = [e for scan in scans for e in scan.events]
    infos = {(s.path, pid): info for s in scans for pid, info in s.processes.items()}
    pairs = pair_spans(events, spans, lambda e: infos[(e.report, e.pid)].role)
    paired_events = {id(e): sp for e, sp in pairs}
    paired_spans = {id(sp) for _, sp in pairs}

    rows: list[Row] = []
    for event in events:
        info = infos[(event.report, event.pid)]
        row = Row(
            stage=event.msg.stage,
            source="nvtx",
            process=info.label,
            role=info.role,
            mark=event.mark,
            start_ns=event.start_ns,
            end_ns=event.end_ns,
            open=event.open,
            report=event.report,
            pid=event.pid,
            rel_start_s=event.rel_start_s,
            rel_end_s=event.rel_end_s,
            attrs=dict(event.msg.attrs),
        )
        span = paired_events.get(id(event))
        if span is not None:
            row.source = "both"
            row.otel_start_ns = span.start_ns
            row.otel_end_ns = span.end_ns
            if event.start_ns is not None:
                row.delta_ns = span.start_ns - event.start_ns
        rows.append(row)
    for span in spans:
        if id(span) in paired_spans:
            continue
        rows.append(
            Row(
                stage=span.stage,
                source="otel",
                process=f"otel:{span.service or '?'}",
                role=_role_from_service(span.service),
                mark=False,
                start_ns=span.start_ns,
                end_ns=span.end_ns,
                open=span.end_ns is None,
                otel_start_ns=span.start_ns,
                otel_end_ns=span.end_ns,
            )
        )
    rows.sort(key=lambda r: (r.start_ns is None, r.start_ns or 0, r.process, r.stage))
    return rows


def timeline_origin(rows: Sequence[Row]) -> int | None:
    starts = [r.start_ns for r in rows if r.start_ns is not None]
    return min(starts) if starts else None


def row_duration_ns(row: Row) -> int | None:
    if row.mark:
        return None
    if row.end_ns is not None and row.start_ns is not None:
        return row.end_ns - row.start_ns
    if row.otel_end_ns is not None and row.otel_start_ns is not None and row.open:
        return row.otel_end_ns - row.otel_start_ns
    return None


# --------------------------------------------------------------------------
# TTFT breakdown
# --------------------------------------------------------------------------


def _first(rows: Iterable[Row]) -> Row | None:
    candidates = sorted(
        (r for r in rows if r.start_ns is not None), key=lambda r: r.start_ns or 0
    )
    return candidates[0] if candidates else None


def _legs_reached(legs: Sequence[Row], rows: Sequence[Row]) -> dict[int, list[Row]]:
    """The worker ``handle_payload`` rows each frontend leg dispatched to, by leg index.

    A ``handle_payload`` belongs to the latest-starting leg containing its start (the same
    rule as :func:`assign_worker_legs`), so a retried or migrated request's second leg is
    not credited with the first worker's stages.
    """
    reached: dict[int, list[Row]] = {}
    for row in rows:
        if row.base != _STAGE_HANDLE or row.start_ns is None:
            continue
        containing = [
            i
            for i, leg in enumerate(legs)
            if leg.start_ns is not None
            and leg.start_ns <= row.start_ns
            and (leg.end_ns is None or row.start_ns <= leg.end_ns)
        ]
        if containing:
            reached.setdefault(containing[-1], []).append(row)
    return reached


def _leg_phases(legs: Sequence[Row], reached: dict[int, list[Row]]) -> list[str]:
    """Phase of each frontend leg: its qualifier, else the role of the worker it reached.

    Non-KV routers emit ``router.route_request`` without a phase; an unqualified leg whose
    workers all share one prefill/decode/... role takes that role, otherwise ``worker``.
    """
    phases = []
    for i, leg in enumerate(legs):
        roles = {r.role for r in reached.get(i, []) if r.role in _WORKER_ROLES}
        phases.append(leg.qualifier or (roles.pop() if len(roles) == 1 else "worker"))
    return phases


def _worker_belongs(row: Row, phase: str, single_leg: bool) -> bool:
    if row.base not in _WORKER_STAGES and row.base != _STAGE_HANDLE:
        return False
    if row.qualifier is not None:
        return row.qualifier == phase or single_leg
    if row.role == phase:
        return True
    return single_leg and row.role not in ("frontend", None)


class _Chain:
    """Emit items for sequential stages with gap/overlap detection."""

    def __init__(
        self, t0: int, ft: int | None, items: list[BreakdownItem], depth: int
    ) -> None:
        self.t0, self.ft, self.items, self.depth = t0, ft, items, depth
        self.cursor: int | None = None
        self.gap_ns = 0

    def ms(self, ns: int) -> float:
        return (ns - self.t0) / 1e6

    def add(
        self, row: Row, name: str, note: str = "", gap_label: str = "gap (unattributed)"
    ) -> None:
        start = row.start_ns
        if start is None:
            return
        if self.ft is not None and start > self.ft:
            return
        end = start if row.mark else row.end_ns
        notes = [note] if note else []
        if not row.mark and end is None:
            end = self.ft
            notes.append("open range" + (", clipped at first token" if self.ft else ""))
        elif end is not None and self.ft is not None and end > self.ft:
            end = self.ft
            notes.append("clipped at first token")
        if self.cursor is not None:
            if start > self.cursor:
                gap = start - self.cursor
                self.gap_ns += gap
                self.items.append(
                    BreakdownItem(
                        "gap", gap_label, self.ms(self.cursor), gap / 1e6, self.depth
                    )
                )
            elif start < self.cursor:
                notes.append(
                    f"overlaps previous by {(self.cursor - start) / 1e6:.3f} ms"
                )
        dur = None if row.mark or end is None else (end - start) / 1e6
        self.items.append(
            BreakdownItem(
                "mark" if row.mark else "segment",
                name,
                self.ms(start),
                dur,
                self.depth,
                "; ".join(notes),
            )
        )
        self.cursor = max(self.cursor or 0, end if end is not None else start)


def build_breakdown(rows: Sequence[Row]) -> Breakdown | None:
    """Compute the TTFT breakdown from merged rows, or None without catalog stages."""
    http = _first(r for r in rows if r.base == _STAGE_HTTP)
    first_token = _first(r for r in rows if r.base == _STAGE_FIRST_TOKEN)
    legs = sorted(
        (r for r in rows if r.base in _STAGE_ROUTE and r.start_ns is not None),
        key=lambda r: r.start_ns or 0,
    )
    if http is None and first_token is None and not legs:
        return None
    notes: list[str] = []
    if http is not None and http.start_ns is not None:
        t0, t0_source = http.start_ns, _STAGE_HTTP + " start"
    else:
        starts = [
            r.start_ns for r in [*legs, first_token] if r is not None and r.start_ns
        ]
        t0, t0_source = min(starts), "first available frontend stage"
        notes.append(
            "http-request missing; t0 is the earliest available frontend stage"
        )
    ft = first_token.start_ns if first_token is not None else None
    if ft is None:
        notes.append(
            "frontend.first_token missing; TTFT unknown and segments are not clipped"
        )

    items: list[BreakdownItem] = []
    ms = lambda ns: (ns - t0) / 1e6  # noqa: E731
    items.append(BreakdownItem("mark", "frontend arrival (http-request)", 0.0, None, 0))

    top = _Chain(t0, ft, items, 0)
    top.cursor = t0
    pre = _first(r for r in rows if r.base == _STAGE_PREPROCESS)
    if pre is not None:
        top.add(pre, _STAGE_PREPROCESS)
    elif first_token is not None:
        notes.append(
            "request.preprocessing not present (needs DYN_LIFECYCLE_TRACE_ENABLED=1)"
        )

    selects = sorted(
        (r for r in rows if r.base == _STAGE_SELECT and r.start_ns is not None),
        key=lambda r: r.start_ns or 0,
    )
    schedules = [
        r for r in rows if r.base == _STAGE_SCHEDULE and r.start_ns is not None
    ]
    single_leg = len(legs) == 1
    reached = _legs_reached(legs, rows)
    leg_phase = _leg_phases(legs, reached)

    # Associate every selection with the next leg that starts after it began.
    select_for_leg: dict[int, Row] = {}
    for sel in selects:
        for li, leg in enumerate(legs):
            if li not in select_for_leg and (leg.start_ns or 0) >= (sel.start_ns or 0):
                select_for_leg[li] = sel
                break
    unmatched = [s for s in selects if s not in select_for_leg.values()]

    sequence: list[tuple[int, Row | None, Row | None]] = []
    for li, leg in enumerate(legs):
        sequence.append((li, select_for_leg.get(li), leg))
    for sel in unmatched:
        sequence.append((-1, sel, None))
    sequence.sort(key=lambda t: ((t[1] or t[2]).start_ns or 0))  # type: ignore[union-attr]

    for li, sel, leg in sequence:
        phase = leg_phase[li] if leg is not None else (sel.qualifier or "worker")  # type: ignore[union-attr]
        if sel is not None:
            note = ""
            sched = [
                s
                for s in schedules
                if sel.start_ns is not None
                and s.start_ns is not None
                and s.start_ns >= sel.start_ns
                and (sel.end_ns is None or s.start_ns <= sel.end_ns)
            ]
            if sched and (d := row_duration_ns(sched[0])) is not None:
                note = f"includes {_STAGE_SCHEDULE} (queue wait) {d / 1e6:.3f} ms"
            top.add(sel, f"{phase} selection ({_STAGE_SELECT})", note)
        if leg is not None:
            top.add(leg, f"{phase} leg ({leg.stage})")
            processes = {r.process for r in reached.get(li, [])}
            _add_leg_children(rows, leg, phase, single_leg, t0, ft, items, processes)

    if first_token is not None and ft is not None:
        items.append(
            BreakdownItem(
                "mark", f"{_STAGE_FIRST_TOKEN}", ms(ft), None, 0, "TTFT point"
            )
        )
    ttft_ms = ms(ft) if ft is not None else None
    unattributed = top.gap_ns / 1e6
    # Depth-0 segments can overlap (SGLang decode starts while prefill still streams), so
    # attributed time is the union of their intervals, not their sum.
    intervals = sorted(
        (it.start_ms, it.start_ms + it.dur_ms)
        for it in items
        if it.depth == 0
        and it.kind == "segment"
        and it.start_ms is not None
        and it.dur_ms is not None
    )
    attributed = 0.0
    merged_start, merged_end = None, None
    for start, end in intervals:
        if merged_end is not None and start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        if merged_end is not None:
            attributed += merged_end - merged_start
        merged_start, merged_end = start, end
    if merged_end is not None:
        attributed += merged_end - merged_start
    return Breakdown(t0_source, ttft_ms, attributed, unattributed, items, notes)


def _add_leg_children(
    rows: Sequence[Row],
    leg: Row,
    phase: str,
    single_leg: bool,
    t0: int,
    ft: int | None,
    items: list[BreakdownItem],
    processes: set[str],
) -> None:
    """Nest the worker stages of the processes this leg reached under it.

    ``processes`` comes from :func:`_legs_reached`; when it is empty (no
    ``handle_payload`` was captured) fall back to matching by phase and role alone.
    """
    kids = sorted(
        (
            r
            for r in rows
            if _worker_belongs(r, phase, single_leg)
            and r.start_ns is not None
            and (not processes or r.process in processes)
        ),
        key=lambda r: r.start_ns or 0,
    )
    if not kids or leg.start_ns is None:
        return
    handle = _first(r for r in kids if r.base == _STAGE_HANDLE)
    subs = [r for r in kids if r.base != _STAGE_HANDLE]
    chain = _Chain(t0, ft, items, 1)
    worker_start = min(r.start_ns for r in kids if r.start_ns is not None)
    if worker_start > leg.start_ns and (ft is None or worker_start <= ft):
        items.append(
            BreakdownItem(
                "gap",
                "transport/queue to worker (includes clock skew if cross-host)",
                (leg.start_ns - t0) / 1e6,
                (worker_start - leg.start_ns) / 1e6,
                1,
            )
        )
    if handle is not None:
        chain.add(handle, f"{_STAGE_HANDLE} ({handle.process})")
        chain.cursor = handle.start_ns  # sub-stages nest inside handle_payload
    for sub in subs:
        chain.add(sub, f"{sub.stage} ({sub.process})")
    egress = next((r for r in subs if r.base == "worker.egress.first_chunk"), None)
    if (
        egress is not None
        and egress.start_ns is not None
        and ft is not None
        and phase != "prefill"
    ):
        items.append(
            BreakdownItem(
                "gap",
                "worker egress -> frontend first_token (transport; negative means clock skew)",
                (egress.start_ns - t0) / 1e6,
                (ft - egress.start_ns) / 1e6,
                1,
            )
        )


# --------------------------------------------------------------------------
# Zoom windows
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ZoomWindow:
    report: str
    pid: int
    process: str
    start_s: float
    end_s: float
    has_open: bool


def zoom_windows(scans: Sequence[ReportScan]) -> list[ZoomWindow]:
    """Per report and process, the report-relative window covering the request."""
    out: list[ZoomWindow] = []
    for scan in scans:
        for pid, info in scan.processes.items():
            evs = [e for e in scan.events if e.pid == pid and e.rel_start_s is not None]
            if not evs:
                continue
            start = min(e.rel_start_s for e in evs if e.rel_start_s is not None)
            ends = [
                e.rel_end_s if e.rel_end_s is not None else e.rel_start_s
                for e in evs
                if (e.rel_end_s is not None or e.rel_start_s is not None)
            ]
            has_open = any(e.open for e in evs)
            end = max(x for x in ends if x is not None)
            if has_open:
                end = max(end, scan.duration_ns / 1e9)
            out.append(ZoomWindow(scan.path, pid, info.label, start, end, has_open))
    return out


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _fmt_dur(row: Row) -> str:
    if row.mark:
        return "mark"
    dur = row_duration_ns(row)
    if dur is None:
        return "open"
    suffix = "+" if row.open else ""
    return f"{dur / 1e6:.3f}{suffix}"


def _fmt_delta(row: Row) -> str:
    if row.delta_ns is None:
        return ""
    return f"{row.delta_ns / 1e3:+.1f}us"


def render_text(report: dict[str, Any]) -> str:
    """Render the report dict (see :func:`build_report`) as text."""
    lines: list[str] = []
    res = report["resolution"]
    lines.append("== Request resolution ==")
    lines.append(f"key:          {res['key']}")
    lines.append(f"resolved as:  {res['kind']} -> rid={res['rid']}  (via {res['via']})")
    lines.append(f"trace id:     {res['trace_id'] or '-'}")
    lines.append(f"x-request-id: {res['xrid'] or '-'}")
    for rep in report["reports"]:
        procs = ", ".join(p["label"] for p in rep["processes"]) or "no events"
        lines.append(
            f"report:       {Path(rep['path']).name}  host={rep['host'] or '?'}  "
            f"events={rep['events']}  [{procs}]"
        )
    otel = report["otel"]
    lines.append(
        f"otel:         {otel['source'] or 'not provided'}"
        + (f"  spans={otel['spans']}" if otel["source"] else "")
    )
    lines.append("")
    lines.append("== Timeline ==")
    header = f"{'t+ms':>10}  {'dur_ms':>10}  {'process':<34} {'stage':<42} {'src':<5} {'d(otel-nvtx)':>13}"
    lines.append(header)
    for row in report["timeline"]:
        t = "?" if row["t_ms"] is None else f"{row['t_ms']:.3f}"
        lines.append(
            f"{t:>10}  {row['dur_text']:>10}  {row['process'][:34]:<34} "
            f"{row['stage'][:42]:<42} {row['source']:<5} {row['delta_text']:>13}"
        )
    lines.append("")
    bd = report["ttft_breakdown"]
    lines.append("== TTFT breakdown ==")
    if bd is None:
        lines.append(
            "no catalog frontend stages (http-request / kv_router.route_request / frontend.first_token) found"
        )
    else:
        ttft = "unknown" if bd["ttft_ms"] is None else f"{bd['ttft_ms']:.3f} ms"
        lines.append(
            f"t0 = {bd['t0_source']}; TTFT = {ttft}; "
            f"attributed {bd['attributed_ms']:.3f} ms, unattributed gaps {bd['unattributed_ms']:.3f} ms"
        )
        lines.append(f"{'t+ms':>10}  {'dur_ms':>10}  stage")
        for it in bd["items"]:
            start = "" if it["start_ms"] is None else f"{it['start_ms']:.3f}"
            dur = "" if it["dur_ms"] is None else f"{it['dur_ms']:.3f}"
            note = f"  [{it['note']}]" if it["note"] else ""
            prefix = "  " * it["depth"] + ("~ " if it["kind"] == "gap" else "")
            lines.append(f"{start:>10}  {dur:>10}  {prefix}{it['name']}{note}")
        for note in bd["notes"]:
            lines.append(f"note: {note}")
    lines.append("")
    lines.append(
        "== nsys zoom windows (report-relative, as shown in the Nsight timeline) =="
    )
    for zw in report["zoom"]:
        extra = " (contains open range; end is end of report)" if zw["has_open"] else ""
        lines.append(
            f"zoom to {zw['start_s']:.4f}s-{zw['end_s']:.4f}s in {Path(zw['report']).name}"
            f"  [{zw['process']}]{extra}"
        )
    if report["warnings"]:
        lines.append("")
        lines.append("== Warnings ==")
        lines.extend(f"warning: {w}" for w in report["warnings"])
    return "\n".join(lines)


def build_report(
    resolution: Resolution,
    scans: Sequence[ReportScan],
    spans: Sequence[OtelSpan],
    otel_description: str | None,
    warnings: list[str],
) -> tuple[dict[str, Any], list[Row]]:
    """Assemble the machine-readable report (also the input of the text renderer)."""
    rows = build_rows(scans, spans)
    origin = timeline_origin(rows)
    breakdown = build_breakdown(rows)

    def t_ms(ns: int | None) -> float | None:
        return None if ns is None or origin is None else (ns - origin) / 1e6

    timeline = []
    for row in rows:
        dur = row_duration_ns(row)
        timeline.append(
            {
                "t_ms": t_ms(row.start_ns),
                "dur_ms": None if dur is None else dur / 1e6,
                "dur_text": _fmt_dur(row),
                "process": row.process,
                "role": row.role,
                "stage": row.stage,
                "source": row.source,
                "mark": row.mark,
                "open": row.open,
                "delta_us": None if row.delta_ns is None else row.delta_ns / 1e3,
                "delta_text": _fmt_delta(row),
                "start_unix_ns": row.start_ns,
                "end_unix_ns": row.end_ns,
                "otel_start_unix_ns": row.otel_start_ns,
                "otel_end_unix_ns": row.otel_end_ns,
                "report": row.report,
                "pid": row.pid,
                "report_rel_start_s": row.rel_start_s,
                "report_rel_end_s": row.rel_end_s,
                "attrs": row.attrs,
            }
        )
    report = {
        "resolution": {
            "key": resolution.key,
            "kind": resolution.kind,
            "rid": resolution.rid,
            "via": resolution.via,
            "trace_id": resolution.trace_id,
            "xrid": resolution.xrid,
            "notes": resolution.notes,
        },
        "reports": [
            {
                "path": s.path,
                "host": s.host,
                "session_start_unix_ns": s.session_start_ns,
                "duration_s": s.duration_ns / 1e9,
                "events": len(s.events),
                "processes": [
                    {
                        "pid": p.pid,
                        "role": p.role,
                        "host": p.host,
                        "name": p.name,
                        "label": p.label,
                    }
                    for p in s.processes.values()
                ],
            }
            for s in scans
        ],
        "otel": {"source": otel_description, "spans": len(spans)},
        "timeline": timeline,
        "ttft_breakdown": None
        if breakdown is None
        else {
            "t0_source": breakdown.t0_source,
            "ttft_ms": breakdown.ttft_ms,
            "attributed_ms": breakdown.attributed_ms,
            "unattributed_ms": breakdown.unattributed_ms,
            "items": [dataclasses.asdict(i) for i in breakdown.items],
            "notes": breakdown.notes,
        },
        "zoom": [
            {
                "report": z.report,
                "pid": z.pid,
                "process": z.process,
                "start_s": z.start_s,
                "end_s": z.end_s,
                "has_open": z.has_open,
            }
            for z in zoom_windows(scans)
        ],
        "warnings": warnings,
    }
    return report, rows


def collect_warnings(
    resolution: Resolution,
    scans: Sequence[ReportScan],
    spans: Sequence[OtelSpan],
    otel: OtelSource | None,
    rows: Sequence[Row],
) -> list[str]:
    """Non-fatal problems worth surfacing."""
    warnings = list(resolution.notes)
    for scan in scans:
        if not scan.events:
            warnings.append(f"rid not seen in {Path(scan.path).name}")
    if not any(s.events for s in scans):
        warnings.append("rid seen in no nsys report (OTel data only)")
    open_events = [e for s in scans for e in s.events if e.open]
    for event in open_events:
        warnings.append(
            f"open range '{event.msg.stage}' in {Path(event.report).name} pid {event.pid}: "
            "end unknown (capture ended before it closed, or the range never closed)"
        )
    roles = {p.role for s in scans for p in s.processes.values()}
    if scans and any(s.events for s in scans) and "frontend" not in roles:
        warnings.append(
            "no frontend report contains this request (TTFT breakdown will be partial)"
        )
    if otel is None:
        warnings.append("OTel not provided (--tempo/--otel-json); only NVTX data shown")
    elif not spans:
        warnings.append(
            "no OTel spans found for this request: the trace may not have been sampled "
            "(OTEL_TRACES_SAMPLE_RATIO) or has aged out of Tempo (the local stack keeps ~1h)"
        )
    if isinstance(otel, TempoOtelSource):
        warnings.extend(otel.notes)
    skews = [abs(r.delta_ns) for r in rows if r.delta_ns is not None]
    if skews and max(skews) > 5_000_000:
        warnings.append(
            f"paired OTel/NVTX start deltas reach {max(skews) / 1e6:.1f} ms; "
            "same-host deltas should be microseconds, so expect clock skew or a mis-pairing"
        )
    return warnings


# --------------------------------------------------------------------------
# Perfetto (Chrome trace) export
# --------------------------------------------------------------------------


def build_perfetto(rows: Sequence[Row]) -> dict[str, Any]:
    """Chrome Trace Event JSON: one process row per captured process plus OTel."""
    origin = timeline_origin(rows) or 0
    pids: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    for row in rows:
        if row.start_ns is None:
            continue
        pid = pids.setdefault(row.process, len(pids) + 1)
        args: dict[str, Any] = {"source": row.source, **row.attrs}
        if row.delta_ns is not None:
            args["otel_minus_nvtx_start_us"] = row.delta_ns / 1e3
        if row.open:
            args["open"] = True
        ts = int(round((row.start_ns - origin) / 1e3))
        if row.mark:
            events.append(
                {
                    "name": row.stage,
                    "cat": "nvtx",
                    "ph": "i",
                    "s": "t",
                    "pid": pid,
                    "tid": 1,
                    "ts": ts,
                    "args": args,
                }
            )
            continue
        dur = row_duration_ns(row)
        dur_us = max(1, int(round((dur or 0) / 1e3)))
        events.append(
            {
                "name": row.stage,
                "cat": "otel" if row.source == "otel" else "nvtx",
                "ph": "X",
                "pid": pid,
                "tid": 1,
                "ts": ts,
                "dur": dur_us,
                "args": args,
            }
        )
    meta: list[dict[str, Any]] = []
    for name, pid in pids.items():
        meta.append(
            {"name": "process_name", "ph": "M", "pid": pid, "args": {"name": name}}
        )
        meta.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 1,
                "args": {"name": "request"},
            }
        )
    events.sort(key=lambda e: (e["ts"], e["pid"]))
    return {"displayTimeUnit": "ms", "traceEvents": meta + events}


# --------------------------------------------------------------------------
# Per-stage summary across all requests
# --------------------------------------------------------------------------


def _p99(sorted_values: Sequence[float]) -> float:
    """Nearest-rank 99th percentile of an ascending, non-empty sequence."""
    return sorted_values[max(math.ceil(0.99 * len(sorted_values)) - 1, 0)]


def build_summary(paths: Sequence[Path]) -> dict[str, Any]:
    """Per-stage duration aggregates over every request-scoped range in ``paths``.

    Counts closed ranges (start/end ranges and push/pop ranges) carrying a payload,
    grouped by stage name; marks and ranges that never closed are not aggregated.
    """
    durations: dict[str, list[float]] = {}
    open_ranges = 0
    for path in paths:
        per_stage, open_count = scan_stage_durations(path)
        open_ranges += open_count
        for stage, values in per_stage.items():
            durations.setdefault(stage, []).extend(values)
    stages = []
    for stage in sorted(durations):
        values = sorted(durations[stage])
        stages.append(
            {
                "stage": stage,
                "count": len(values),
                "mean_ms": statistics.fmean(values),
                "median_ms": statistics.median(values),
                "p99_ms": _p99(values),
                "max_ms": values[-1],
            }
        )
    return {
        "reports": [str(p) for p in paths],
        "open_ranges_skipped": open_ranges,
        "stages": stages,
    }


def render_summary(summary: dict[str, Any]) -> str:
    stages = summary["stages"]
    lines = [f"== Per-stage summary ({len(summary['reports'])} report(s)) =="]
    if not stages:
        lines.append(
            "no request-scoped ranges found (were the processes run with DYN_NVTX=1?)"
        )
    else:
        width = max(len(s["stage"]) for s in stages)
        lines.append(
            f"{'stage':<{width}}  {'count':>8}  {'mean ms':>10}  {'median ms':>10}  "
            f"{'p99 ms':>10}  {'max ms':>10}"
        )
        for s in stages:
            lines.append(
                f"{s['stage']:<{width}}  {s['count']:>8}  {s['mean_ms']:>10.3f}  "
                f"{s['median_ms']:>10.3f}  {s['p99_ms']:>10.3f}  {s['max_ms']:>10.3f}"
            )
    if summary["open_ranges_skipped"]:
        lines.append(
            f"skipped {summary['open_ranges_skipped']} range(s) that never closed"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show one request's Dynamo-level timeline across nsys reports "
            "(DYN_NVTX=1) joined with its OTel spans."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="nsys .sqlite exports or .nsys-rep files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--request",
        help="Request id, chatcmpl-<id>, W3C trace id (32 hex), or x-request-id.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Instead of one request, print per-stage count/mean/median/p99/max "
            "across ALL requests in the reports (honors --json)."
        ),
    )
    parser.add_argument(
        "--nsys", default="nsys", help="nsys binary used to export .nsys-rep files."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tempo", help="Tempo base URL, e.g. http://localhost:3200.")
    source.add_argument("--otel-json", help="Saved Tempo response or OTLP JSON file.")
    parser.add_argument(
        "--otel-margin-s",
        type=int,
        default=600,
        help="Margin around the nsys event times for Tempo search windows (default: 600).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    parser.add_argument(
        "--perfetto",
        metavar="OUT",
        help="Also write a Chrome Trace JSON (Perfetto UI) with one process per captured process.",
    )
    args = parser.parse_args(argv)
    if not args.summary and not args.request:
        parser.error("--request is required unless --summary is given")
    return args


def _window_fn(margin_s: int) -> Callable[[list[ReportScan]], tuple[int, int] | None]:
    def window(scans: list[ReportScan]) -> tuple[int, int] | None:
        times = [e.start_ns for s in scans for e in s.events if e.start_ns is not None]
        if not times:
            return None
        return (int(min(times) / 1e9) - margin_s, int(max(times) / 1e9) + margin_s)

    return window


def _load_spans(
    otel: OtelSource, resolution: Resolution, scans: Sequence[ReportScan], margin_s: int
) -> list[OtelSpan]:
    trace_id = resolution.trace_id
    if trace_id is None and resolution.rid is not None:
        ids = otel.trace_ids_for(
            _RID_ATTRS, resolution.rid, _window_fn(margin_s)(list(scans))
        )
        if len(ids) > 1:
            raise TimelineError(
                f"rid {resolution.rid} matches several OTel traces: {', '.join(ids)}"
            )
        trace_id = ids[0] if ids else None
    if trace_id is None:
        return []
    resolution.trace_id = trace_id
    spans = otel.spans_for_trace(trace_id) or []
    # Keep spans of this request: drop those explicitly tagged with another rid.
    return [s for s in spans if s.rid is None or s.rid == resolution.rid]


def run(
    args: argparse.Namespace,
    fetch: HttpGet | None = None,
    export_runner: Callable[[list[str]], None] | None = None,
) -> int:
    """Execute the CLI; ``fetch`` and ``export_runner`` are injectable for tests."""
    try:
        inputs = expand_inputs(args.inputs)
        if not inputs:
            raise TimelineError("no input files matched")
        sqlite_paths = list(
            dict.fromkeys(ensure_sqlite(p, args.nsys, export_runner) for p in inputs)
        )
        if args.summary:
            summary = build_summary(sqlite_paths)
            if args.json:
                json.dump(summary, sys.stdout, indent=2)
                sys.stdout.write("\n")
            else:
                print(render_summary(summary))
            return 0
        otel: OtelSource | None = None
        if args.tempo:
            otel = TempoOtelSource(args.tempo, fetch)
        elif args.otel_json:
            otel = FileOtelSource(Path(args.otel_json))
        resolution, scans = resolve_key(
            args.request, sqlite_paths, otel, _window_fn(args.otel_margin_s)
        )
        assign_worker_legs(scans)
        spans = (
            _load_spans(otel, resolution, scans, args.otel_margin_s)
            if otel is not None
            else []
        )
        rows = build_rows(scans, spans)
        warnings = collect_warnings(resolution, scans, spans, otel, rows)
        report, rows = build_report(
            resolution, scans, spans, otel.description if otel else None, warnings
        )
        if args.perfetto:
            out = Path(args.perfetto)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(build_perfetto(rows), f, separators=(",", ":"))
                f.write("\n")
            print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)
    except TimelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(render_text(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
