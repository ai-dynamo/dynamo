# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nsys_request_timeline using synthetic nsys SQLite exports (no network)."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from benchmarks.request_trace import nsys_request_timeline as nrt

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

RID = "11111111-2222-3333-4444-555555555555"
OTHER_RID = "99999999-2222-3333-4444-555555555555"
# Leading zero byte so Tempo's search response (stripped zeros) must be re-padded.
TRACE = "00f7651916cd43dd8448eb211c80319c"
XRID = "my-req-1"

MS = 1_000_000
BASE = 1_700_000_000_000_000_000  # unix ns of "t=0 ms" in the scenario
DURATION = 2_000 * MS  # every synthetic session lasts 2 s

# Sessions start at different instants, as with real per-process captures.
SESSION_START = {
    "frontend": BASE,
    "prefill": BASE + 1_000 * MS,
    "decode": BASE - 500 * MS,
}

SCHEMA = """
CREATE TABLE NVTX_EVENTS (
    start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL, rangeId INTEGER,
    category INTEGER, color INTEGER, text TEXT, globalTid INTEGER, endGlobalTid INTEGER,
    textId INTEGER, domainId INTEGER, uint64Value INTEGER, int64Value INTEGER,
    doubleValue REAL, uint32Value INTEGER, int32Value INTEGER, floatValue REAL,
    jsonTextId INTEGER, jsonText TEXT, binaryData TEXT);
CREATE TABLE StringIds (id INTEGER NOT NULL PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE TARGET_INFO_SESSION_START_TIME (
    utcEpochNs INTEGER, utcTime TEXT, localTime TEXT, systemClockNs INTEGER);
CREATE TABLE ANALYSIS_DETAILS (
    globalVid INTEGER NOT NULL, duration INTEGER NOT NULL,
    startTime INTEGER NOT NULL, stopTime INTEGER NOT NULL);
CREATE TABLE ThreadNames (nameId INTEGER NOT NULL, priority INTEGER, globalTid INTEGER);
CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER, name TEXT);
CREATE TABLE TARGET_INFO_SYSTEM_ENV (
    globalVid INTEGER, devStateName TEXT NOT NULL, name TEXT NOT NULL,
    nameEnum INTEGER NOT NULL, value TEXT NOT NULL);
"""


def gtid(pid: int, tid: int) -> int:
    return (pid << 24) | tid


class ReportBuilder:
    """Writes a synthetic nsys SQLite export with the real column layout."""

    def __init__(self, path: Path, session_start_ns: int, pid: int, host: str) -> None:
        self.path = path
        self.pid = pid
        self.session_start_ns = session_start_ns
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, 'u', 'l', 0)",
            (session_start_ns,),
        )
        self.conn.execute(
            "INSERT INTO ANALYSIS_DETAILS VALUES (?, ?, 0, ?)",
            (gtid(pid, 0), DURATION, DURATION),
        )
        self.conn.execute(
            "INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES (NULL, 'd', 'Hostname', 1, ?)",
            (host,),
        )
        self.conn.execute(
            "INSERT INTO PROCESSES VALUES (?, ?, ?)", (gtid(pid, 0), pid, f"proc-{pid}")
        )
        self._next_string = 1
        self.name_thread(1, "main-thread")

    def name_thread(self, tid: int, name: str) -> None:
        sid = self._string_id(name)
        self.conn.execute(
            "INSERT INTO ThreadNames VALUES (?, 20, ?)", (sid, gtid(self.pid, tid))
        )

    def _string_id(self, value: str) -> int:
        row = self.conn.execute(
            "SELECT id FROM StringIds WHERE value=?", (value,)
        ).fetchone()
        if row:
            return int(row[0])
        sid = self._next_string
        self._next_string += 1
        self.conn.execute("INSERT INTO StringIds VALUES (?, ?)", (sid, value))
        return sid

    def _insert(
        self,
        start: int,
        end: int | None,
        etype: int,
        msg: str,
        tid: int,
        registered: bool,
        rid: str | None,
    ) -> None:
        """Insert one event; ``rid`` becomes the uint64Value payload (None: no payload)."""
        text, text_id = (None, self._string_id(msg)) if registered else (msg, None)
        key = None if rid is None else nrt.rid_key(rid)
        self.conn.execute(
            "INSERT INTO NVTX_EVENTS (start, end, eventType, text, globalTid, textId, "
            "domainId, uint64Value) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
            (start, end, etype, text, gtid(self.pid, tid), text_id, key),
        )

    def _rel(self, abs_ms: float) -> int:
        return int(abs_ms * MS) + BASE - self.session_start_ns

    def rng(
        self,
        stage: str,
        start_ms: float,
        end_ms: float | None,
        tid: int = 1,
        registered: bool = False,
        rid: str | None = RID,
    ) -> None:
        """A start/end range; ``end_ms=None`` makes an unclosed (open) range."""
        end: int | None
        if end_ms is None:
            # nsys keeps a raw un-rebased (huge) timestamp for unclosed ranges.
            end = self.session_start_ns + self._rel(start_ms)
        else:
            end = self._rel(end_ms)
        self._insert(self._rel(start_ms), end, 60, stage, tid, registered, rid)

    def mark(
        self,
        stage: str,
        at_ms: float,
        tid: int = 1,
        registered: bool = False,
        rid: str | None = RID,
    ) -> None:
        self._insert(self._rel(at_ms), None, 34, stage, tid, registered, rid)

    def identity(self, rid: str, at_ms: float, tid: int = 1, **attrs: str) -> None:
        """A ``dynamo.request rid=<rid> k=v...`` identity mark (attrs in given order)."""
        text = f"dynamo.request rid={rid}" + "".join(
            f" {k}={v}" for k, v in attrs.items()
        )
        self._insert(self._rel(at_ms), None, 34, text, tid, False, rid)

    def noise(self) -> None:
        # Rows that must be ignored: another request's events and identity (same stage
        # names, different key), a static range without payload, and an identity mark
        # of another request that merely mentions this rid in an attribute.
        self.identity(OTHER_RID, 49, trace="f" * 32, engine_rid=RID)
        self.rng("http-request", 50, 60, rid=OTHER_RID)
        self.rng("detokenize", 70, 71, rid=None)
        self.rng("worker.handler.prepare:decode", 72, 73, rid=OTHER_RID)

    def finish(self) -> Path:
        self.conn.commit()
        self.conn.close()
        return self.path


def build_scenario(tmp_path: Path) -> dict[str, Path]:
    """Disaggregated request over a frontend, a prefill worker and a decode worker."""
    fe = ReportBuilder(
        tmp_path / "frontend.sqlite", SESSION_START["frontend"], 1000, "node-a"
    )
    fe.noise()
    fe.identity(RID, 100, trace=TRACE, xrid=XRID)
    fe.rng("http-request", 100, 400)
    fe.rng("request.preprocessing", 101, 112)
    fe.rng("kv_router.select_worker", 113, 120)
    fe.rng("kv_router.schedule", 114, 117)
    fe.rng("kv_router.route_request:prefill", 121, 300, registered=True)
    fe.rng("kv_router.select_worker", 301, 305)
    fe.rng("kv_router.route_request:decode", 306, 400)
    fe.mark("frontend.first_token", 330, registered=True)

    pf = ReportBuilder(
        tmp_path / "prefill.sqlite", SESSION_START["prefill"], 2000, "node-b"
    )
    pf.identity(RID, 123, trace=TRACE, xrid=XRID)
    pf.identity(RID, 127, engine_rid="abc")  # a second identity mark, merged
    pf.rng("handle_payload", 123, 298)
    pf.rng("worker.admission_gate.wait", 124, 126)
    pf.rng("worker.handler.prepare:prefill", 126, 127)
    pf.rng("worker.engine.first_output:prefill", 127, 295)
    pf.mark("worker.egress.first_chunk", 296)

    dc = ReportBuilder(
        tmp_path / "decode.sqlite", SESSION_START["decode"], 3000, "node-c"
    )
    dc.identity(RID, 308)
    dc.rng("handle_payload", 308, 399)
    dc.rng("worker.admission_gate.wait", 309, 310)
    dc.rng("worker.handler.prepare:decode", 310, 311)
    dc.rng("worker.engine.first_output:decode", 311, 325)
    dc.mark("worker.egress.first_chunk", 326)
    dc.rng("response.streaming.decode", 312, None)  # never closed
    return {"frontend": fe.finish(), "prefill": pf.finish(), "decode": dc.finish()}


@pytest.fixture
def reports(tmp_path: Path) -> dict[str, Path]:
    return build_scenario(tmp_path)


def b64(hex_id: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_id)).decode()


def _span(
    name: str,
    span_id: str,
    start_ms: float,
    end_ms: float,
    attrs: dict[str, Any],
    delta_us: float = 0.0,
) -> dict[str, Any]:
    def anyval(v: Any) -> dict[str, Any]:
        return {"intValue": str(v)} if isinstance(v, int) else {"stringValue": v}

    return {
        "traceId": b64(TRACE),
        "spanId": b64(span_id),
        "parentSpanId": "",
        "name": name,
        "kind": "SPAN_KIND_INTERNAL",
        "startTimeUnixNano": str(int(BASE + start_ms * MS + delta_us * 1000)),
        "endTimeUnixNano": str(int(BASE + end_ms * MS)),
        "attributes": [{"key": k, "value": anyval(v)} for k, v in attrs.items()],
    }


def otel_batches() -> list[dict[str, Any]]:
    def res(service: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service}}
                ]
            },
            "scopeSpans": [{"scope": {"name": "dynamo"}, "spans": spans}],
        }

    return [
        res(
            "dynamo-frontend",
            [
                _span(
                    "http-request",
                    "0000000000000001",
                    100,
                    400,
                    {"request_id": RID, "x_request_id": XRID},
                    delta_us=3,
                ),
                _span(
                    "kv_router.route_request",
                    "0000000000000002",
                    121,
                    300,
                    {"request_id": RID, "phase": "Prefill"},
                    delta_us=7,
                ),
                _span(
                    "kv_router.route_request",
                    "0000000000000003",
                    306,
                    400,
                    {"request_id": RID, "phase": "decode"},
                    delta_us=-2,
                ),
                _span(
                    "tokenize.other",
                    "0000000000000004",
                    102,
                    103,
                    {"dynamo.request.id": RID},
                ),
                _span(
                    "unrelated", "0000000000000005", 102, 103, {"request_id": OTHER_RID}
                ),
            ],
        ),
        res(
            "dynamo-prefill",
            [
                _span(
                    "handle_payload",
                    "0000000000000006",
                    123,
                    298,
                    {"request_id": RID},
                    delta_us=11,
                )
            ],
        ),
    ]


def tempo_v1() -> dict[str, Any]:
    return {"batches": otel_batches()}


def tempo_v2() -> dict[str, Any]:
    return {"trace": {"resourceSpans": otel_batches()}}


def fake_tempo(calls: list[str] | None = None, v2: bool = True) -> Any:
    def fetch(url: str) -> tuple[int, Any]:
        if calls is not None:
            calls.append(url)
        if f"/api/v2/traces/{TRACE}" in url:
            return (200, tempo_v2()) if v2 else (404, None)
        if f"/api/traces/{TRACE}" in url:
            return 200, tempo_v1()
        if "/api/search" in url:
            if "span.request_id" in url or "x_request_id" in url:
                return 200, {
                    "traces": [
                        {
                            "traceID": TRACE.lstrip("0"),
                            "rootServiceName": "dynamo-frontend",
                        }
                    ]
                }
            return 200, {"traces": []}
        return 404, None

    return fetch


def paths(reports: dict[str, Path]) -> list[Path]:
    return [reports["frontend"], reports["prefill"], reports["decode"]]


# ----------------------------------------------------------------------------
# Parsing and scanning
# ----------------------------------------------------------------------------


# Pinned with the Rust implementation (lib/runtime/src/nvtx.rs): FNV-1a-64, top bit cleared.
RID_KEY_VECTORS = [
    ("", 5472609002491880229),
    ("a", 3414815163700866188),
    ("6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42", 1346822055457142067),
]


@pytest.mark.parametrize(("rid", "expected"), RID_KEY_VECTORS)
def test_rid_key_vectors(rid: str, expected: int) -> None:
    assert nrt.rid_key(rid) == expected
    assert 0 <= expected < 1 << 63  # positive in SQLite's signed INTEGER


def test_rid_key_hashes_utf8_bytes() -> None:
    assert nrt.rid_key("é") != nrt.rid_key("e")
    assert nrt.rid_key("é") == nrt.rid_key(b"\xc3\xa9".decode("utf-8"))


def test_parse_identity_grammar() -> None:
    msg = nrt.parse_identity(
        f"dynamo.request rid={RID} trace={TRACE} xrid=a_b engine_rid=e1"
    )
    assert msg is not None
    assert (msg.stage, msg.rid) == ("dynamo.request", RID)
    assert msg.attrs == {"trace": TRACE, "xrid": "a_b", "engine_rid": "e1"}
    bare = nrt.parse_identity(f"dynamo.request rid={RID}")
    assert bare is not None and bare.attrs == {}
    assert nrt.parse_identity("worker.egress.chunk") is None
    assert nrt.parse_identity(f"http-request rid={RID}") is None  # v1 format dropped
    assert nrt.parse_identity(f"dynamo.request rid={RID}  trace=x") is None
    assert nrt.parse_identity("") is None
    assert nrt.parse_identity(None) is None


def test_scan_report_ascii_and_registered_strings(reports: dict[str, Path]) -> None:
    scan = nrt.scan_report(reports["frontend"], nrt.rid_key(RID))
    stages = {e.msg.stage for e in scan.events}
    # route_request:prefill and frontend.first_token were stored through StringIds/textId.
    assert "kv_router.route_request:prefill" in stages
    assert "frontend.first_token" in stages
    assert "http-request" in stages
    # The other request's events, the static range and identity marks are not events.
    assert all(e.msg.rid == RID for e in scan.events)
    assert scan.rids == {RID}
    assert scan.identities == {1000: {"trace": TRACE, "xrid": XRID}}
    assert len(scan.events) == 8
    http = next(e for e in scan.events if e.msg.stage == "http-request")
    assert http.start_ns == BASE + 100 * MS and http.end_ns == BASE + 400 * MS
    assert http.rel_start_s == pytest.approx(0.100) and http.rel_end_s == pytest.approx(
        0.400
    )
    assert http.pid == 1000 and http.tid == 1 and http.thread_name == "main-thread"
    assert scan.host == "node-a"
    assert scan.processes[1000].role == "frontend"
    assert scan.processes[1000].name == "proc-1000"
    assert scan.processes[1000].label == "frontend/pid1000@node-a"


def test_scan_uses_each_reports_session_start(reports: dict[str, Path]) -> None:
    scan = nrt.scan_report(reports["prefill"], nrt.rid_key(RID))
    hp = next(e for e in scan.events if e.msg.stage == "handle_payload")
    assert (
        hp.start_ns == BASE + 123 * MS
    )  # absolute, independent of the 1 s later session start
    assert hp.rel_start_s == pytest.approx(
        -0.877
    )  # report-relative time may be negative in the fixture
    assert scan.processes[2000].role == "prefill"


def test_open_range_is_flagged(reports: dict[str, Path]) -> None:
    scan = nrt.scan_report(reports["decode"], nrt.rid_key(RID))
    streaming = next(
        e for e in scan.events if e.msg.stage == "response.streaming.decode"
    )
    assert streaming.open and streaming.end_ns is None
    closed = next(e for e in scan.events if e.msg.stage == "handle_payload")
    assert not closed.open and closed.end_ns == BASE + 399 * MS
    assert scan.processes[3000].role == "decode"


def test_non_nsys_database_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sqlite"
    sqlite3.connect(bad).close()
    with pytest.raises(nrt.TimelineError, match="not an nsys SQLite export"):
        nrt.scan_report(bad, 1)


# ----------------------------------------------------------------------------
# Key resolution
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "kind", "via_fragment"),
    [
        (RID, "rid", "payload key"),
        (f"chatcmpl-{RID}", "rid", "payload key"),
        (TRACE, "trace", "identity marks (trace=)"),
        (XRID, "xrid", "identity marks (xrid=)"),
    ],
)
def test_resolve_key_variants(
    reports: dict[str, Path], key: str, kind: str, via_fragment: str
) -> None:
    res, scans = nrt.resolve_key(key, paths(reports), None)
    assert (res.kind, res.rid) == (kind, RID)
    assert via_fragment in res.via
    assert res.trace_id == TRACE and res.xrid == XRID
    # Second scan on the resolved rid finds the events in all three reports.
    assert [len(s.events) for s in scans] == [8, 5, 6]


def test_resolve_key_not_found(reports: dict[str, Path]) -> None:
    with pytest.raises(nrt.TimelineError, match="not found"):
        nrt.resolve_key("no-such-request", paths(reports), None)


def test_resolve_key_ambiguous_xrid(reports: dict[str, Path], tmp_path: Path) -> None:
    extra = ReportBuilder(tmp_path / "frontend2.sqlite", BASE, 1001, "node-a")
    extra.identity(OTHER_RID, 10, xrid=XRID)
    extra.rng("http-request", 10, 20, rid=OTHER_RID)
    extra.finish()
    with pytest.raises(nrt.TimelineError) as info:
        nrt.resolve_key(XRID, [*paths(reports), extra.path], None)
    message = str(info.value)
    assert "ambiguous" in message and RID in message and OTHER_RID in message


def test_resolve_key_via_otel_only(tmp_path: Path) -> None:
    empty = ReportBuilder(tmp_path / "empty.sqlite", BASE, 1, "n").finish()
    inner = fake_tempo()

    def fetch(url: str) -> tuple[int, Any]:
        status, doc = inner(url)
        if (
            isinstance(doc, dict) and "trace" in doc
        ):  # single-request trace: drop the other rid
            for rs in doc["trace"]["resourceSpans"]:
                for scope in rs["scopeSpans"]:
                    scope["spans"] = [
                        s for s in scope["spans"] if s["name"] != "unrelated"
                    ]
        return status, doc

    source = nrt.TempoOtelSource("http://tempo:3200", fetch)
    res, scans = nrt.resolve_key(TRACE, [empty], source)
    assert res.rid == RID and res.trace_id == TRACE and "OTel" in res.via
    assert scans[0].events == []
    res, _ = nrt.resolve_key(XRID, [empty], source)
    assert res.rid == RID


def test_resolve_trace_with_several_rids_is_ambiguous(tmp_path: Path) -> None:
    empty = ReportBuilder(tmp_path / "empty.sqlite", BASE, 1, "n").finish()
    source = nrt.TempoOtelSource(
        "http://tempo:3200", fake_tempo()
    )  # trace holds two rids
    with pytest.raises(nrt.TimelineError, match="ambiguous"):
        nrt.resolve_key(TRACE, [empty], source)


# ----------------------------------------------------------------------------
# OTel parsing and Tempo access
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc", [tempo_v1(), tempo_v2(), {"resourceSpans": otel_batches()}]
)
def test_parse_otel_json_formats(doc: dict[str, Any]) -> None:
    spans = nrt.parse_otel_json(doc)
    assert len(spans) == 6
    http = next(s for s in spans if s.name == "http-request")
    assert http.trace_id == TRACE  # base64 -> hex
    assert http.span_id == "0000000000000001"
    assert http.start_ns == BASE + 100 * MS + 3000  # string nanos -> int
    assert http.service == "dynamo-frontend"
    assert http.rid == RID
    phase = next(s for s in spans if s.attrs.get("phase") == "Prefill")
    assert (
        phase.stage == "kv_router.route_request:prefill"
    )  # lowercased phase qualifier
    assert next(s for s in spans if s.name == "tokenize.other").rid == RID


def test_parse_otel_json_hex_ids_and_int_values() -> None:
    doc = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": TRACE,
                                "spanId": "00000000000000aa",
                                "name": "x",
                                "startTimeUnixNano": "5",
                                "endTimeUnixNano": "9",
                                "attributes": [
                                    {"key": "n", "value": {"intValue": "42"}},
                                    {"key": "f", "value": {"doubleValue": 1.5}},
                                    {"key": "b", "value": {"boolValue": True}},
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    (span,) = nrt.parse_otel_json(doc)
    assert span.trace_id == TRACE and span.span_id == "00000000000000aa"
    assert span.attrs == {"n": 42, "f": 1.5, "b": True}


def test_tempo_search_pads_trace_id_and_uses_traceql() -> None:
    calls: list[str] = []
    source = nrt.TempoOtelSource("http://tempo:3200/", fake_tempo(calls))
    ids = source.trace_ids_for(("request_id", "dynamo.request.id"), RID, (1000, 2000))
    assert ids == [TRACE]  # re-padded to 32 chars
    url = calls[0]
    assert url.startswith("http://tempo:3200/api/search?")
    assert "start=1000" in url and "end=2000" in url and "limit=20" in url
    assert "span.request_id" in url and "%22" in url


def test_tempo_search_quotes_dotted_attribute_and_falls_back() -> None:
    calls: list[str] = []

    def fetch(url: str) -> tuple[int, Any]:
        calls.append(url)
        if "dynamo.request.id" in url:
            return 200, {"traces": [{"traceID": "abc"}]}
        return 200, {"traces": []}

    ids = nrt.TempoOtelSource("http://t", fetch).trace_ids_for(
        ("request_id", "dynamo.request.id"), RID, None
    )
    assert ids == ["0" * 29 + "abc"]
    assert len(calls) == 2 and "span.%22dynamo.request.id%22" in calls[1]
    assert "start=" not in calls[0]  # no window -> Tempo's default recent-data search


@pytest.mark.parametrize("v2", [True, False])
def test_tempo_get_trace_v2_then_v1_fallback(v2: bool) -> None:
    calls: list[str] = []
    spans = nrt.TempoOtelSource("http://t", fake_tempo(calls, v2=v2)).spans_for_trace(
        TRACE
    )
    assert spans is not None and len(spans) == 6
    assert ("/api/v2/traces/" in calls[0]) and (len(calls) == (1 if v2 else 2))


def test_tempo_missing_trace_returns_none_and_errors_surface() -> None:
    assert (
        nrt.TempoOtelSource("http://t", lambda url: (404, None)).spans_for_trace(TRACE)
        is None
    )
    with pytest.raises(nrt.TimelineError, match="HTTP 500"):
        nrt.TempoOtelSource("http://t", lambda url: (500, None)).spans_for_trace(TRACE)


# ----------------------------------------------------------------------------
# Join, pairing and breakdown
# ----------------------------------------------------------------------------


def _joined(
    reports: dict[str, Path]
) -> tuple[list[nrt.ReportScan], list[nrt.OtelSpan], list[nrt.Row]]:
    _, scans = nrt.resolve_key(RID, paths(reports), None)
    spans = [s for s in nrt.parse_otel_json(tempo_v1()) if s.rid in (None, RID)]
    return scans, spans, nrt.build_rows(scans, spans)


def test_pairing_deltas_and_sources(reports: dict[str, Path]) -> None:
    _, _, rows = _joined(reports)
    by_stage = {(r.stage, r.process.split("/")[0]): r for r in rows}
    http = by_stage[("http-request", "frontend")]
    assert http.source == "both" and http.delta_ns == 3000
    prefill_leg = by_stage[("kv_router.route_request:prefill", "frontend")]
    assert prefill_leg.source == "both" and prefill_leg.delta_ns == 7000
    decode_leg = by_stage[("kv_router.route_request:decode", "frontend")]
    assert decode_leg.delta_ns == -2000
    assert by_stage[("handle_payload", "prefill")].delta_ns == 11000
    # Decode-side handle_payload has no OTel span; the OTel-only span is kept.
    assert by_stage[("handle_payload", "decode")].source == "nvtx"
    assert by_stage[("tokenize.other", "otel:dynamo-frontend")].source == "otel"
    assert not any(
        r.stage == "unrelated" for r in rows
    )  # other rid filtered at span load
    starts = [r.start_ns for r in rows if r.start_ns is not None]
    assert starts == sorted(starts)


def test_pairing_uses_occurrence_order_for_retries() -> None:
    def ev(stage: str, start: int) -> nrt.NvtxEvent:
        return nrt.NvtxEvent(
            nrt.NvtxMessage(stage, RID, {}),
            "r",
            1,
            1,
            False,
            start,
            start + 5,
            False,
            0.0,
            0.0,
        )

    def sp(name: str, start: int) -> nrt.OtelSpan:
        return nrt.OtelSpan(
            TRACE, f"{start:016x}", "", name, "svc", start, start + 5, {}
        )

    events = [ev("handle_payload", 1000), ev("handle_payload", 5000)]
    spans = [sp("handle_payload", 5002), sp("handle_payload", 1001)]
    pairs = nrt.pair_spans(events, spans)
    deltas = sorted(s.start_ns - e.start_ns for e, s in pairs)
    assert deltas == [1, 2]


def test_ttft_breakdown_math(reports: dict[str, Path]) -> None:
    _, _, rows = _joined(reports)
    bd = nrt.build_breakdown(rows)
    assert bd is not None
    assert bd.ttft_ms == pytest.approx(
        230.0
    )  # first_token (330) - http-request start (100)
    top = [i for i in bd.items if i.depth == 0]
    segs = {i.name.split(" (")[0]: i for i in top if i.kind == "segment"}
    assert segs["request.preprocessing"].dur_ms == pytest.approx(11.0)
    assert segs["prefill selection"].dur_ms == pytest.approx(7.0)
    assert "queue wait) 3.000 ms" in segs["prefill selection"].note
    assert segs["prefill leg"].dur_ms == pytest.approx(179.0)
    assert segs["decode selection"].dur_ms == pytest.approx(4.0)
    # The decode leg runs to 400 ms but is clipped at the first token (330 ms).
    assert segs["decode leg"].dur_ms == pytest.approx(24.0)
    assert "clipped at first token" in segs["decode leg"].note
    gaps = [i.dur_ms for i in top if i.kind == "gap"]
    assert gaps == [pytest.approx(1.0)] * 5
    assert bd.unattributed_ms == pytest.approx(5.0)
    assert bd.attributed_ms == pytest.approx(225.0)
    assert bd.attributed_ms + bd.unattributed_ms == pytest.approx(bd.ttft_ms)
    # Worker children nest under their legs and carry the transport gaps.
    children = [i for i in bd.items if i.depth == 1]
    names = [i.name for i in children]
    assert any(n.startswith("worker.engine.first_output:prefill") for n in names)
    assert any(n.startswith("worker.engine.first_output:decode") for n in names)
    transport = [i for i in children if i.name.startswith("transport/queue to worker")]
    assert [round(i.dur_ms or 0, 3) for i in transport] == [2.0, 2.0]
    egress = next(i for i in children if i.name.startswith("worker egress -> frontend"))
    assert egress.dur_ms == pytest.approx(4.0)
    assert bd.items[-1].name == "frontend.first_token"
    assert bd.items[-1].start_ms == pytest.approx(230.0)


def test_breakdown_aggregated_mode_single_leg() -> None:
    def row(
        stage: str,
        start_ms: float,
        end_ms: float | None,
        process: str,
        role: str,
        mark: bool = False,
    ) -> nrt.Row:
        return nrt.Row(
            stage,
            "nvtx",
            process,
            role,
            mark,
            int(start_ms * MS),
            None if mark or end_ms is None else int(end_ms * MS),
            False,
        )

    rows = [
        row("http-request", 0, 100, "frontend/pid1", "frontend"),
        row("kv_router.select_worker", 2, 5, "frontend/pid1", "frontend"),
        row("router.route_request", 6, 100, "frontend/pid1", "frontend"),
        row("worker.admission_gate.wait", 8, 9, "worker/pid2", "worker"),
        row(
            "worker.engine.first_output:aggregated", 9, 40, "worker/pid2", "aggregated"
        ),
        row(
            "worker.egress.first_chunk",
            41,
            None,
            "worker/pid2",
            "aggregated",
            mark=True,
        ),
        row("frontend.first_token", 43, None, "frontend/pid1", "frontend", mark=True),
    ]
    bd = nrt.build_breakdown(rows)
    assert bd is not None and bd.ttft_ms == pytest.approx(43.0)
    assert any(i.name.startswith("worker leg") for i in bd.items)
    assert any(i.name.startswith("worker.admission_gate.wait") for i in bd.items)


def test_breakdown_concurrent_prefill_decode_reports_overlap() -> None:
    def row(stage: str, s: float, e: float | None, mark: bool = False) -> nrt.Row:
        return nrt.Row(
            stage,
            "nvtx",
            "frontend/pid1",
            "frontend",
            mark,
            int(s * MS),
            None if mark or e is None else int(e * MS),
            False,
        )

    rows = [
        row("http-request", 0, 90),
        row("kv_router.route_request:prefill", 5, 50),
        row("kv_router.route_request:decode", 10, 90),
        row("frontend.first_token", 60, None, mark=True),
    ]
    bd = nrt.build_breakdown(rows)
    assert bd is not None
    decode = next(i for i in bd.items if i.name.startswith("decode leg"))
    assert "overlaps previous by 40.000 ms" in decode.note


def test_breakdown_missing_stages_returns_none() -> None:
    assert nrt.build_breakdown([]) is None
    other = nrt.Row("handle_payload", "nvtx", "w", "worker", False, 1, 2, False)
    assert nrt.build_breakdown([other]) is None


def test_zoom_windows(reports: dict[str, Path]) -> None:
    _, scans = nrt.resolve_key(RID, paths(reports), None)
    zooms = {Path(z.report).stem: z for z in nrt.zoom_windows(scans)}
    assert zooms["frontend"].start_s == pytest.approx(0.100)
    assert zooms["frontend"].end_s == pytest.approx(0.400)
    # prefill session starts 1 s later than BASE, so times are negative relative to it
    assert zooms["prefill"].start_s == pytest.approx(-0.877)
    assert zooms["decode"].has_open and zooms["decode"].end_s == pytest.approx(
        DURATION / 1e9
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _args(reports: dict[str, Path], *extra: str) -> Any:
    return nrt.parse_args([*map(str, paths(reports)), *extra])


def test_cli_text_output_end_to_end(
    reports: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(
        reports, "--request", f"chatcmpl-{RID}", "--tempo", "http://tempo:3200"
    )
    assert nrt.run(args, fetch=fake_tempo()) == 0
    out = capsys.readouterr().out
    assert "== Request resolution ==" in out and f"rid={RID}" in out
    assert f"trace id:     {TRACE}" in out
    assert "== Timeline ==" in out and "kv_router.route_request:prefill" in out
    assert "both" in out and "+3.0us" in out
    assert "== TTFT breakdown ==" in out and "TTFT = 230.000 ms" in out
    assert "zoom to 0.1000s-0.4000s in frontend.sqlite" in out
    assert "open range 'response.streaming.decode'" in out


def test_cli_json_output(
    reports: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    otel_file = tmp_path / "trace.json"
    otel_file.write_text(json.dumps(tempo_v2()))
    args = _args(reports, "--request", XRID, "--otel-json", str(otel_file), "--json")
    assert nrt.run(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["resolution"]["rid"] == RID and doc["resolution"]["kind"] == "xrid"
    assert doc["otel"]["spans"] == 5  # the span of another rid is dropped
    assert doc["ttft_breakdown"]["ttft_ms"] == pytest.approx(230.0)
    both = [r for r in doc["timeline"] if r["source"] == "both"]
    assert {r["stage"] for r in both} >= {"http-request", "handle_payload"}
    assert doc["timeline"][0]["t_ms"] == 0.0
    assert len(doc["zoom"]) == 3


def test_cli_warns_without_otel_and_missing_frontend(
    reports: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = nrt.parse_args(
        [str(reports["prefill"]), str(reports["decode"]), "--request", RID, "--json"]
    )
    assert nrt.run(args) == 0
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("no frontend report" in w for w in warnings)
    assert any("OTel not provided" in w for w in warnings)
    assert any("open range" in w for w in warnings)


def test_cli_rid_absent_from_reports_errors(
    reports: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(reports, "--request", "chatcmpl-" + OTHER_RID.replace("9999", "7777"))
    assert nrt.run(args) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_tempo_trace_not_sampled_warns(
    reports: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = _args(reports, "--request", RID, "--tempo", "http://t", "--json")
    assert nrt.run(args, fetch=lambda url: (404, None)) == 0
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("sampled" in w for w in warnings)


def test_perfetto_output(reports: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "out" / "trace.perfetto.json"
    args = _args(
        reports,
        "--request",
        RID,
        "--tempo",
        "http://t",
        "--perfetto",
        str(out),
        "--json",
    )
    assert nrt.run(args, fetch=fake_tempo()) == 0
    trace = json.loads(out.read_text())
    assert trace["displayTimeUnit"] == "ms"
    events = trace["traceEvents"]
    names = {
        e["args"]["name"]
        for e in events
        if e["ph"] == "M" and e["name"] == "process_name"
    }
    assert any(n.startswith("frontend/pid1000") for n in names)
    assert any(n.startswith("prefill/pid2000") for n in names)
    assert any(n.startswith("decode/pid3000") for n in names)
    assert "otel:dynamo-frontend" in names
    http = next(e for e in events if e["ph"] == "X" and e["name"] == "http-request")
    assert http["ts"] == 0 and http["dur"] == 300_000  # microseconds
    first_token = next(
        e for e in events if e["ph"] == "i" and e["name"] == "frontend.first_token"
    )
    assert first_token["ts"] == 230_000


# ----------------------------------------------------------------------------
# .nsys-rep export handling
# ----------------------------------------------------------------------------


def test_ensure_sqlite_exports_and_reuses_fresh_export(tmp_path: Path) -> None:
    rep = tmp_path / "frontend.nsys-rep"
    rep.write_bytes(b"x")
    commands: list[list[str]] = []

    def runner(cmd: list[str]) -> None:
        commands.append(cmd)
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"sqlite")

    out = nrt.ensure_sqlite(rep, "my-nsys", runner)
    assert out == tmp_path / "frontend.sqlite"
    assert commands[0][:5] == [
        "my-nsys",
        "export",
        "--type",
        "sqlite",
        "--force-overwrite",
    ]
    assert commands[0][-1] == str(rep)
    assert (
        nrt.ensure_sqlite(rep, "my-nsys", runner) == out and len(commands) == 1
    )  # reused
    os.utime(
        rep, (out.stat().st_mtime + 10,) * 2
    )  # report is now newer than the export
    nrt.ensure_sqlite(rep, "my-nsys", runner)
    assert len(commands) == 2


def test_expand_inputs_directories_and_globs(
    reports: dict[str, Path], tmp_path: Path
) -> None:
    found = nrt.expand_inputs([str(tmp_path)])
    assert {p.name for p in found} == {
        "frontend.sqlite",
        "prefill.sqlite",
        "decode.sqlite",
    }
    globbed = nrt.expand_inputs(
        [str(tmp_path / "*.sqlite"), str(tmp_path / "frontend.sqlite")]
    )
    assert len(globbed) == 3


def test_generic_workers_are_named_after_the_leg_that_dispatched_to_them(
    tmp_path: Path,
) -> None:
    # Rust mock workers emit only handle_payload (no lifecycle or handler ranges), and
    # containers commonly share one pid and host, so neither role nor label is unique.
    fe = ReportBuilder(tmp_path / "frontend.sqlite", BASE, 24, "node")
    fe.rng("http-request", 0, 300)
    fe.rng("kv_router.route_request:prefill", 10, 100)
    fe.rng("kv_router.route_request:decode", 110, 300)
    fe.mark("frontend.first_token", 150)
    pf = ReportBuilder(tmp_path / "prefill.sqlite", BASE, 24, "node")
    pf.rng("handle_payload", 12, 98)
    dc = ReportBuilder(tmp_path / "decode.sqlite", BASE, 24, "node")
    dc.rng("handle_payload", 112, 298)
    stray = ReportBuilder(tmp_path / "stray.sqlite", BASE, 24, "node")
    stray.rng("handle_payload", 400, 410)  # outside every leg
    scans = nrt._scan_rid([b.finish() for b in (fe, pf, dc, stray)], RID)

    nrt.assign_worker_legs(scans)

    roles = {Path(s.path).stem: s.processes[24].role for s in scans}
    assert roles == {
        "frontend": "frontend",
        "prefill": "prefill",
        "decode": "decode",
        "stray": "worker",
    }
    labels = [s.processes[24].label for s in scans]
    assert len(set(labels)) == len(labels)
    assert "[" not in scans[1].processes[24].label, "unique labels stay short"


def test_lifecycle_span_names_reveal_the_worker_role() -> None:
    msgs = [
        nrt.NvtxMessage(stage, RID, {})
        for stage in (
            "handle_payload",
            "worker.operation.decode",
            "response.streaming.decode",
        )
    ]
    assert nrt.infer_role(msgs) == "decode"


def test_breakdown_names_unqualified_router_legs_after_the_worker_reached() -> None:
    # Non-KV routers emit router.route_request without a phase; disaggregated vLLM then
    # has two such legs. Each takes the role of the worker whose handle_payload it holds.
    def row(
        stage: str,
        s: float,
        e: float | None,
        process: str,
        role: str,
        mark: bool = False,
    ) -> nrt.Row:
        return nrt.Row(
            stage,
            "nvtx",
            process,
            role,
            mark,
            int(s * MS),
            None if mark or e is None else int(e * MS),
            False,
        )

    rows = [
        row("http-request", 0, 500, "frontend/pid1", "frontend"),
        row("router.route_request", 1, 58, "frontend/pid1", "frontend"),
        row("handle_payload", 1.2, 58, "prefill/pid7", "prefill"),
        row("worker.engine.first_output:prefill", 2, 55, "prefill/pid7", "prefill"),
        row("router.route_request", 58.3, 500, "frontend/pid1", "frontend"),
        row("handle_payload", 58.6, 500, "decode/pid8", "decode"),
        row("worker.engine.first_output:decode", 59.3, 296, "decode/pid8", "decode"),
        row(
            "worker.egress.first_chunk", 298.9, None, "decode/pid8", "decode", mark=True
        ),
        row("frontend.first_token", 299, None, "frontend/pid1", "frontend", mark=True),
    ]
    bd = nrt.build_breakdown(rows)
    assert bd is not None
    names = [i.name for i in bd.items]
    assert any(n.startswith("prefill leg (router.route_request)") for n in names)
    assert any(n.startswith("decode leg (router.route_request)") for n in names)
    assert any(n.startswith("worker.engine.first_output:decode") for n in names)


def _brow(
    stage: str,
    s: float,
    e: float | None = None,
    mark: bool = False,
    role: str = "frontend",
    proc: str = "fe",
) -> nrt.Row:
    return nrt.Row(
        stage=stage,
        source="nvtx",
        process=proc,
        role=role,
        mark=mark,
        start_ns=int(s * MS),
        end_ns=None if e is None else int(e * MS),
        open=False,
    )


def test_breakdown_attributed_time_never_exceeds_ttft_when_legs_overlap() -> None:
    # SGLang-style: the prefill leg keeps streaming after decode is selected and dispatched.
    rows = [
        _brow("http-request", 0, 120),
        _brow("kv_router.select_worker", 2, 9),
        _brow("kv_router.route_request:prefill", 10, 100),
        _brow("kv_router.select_worker", 40, 45),
        _brow("kv_router.route_request:decode", 46, 120),
        _brow("frontend.first_token", 80, mark=True),
    ]
    bd = nrt.build_breakdown(rows)
    assert bd is not None and bd.ttft_ms == pytest.approx(80.0)
    assert bd.attributed_ms + bd.unattributed_ms == pytest.approx(bd.ttft_ms)


def test_breakdown_ties_worker_stages_to_the_leg_that_reached_them() -> None:
    # A migrated aggregated request: attempt 1 on worker A dies, attempt 2 runs on B.
    rows = [
        _brow("http-request", 0, 120),
        _brow("kv_router.route_request:aggregated", 5, 30),
        _brow("handle_payload", 6, 30, role="aggregated", proc="A"),
        _brow("worker.handler.prepare:aggregated", 7, 9, role="aggregated", proc="A"),
        _brow("kv_router.route_request:aggregated", 31, 120),
        _brow("handle_payload", 32, 120, role="aggregated", proc="B"),
        _brow("worker.handler.prepare:aggregated", 33, 35, role="aggregated", proc="B"),
        _brow(
            "worker.engine.first_output:aggregated", 35, 60, role="aggregated", proc="B"
        ),
        _brow("worker.egress.first_chunk", 61, mark=True, role="aggregated", proc="B"),
        _brow("frontend.first_token", 62, mark=True),
    ]
    bd = nrt.build_breakdown(rows)
    assert bd is not None
    legs: list[list[str]] = []
    for item in bd.items:
        if item.depth == 0 and " leg (" in item.name:
            legs.append([])
        elif item.depth == 1 and legs and item.kind != "gap":
            legs[-1].append(item.name)
    assert len(legs) == 2
    assert all("(A)" in name for name in legs[0]) and legs[0]
    assert all("(B)" in name for name in legs[1]) and legs[1]
    # Leg 2 still reports its own transport gap to worker B.
    assert any(
        i.depth == 1
        and i.kind == "gap"
        and i.name.startswith("transport/queue")
        and i.start_ms == pytest.approx(31.0)
        for i in bd.items
    )


def test_pairing_respects_worker_role_under_clock_skew() -> None:
    # Cross-host skew larger than the prefill->decode gap: by start time alone the decode
    # worker's handle_payload is nearest the prefill worker's span.
    def ev(report: str, start: int) -> nrt.NvtxEvent:
        return nrt.NvtxEvent(
            nrt.NvtxMessage("handle_payload", RID, {}),
            report,
            1,
            1,
            False,
            start,
            start + 5,
            False,
            0.0,
            0.0,
        )

    def sp(service: str, start: int) -> nrt.OtelSpan:
        return nrt.OtelSpan(
            TRACE,
            f"{service[:8]:0<16}",
            "",
            "handle_payload",
            service,
            start,
            start + 5,
            {"request_id": RID},
        )

    prefill_ev, decode_ev = ev("prefill.sqlite", 100 * MS), ev(
        "decode.sqlite", 110 * MS
    )
    spans = [sp("dynamo-prefill", 108 * MS), sp("dynamo-decode", 100 * MS)]
    roles = {"prefill.sqlite": "prefill", "decode.sqlite": "decode"}
    pairs = nrt.pair_spans([prefill_ev, decode_ev], spans, lambda e: roles[e.report])
    assert {(e.report, s.service) for e, s in pairs} == {
        ("prefill.sqlite", "dynamo-prefill"),
        ("decode.sqlite", "dynamo-decode"),
    }


def test_xrid_lookup_matches_the_sanitized_nvtx_value(tmp_path: Path) -> None:
    raw = "client id\twith space"
    fe = ReportBuilder(tmp_path / "frontend.sqlite", BASE, 24, "node")
    fe.identity(RID, 0, xrid=nrt.sanitize_nvtx_value(raw))
    path = fe.finish()
    assert nrt.sanitize_nvtx_value(raw) == "client_id_with_space"
    assert (
        nrt.sanitize_nvtx_value("a\x1cb") == "ab"
    ), "Rust drops U+001C (not White_Space)"
    assert len(nrt.sanitize_nvtx_value("é" * 100).encode()) == 128
    assert nrt._identity_matches([path], "xrid", raw) == {(RID, nrt.rid_key(RID))}


# ----------------------------------------------------------------------------
# Contract v2: payload key lookup, identity marks, --summary
# ----------------------------------------------------------------------------


def test_lookup_by_key_ignores_other_requests(reports: dict[str, Path]) -> None:
    other = nrt.scan_report(reports["frontend"], nrt.rid_key(OTHER_RID))
    assert {e.msg.stage for e in other.events} == {
        "http-request",
        "worker.handler.prepare:decode",
    }
    assert all(e.msg.rid == OTHER_RID for e in other.events)
    assert other.rids == {OTHER_RID}
    assert other.identities == {1000: {"trace": "f" * 32, "engine_rid": RID}}
    mine = nrt.scan_report(reports["frontend"], nrt.rid_key(RID))
    assert len(mine.events) == 8
    # An unknown key matches nothing, including the payload-less static range.
    assert nrt.scan_report(reports["frontend"], nrt.rid_key("nobody")).events == []


def test_resolve_by_rid_needs_events_with_that_key(reports: dict[str, Path]) -> None:
    # OTHER_RID has events in the frontend report, so it resolves (key lookup only).
    res, scans = nrt.resolve_key(OTHER_RID, paths(reports), None)
    assert res.rid == OTHER_RID and res.trace_id == "f" * 32
    assert [len(s.events) for s in scans] == [2, 0, 0]


def test_identity_marks_merge_attrs(tmp_path: Path) -> None:
    b = ReportBuilder(tmp_path / "w.sqlite", BASE, 77, "n")
    b.identity(RID, 1, trace=TRACE)
    b.identity(RID, 2, xrid=XRID, trace="ignored-later-duplicate")
    b.identity(RID, 3, engine_rid="e9")
    b.rng("handle_payload", 1, 5)
    b.rng("worker.engine.first_output:decode", 2, 4, tid=2)
    scan = nrt.scan_report(b.finish(), nrt.rid_key(RID))
    merged = {"trace": TRACE, "xrid": XRID, "engine_rid": "e9"}
    assert scan.identities == {77: merged}
    assert [e.msg.attrs for e in scan.events] == [merged, merged]
    rows = nrt.build_rows([scan], [])
    assert all(r.attrs == merged for r in rows)
    res = nrt.Resolution(RID, "rid", RID, "x")
    nrt._fill_ids_from_events(res, [scan])
    assert (res.trace_id, res.xrid) == (TRACE, XRID)


def test_resolve_trace_and_xrid_through_identity_marks(
    reports: dict[str, Path],
) -> None:
    expected = {(RID, nrt.rid_key(RID))}
    assert nrt._identity_matches(paths(reports), "trace", TRACE) == expected
    assert nrt._identity_matches(paths(reports), "xrid", XRID) == expected
    # Exact token match: a prefix or superstring of the id must not resolve.
    assert nrt._identity_matches(paths(reports), "xrid", XRID[:-1]) == set()
    assert nrt._identity_matches(paths(reports), "xrid", XRID + "x") == set()
    res, scans = nrt.resolve_key(TRACE.upper(), paths(reports), None)
    assert (res.kind, res.rid, res.trace_id) == ("trace", RID, TRACE)
    assert sum(len(s.events) for s in scans) == 19
    # A UUID-shaped x-request-id falls through the rid interpretation to identity marks.
    uuid_xrid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    extra = ReportBuilder(reports["frontend"].parent / "x.sqlite", BASE, 5, "n")
    extra.identity(RID, 1, xrid=uuid_xrid)
    extra.rng("http-request", 1, 2)
    res, _ = nrt.resolve_key(uuid_xrid, [extra.finish()], None)
    assert (res.kind, res.rid) == ("xrid", RID)


def test_resolution_uses_identity_rid_when_key_has_no_identity(
    tmp_path: Path,
) -> None:
    b = ReportBuilder(tmp_path / "w.sqlite", BASE, 77, "n")
    b.rng("handle_payload", 1, 5)  # no identity mark was captured
    scan = nrt.scan_report(b.finish(), nrt.rid_key(RID))
    assert scan.rids == set() and len(scan.events) == 1
    res, _ = nrt.resolve_key(RID, [b.path], None)
    assert res.rid == RID and res.trace_id is None


def test_summary_math(tmp_path: Path) -> None:
    a = ReportBuilder(tmp_path / "a.sqlite", BASE, 1, "n")
    b = ReportBuilder(tmp_path / "b.sqlite", BASE + 1_000 * MS, 2, "n")
    durations = [float(d) for d in range(1, 101)]  # 1..100 ms, split over two reports
    for i, d in enumerate(durations):
        target = a if i % 2 == 0 else b
        target.rng("http-request", 10, 10 + d, rid=f"req-{i}")
    a.rng("kv_router.schedule", 5, 7, rid="req-0")
    a.rng("kv_router.schedule", 5, 9, rid="req-1")
    a.rng("kv_router.schedule", 5, 6, rid="req-2")
    a.mark("frontend.first_token", 50, rid="req-0")  # marks are not aggregated
    a.identity("req-0", 1, trace=TRACE)  # neither are identity marks
    a.rng("detokenize", 70, 71, rid=None)  # no payload: static range
    a.rng("response.streaming.decode", 12, None, rid="req-0")  # never closed
    summary = nrt.build_summary([a.finish(), b.finish()])
    stages = {s["stage"]: s for s in summary["stages"]}
    assert list(stages) == ["http-request", "kv_router.schedule"]
    http = stages["http-request"]
    assert http["count"] == 100
    assert http["mean_ms"] == pytest.approx(50.5)
    assert http["median_ms"] == pytest.approx(50.5)
    assert http["p99_ms"] == pytest.approx(
        99.0
    )  # nearest rank: ceil(0.99 * 100) = 99th
    assert http["max_ms"] == pytest.approx(100.0)
    sched = stages["kv_router.schedule"]
    assert sched["count"] == 3
    assert sched["mean_ms"] == pytest.approx(7 / 3)
    assert (sched["median_ms"], sched["p99_ms"], sched["max_ms"]) == (2.0, 4.0, 4.0)
    assert summary["open_ranges_skipped"] == 1
    text = nrt.render_summary(summary)
    assert "http-request" in text and "p99 ms" in text and "never closed" in text


def test_cli_summary(
    reports: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    args = nrt.parse_args([*map(str, paths(reports)), "--summary", "--json"])
    assert nrt.run(args) == 0
    doc = json.loads(capsys.readouterr().out)
    stages = {s["stage"]: s for s in doc["stages"]}
    # Two requests' http-request ranges (RID 300 ms, OTHER_RID 10 ms) in the frontend.
    assert stages["http-request"]["count"] == 2
    assert stages["http-request"]["max_ms"] == pytest.approx(300.0)
    assert stages["http-request"]["mean_ms"] == pytest.approx(155.0)
    assert "detokenize" not in stages and "frontend.first_token" not in stages
    assert doc["open_ranges_skipped"] == 1
    assert nrt.run(nrt.parse_args([*map(str, paths(reports)), "--summary"])) == 0
    assert "== Per-stage summary (3 report(s)) ==" in capsys.readouterr().out


def test_request_is_required_without_summary(reports: dict[str, Path]) -> None:
    with pytest.raises(SystemExit):
        nrt.parse_args([str(reports["frontend"])])
