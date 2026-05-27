#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Render a two-column CD timeline as a self-contained HTML file.

Reads kvbm_audit tracing events from hub.log, prefill.log, decode.log
in an experiment directory and emits trace.html alongside them. If the
current Dynamo/vLLM path does not emit kvbm_audit, falls back to a small
set of operational breadcrumbs from the same logs: request receive/complete,
endpoint registration, side-channel setup, cache stats, and KV transfer
metrics. The HTML shows three lanes (Decode | Hub | Prefill) ordered by
timestamp. Events that share a request_id are linked via a sidebar that
filters to that request_id.

Usage:
    kvbm-xdc-trace.py <experiment-dir>
    kvbm-xdc-trace.py /tmp/kvbm-xdc/<run-label>
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ANSI colour escape sequences emitted by `tracing_subscriber`'s
# default formatter when stdout is a tty (or in our case piped to a
# file but inheriting the terminal mode).
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# A kvbm_audit log line looks like (after ANSI stripping):
#
#   2026-04-30T08:46:09.231194Z  INFO kvbm_audit: event="gnmt_entry" role="prefill" request_id="cmpl-..." num_computed_tokens=0
#
# `tracing` adds nesting indicators (`get_num_new_matched_tokens:`)
# between the level and the target, and a closing `[role=Decode]`
# style suffix. We accept both shapes and key off the literal
# `kvbm_audit:` token to identify audit lines.
TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+"
    r"\w+\s+"  # log level
    r".*?kvbm_audit:\s*(?P<rest>.*)$"
)
FALLBACK_AUDIT_RE = re.compile(r".*?kvbm_audit:\s*(?P<rest>.*)$")

# Field tokenizer: matches `key="quoted value"` or `key=bareword`.
FIELD_RE = re.compile(r"(?P<key>\w+)=(?:\"(?P<qv>(?:[^\"\\]|\\.)*)\"|(?P<bv>\S+))")
KV_TRANSFER_METRICS_RE = re.compile(
    r"Num successful transfers=(?P<successful_transfers>\d+),\s*"
    r"Avg xfer time \(ms\)=(?P<avg_xfer_ms>[\d.]+),\s*"
    r"P90 xfer time \(ms\)=(?P<p90_xfer_ms>[\d.]+),\s*"
    r"Avg post time \(ms\)=(?P<avg_post_ms>[\d.]+),\s*"
    r"P90 post time \(ms\)=(?P<p90_post_ms>[\d.]+),\s*"
    r"Avg MB per transfer=(?P<avg_mb_per_transfer>[\d.]+),\s*"
    r"Throughput \(MB/s\)=(?P<throughput_mb_s>[\d.]+),\s*"
    r"Avg number of descriptors=(?P<avg_descriptors>[\d.]+)"
)
CACHE_STATS_RE = re.compile(
    r"Prefix cache hit rate:\s*(?P<prefix_cache_hit_rate_pct>[\d.]+)%,\s*"
    r"External prefix cache hit rate:\s*(?P<external_prefix_cache_hit_rate_pct>[\d.]+)%"
)
RAW_TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+" r"\w+\s+(?P<rest>.*)$"
)
GLOG_TS_RE = re.compile(
    r"^(?P<level>[IWEF])(?P<mmdd>\d{4})\s+"
    r"(?P<clock>\d{2}:\d{2}:\d{2}\.\d+)\s+\d+\s+(?P<rest>.*)$"
)
URL_RE = re.compile(r"\b(?:https?|nats)://[^\s\"'<>]+")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b")
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,}[0-9a-fA-F:]{1,}(?:%\w+)?(?::\d+)?\b")
ABS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"/(?:Users|home|tmp|private|workspace|mnt|scratch|var|opt)"
    r"/[^\s\"'<>:,;)]*"
)
INTERNAL_HOST_RE = re.compile(
    r"\b(?=[A-Za-z0-9.-]*[.-])(?=[A-Za-z0-9.-]*\d)"
    r"[A-Za-z0-9.-]*(?:prod|cluster|login|compute|gpu|dgx|h100|a100)"
    r"[A-Za-z0-9.-]*(?::\d+)?\b",
    re.IGNORECASE,
)


def scrub_value(value: str) -> str:
    value = URL_RE.sub("<url>", value)
    value = IPV4_RE.sub("<addr>", value)
    value = IPV6_RE.sub("<addr>", value)
    value = ABS_PATH_RE.sub("<path>", value)
    value = INTERNAL_HOST_RE.sub("<host>", value)
    return value


@dataclass
class Event:
    ts: str
    source: str  # "decode" | "hub" | "prefill"
    fields: dict = field(default_factory=dict)
    line_no: int = 0
    sort_ns: int | None = None

    @property
    def event(self) -> str:
        return self.fields.get("event", "?")

    @property
    def request_id(self) -> str | None:
        return self.fields.get("request_id")

    @property
    def role(self) -> str:
        # role may be missing on hub events; fall back to source
        return self.fields.get("role", self.source)


def parse_log(path: Path, source: str) -> list[Event]:
    if not path.exists():
        return []
    audit_events: list[Event] = []
    fallback_events: list[Event] = []
    for line_no, line in enumerate(
        path.read_text(errors="replace").splitlines(), start=1
    ):
        line = ANSI_RE.sub("", line)
        m = TS_RE.match(line)
        if m:
            ts = m.group("ts")
            rest = m.group("rest")
            fields = parse_fields(rest)
            audit_events.append(
                Event(
                    ts=ts,
                    source=source,
                    fields=fields,
                    line_no=line_no,
                    sort_ns=audit_ts_ns(fields),
                )
            )
            continue

        m = FALLBACK_AUDIT_RE.match(line)
        if m:
            fields = parse_fields(m.group("rest"))
            audit_events.append(
                Event(
                    ts="",
                    source=source,
                    fields=fields,
                    line_no=line_no,
                    sort_ns=audit_ts_ns(fields),
                )
            )
            continue

        fallback = parse_fallback_line(line, source)
        if fallback is not None:
            fallback.line_no = line_no
            fallback_events.append(fallback)
    return audit_events if audit_events else fallback_events


def parse_fields(rest: str) -> dict:
    fields = {}
    for fm in FIELD_RE.finditer(rest):
        v = fm.group("qv") if fm.group("qv") is not None else fm.group("bv")
        # Unescape quoted strings (basic)
        if fm.group("qv") is not None:
            v = v.replace('\\"', '"').replace("\\\\", "\\")
        else:
            v = v.rstrip(",")
        fields[fm.group("key")] = scrub_value(v)
    return fields


def parse_ts_ns(ts: str) -> int | None:
    if not ts or "T" not in ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except ValueError:
        return None


def audit_ts_ns(fields: dict) -> int | None:
    try:
        return int(fields.get("audit_ts_ns", ""))
    except ValueError:
        return None


SOURCE_TO_LANE = {"decode": 0, "hub": 1, "prefill": 2}


def event_sort_key(event: Event) -> tuple[int, int, int]:
    sort_ns = event.sort_ns
    if sort_ns is None:
        sort_ns = parse_ts_ns(event.ts)
    if sort_ns is None:
        sort_ns = event.line_no
    return (sort_ns, SOURCE_TO_LANE.get(event.source, 1), event.line_no)


def event_display_ts(event: Event) -> str:
    if event.sort_ns is not None:
        dt = datetime.fromtimestamp(event.sort_ns / 1_000_000_000, timezone.utc)
        return dt.strftime("%H:%M:%S.%f")[:15]
    if "T" in event.ts:
        return event.ts.split("T", 1)[-1].rstrip("Z")[:15]
    return f"line {event.line_no}"


def parse_fallback_line(line: str, source: str) -> Event | None:
    m = RAW_TS_RE.match(line)
    if m:
        ts = m.group("ts")
        rest = m.group("rest")
        level = None
    else:
        gm = GLOG_TS_RE.match(line)
        if not gm:
            return None
        mmdd = gm.group("mmdd")
        ts = (
            f"{datetime.now(timezone.utc).year}-"
            f"{mmdd[:2]}-{mmdd[2:]}T{gm.group('clock')}Z"
        )
        rest = gm.group("rest")
        level = gm.group("level")
    fields = parse_fields(rest)
    role = fields.get("component", source)
    event: str | None = None

    if "request received" in rest:
        event = "request_received"
    elif "request completed" in rest:
        event = "request_completed"
    elif "KV Transfer metrics:" in rest:
        event = "kv_transfer_metrics"
        if metrics := KV_TRANSFER_METRICS_RE.search(rest):
            fields.update(metrics.groupdict())
    elif "External prefix cache hit rate:" in rest:
        event = "cache_stats"
        if stats := CACHE_STATS_RE.search(rest):
            fields.update(stats.groupdict())
    elif "Using existing VLLM_NIXL_SIDE_CHANNEL_HOST" in rest:
        event = "side_channel_host"
    elif "Registered endpoint" in rest and ".generate" in rest:
        event = "generate_endpoint_registered"
    elif "VllmWorker for" in rest and "has been initialized" in rest:
        event = "worker_initialized"
    elif "registerMem: registration failed" in rest:
        event = "nixl_register_mem_failed"
    elif "createXferReq: no potential backend found" in rest:
        event = "nixl_create_xfer_req_no_backend"
    elif "Recovered from KV load failure" in rest:
        event = "kv_load_recovered_from_failure"
    elif "No POSIX plugin found" in rest or "No UCX plugin found" in rest:
        event = "nixl_plugin_missing"

    if event is None:
        return None

    fields["event"] = event
    fields.setdefault("role", role)
    if level is not None:
        fields["level"] = level
    fields["message"] = scrub_value(rest)
    return Event(ts=ts, source=source, fields=fields)


def render_html(events: list[Event], out_path: Path, exp_label: str) -> None:
    # Sort stably by timestamp.
    events.sort(key=event_sort_key)

    # Distinct request_ids for the sidebar.
    request_ids = sorted({e.request_id for e in events if e.request_id})

    rows_html: list[str] = []
    for e in events:
        rid = e.request_id or ""
        role = e.role
        # Lane: decode | hub | prefill
        if role == "decode":
            lane = 0
        elif role == "prefill":
            lane = 2
        elif e.source == "hub":
            lane = 1
        else:
            # role="both" (e.g. process_finished_onboarding events
            # emitted by ConnectorLeader, agnostic of role) - drop
            # them in the same lane as the source file.
            lane = {"decode": 0, "hub": 1, "prefill": 2}.get(e.source, 1)

        cells = ["", "", ""]
        # Render fields excluding event/role/request_id (shown elsewhere).
        body_fields = {
            k: v
            for k, v in e.fields.items()
            if k not in {"event", "role", "request_id"}
        }
        body = f"<div class='ev'>{html.escape(e.event)}</div>" + "".join(
            f"<div class='kv'><span class='k'>{html.escape(k)}</span>"
            f"<span class='v'>{html.escape(scrub_value(v))}</span></div>"
            for k, v in body_fields.items()
        )
        ts_short = event_display_ts(e)
        cells[lane] = (
            f"<div class='cell' data-rid='{html.escape(rid)}'>"
            f"<div class='ts'>{html.escape(ts_short)}</div>"
            f"{body}"
            f"<div class='rid'>{html.escape(rid)}</div>"
            f"</div>"
        )
        rid_attr = html.escape(rid) if rid else ""
        rows_html.append(
            f"<tr data-rid='{rid_attr}'>"
            f"<td class='lane decode'>{cells[0]}</td>"
            f"<td class='lane hub'>{cells[1]}</td>"
            f"<td class='lane prefill'>{cells[2]}</td>"
            f"</tr>"
        )

    sidebar = "".join(
        f"<button class='rid-btn' data-rid='{html.escape(r)}'>{html.escape(r)}</button>"
        for r in request_ids
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kvbm-xdc-trace - {html.escape(exp_label)}</title>
<style>
  body {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
          margin: 0; background: #0d1117; color: #e6edf3; }}
  header {{ padding: 8px 16px; background: #161b22; border-bottom: 1px solid #30363d;
            display: flex; align-items: center; gap: 16px; }}
  header h1 {{ font-size: 14px; margin: 0; }}
  header .stats {{ color: #8b949e; }}
  .layout {{ display: flex; height: calc(100vh - 41px); }}
  aside {{ width: 280px; padding: 8px; overflow-y: auto; border-right: 1px solid #30363d;
           background: #0d1117; }}
  aside h2 {{ font-size: 11px; color: #8b949e; text-transform: uppercase;
              margin: 8px 0 4px; }}
  .rid-btn {{ display: block; width: 100%; text-align: left; background: transparent;
              border: 1px solid #30363d; color: #e6edf3; padding: 4px 6px;
              margin-bottom: 4px; cursor: pointer; border-radius: 4px;
              font-family: inherit; font-size: 11px; word-break: break-all; }}
  .rid-btn.active {{ background: #1f6feb; border-color: #1f6feb; }}
  .rid-btn:hover {{ background: #21262d; }}
  main {{ flex: 1; overflow-y: auto; padding: 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ position: sticky; top: 0; background: #161b22; padding: 6px 8px;
        border-bottom: 1px solid #30363d; text-align: left; }}
  th.decode {{ color: #58a6ff; }}
  th.hub {{ color: #d2a8ff; }}
  th.prefill {{ color: #f0883e; }}
  td {{ vertical-align: top; padding: 0 8px; border-bottom: 1px solid #161b22;
        width: 33.33%; }}
  td.lane {{ height: 1px; }}
  .cell {{ background: #161b22; padding: 4px 6px; margin: 4px 0; border-radius: 4px;
           border-left: 3px solid #30363d; }}
  td.decode .cell {{ border-left-color: #58a6ff; }}
  td.hub .cell {{ border-left-color: #d2a8ff; }}
  td.prefill .cell {{ border-left-color: #f0883e; }}
  .ts {{ color: #8b949e; font-size: 10px; }}
  .ev {{ font-weight: 600; color: #e6edf3; }}
  .kv {{ font-size: 11px; color: #c9d1d9; }}
  .kv .k {{ color: #7ee787; margin-right: 4px; }}
  .kv .v {{ color: #d2a8ff; word-break: break-all; }}
  .rid {{ color: #6e7681; font-size: 10px; margin-top: 4px; }}
  tr.dimmed {{ opacity: 0.15; }}
</style>
</head>
<body>
  <header>
    <h1>kvbm-xdc-trace - {html.escape(exp_label)}</h1>
    <span class='stats'>{len(events)} trace events - {len(request_ids)} request_ids</span>
  </header>
  <div class='layout'>
    <aside>
      <h2>Filter by request_id</h2>
      <button class='rid-btn active' data-rid=''>(all)</button>
      {sidebar}
    </aside>
    <main>
      <table>
        <thead>
          <tr>
            <th class='decode'>decode</th>
            <th class='hub'>hub</th>
            <th class='prefill'>prefill</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </main>
  </div>
<script>
  const buttons = document.querySelectorAll('.rid-btn');
  buttons.forEach(b => b.addEventListener('click', () => {{
    const target = b.dataset.rid;
    buttons.forEach(x => x.classList.toggle('active', x === b));
    document.querySelectorAll('tbody tr').forEach(tr => {{
      const rid = tr.dataset.rid || '';
      tr.classList.toggle('dimmed', !!target && rid !== target);
    }});
  }}));
</script>
</body>
</html>
"""
    out_path.write_text(page)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    exp_dir = Path(sys.argv[1]).resolve()
    if not exp_dir.is_dir():
        print(f"not a directory: {exp_dir}", file=sys.stderr)
        return 2

    events: list[Event] = []
    for source in ("decode", "hub", "prefill"):
        events.extend(parse_log(exp_dir / f"{source}.log", source))

    out = exp_dir / "trace.html"
    render_html(events, out, exp_dir.name)
    print(f"{len(events)} trace events -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
