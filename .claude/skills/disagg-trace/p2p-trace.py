#!/usr/bin/env python3
"""Render a three-column P2P audit timeline as a self-contained HTML file.

Reads kvbm_audit tracing events from instance_a.log, hub.log, instance_b.log
in an experiment directory and emits trace.html alongside them. The HTML
shows three lanes (Instance A | Hub | Instance B) ordered by timestamp,
with one row per audit event. Events that share a request_id are linked
via a sidebar that filters to that request_id.

Forked from cd-trace.py — the only structural difference is the lane
labels and the source files. P2P has two same-role instances (no
prefill/decode asymmetry), so lane assignment is by source FILE rather
than by the `role` field.

Usage:
    p2p-trace.py <experiment-dir>
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

TS_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+"
    r"\w+\s+"
    r".*?kvbm_audit:\s*(?P<rest>.*)$"
)

FIELD_RE = re.compile(r"(?P<key>\w+)=(?:\"(?P<qv>(?:[^\"\\]|\\.)*)\"|(?P<bv>\S+))")


@dataclass
class Event:
    ts: str
    source: str  # "instance_a" | "hub" | "instance_b"
    fields: dict = field(default_factory=dict)

    @property
    def event(self) -> str:
        return self.fields.get("event", "?")

    @property
    def request_id(self) -> str | None:
        return self.fields.get("request_id")


def parse_log(path: Path, source: str) -> list[Event]:
    if not path.exists():
        return []
    events: list[Event] = []
    for line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", line)
        m = TS_RE.match(line)
        if not m:
            continue
        ts = m.group("ts")
        rest = m.group("rest")
        fields = {}
        for fm in FIELD_RE.finditer(rest):
            v = fm.group("qv") if fm.group("qv") is not None else fm.group("bv")
            if fm.group("qv") is not None:
                v = v.replace('\\"', '"').replace("\\\\", "\\")
            fields[fm.group("key")] = v
        events.append(Event(ts=ts, source=source, fields=fields))
    return events


SOURCE_TO_LANE = {"instance_a": 0, "hub": 1, "instance_b": 2}
LANE_LABELS = ("instance A", "hub", "instance B")


def render_html(events: list[Event], out_path: Path, exp_label: str) -> None:
    events.sort(key=lambda e: e.ts)
    request_ids = sorted({e.request_id for e in events if e.request_id})

    rows_html: list[str] = []
    for e in events:
        rid = e.request_id or ""
        lane = SOURCE_TO_LANE.get(e.source, 1)

        cells = ["", "", ""]
        body_fields = {
            k: v
            for k, v in e.fields.items()
            if k not in {"event", "role", "request_id"}
        }
        body = f"<div class='ev'>{html.escape(e.event)}</div>" + "".join(
            f"<div class='kv'><span class='k'>{html.escape(k)}</span>"
            f"<span class='v'>{html.escape(v)}</span></div>"
            for k, v in body_fields.items()
        )
        ts_short = e.ts.split("T", 1)[-1].rstrip("Z")[:15]
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
            f"<td class='lane a'>{cells[0]}</td>"
            f"<td class='lane hub'>{cells[1]}</td>"
            f"<td class='lane b'>{cells[2]}</td>"
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
<title>p2p-trace — {html.escape(exp_label)}</title>
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
  th.a {{ color: #58a6ff; }}
  th.hub {{ color: #d2a8ff; }}
  th.b {{ color: #f0883e; }}
  td {{ vertical-align: top; padding: 0 8px; border-bottom: 1px solid #161b22;
        width: 33.33%; }}
  td.lane {{ height: 1px; }}
  .cell {{ background: #161b22; padding: 4px 6px; margin: 4px 0; border-radius: 4px;
           border-left: 3px solid #30363d; }}
  td.a .cell {{ border-left-color: #58a6ff; }}
  td.hub .cell {{ border-left-color: #d2a8ff; }}
  td.b .cell {{ border-left-color: #f0883e; }}
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
    <h1>p2p-trace · {html.escape(exp_label)}</h1>
    <span class='stats'>{len(events)} events · {len(request_ids)} request_ids</span>
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
            <th class='a'>{LANE_LABELS[0]}</th>
            <th class='hub'>{LANE_LABELS[1]}</th>
            <th class='b'>{LANE_LABELS[2]}</th>
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
    for source in ("instance_a", "hub", "instance_b"):
        events.extend(parse_log(exp_dir / f"{source}.log", source))

    out = exp_dir / "trace.html"
    render_html(events, out, exp_dir.name)
    print(f"{len(events)} audit events → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
