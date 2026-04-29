# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified HTML report generator.

Combines data from two sources into one self-contained report:

  **From merge_result.json (Rust merger):**
  1. KPI headline cards (p99/p50 TTFT, bottleneck, request count, components)
  2. p99 critical-path stacked bar
  3. Top-10 slowest requests with Perfetto deep-links
  4. View A: Component utilization heat-strip (Plotly heatmap)
  5. View B: Per-shard GPU strips with TP imbalance
  6. View C: Perfetto deep-link with pre-pinned tracks
  7. View D: Causality DAG (Plotly scatter graph)

  **From Python analyzers (optional, gracefully degraded):**
  8. Per-stage latency distribution chart
  9. Kernel hotlist table with TP CV% and stage attribution
  10. GPU utilization per device with idle gap analysis
  11. Communication breakdown (NCCL/NiXL/NATS)
  12. Baseline comparison/diff
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

log = logging.getLogger("report_gen")


def _esc(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log.warning("Could not load %s: %s", path, e)
        return None


# -- Color palette (Tatva-inspired, dark theme) --------------------------------

BG_PRIMARY = "#141414"
BG_SECONDARY = "#1a1a1a"
BG_SURFACE = "#222222"
BORDER = "#333333"
TEXT_PRIMARY = "#f5f5f5"
TEXT_SECONDARY = "#999999"
TEXT_TERTIARY = "#666666"

STAGE_COLORS = [
    "#516CDC", "#76B900", "#F0783C", "#C84673", "#5F9637",
    "#3842B4", "#D25A1E", "#9D2055", "#06b6d4", "#a855f7",
    "#f59e0b", "#ef4444",
]


def _stage_color(idx: int) -> str:
    return STAGE_COLORS[idx % len(STAGE_COLORS)]


# -- Main entry point ----------------------------------------------------------

def generate_report(
    merge_result: Optional[dict] = None,
    stage_attr: Optional[dict] = None,
    gpu_util: Optional[dict] = None,
    kernels: Optional[dict] = None,
    comm: Optional[dict] = None,
    trace_url: Optional[str] = None,
    trace_files: Optional[list] = None,
    title: str = "sysprofile report",
) -> str:
    mr = merge_result or {}
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_id = mr.get("run_id", "unknown")
    capture_ms = mr.get("capture_duration_ms", 0)

    parts = [
        _header(run_id, capture_ms, title, now),
    ]

    if mr:
        parts.append(_section_headline(mr))
        parts.append(_section_top10(mr, trace_url))
        parts.append(_section_view_a(mr))
        parts.append(_section_view_b(mr))
        parts.append(_section_view_c(mr, trace_url, trace_files))
        parts.append(_section_view_d(mr))

    if stage_attr:
        parts.append(_section_stage_latency(stage_attr))
    if kernels:
        parts.append(_section_kernel_hotlist(kernels))
    if gpu_util:
        parts.append(_section_gpu_util(gpu_util))
    if comm:
        parts.append(_section_comm_breakdown(comm))

    parts.append(_footer(mr))
    return "".join(parts)


# -- Header & CSS --------------------------------------------------------------

def _header(run_id: str, duration_ms: float, title: str, now: str) -> str:
    dur_s = duration_ms / 1000.0
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}: {_esc(run_id)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: {BG_PRIMARY}; --bg2: {BG_SECONDARY}; --bg3: {BG_SURFACE};
  --border: {BORDER}; --text: {TEXT_PRIMARY}; --text2: {TEXT_SECONDARY}; --text3: {TEXT_TERTIARY};
  --green: #76B900; --indigo: #516CDC; --orange: #F0783C; --red: #B81514;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--text); font-family: "Matter", -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif; font-size: 14px; line-height: 1.6; }}
.mono {{ font-family: "Matter Mono", "JetBrains Mono", "SF Mono", "Cascadia Code", monospace; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
.header {{ display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 32px; padding-bottom: 16px; border-bottom: 1px solid var(--border); }}
.header h1 {{ font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }}
.header .meta {{ font-size: 12px; color: var(--text3); }}
.section {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 24px; overflow: hidden; }}
.section-h {{ padding: 16px 20px; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text2); border-bottom: 1px solid var(--border); }}
.section-body {{ padding: 20px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 10px 14px; color: var(--text3); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); }}
td {{ padding: 10px 14px; border-bottom: 1px solid rgba(51,51,51,0.5); }}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: var(--bg3); }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.kpi {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
.kpi-label {{ font-size: 11px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }}
.kpi-value {{ font-size: 28px; font-weight: 600; font-family: "Matter Mono", monospace; letter-spacing: -0.02em; }}
.kpi-unit {{ font-size: 14px; color: var(--text3); }}
.kpi-hint {{ font-size: 11px; color: var(--text2); margin-top: 6px; }}
.ok {{ color: var(--green); }}
.warn {{ color: var(--orange); }}
.bad {{ color: var(--red); }}
.perf-link {{ display: inline-flex; align-items: center; gap: 4px; font-size: 11px; padding: 4px 10px; background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; color: var(--indigo); text-decoration: none; transition: background 0.15s; }}
.perf-link:hover {{ background: rgba(81,108,220,0.1); }}
.cp-bar {{ display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin: 12px 0; }}
.cp-seg {{ display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 500; color: white; min-width: 3px; transition: opacity 0.15s; cursor: default; }}
.cp-seg:hover {{ opacity: 0.85; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }}
.warning-banner {{ background: rgba(240,120,60,0.1); border: 1px solid rgba(240,120,60,0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; font-size: 12px; color: var(--orange); }}
.plotly-chart {{ width: 100%; }}
.bar-outer {{ background: #2a2a2a; border-radius: 4px; height: 16px; width: 100%; }}
.bar-inner {{ height: 100%; border-radius: 4px; min-width: 2px; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
.tag-warn {{ background: rgba(240,120,60,0.15); color: var(--orange); }}
.tag-ok {{ background: rgba(118,185,0,0.15); color: var(--green); }}
.tag-bad {{ background: rgba(184,21,20,0.15); color: var(--red); }}
.chart-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
.chart-label {{ width: 200px; font-size: 12px; color: var(--text2); text-align: right; flex-shrink: 0; }}
.chart-bars {{ flex: 1; display: flex; gap: 2px; align-items: center; }}
.chart-bar {{ height: 18px; border-radius: 3px; min-width: 2px; }}
.chart-val {{ font-size: 10px; color: var(--text3); margin-left: 4px; white-space: nowrap; }}
.footer {{ text-align: center; padding: 32px; font-size: 11px; color: var(--text3); border-top: 1px solid var(--border); margin-top: 16px; }}
.divider {{ border-top: 2px solid var(--border); margin: 32px 0 24px 0; }}
.divider-label {{ font-size: 12px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>{_esc(title)}</h1>
  <div class="meta">Run {_esc(run_id)} &middot; {dur_s:.1f}s capture &middot; {now}</div>
</div>
"""


# -- Section 1: KPI headline + p99 critical-path bar --------------------------

def _kpi_card(label: str, value: str, unit: str, hint: str, cls: str = "") -> str:
    color_class = f" {cls}" if cls else ""
    return f"""<div class="kpi">
  <div class="kpi-label">{_esc(label)}</div>
  <div class="kpi-value{color_class}">{_esc(value)}<span class="kpi-unit">{_esc(unit)}</span></div>
  <div class="kpi-hint">{_esc(hint)}</div>
</div>
"""


def _section_headline(mr: dict) -> str:
    s = '<div class="kpis">\n'
    p99 = mr.get("p99_total_ms", 0)
    p50 = mr.get("p50_total_ms", 0)
    s += _kpi_card("p99 TTFT", f"{p99:.1f}", "ms", "End-to-end critical path",
                   "warn" if p99 > 500 else "ok")
    s += _kpi_card("p50 TTFT", f"{p50:.1f}", "ms", "Median critical path", "ok")
    s += _kpi_card("Requests", str(mr.get("total_requests", 0)), "",
                   "Captured with traceparent", "")

    components = mr.get("components", [])
    comp_names = ", ".join(c.get("name", "") for c in components)
    s += _kpi_card("Components", str(len(components)), "", comp_names, "")

    cp = mr.get("p99_critical_path", [])
    if cp:
        bn = max(cp, key=lambda x: x.get("fraction", 0))
        s += _kpi_card("Bottleneck",
                       bn.get("stage", "").replace("dynamo.", ""), "",
                       f"{bn.get('fraction', 0) * 100:.1f}% of p99 critical path", "warn")
    s += "</div>\n"

    clock = mr.get("clock_alignment", {})
    if clock.get("max_residual_ns", 0) > 100_000:
        s += f'<div class="warning-banner">Clock alignment residual: {clock["max_residual_ns"] / 1000:.0f}&micro;s. Cross-host event ordering may be unreliable for events shorter than this.</div>\n'

    if cp:
        s += '<div class="section"><div class="section-h">p99 Critical-Path Attribution</div><div class="section-body">\n'
        s += f'<div style="font-size:12px;color:var(--text2);margin-bottom:8px">Total p99 = {mr.get("p99_total_ms", 0):.1f}ms across {len(cp)} stages ({mr.get("total_requests", 0)} requests sampled)</div>\n'
        s += '<div class="cp-bar">\n'
        for i, stage in enumerate(cp):
            pct = stage.get("fraction", 0) * 100
            color = _stage_color(i)
            name = stage.get("stage", "").replace("dynamo.", "")
            dur = stage.get("duration_ms", 0)
            label = ""
            if pct >= 8:
                label = f"{name} {pct:.0f}%"
            elif pct >= 3:
                label = f"{pct:.0f}%"
            s += f'  <div class="cp-seg" style="width:{pct:.1f}%;background:{color}" title="{_esc(stage.get("stage", ""))}: {pct:.1f}% ({dur:.1f}ms)">{_esc(label)}</div>\n'
        s += '</div>\n<div class="legend">\n'
        for i, stage in enumerate(cp):
            name = stage.get("stage", "").replace("dynamo.", "")
            pct = stage.get("fraction", 0) * 100
            dur = stage.get("duration_ms", 0)
            s += f'  <div class="legend-item"><div class="legend-dot" style="background:{_stage_color(i)}"></div>{_esc(name)} ({pct:.1f}%, {dur:.2f}ms)</div>\n'
        s += '</div>\n</div></div>\n'

    return s


# -- Section 2: Top-10 slowest requests ---------------------------------------

def _section_top10(mr: dict, trace_url: Optional[str]) -> str:
    reqs = mr.get("top_slow_requests", [])
    if not reqs:
        return ""
    s = '<div class="section"><div class="section-h">Top-10 Slowest Requests</div><div class="section-body">\n'
    s += '<table><thead><tr><th>#</th><th>Trace ID</th><th>Duration</th><th>Dominant Stage</th><th>Stages</th><th></th></tr></thead><tbody>\n'

    for i, req in enumerate(reqs):
        tp = req.get("traceparent", "")
        parts = tp.split("-")
        trace_id = parts[1][:12] + "..." if len(parts) > 1 and len(parts[1]) > 12 else tp
        stages = req.get("stages", [])
        dominant = ""
        if stages:
            bn = max(stages, key=lambda x: x.get("fraction", 0))
            dominant = f"{bn.get('stage', '').replace('dynamo.', '')} ({bn.get('fraction', 0) * 100:.0f}%)"
        mini_bar = _build_mini_bar(stages)
        perfetto = ""
        if trace_url:
            perfetto = f'<a href="https://ui.perfetto.dev/#!/?url={_esc(trace_url)}" class="perf-link" target="_blank">Open in Perfetto</a>'
        s += f'<tr><td class="mono">{i + 1}</td><td class="mono" style="color:var(--text)">{_esc(trace_id)}</td><td class="mono">{req.get("total_duration_ms", 0):.1f}ms</td><td style="font-size:12px;color:var(--text2)">{_esc(dominant)}</td><td style="width:200px">{mini_bar}</td><td>{perfetto}</td></tr>\n'

    s += '</tbody></table>\n</div></div>\n'
    return s


def _build_mini_bar(stages: list) -> str:
    bar = '<div style="display:flex;height:16px;border-radius:3px;overflow:hidden">'
    for i, stage in enumerate(stages):
        pct = stage.get("fraction", 0) * 100
        name = stage.get("stage", "").replace("dynamo.", "")
        bar += f'<div style="width:{pct:.1f}%;background:{_stage_color(i)}" title="{_esc(name)}: {pct:.1f}%"></div>'
    bar += '</div>'
    return bar


# -- Section 3: View A — Component utilization heat-strip ---------------------

def _section_view_a(mr: dict) -> str:
    util = mr.get("component_utilization", [])
    s = '<div class="section"><div class="section-h">View A &mdash; Component Utilization Heat-Strip</div><div class="section-body">\n'

    if not util:
        s += '<p style="color:var(--text3)">No utilization data available.</p>\n</div></div>\n'
        return s

    comp_labels = [f"{c['component']} @ {c['host']}" for c in util]
    x_labels = [f"{b['start_ms']:.0f}" for b in util[0].get("bins", [])] if util else []
    z_data = [[b.get("utilization", 0) for b in c.get("bins", [])] for c in util]

    height = 60 + len(comp_labels) * 30
    s += f'''<div id="heatmap-a" class="plotly-chart" style="height:{height}px"></div>
<script>
Plotly.newPlot("heatmap-a", [{{
  z: {json.dumps(z_data)},
  x: {json.dumps(x_labels)},
  y: {json.dumps(comp_labels)},
  type: "heatmap",
  colorscale: [[0,"{BG_SECONDARY}"],[0.3,"#1a3a00"],[0.7,"#4a7a00"],[1.0,"#76B900"]],
  showscale: true,
  colorbar: {{ title: "Utilization", titleside: "right", tickformat: ".0%", len: 0.8 }},
  hovertemplate: "%{{y}}<br>t=%{{x}}ms<br>util=%{{z:.1%}}<extra></extra>"
}}], {{
  paper_bgcolor: "{BG_PRIMARY}",
  plot_bgcolor: "{BG_SECONDARY}",
  font: {{ color: "{TEXT_PRIMARY}", family: "Matter, system-ui, sans-serif", size: 11 }},
  margin: {{ l: 200, r: 80, t: 20, b: 50 }},
  xaxis: {{ title: "Time (ms from capture start)", gridcolor: "{BORDER}" }},
  yaxis: {{ autorange: "reversed", gridcolor: "{BORDER}" }}
}}, {{ responsive: true }});
</script>
'''

    s += '<table style="margin-top:16px"><thead><tr><th>Component</th><th>Host</th><th>Utilization</th><th></th></tr></thead><tbody>\n'
    for c in util:
        ou = c.get("overall_utilization", 0)
        pct = ou * 100
        color = "var(--green)" if ou > 0.9 else "var(--orange)" if ou > 0.5 else "var(--red)"
        s += f'<tr><td class="mono">{_esc(c["component"])}</td><td class="mono" style="color:var(--text2)">{_esc(c["host"])}</td><td class="mono">{pct:.1f}%</td>'
        s += f'<td style="width:120px"><div class="bar-outer"><div class="bar-inner" style="background:{color};width:{min(pct, 100):.1f}%"></div></div></td></tr>\n'
    s += '</tbody></table>\n</div></div>\n'
    return s


# -- Section 4: View B — Per-shard GPU strips ---------------------------------

def _section_view_b(mr: dict) -> str:
    util = mr.get("component_utilization", [])
    compute = [c for c in util if any(k in c.get("component", "")
               for k in ("prefill", "decode", "engine"))]

    s = '<div class="section"><div class="section-h">View B &mdash; Per-Shard GPU Strips with TP Imbalance</div><div class="section-body">\n'

    if not compute:
        s += '<p style="color:var(--text3)">No GPU compute components found. View B requires engine-prefill and engine-decode traces with per-rank data.</p>\n'
        s += '</div></div>\n'
        return s

    labels = [f"{c['component']} @ {c['host']}" for c in compute]
    z_data = [[b.get("utilization", 0) for b in c.get("bins", [])] for c in compute]
    x_labels = [f"{b['start_ms']:.0f}" for b in compute[0].get("bins", [])] if compute else []

    height = 80 + len(labels) * 35
    s += f'''<div id="heatmap-b" class="plotly-chart" style="height:{height}px"></div>
<script>
Plotly.newPlot("heatmap-b", [{{
  z: {json.dumps(z_data)},
  x: {json.dumps(x_labels)},
  y: {json.dumps(labels)},
  type: "heatmap",
  colorscale: [[0,"{BG_SECONDARY}"],[0.5,"#3a5a10"],[1.0,"#76B900"]],
  showscale: true,
  colorbar: {{ title: "GPU Busy", titleside: "right", tickformat: ".0%", len: 0.8 }}
}}], {{
  paper_bgcolor: "{BG_PRIMARY}",
  plot_bgcolor: "{BG_SECONDARY}",
  font: {{ color: "{TEXT_PRIMARY}", family: "Matter, system-ui, sans-serif", size: 11 }},
  margin: {{ l: 220, r: 80, t: 20, b: 50 }},
  xaxis: {{ title: "Time (ms)", gridcolor: "{BORDER}" }},
  yaxis: {{ autorange: "reversed", gridcolor: "{BORDER}" }}
}}, {{ responsive: true }});
</script>
'''
    s += '<p style="font-size:11px;color:var(--text3);margin-top:12px">Straggler ratio = max(rank_busy) / mean(rank_busy) per iteration. Values &gt; 1.15 indicate TP imbalance.</p>\n'
    s += '</div></div>\n'
    return s


# -- Section 5: View C — Perfetto deep-link -----------------------------------

def _section_view_c(mr: dict, trace_url: Optional[str], trace_files: Optional[list] = None) -> str:
    s = '<div class="section"><div class="section-h">View C &mdash; Full Trace in Perfetto</div><div class="section-body">\n'

    if trace_url:
        s += f'''<div style="display:flex;flex-direction:column;align-items:center;gap:16px;padding:24px">
  <a href="https://ui.perfetto.dev/#!/?url={_esc(trace_url)}" target="_blank" class="perf-link" style="font-size:14px;padding:12px 24px;border-radius:8px">
    Open merged trace in Perfetto UI
  </a>
  <p style="font-size:12px;color:var(--text3);max-width:600px;text-align:center">
    Opens <code>{_esc(trace_url)}</code> with orchestration-plane tracks pre-pinned.
  </p>
</div>
'''
    else:
        files = trace_files or []
        # JavaScript: fetch trace from same-origin, open Perfetto, send via postMessage
        s += '''<script>
function openInPerfetto(filename) {
  var btn = event.target;
  btn.textContent = 'Loading...';
  var base = window.location.href.substring(0, window.location.href.lastIndexOf('/') + 1);
  fetch(base + filename)
    .then(function(r) { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(function(buf) {
      var win = window.open('https://ui.perfetto.dev');
      var timer = setInterval(function() {
        win.postMessage('PING', 'https://ui.perfetto.dev');
      }, 100);
      var handler = function(e) {
        if (e.data !== 'PONG') return;
        clearInterval(timer);
        window.removeEventListener('message', handler);
        win.postMessage({
          perfetto: { buffer: buf, title: filename, keepApiOpen: false }
        }, 'https://ui.perfetto.dev');
        btn.textContent = filename;
      };
      window.addEventListener('message', handler);
    })
    .catch(function(err) {
      btn.textContent = filename;
      alert('Failed to fetch ' + filename + ': ' + err.message +
            '\\n\\nMake sure you opened this report via http://localhost, not file://');
    });
}
</script>
'''
        s += '<div style="display:flex;flex-direction:column;align-items:center;gap:20px;padding:24px">\n'
        s += '  <div style="background:var(--bg3);border-radius:12px;padding:24px 32px;max-width:700px;width:100%">\n'

        if files:
            s += '    <p style="font-size:14px;font-weight:600;color:var(--text1);margin:0 0 4px 0">Open in Perfetto</p>\n'
            s += '    <p style="font-size:13px;color:var(--text2);margin:0 0 12px 0">'
            s += 'Serve this directory and open the report from localhost:</p>\n'
            s += '    <pre class="mono" style="background:var(--bg1);padding:12px 16px;border-radius:6px;font-size:12px;color:var(--text2);overflow-x:auto;margin:0 0 4px 0">'
            s += 'cd /path/to/output &amp;&amp; python3 -m http.server 9001</pre>\n'
            s += '    <p style="font-size:12px;color:var(--text3);margin:0 0 16px 0">Then open <code>http://localhost:9001/report.html</code> and click a trace below:</p>\n'
            s += '    <div style="display:flex;flex-wrap:wrap;gap:8px">\n'
            for fname in sorted(files):
                s += f'      <button onclick="openInPerfetto(\'{_esc(fname)}\')" class="perf-link" style="font-size:12px;padding:8px 16px;border-radius:6px;border:none;cursor:pointer">{_esc(fname)}</button>\n'
            s += '    </div>\n'
        else:
            s += '    <p style="font-size:12px;color:var(--text3)">No trace files found.</p>\n'

        s += '    <hr style="border:none;border-top:1px solid var(--border);margin:20px 0 16px 0">\n'
        s += '    <p style="font-size:13px;font-weight:600;color:var(--text1);margin:0 0 8px 0">Alternative &mdash; Drag &amp; Drop</p>\n'
        s += '    <p style="font-size:13px;color:var(--text2);margin:0">'
        s += 'Open <a href="https://ui.perfetto.dev" target="_blank" style="color:var(--accent)">ui.perfetto.dev</a> '
        s += 'and drag any <code>.pftrace.gz</code> file onto the page.</p>\n'

        s += '  </div>\n'
        s += '</div>\n'

    s += '</div></div>\n'
    return s


# -- Section 6: View D — Causality DAG ----------------------------------------

def _section_view_d(mr: dict) -> str:
    edges = mr.get("causality_edges", [])
    s = '<div class="section"><div class="section-h">View D &mdash; Causality DAG</div><div class="section-body">\n'

    if not edges:
        s += '<p style="color:var(--text3)">No causality edges computed. Need multiple stages per request for DAG construction.</p>\n</div></div>\n'
        return s

    stages = []
    for edge in edges:
        for key in ("from_stage", "to_stage"):
            if edge[key] not in stages:
                stages.append(edge[key])

    stage_order = [
        "dynamo.frontend.recv", "dynamo.frontend.preprocess",
        "dynamo.router.schedule", "dynamo.router.kv_lookup",
        "dynamo.router.metrics",
        "dynamo.transport.send", "dynamo.transport.recv",
        "dynamo.kvbm.tier_lookup",
        "dynamo.prefill.recv", "dynamo.prefill.compute",
        "dynamo.nixl.transfer.send", "dynamo.nixl.transfer.recv",
        "dynamo.decode.recv", "dynamo.decode.first_token",
        "dynamo.decode.compute", "dynamo.decode.detok_send",
    ]

    component_y = {
        "frontend": 0.0, "router": 1.0, "transport": 2.0, "kvbm": 1.5,
        "prefill": 3.0, "nixl": 2.5, "decode": 4.0, "planner": 0.5,
    }

    def pos(stage):
        try:
            idx = stage_order.index(stage)
        except ValueError:
            idx = stages.index(stage) + len(stage_order) if stage in stages else len(stage_order)
        x = idx * 1.2
        comp = stage.split(".")[1] if "." in stage else ""
        y = component_y.get(comp, 2.0)
        return x, y

    node_x = [pos(st)[0] for st in stages]
    node_y = [pos(st)[1] for st in stages]
    node_labels = [st.replace("dynamo.", "") for st in stages]
    node_colors = [_stage_color(i) for i in range(len(stages))]

    edge_x: list = []
    edge_y: list = []
    for edge in edges:
        fi = stages.index(edge["from_stage"]) if edge["from_stage"] in stages else None
        ti = stages.index(edge["to_stage"]) if edge["to_stage"] in stages else None
        if fi is not None and ti is not None:
            edge_x.extend([node_x[fi], node_x[ti], None])
            edge_y.extend([node_y[fi], node_y[ti], None])

    s += f'''<div id="dag-d" class="plotly-chart" style="height:500px"></div>
<script>
(function() {{
  var edgeTrace = {{
    x: {json.dumps(edge_x)},
    y: {json.dumps(edge_y)},
    mode: "lines",
    type: "scatter",
    line: {{ color: "rgba(153,153,153,0.4)", width: 1.5 }},
    hoverinfo: "none"
  }};
  var nodeTrace = {{
    x: {json.dumps(node_x)},
    y: {json.dumps(node_y)},
    mode: "markers+text",
    type: "scatter",
    text: {json.dumps(node_labels)},
    textposition: "top center",
    textfont: {{ size: 10, color: "{TEXT_SECONDARY}" }},
    marker: {{
      size: 14,
      color: {json.dumps(node_colors)},
      line: {{ width: 1, color: "{BORDER}" }}
    }},
    hovertemplate: "%{{text}}<extra></extra>"
  }};
  Plotly.newPlot("dag-d", [edgeTrace, nodeTrace], {{
    paper_bgcolor: "{BG_PRIMARY}",
    plot_bgcolor: "{BG_SECONDARY}",
    font: {{ color: "{TEXT_PRIMARY}", family: "Matter, system-ui, sans-serif", size: 11 }},
    showlegend: false,
    margin: {{ l: 40, r: 40, t: 20, b: 40 }},
    xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
    yaxis: {{ showgrid: false, zeroline: false, showticklabels: false, autorange: "reversed" }}
  }}, {{ responsive: true }});
}})();
</script>
'''

    s += '<table style="margin-top:16px"><thead><tr><th>From</th><th>To</th><th>Weight</th><th>Count</th></tr></thead><tbody>\n'
    for edge in edges[:15]:
        s += f'<tr><td class="mono" style="font-size:12px">{_esc(edge["from_stage"].replace("dynamo.", ""))}</td>'
        s += f'<td class="mono" style="font-size:12px">{_esc(edge["to_stage"].replace("dynamo.", ""))}</td>'
        s += f'<td class="mono">{edge.get("weight", 0):.2f}</td><td class="mono">{edge.get("count", 0)}</td></tr>\n'
    s += '</tbody></table>\n'
    s += '<p style="font-size:11px;color:var(--text3);margin-top:8px">Edge weight = (requests where edge is on critical path) / total requests.</p>\n'
    s += '</div></div>\n'
    return s


# -- Python analyzer sections --------------------------------------------------

def _section_stage_latency(stage_attr: dict) -> str:
    report = stage_attr.get("report", {})
    per_stage = report.get("per_stage_percentiles", {})
    if not per_stage:
        return ""

    stage_order = report.get("stage_order", list(per_stage.keys()))
    max_ms = max((s.get("p99", 0) for s in per_stage.values()), default=1) or 1

    s = '<div class="divider"></div>\n<div class="divider-label">Deep Analysis (Python Analyzers)</div>\n'
    s += '<div class="section"><div class="section-h">Per-Stage Latency Distribution</div><div class="section-body">\n'
    s += '<div class="legend">'
    s += '<div class="legend-item"><div class="legend-dot" style="background:#516CDC"></div>p50</div>'
    s += '<div class="legend-item"><div class="legend-dot" style="background:#76B900"></div>p95</div>'
    s += '<div class="legend-item"><div class="legend-dot" style="background:#F0783C"></div>p99</div>'
    s += '</div>\n'

    for stage in stage_order:
        if stage not in per_stage:
            continue
        v = per_stage[stage]
        p50 = v.get("p50", 0)
        p95 = v.get("p95", 0)
        p99 = v.get("p99", 0)
        s += '<div class="chart-row">'
        s += f'<div class="chart-label mono">{_esc(stage)}</div>'
        s += '<div class="chart-bars">'
        s += f'<div class="chart-bar" style="width:{p50 / max_ms * 100:.1f}%;background:#516CDC"></div>'
        s += f'<div class="chart-bar" style="width:{max(0, p95 - p50) / max_ms * 100:.1f}%;background:#76B900"></div>'
        s += f'<div class="chart-bar" style="width:{max(0, p99 - p95) / max_ms * 100:.1f}%;background:#F0783C"></div>'
        s += f'<div class="chart-val">{p99:.1f}ms</div>'
        s += '</div></div>\n'
    s += '</div></div>\n'
    return s


def _section_kernel_hotlist(kernels: dict) -> str:
    top = kernels.get("top_kernels", [])
    if not top:
        return ""

    s = '<div class="section"><div class="section-h">Kernel Hotlist</div><div class="section-body">\n'
    s += '<table><thead><tr><th>#</th><th>Kernel</th><th>Total</th><th>Count</th><th>Avg</th><th>% of GPU</th><th></th><th>TP CV%</th></tr></thead><tbody>\n'
    for i, k in enumerate(top[:15]):
        pct = k.get("pct_of_total", 0)
        cv = k.get("shard_imbalance_cv_pct")
        cv_tag = ""
        if cv is not None and cv > 15:
            cv_tag = '<span class="tag tag-warn">imbalance</span>'
        elif cv is not None and cv > 5:
            cv_tag = '<span class="tag tag-warn">minor</span>'

        color = "var(--green)" if pct < 30 else "var(--orange)" if pct < 60 else "var(--red)"
        s += f'<tr><td>{i + 1}</td>'
        s += f'<td class="mono" style="color:var(--text);max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_esc(k.get("kernel", ""))}</td>'
        s += f'<td class="mono">{k.get("total_ms", 0):.1f}ms</td>'
        s += f'<td class="mono">{k.get("count", 0)}</td>'
        s += f'<td class="mono">{k.get("avg_us", 0):.0f}us</td>'
        s += f'<td class="mono">{pct:.1f}%</td>'
        s += f'<td style="width:120px"><div class="bar-outer"><div class="bar-inner" style="background:{color};width:{min(pct, 100):.1f}%"></div></div></td>'
        if cv is not None:
            s += f'<td>{cv:.1f}% {cv_tag}</td>'
        else:
            s += '<td>&mdash;</td>'
        s += '</tr>\n'

        stages = k.get("per_stage_ms", {})
        if stages:
            stage_str = ", ".join(f'{st}={v:.1f}ms' for st, v in sorted(stages.items(), key=lambda x: -x[1]))
            s += f'<tr><td></td><td colspan="7" style="font-size:11px;color:var(--text3);padding-top:0">stages: {_esc(stage_str)}</td></tr>\n'

    s += '</tbody></table>\n'

    diffs = kernels.get("diff_vs_baseline", [])
    if diffs:
        regressions = [d for d in diffs if d.get("delta_ms", 0) > 0]
        improvements = [d for d in diffs if d.get("delta_ms", 0) < 0]
        if regressions:
            s += '<div style="margin-top:20px"><strong style="color:var(--red)">Regressions vs Baseline:</strong></div>'
            s += '<table style="margin-top:8px"><thead><tr><th>Kernel</th><th>Before</th><th>After</th><th>Delta</th></tr></thead><tbody>'
            for d in sorted(regressions, key=lambda x: -x.get("delta_ms", 0))[:5]:
                s += f'<tr><td class="mono">{_esc(d.get("kernel", ""))}</td>'
                s += f'<td class="mono">{d.get("baseline_ms", 0):.1f}ms</td>'
                s += f'<td class="mono">{d.get("current_ms", 0):.1f}ms</td>'
                s += f'<td class="mono bad">+{d.get("delta_ms", 0):.1f}ms</td></tr>'
            s += '</tbody></table>'
        if improvements:
            s += '<div style="margin-top:12px"><strong style="color:var(--green)">Improvements vs Baseline:</strong></div>'
            s += '<table style="margin-top:8px"><thead><tr><th>Kernel</th><th>Before</th><th>After</th><th>Delta</th></tr></thead><tbody>'
            for d in sorted(improvements, key=lambda x: x.get("delta_ms", 0))[:5]:
                s += f'<tr><td class="mono">{_esc(d.get("kernel", ""))}</td>'
                s += f'<td class="mono">{d.get("baseline_ms", 0):.1f}ms</td>'
                s += f'<td class="mono">{d.get("current_ms", 0):.1f}ms</td>'
                s += f'<td class="mono ok">{d.get("delta_ms", 0):.1f}ms</td></tr>'
            s += '</tbody></table>'

    s += '</div></div>\n'
    return s


def _section_gpu_util(gpu_util: dict) -> str:
    reports = gpu_util.get("gpu_track_reports", [])
    if not reports:
        return ""

    s = '<div class="section"><div class="section-h">GPU Utilization (Per-Device)</div><div class="section-body">\n'
    s += '<table><thead><tr><th>Process / GPU</th><th>Utilization</th><th></th><th>Busy</th><th>Capture</th><th>Suspicious Idle</th></tr></thead><tbody>\n'
    for r in reports:
        util = r.get("utilization", 0) * 100
        color = "var(--green)" if util > 90 else "var(--orange)" if util > 70 else "var(--red)"
        susp = r.get("suspicious_gaps_ms", 0)
        s += f'<tr><td class="mono" style="color:var(--text)">{_esc(r.get("process_name", ""))} / {_esc(r.get("track_name", ""))}</td>'
        s += f'<td class="mono">{util:.1f}%</td>'
        s += f'<td style="width:100px"><div class="bar-outer"><div class="bar-inner" style="background:{color};width:{min(util, 100):.1f}%"></div></div></td>'
        s += f'<td class="mono">{r.get("busy_ms", 0):.1f}ms</td>'
        s += f'<td class="mono">{r.get("capture_duration_ms", 0):.1f}ms</td>'
        s += f'<td class="mono {"warn" if susp > 10 else ""}">{susp:.1f}ms</td></tr>\n'

        stages = r.get("busy_by_stage_ms", {})
        if stages:
            stage_str = ", ".join(f'{st}={v:.1f}ms' for st, v in sorted(stages.items(), key=lambda x: -x[1]))
            s += f'<tr><td></td><td colspan="5" style="font-size:11px;color:var(--text3);padding-top:0">stages: {_esc(stage_str)}</td></tr>\n'
    s += '</tbody></table>\n</div></div>\n'
    return s


def _section_comm_breakdown(comm: dict) -> str:
    cats = comm.get("per_category", {})
    has_data = any(c.get("op_count", 0) > 0 for c in cats.values())
    if not has_data:
        return ""

    s = '<div class="section"><div class="section-h">Communication Breakdown</div><div class="section-body">\n'
    for cat_name in ("nccl", "nixl", "nats"):
        cat = cats.get(cat_name, {})
        if cat.get("op_count", 0) == 0:
            continue
        s += f'<div style="margin-bottom:16px"><strong>{_esc(cat_name.upper())}</strong>'
        s += f' &mdash; {cat.get("total_duration_ms", 0):.1f}ms total, {cat.get("op_count", 0)} ops'
        if cat.get("total_bytes", 0) > 0:
            gb = cat["total_bytes"] / (1024**3)
            s += f', {gb:.2f} GB, {cat.get("avg_bandwidth_gbps", 0):.1f} Gbps avg'
        s += '</div>'
        s += '<table><thead><tr><th>Operation</th><th>Count</th><th>Total</th><th>Avg</th><th>p99</th>'
        if cat.get("total_bytes", 0) > 0:
            s += '<th>BW</th>'
        s += '</tr></thead><tbody>'
        for op in cat.get("operations", [])[:10]:
            s += f'<tr><td class="mono">{_esc(op.get("op_name", ""))}</td>'
            s += f'<td class="mono">{op.get("count", 0)}</td>'
            s += f'<td class="mono">{op.get("total_duration_ms", 0):.1f}ms</td>'
            s += f'<td class="mono">{op.get("avg_duration_us", 0):.0f}us</td>'
            s += f'<td class="mono">{op.get("p99_duration_us", 0):.0f}us</td>'
            if cat.get("total_bytes", 0) > 0:
                s += f'<td class="mono">{op.get("bandwidth_gbps", 0):.1f}</td>'
            s += '</tr>'
        s += '</tbody></table>'

    nats_prop = comm.get("nats_propagation", {})
    if nats_prop.get("pairs_analyzed", 0) > 0:
        s += '<div style="margin-top:16px"><strong>NATS Propagation</strong>'
        s += f' &mdash; {nats_prop["pairs_analyzed"]} pairs, median {nats_prop.get("median_lag_ns", 0) / 1000:.1f}us'
        violations = nats_prop.get("causality_violations", 0)
        if violations:
            s += f' <span class="tag tag-bad">{violations} causality violations</span>'
        s += '</div>'
    s += '</div></div>\n'
    return s


# -- Footer --------------------------------------------------------------------

def _footer(mr: dict) -> str:
    total_slices = mr.get("total_slices", 0)
    n_components = len(mr.get("components", []))
    clock = mr.get("clock_alignment", {})
    clock_info = ""
    if clock.get("max_residual_ns", 0) > 0:
        clock_info = f' &middot; Clock residual: {clock["max_residual_ns"] / 1000:.0f}&micro;s ({clock.get("method", "")})'
    else:
        clock_info = f' &middot; Clock: {clock.get("method", "")}'

    return f"""<div class="footer">
  Generated by <strong>dynamo-sysprofile</strong> &middot;
  {total_slices} slices across {n_components} components{clock_info} &middot;
  <a href="https://github.com/ai-dynamo/dynamo" style="color:var(--indigo)">ai-dynamo/dynamo</a>
</div>
</div>
</body>
</html>"""


# -- CLI entry point -----------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate unified sysprofile HTML report")
    parser.add_argument("--merge-result", help="Path to merge_result.json from Rust merger")
    parser.add_argument("--stage-attr", help="Path to stage_attr.json")
    parser.add_argument("--gpu-util", help="Path to gpu_util.json")
    parser.add_argument("--kernels", help="Path to kernels.json")
    parser.add_argument("--comm", help="Path to comm.json")
    parser.add_argument("--trace-url", help="URL to merged .pftrace.gz for Perfetto deep-links")
    parser.add_argument("--title", default="sysprofile report")
    parser.add_argument("--output", required=True, help="Output HTML path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    html = generate_report(
        merge_result=_load_json(args.merge_result),
        stage_attr=_load_json(args.stage_attr),
        gpu_util=_load_json(args.gpu_util),
        kernels=_load_json(args.kernels),
        comm=_load_json(args.comm),
        trace_url=args.trace_url,
        title=args.title,
    )

    with open(args.output, "w") as f:
        f.write(html)
    log.info("Report written to %s (%d bytes)", args.output, len(html))


if __name__ == "__main__":
    main()
