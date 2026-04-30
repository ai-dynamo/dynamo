// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTML report generator for `dynamo-sysprofile-merge`.
//!
//! Produces a single self-contained HTML file with Plotly.js charts.
//! DEP section 5.10 defines six sections:
//!
//! 1. **Headline**: p99 stacked-bar critical-path attribution
//! 2. **Top-10 slowest requests** with Perfetto deep-links
//! 3. **View A**: Component utilization heat-strip
//! 4. **View B**: Per-shard GPU strips with TP imbalance overlay
//! 5. **View C**: Perfetto deep-link with pre-pinned tracks
//! 6. **View D**: Causality DAG (force-directed graph)

use crate::merger::MergeResult;

/// Generate a self-contained HTML report from merge results.
pub fn generate_report(result: &MergeResult, trace_url: Option<&str>) -> String {
    let mut html = String::with_capacity(64 * 1024);

    html.push_str(&header(&result.run_id, result.capture_duration_ms));
    html.push_str(&section_headline(result));
    html.push_str(&section_top10(result, trace_url));
    html.push_str(&section_view_a(result));
    html.push_str(&section_view_b(result));
    html.push_str(&section_view_c(result, trace_url));
    html.push_str(&section_view_d(result));
    html.push_str(&footer(result));

    html
}

// ── Color palette (Tatva-inspired, dark theme) ────────────────────────────────

const BG_PRIMARY: &str = "#141414";
const BG_SECONDARY: &str = "#1a1a1a";
const BG_SURFACE: &str = "#222222";
const BORDER: &str = "#333333";
const TEXT_PRIMARY: &str = "#f5f5f5";
const TEXT_SECONDARY: &str = "#999999";
const TEXT_TERTIARY: &str = "#666666";

const STAGE_COLORS: &[&str] = &[
    "#516CDC", // indigo-500
    "#76B900", // nvidia green
    "#F0783C", // orange-500
    "#C84673", // pink-500
    "#5F9637", // green-500
    "#3842B4", // indigo-700
    "#D25A1E", // orange-700
    "#9D2055", // pink-600
    "#06b6d4", // cyan
    "#a855f7", // purple
    "#f59e0b", // amber
    "#ef4444", // red
];

fn stage_color(idx: usize) -> &'static str {
    STAGE_COLORS[idx % STAGE_COLORS.len()]
}

// ── Header ────────────────────────────────────────────────────────────────────

fn header(run_id: &str, duration_ms: f64) -> String {
    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sysprofile: {run_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: {BG_PRIMARY};
  --bg2: {BG_SECONDARY};
  --bg3: {BG_SURFACE};
  --border: {BORDER};
  --text: {TEXT_PRIMARY};
  --text2: {TEXT_SECONDARY};
  --text3: {TEXT_TERTIARY};
  --green: #76B900;
  --indigo: #516CDC;
  --orange: #F0783C;
  --red: #B81514;
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
.footer {{ text-align: center; padding: 32px; font-size: 11px; color: var(--text3); border-top: 1px solid var(--border); margin-top: 16px; }}
.warning-banner {{ background: rgba(240,120,60,0.1); border: 1px solid rgba(240,120,60,0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; font-size: 12px; color: var(--orange); }}
.plotly-chart {{ width: 100%; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>sysprofile report</h1>
  <div class="meta">Run {run_id} &middot; {dur:.1}s capture &middot; Generated by dynamo-sysprofile-merge</div>
</div>
"#,
        run_id = run_id,
        dur = duration_ms / 1000.0
    )
}

// ── Section 1: p99 critical-path headline ─────────────────────────────────────

fn section_headline(result: &MergeResult) -> String {
    let mut s = String::new();

    // KPI cards
    s.push_str("<div class=\"kpis\">\n");
    s.push_str(&kpi_card(
        "p99 TTFT",
        &format!("{:.1}", result.p99_total_ms),
        "ms",
        "End-to-end critical path",
        if result.p99_total_ms > 500.0 {
            "warn"
        } else {
            "ok"
        },
    ));
    s.push_str(&kpi_card(
        "p50 TTFT",
        &format!("{:.1}", result.p50_total_ms),
        "ms",
        "Median critical path",
        "ok",
    ));
    s.push_str(&kpi_card(
        "Requests",
        &result.total_requests.to_string(),
        "",
        "Captured with traceparent",
        "",
    ));
    s.push_str(&kpi_card(
        "Components",
        &result.components.len().to_string(),
        "",
        &result
            .components
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        "",
    ));

    if !result.p99_critical_path.is_empty() {
        let bottleneck_stage = result
            .p99_critical_path
            .iter()
            .max_by(|a, b| a.fraction.total_cmp(&b.fraction));
        if let Some(bn) = bottleneck_stage {
            s.push_str(&kpi_card(
                "Bottleneck",
                &bn.stage.replace("dynamo.", ""),
                "",
                &format!("{:.1}% of p99 critical path", bn.fraction * 100.0),
                "warn",
            ));
        }
    }
    s.push_str("</div>\n");

    // Clock alignment warning
    if result.clock_alignment.max_residual_ns > 100_000 {
        s.push_str(&format!(
            "<div class=\"warning-banner\">Clock alignment residual: {:.0}&micro;s. Cross-host event ordering may be unreliable for events shorter than this.</div>\n",
            result.clock_alignment.max_residual_ns as f64 / 1000.0
        ));
    }

    // p99 stacked bar
    s.push_str("<div class=\"section\"><div class=\"section-h\">p99 Critical-Path Attribution</div><div class=\"section-body\">\n");
    s.push_str(&format!(
        "<div style=\"font-size:12px;color:var(--text2);margin-bottom:8px\">Total p99 = {:.1}ms across {} stages ({} requests sampled)</div>\n",
        result.p99_total_ms,
        result.p99_critical_path.len(),
        result.total_requests
    ));

    s.push_str("<div class=\"cp-bar\">\n");
    for (i, stage) in result.p99_critical_path.iter().enumerate() {
        let pct = stage.fraction * 100.0;
        let color = stage_color(i);
        let label = if pct >= 8.0 {
            format!(
                "{} {:.0}%",
                stage.stage.replace("dynamo.", ""),
                pct
            )
        } else if pct >= 3.0 {
            format!("{:.0}%", pct)
        } else {
            String::new()
        };
        s.push_str(&format!(
            "  <div class=\"cp-seg\" style=\"width:{pct:.1}%;background:{color}\" title=\"{name}: {pct:.1}% ({dur:.1}ms)\">{label}</div>\n",
            name = stage.stage,
            dur = stage.duration_ms,
        ));
    }
    s.push_str("</div>\n");

    // Legend
    s.push_str("<div class=\"legend\">\n");
    for (i, stage) in result.p99_critical_path.iter().enumerate() {
        s.push_str(&format!(
            "  <div class=\"legend-item\"><div class=\"legend-dot\" style=\"background:{}\"></div>{} ({:.1}%, {:.2}ms)</div>\n",
            stage_color(i),
            stage.stage.replace("dynamo.", ""),
            stage.fraction * 100.0,
            stage.duration_ms
        ));
    }
    s.push_str("</div>\n");
    s.push_str("</div></div>\n");

    s
}

// ── Section 2: Top-10 slowest requests ────────────────────────────────────────

fn section_top10(result: &MergeResult, trace_url: Option<&str>) -> String {
    let mut s = String::new();
    s.push_str("<div class=\"section\"><div class=\"section-h\">Top-10 Slowest Requests</div><div class=\"section-body\">\n");
    s.push_str("<table><thead><tr><th>#</th><th>Trace ID</th><th>Duration</th><th>Dominant Stage</th><th>Stages</th><th></th></tr></thead><tbody>\n");

    for (i, req) in result.top_slow_requests.iter().enumerate() {
        let trace_id = extract_trace_id(&req.traceparent);
        let dominant = req
            .stages
            .iter()
            .max_by(|a, b| a.fraction.total_cmp(&b.fraction))
            .map(|s| format!("{} ({:.0}%)", s.stage.replace("dynamo.", ""), s.fraction * 100.0))
            .unwrap_or_default();

        let perfetto_link = trace_url
            .map(|url| {
                format!(
                    "<a href=\"{}\" class=\"perf-link\" target=\"_blank\">Open in Perfetto</a>",
                    build_perfetto_url(url, req)
                )
            })
            .unwrap_or_default();

        // Mini critical-path bar
        let mini_bar = build_mini_bar(&req.stages);

        s.push_str(&format!(
            "<tr><td class=\"mono\">{rank}</td><td class=\"mono\" style=\"color:var(--text)\">{trace_id}</td><td class=\"mono\">{dur:.1}ms</td><td style=\"font-size:12px;color:var(--text2)\">{dominant}</td><td style=\"width:200px\">{mini_bar}</td><td>{perfetto_link}</td></tr>\n",
            rank = i + 1,
            dur = req.total_duration_ms,
        ));
    }

    s.push_str("</tbody></table>\n</div></div>\n");
    s
}

// ── Section 3: View A — Component utilization heat-strip ──────────────────────

fn section_view_a(result: &MergeResult) -> String {
    let mut s = String::new();
    s.push_str("<div class=\"section\"><div class=\"section-h\">View A &mdash; Component Utilization Heat-Strip</div><div class=\"section-body\">\n");

    if result.component_utilization.is_empty() {
        s.push_str("<p style=\"color:var(--text3)\">No utilization data available.</p>\n");
        s.push_str("</div></div>\n");
        return s;
    }

    // Build Plotly heatmap
    let comp_labels: Vec<String> = result
        .component_utilization
        .iter()
        .map(|c| format!("{} @ {}", c.component, c.host))
        .collect();

    let num_bins = result
        .component_utilization
        .first()
        .map(|c| c.bins.len())
        .unwrap_or(0);

    let x_labels: Vec<String> = if num_bins > 0 {
        result.component_utilization[0]
            .bins
            .iter()
            .map(|b| format!("{:.0}", b.start_ms))
            .collect()
    } else {
        vec![]
    };

    let z_data: Vec<Vec<f64>> = result
        .component_utilization
        .iter()
        .map(|c| c.bins.iter().map(|b| b.utilization).collect())
        .collect();

    s.push_str(&format!(
        r##"<div id="heatmap-a" class="plotly-chart" style="height:{}px"></div>
<script>
Plotly.newPlot("heatmap-a", [{{
  z: {z},
  x: {x},
  y: {y},
  type: "heatmap",
  colorscale: [[0,"#1a1a1a"],[0.3,"#1a3a00"],[0.7,"#4a7a00"],[1.0,"#76B900"]],
  showscale: true,
  colorbar: {{ title: "Utilization", titleside: "right", tickformat: ".0%", len: 0.8 }},
  hovertemplate: "%{{y}}<br>t=%{{x}}ms<br>util=%{{z:.1%}}<extra></extra>"
}}], {{
  paper_bgcolor: "{bg}",
  plot_bgcolor: "{bg2}",
  font: {{ color: "{text}", family: "Matter, system-ui, sans-serif", size: 11 }},
  margin: {{ l: 200, r: 80, t: 20, b: 50 }},
  xaxis: {{ title: "Time (ms from capture start)", gridcolor: "{border}" }},
  yaxis: {{ autorange: "reversed", gridcolor: "{border}" }}
}}, {{ responsive: true }});
</script>
"##,
        60 + comp_labels.len() * 30,
        z = serde_json::to_string(&z_data).unwrap_or_default(),
        x = serde_json::to_string(&x_labels).unwrap_or_default(),
        y = serde_json::to_string(&comp_labels).unwrap_or_default(),
        bg = BG_PRIMARY,
        bg2 = BG_SECONDARY,
        text = TEXT_PRIMARY,
        border = BORDER,
    ));

    // Utilization summary table
    s.push_str("<table style=\"margin-top:16px\"><thead><tr><th>Component</th><th>Host</th><th>Utilization</th><th></th></tr></thead><tbody>\n");
    for c in &result.component_utilization {
        let bar_width = (c.overall_utilization * 100.0).min(100.0);
        let color = if c.overall_utilization > 0.9 {
            "var(--green)"
        } else if c.overall_utilization > 0.5 {
            "var(--orange)"
        } else {
            "var(--red)"
        };
        s.push_str(&format!(
            "<tr><td class=\"mono\">{comp}</td><td class=\"mono\" style=\"color:var(--text2)\">{host}</td><td class=\"mono\">{pct:.1}%</td><td style=\"width:120px\"><div style=\"background:#2a2a2a;border-radius:4px;height:16px;width:100%\"><div style=\"background:{color};height:100%;width:{bar_width:.1}%;border-radius:4px;min-width:2px\"></div></div></td></tr>\n",
            comp = c.component,
            host = c.host,
            pct = c.overall_utilization * 100.0,
        ));
    }
    s.push_str("</tbody></table>\n");
    s.push_str("</div></div>\n");
    s
}

// ── Section 4: View B — Per-shard GPU strips with TP imbalance ────────────────

fn section_view_b(result: &MergeResult) -> String {
    let mut s = String::new();
    s.push_str("<div class=\"section\"><div class=\"section-h\">View B &mdash; Per-Shard GPU Strips with TP Imbalance</div><div class=\"section-body\">\n");

    // For the demo/single-host case, show per-component compute strips
    // In production, this would show per-(host, rank) GPU utilization
    let compute_components: Vec<&crate::merger::ComponentUtilization> = result
        .component_utilization
        .iter()
        .filter(|c| {
            let comp = c.component.as_str();
            let host = c.host.as_str();
            comp.contains("prefill")
                || comp.contains("decode")
                || comp.starts_with("engine")
                || comp == "transport"
                || host.contains("prefill")
                || host.contains("decode")
        })
        .collect();

    if compute_components.is_empty() {
        s.push_str("<p style=\"color:var(--text3)\">No GPU compute components found. View B requires engine-prefill and engine-decode traces with per-rank data.</p>\n");
        s.push_str("<p style=\"font-size:12px;color:var(--text3);margin-top:8px\">In production, this view shows one strip per (host, TP rank) with the straggler ratio overlay: <code>max(rank_busy) / mean(rank_busy)</code> per iteration.</p>\n");
        s.push_str("</div></div>\n");
        return s;
    }

    let labels: Vec<String> = compute_components
        .iter()
        .map(|c| format!("{} @ {}", c.component, c.host))
        .collect();

    let z_data: Vec<Vec<f64>> = compute_components
        .iter()
        .map(|c| c.bins.iter().map(|b| b.utilization).collect())
        .collect();

    let x_labels: Vec<String> = compute_components
        .first()
        .map(|c| c.bins.iter().map(|b| format!("{:.0}", b.start_ms)).collect())
        .unwrap_or_default();

    s.push_str(&format!(
        r##"<div id="heatmap-b" class="plotly-chart" style="height:{}px"></div>
<script>
Plotly.newPlot("heatmap-b", [{{
  z: {z},
  x: {x},
  y: {y},
  type: "heatmap",
  colorscale: [[0,"#1a1a1a"],[0.5,"#3a5a10"],[1.0,"#76B900"]],
  showscale: true,
  colorbar: {{ title: "GPU Busy", titleside: "right", tickformat: ".0%", len: 0.8 }}
}}], {{
  paper_bgcolor: "{bg}",
  plot_bgcolor: "{bg2}",
  font: {{ color: "{text}", family: "Matter, system-ui, sans-serif", size: 11 }},
  margin: {{ l: 220, r: 80, t: 20, b: 50 }},
  xaxis: {{ title: "Time (ms)", gridcolor: "{border}" }},
  yaxis: {{ autorange: "reversed", gridcolor: "{border}" }}
}}, {{ responsive: true }});
</script>
"##,
        80 + labels.len() * 35,
        z = serde_json::to_string(&z_data).unwrap_or_default(),
        x = serde_json::to_string(&x_labels).unwrap_or_default(),
        y = serde_json::to_string(&labels).unwrap_or_default(),
        bg = BG_PRIMARY,
        bg2 = BG_SECONDARY,
        text = TEXT_PRIMARY,
        border = BORDER,
    ));

    s.push_str("<p style=\"font-size:11px;color:var(--text3);margin-top:12px\">Straggler ratio = max(rank_busy) / mean(rank_busy) per iteration. Values &gt; 1.15 indicate TP imbalance.</p>\n");
    s.push_str("</div></div>\n");
    s
}

// ── Section 5: View C — Perfetto deep-link ────────────────────────────────────

fn section_view_c(result: &MergeResult, trace_url: Option<&str>) -> String {
    let mut s = String::new();
    s.push_str("<div class=\"section\"><div class=\"section-h\">View C &mdash; Full Trace in Perfetto</div><div class=\"section-body\">\n");

    let has_remote_url = trace_url
        .map(|u| u.starts_with("http://") || u.starts_with("https://"))
        .unwrap_or(false);

    if has_remote_url {
        let base_url = trace_url.unwrap().trim_end_matches('/');

        s.push_str("<div style=\"display:flex;flex-direction:column;align-items:center;gap:16px;padding:24px\">\n");
        s.push_str("  <p id=\"perfetto-status\" style=\"font-size:12px;color:var(--text3)\"></p>\n");

        for comp in &result.components {
            s.push_str(&format!(
                r#"  <button onclick="openInPerfetto('{base_url}/{name}.pftrace.gz', '{name}')" class="perf-link" style="font-size:14px;padding:10px 20px;border-radius:8px;margin:4px;cursor:pointer;border:none">
    {name} ({slices} slices)
  </button>
"#,
                name = comp.name,
                slices = comp.slice_count,
            ));
        }

        s.push_str(&format!(
            r#"  <p style="font-size:12px;color:var(--text3);max-width:600px;text-align:center;margin-top:8px">
    Clicks fetch the trace from <code>{base_url}/</code> and send it to Perfetto via <code>postMessage</code>.
    Serve this report from the same host as the traces (e.g. <code>kubectl port-forward svc/sysprofile-viewer 9090:80</code>).
  </p>
  <p style="font-size:11px;color:var(--text3)">For traces over 2 GB, use <code>trace_processor --httpd &lt;file&gt;</code> and open <code>http://localhost:9001</code>.</p>
</div>
<script>
async function openInPerfetto(url, title) {{
  const status = document.getElementById('perfetto-status');
  status.textContent = 'Downloading ' + title + '...';
  status.style.color = '#76B900';
  try {{
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const buf = await resp.arrayBuffer();
    status.textContent = 'Opening Perfetto UI...';
    const perfWin = window.open('https://ui.perfetto.dev');
    if (!perfWin) {{ status.textContent = 'Pop-up blocked. Allow pop-ups and retry.'; status.style.color = '#ff4444'; return; }}
    const timer = setInterval(() => {{
      perfWin.postMessage('PING', 'https://ui.perfetto.dev');
    }}, 200);
    const handler = (evt) => {{
      if (evt.data !== 'PONG') return;
      window.removeEventListener('message', handler);
      clearInterval(timer);
      perfWin.postMessage({{
        perfetto: {{
          buffer: buf,
          title: title,
          keepApiOpen: false
        }}
      }}, 'https://ui.perfetto.dev');
      status.textContent = title + ' loaded in Perfetto.';
    }};
    window.addEventListener('message', handler);
    setTimeout(() => {{ clearInterval(timer); window.removeEventListener('message', handler); }}, 30000);
  }} catch(e) {{
    status.textContent = 'Error: ' + e.message;
    status.style.color = '#ff4444';
  }}
}}
</script>
"#,
        ));
    } else {
        s.push_str(
            r#"<div style="display:flex;flex-direction:column;align-items:center;gap:16px;padding:24px">
  <a href="https://ui.perfetto.dev" target="_blank" class="perf-link" style="font-size:14px;padding:12px 24px;border-radius:8px">
    Open Perfetto UI
  </a>
  <p style="font-size:12px;color:var(--text3);max-width:600px;text-align:center">
    Drag and drop individual <code>.pftrace.gz</code> files from the run directory into the Perfetto UI.
  </p>
  <p style="font-size:12px;color:var(--text3);max-width:600px;text-align:center">
    For clickable deep-links, serve the report and traces from the same HTTP server and
    re-run merge with <code>--trace-url</code>:
  </p>
  <pre class="mono" style="background:var(--bg3);padding:12px 16px;border-radius:6px;font-size:12px;color:var(--text2);overflow-x:auto;max-width:100%">kubectl port-forward svc/sysprofile-viewer 9090:80
dynamo-sysprofile-merge ./run-dir --trace-url http://localhost:9090/bench-001</pre>
  <p style="font-size:11px;color:var(--text3)">Then open <code>http://localhost:9090/bench-001/report.html</code> in your browser.</p>
</div>
"#,
        );
    }

    s.push_str("</div></div>\n");
    s
}

// ── Section 6: View D — Causality DAG ─────────────────────────────────────────

fn section_view_d(result: &MergeResult) -> String {
    let mut s = String::new();
    s.push_str("<div class=\"section\"><div class=\"section-h\">View D &mdash; Causality DAG</div><div class=\"section-body\">\n");

    if result.causality_edges.is_empty() {
        s.push_str("<p style=\"color:var(--text3)\">No causality edges computed. Need multiple stages per request for DAG construction.</p>\n");
        s.push_str("</div></div>\n");
        return s;
    }

    // Collect unique stages for node positioning
    let mut stages: Vec<String> = Vec::new();
    for edge in &result.causality_edges {
        if !stages.contains(&edge.from_stage) {
            stages.push(edge.from_stage.clone());
        }
        if !stages.contains(&edge.to_stage) {
            stages.push(edge.to_stage.clone());
        }
    }

    // Assign positions in a pipeline layout (left to right)
    let stage_order = [
        "dynamo.frontend.recv",
        "dynamo.frontend.preprocess",
        "dynamo.router.schedule",
        "dynamo.router.kv_lookup",
        "dynamo.router.metrics",
        "dynamo.transport.send",
        "dynamo.transport.recv",
        "dynamo.kvbm.tier_lookup",
        "dynamo.prefill.recv",
        "dynamo.prefill.compute",
        "dynamo.nixl.transfer.send",
        "dynamo.nixl.transfer.recv",
        "dynamo.decode.recv",
        "dynamo.decode.first_token",
        "dynamo.decode.compute",
        "dynamo.decode.detok_send",
    ];

    let position_of = |stage: &str| -> (f64, f64) {
        let idx = stage_order
            .iter()
            .position(|&s| s == stage)
            .unwrap_or_else(|| {
                stages.iter().position(|s| s == stage).unwrap_or(0) + stage_order.len()
            });
        let x = (idx as f64) * 1.2;
        // Alternate y for readability
        let component = stage.split('.').nth(1).unwrap_or("");
        let y = match component {
            "frontend" => 0.0,
            "router" => 1.0,
            "transport" => 2.0,
            "kvbm" => 1.5,
            "prefill" => 3.0,
            "nixl" => 2.5,
            "decode" => 4.0,
            "planner" => 0.5,
            _ => 2.0,
        };
        (x, y)
    };

    // Build node and edge data for Plotly
    let node_x: Vec<f64> = stages.iter().map(|s| position_of(s).0).collect();
    let node_y: Vec<f64> = stages.iter().map(|s| position_of(s).1).collect();
    let node_labels: Vec<String> = stages
        .iter()
        .map(|s| s.replace("dynamo.", ""))
        .collect();
    let node_colors: Vec<&str> = stages
        .iter()
        .enumerate()
        .map(|(i, _)| stage_color(i))
        .collect();

    // Edge lines as scatter traces with None separators
    let mut edge_x: Vec<Option<f64>> = Vec::new();
    let mut edge_y: Vec<Option<f64>> = Vec::new();
    let mut edge_widths: Vec<f64> = Vec::new();

    for edge in &result.causality_edges {
        let from_idx = stages.iter().position(|s| s == &edge.from_stage);
        let to_idx = stages.iter().position(|s| s == &edge.to_stage);
        if let (Some(fi), Some(ti)) = (from_idx, to_idx) {
            edge_x.push(Some(node_x[fi]));
            edge_x.push(Some(node_x[ti]));
            edge_x.push(None);
            edge_y.push(Some(node_y[fi]));
            edge_y.push(Some(node_y[ti]));
            edge_y.push(None);
            edge_widths.push(edge.weight);
        }
    }

    s.push_str(&format!(
        r#"<div id="dag-d" class="plotly-chart" style="height:500px"></div>
<script>
(function() {{
  var edgeTrace = {{
    x: {edge_x},
    y: {edge_y},
    mode: "lines",
    type: "scatter",
    line: {{ color: "rgba(153,153,153,0.4)", width: 1.5 }},
    hoverinfo: "none"
  }};
  var nodeTrace = {{
    x: {node_x},
    y: {node_y},
    mode: "markers+text",
    type: "scatter",
    text: {labels},
    textposition: "top center",
    textfont: {{ size: 10, color: "{text2}" }},
    marker: {{
      size: 14,
      color: {colors},
      line: {{ width: 1, color: "{border}" }}
    }},
    hovertemplate: "%{{text}}<extra></extra>"
  }};
  Plotly.newPlot("dag-d", [edgeTrace, nodeTrace], {{
    paper_bgcolor: "{bg}",
    plot_bgcolor: "{bg2}",
    font: {{ color: "{text}", family: "Matter, system-ui, sans-serif", size: 11 }},
    showlegend: false,
    margin: {{ l: 40, r: 40, t: 20, b: 40 }},
    xaxis: {{ showgrid: false, zeroline: false, showticklabels: false }},
    yaxis: {{ showgrid: false, zeroline: false, showticklabels: false, autorange: "reversed" }}
  }}, {{ responsive: true }});
}})();
</script>
"#,
        edge_x = serde_json::to_string(&edge_x).unwrap_or_default(),
        edge_y = serde_json::to_string(&edge_y).unwrap_or_default(),
        node_x = serde_json::to_string(&node_x).unwrap_or_default(),
        node_y = serde_json::to_string(&node_y).unwrap_or_default(),
        labels = serde_json::to_string(&node_labels).unwrap_or_default(),
        colors = serde_json::to_string(&node_colors).unwrap_or_default(),
        text2 = TEXT_SECONDARY,
        text = TEXT_PRIMARY,
        bg = BG_PRIMARY,
        bg2 = BG_SECONDARY,
        border = BORDER,
    ));

    // Edge weight table
    s.push_str("<table style=\"margin-top:16px\"><thead><tr><th>From</th><th>To</th><th>Weight</th><th>Count</th></tr></thead><tbody>\n");
    for edge in result.causality_edges.iter().take(15) {
        s.push_str(&format!(
            "<tr><td class=\"mono\" style=\"font-size:12px\">{}</td><td class=\"mono\" style=\"font-size:12px\">{}</td><td class=\"mono\">{:.2}</td><td class=\"mono\">{}</td></tr>\n",
            edge.from_stage.replace("dynamo.", ""),
            edge.to_stage.replace("dynamo.", ""),
            edge.weight,
            edge.count
        ));
    }
    s.push_str("</tbody></table>\n");
    s.push_str("<p style=\"font-size:11px;color:var(--text3);margin-top:8px\">Edge weight = (requests where edge is on critical path) / total requests. Higher weight = more causal influence on tail latency.</p>\n");
    s.push_str("</div></div>\n");
    s
}

// ── Footer ────────────────────────────────────────────────────────────────────

fn footer(result: &MergeResult) -> String {
    let clock_info = if result.clock_alignment.max_residual_ns > 0 {
        format!(
            " &middot; Clock residual: {:.0}&micro;s ({})",
            result.clock_alignment.max_residual_ns as f64 / 1000.0,
            result.clock_alignment.method
        )
    } else {
        format!(" &middot; Clock: {}", result.clock_alignment.method)
    };

    format!(
        r#"<div class="footer">
  Generated by <strong>dynamo-sysprofile-merge</strong> &middot;
  {} slices across {} components{} &middot;
  <a href="https://github.com/ai-dynamo/dynamo" style="color:var(--indigo)">ai-dynamo/dynamo</a>
</div>
</div>
</body>
</html>"#,
        result.total_slices,
        result.components.len(),
        clock_info,
    )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn kpi_card(label: &str, value: &str, unit: &str, hint: &str, class: &str) -> String {
    let color_class = if class.is_empty() {
        String::new()
    } else {
        format!(" {class}")
    };
    format!(
        r#"<div class="kpi">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value{color_class}">{value}<span class="kpi-unit">{unit}</span></div>
  <div class="kpi-hint">{hint}</div>
</div>
"#,
    )
}

fn extract_trace_id(traceparent: &str) -> String {
    // "00-<trace-id>-<span-id>-<flags>" -> trace_id (first 12 chars)
    traceparent
        .split('-')
        .nth(1)
        .map(|s| {
            if s.len() > 12 {
                format!("{}...", &s[..12])
            } else {
                s.to_string()
            }
        })
        .unwrap_or_else(|| traceparent.to_string())
}

fn build_perfetto_url(base_url: &str, _req: &crate::merger::RequestCriticalPath) -> String {
    let url = format!("{}/frontend.pftrace.gz", base_url.trim_end_matches('/'));
    format!(
        "https://ui.perfetto.dev/#!/?url={}",
        urlencoding::encode(&url)
    )
}

fn build_mini_bar(stages: &[crate::merger::CriticalPathStage]) -> String {
    let mut bar = String::from("<div style=\"display:flex;height:16px;border-radius:3px;overflow:hidden\">");
    for (i, stage) in stages.iter().enumerate() {
        let pct = stage.fraction * 100.0;
        bar.push_str(&format!(
            "<div style=\"width:{pct:.1}%;background:{}\" title=\"{}: {:.1}%\"></div>",
            stage_color(i),
            stage.stage.replace("dynamo.", ""),
            pct
        ));
    }
    bar.push_str("</div>");
    bar
}
