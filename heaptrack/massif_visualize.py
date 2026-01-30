#!/usr/bin/env python3
"""
Visualize massif.out files from heaptrack_print --print-massif.

Usage:
  python3 massif_visualize.py <massif.out> --html > chart.html
  python3 massif_visualize.py <massif.out> --html --rss rss.csv > chart.html
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Snapshot:
    time: float
    mem_heap_b: int


@dataclass
class RssSample:
    elapsed_s: float
    rss_bytes: int


def parse_massif(path: Path) -> List[Snapshot]:
    snapshots = []
    current = {}

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("snapshot="):
                if current.get("time") is not None:
                    snapshots.append(
                        Snapshot(
                            time=current.get("time", 0.0),
                            mem_heap_b=current.get("mem_heap_B", 0),
                        )
                    )
                current = {}
            elif "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                val = val.strip()
                if not val:
                    continue
                if key == "time":
                    current[key] = float(val)
                elif key == "mem_heap_B":
                    current[key] = int(val)

    if current.get("time") is not None:
        snapshots.append(
            Snapshot(
                time=current.get("time", 0.0),
                mem_heap_b=current.get("mem_heap_B", 0),
            )
        )

    return snapshots


def parse_rss_csv(path: Path) -> List[RssSample]:
    samples = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                samples.append(
                    RssSample(
                        elapsed_s=float(row["elapsed_s"]),
                        rss_bytes=int(row["rss_bytes"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return samples


def format_bytes(b: int) -> str:
    if b >= 1024 * 1024 * 1024:
        return f"{b / 1024 / 1024 / 1024:.2f} GB"
    elif b >= 1024 * 1024:
        return f"{b / 1024 / 1024:.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    return f"{b} B"


def to_html(
    snapshots: List[Snapshot],
    rss_samples: Optional[List[RssSample]] = None,
    title: str = "Memory Profile",
) -> str:
    if not snapshots:
        return "<html><body>No data</body></html>"

    # Downsample if needed
    max_points = 500
    if len(snapshots) > max_points:
        step = len(snapshots) / max_points
        snapshots = [snapshots[int(i * step)] for i in range(max_points)]

    if rss_samples and len(rss_samples) > max_points:
        step = len(rss_samples) / max_points
        rss_samples = [rss_samples[int(i * step)] for i in range(max_points)]

    # Stats
    peak = max(snapshots, key=lambda s: s.mem_heap_b)
    first = snapshots[0]
    last = snapshots[-1]

    heap_data = [{"x": s.time, "y": s.mem_heap_b / 1024 / 1024} for s in snapshots]
    rss_data = []
    rss_stats = ""
    slider_html = ""

    if rss_samples:
        rss_data = [
            {"x": r.elapsed_s, "y": r.rss_bytes / 1024 / 1024} for r in rss_samples
        ]
        peak_rss = max(rss_samples, key=lambda r: r.rss_bytes)
        last_rss = rss_samples[-1]

        rss_stats = f"""
        <div class="stat">
            <div class="label">Peak RSS</div>
            <div class="value">{format_bytes(peak_rss.rss_bytes)}</div>
        </div>
        <div class="stat">
            <div class="label">Final RSS</div>
            <div class="value">{format_bytes(last_rss.rss_bytes)}</div>
        </div>
        """

        slider_html = """
        <div class="slider">
            <label>RSS Time Offset: <span id="offsetVal">0.0s</span></label>
            <input type="range" id="offset" min="-60" max="60" step="0.5" value="0">
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f5f5f5; }}
        .stats {{ display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }}
        .stat {{ background: #fff; padding: 12px 16px; border-radius: 6px; }}
        .stat .label {{ font-size: 11px; color: #666; text-transform: uppercase; }}
        .stat .value {{ font-size: 20px; font-weight: bold; }}
        .chart {{ height: 400px; background: #fff; border-radius: 6px; padding: 15px; }}
        .slider {{ background: #fff; padding: 15px; border-radius: 6px; margin-bottom: 15px; }}
        .slider input {{ width: 300px; }}
    </style>
</head>
<body>
    <h2>{title}</h2>

    <div class="stats">
        <div class="stat">
            <div class="label">Initial Heap</div>
            <div class="value">{format_bytes(first.mem_heap_b)}</div>
        </div>
        <div class="stat">
            <div class="label">Peak Heap</div>
            <div class="value">{format_bytes(peak.mem_heap_b)}</div>
        </div>
        <div class="stat">
            <div class="label">Final Heap</div>
            <div class="value">{format_bytes(last.mem_heap_b)}</div>
        </div>
        {rss_stats}
    </div>

    {slider_html}

    <div class="chart">
        <canvas id="chart"></canvas>
    </div>

    <script>
        const heapData = {json.dumps(heap_data)};
        const rssData = {json.dumps(rss_data)};

        function buildDatasets(offset) {{
            const ds = [{{
                label: 'Heap (MB)',
                data: heapData,
                borderColor: 'rgb(75, 192, 192)',
                fill: false,
                pointRadius: 0,
            }}];
            if (rssData.length) {{
                ds.push({{
                    label: 'RSS (MB)',
                    data: rssData.map(p => ({{ x: p.x + offset, y: p.y }})),
                    borderColor: 'rgb(255, 99, 132)',
                    fill: false,
                    pointRadius: 0,
                }});
            }}
            return ds;
        }}

        const chart = new Chart(document.getElementById('chart'), {{
            type: 'scatter',
            data: {{ datasets: buildDatasets(0) }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Time (s)' }} }},
                    y: {{ title: {{ display: true, text: 'Memory (MB)' }}, beginAtZero: true }}
                }}
            }}
        }});

        const slider = document.getElementById('offset');
        const valDisplay = document.getElementById('offsetVal');
        if (slider) {{
            slider.oninput = function() {{
                const v = parseFloat(this.value);
                valDisplay.textContent = v.toFixed(1) + 's';
                chart.data.datasets = buildDatasets(v);
                chart.update('none');
            }};
        }}
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("massif_file", type=Path)
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--rss", type=Path)
    args = parser.parse_args()

    snapshots = parse_massif(args.massif_file)
    if not snapshots:
        print("No data", file=sys.stderr)
        sys.exit(1)

    rss_samples = None
    if args.rss and args.rss.exists():
        rss_samples = parse_rss_csv(args.rss)

    if args.html:
        title = args.massif_file.parent.name
        print(to_html(snapshots, rss_samples, title=title))
    else:
        peak = max(snapshots, key=lambda s: s.mem_heap_b)
        print(f"Snapshots: {len(snapshots)}")
        print(f"Peak heap: {format_bytes(peak.mem_heap_b)} at t={peak.time:.1f}s")
        print(f"Final heap: {format_bytes(snapshots[-1].mem_heap_b)}")


if __name__ == "__main__":
    main()
