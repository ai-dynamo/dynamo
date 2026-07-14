# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write Markdown and CSV summaries for the custom-encoder ablation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.custom_encoder.benchmark.run_ablation import VARIANTS  # noqa: E402
from examples.custom_encoder.benchmark.run_image_sweep import RATES  # noqa: E402


def _metric(data: dict[str, Any], name: str, statistic: str = "avg") -> float:
    return float(data[name][statistic])


def _load_rows(root: Path) -> list[dict[str, Any]]:
    labels = {variant[0] for variant in VARIANTS}
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("profile_export_aiperf.json")):
        label = path.parents[1].name
        if label not in labels:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = next(value for value in VARIANTS if value[0] == label)
        rows.append(
            {
                "variant": label,
                "offered_qps": int(data["input_config"]["loadgen"]["request_rate"]),
                "graph_buckets": variant[1],
                "max_batch_cost": variant[2],
                "cuda_graphs_disabled": variant[3],
                "ttft_avg_ms": _metric(data, "time_to_first_token"),
                "ttft_p99_ms": _metric(data, "time_to_first_token", "p99"),
                "e2e_avg_ms": _metric(data, "request_latency"),
                "e2e_p99_ms": _metric(data, "request_latency", "p99"),
                "throughput_req_s": _metric(data, "request_throughput"),
                "artifact": str(path.relative_to(root)),
                "command": str((path.parent / "command.txt").relative_to(root)),
            }
        )
    return rows


def _delta(value: float, baseline: float) -> str:
    return f"{(value / baseline - 1) * 100:+.1f}%"


def _markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    by_key = {(row["variant"], row["offered_qps"]): row for row in rows}
    lines = [
        "# Qwen2.5-VL custom encoder ablation",
        "",
        "All cells use 1,000 streaming requests, exact ISL 515, exact OSL 70, "
        "and the same unique 500×500 JPEG pools. Deltas are relative to the full "
        "`1,2,4,8` graph ladder; lower latency and higher throughput are better.",
        "",
    ]
    for rate in RATES:
        baseline = by_key[("custom-graph-full", rate)]
        lines.extend(
            [
                f"## Offered QPS {rate}",
                "",
                "| Variant | Graphs | Buckets | Max batch | TTFT avg | Δ | E2E avg | Δ | Throughput | Δ |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, buckets, max_batch_cost, disabled in VARIANTS:
            row = by_key[(label, rate)]
            lines.append(
                f"| {label} | {'off' if disabled else 'on'} | {buckets} | "
                f"{max_batch_cost} | {row['ttft_avg_ms']:.1f} ms | "
                f"{_delta(row['ttft_avg_ms'], baseline['ttft_avg_ms'])} | "
                f"{row['e2e_avg_ms']:.1f} ms | "
                f"{_delta(row['e2e_avg_ms'], baseline['e2e_avg_ms'])} | "
                f"{row['throughput_req_s']:.3f} req/s | "
                f"{_delta(row['throughput_req_s'], baseline['throughput_req_s'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "| QPS | Variant | AIPerf JSON | Exact command |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for rate in RATES:
        for label, _buckets, _max_batch_cost, _disabled in VARIANTS:
            row = by_key[(label, rate)]
            lines.append(
                f"| {rate} | {label} | [artifact]({row['artifact']}) | "
                f"[command]({row['command']}) |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = _load_rows(root)
    expected = len(VARIANTS) * len(RATES)
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} ablation cells, found {len(rows)}")
    args.markdown.write_text(_markdown(root, rows), encoding="utf-8")
    with args.csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"markdown={args.markdown}")
    print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
