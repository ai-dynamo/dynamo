# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write aligned Markdown and CSV summaries for the nine-cell image sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

RUNTIMES = [
    ("vllm-serve", "vllm"),
    ("dynamo-native", "native"),
    ("dynamo-custom-encoder", "custom"),
]
RATES = (16, 24, 32)
METRICS = [
    ("TTFT avg (ms)", "time_to_first_token", "avg", False, 1),
    ("TTFT p50 (ms)", "time_to_first_token", "p50", False, 1),
    ("TTFT p90 (ms)", "time_to_first_token", "p90", False, 1),
    ("TTFT p99 (ms)", "time_to_first_token", "p99", False, 1),
    ("E2E latency avg (ms)", "request_latency", "avg", False, 1),
    ("E2E latency p50 (ms)", "request_latency", "p50", False, 1),
    ("E2E latency p90 (ms)", "request_latency", "p90", False, 1),
    ("E2E latency p99 (ms)", "request_latency", "p99", False, 1),
    ("Throughput (req/s)", "request_throughput", "avg", True, 3),
]


def _metric(data: dict[str, Any], name: str, statistic: str = "avg") -> float:
    return float(data[name][statistic])


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("profile_export_aiperf.json")):
        runtime = path.parents[1].name
        if runtime not in dict(RUNTIMES):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rate = int(data["input_config"]["loadgen"]["request_rate"])
        row: dict[str, Any] = {
            "runtime": runtime,
            "offered_qps": rate,
            "requests": _metric(data, "request_count"),
            "artifact": str(path.relative_to(root)),
            "command": str((path.parent / "command.txt").relative_to(root)),
        }
        for title, metric_name, statistic, _higher, _precision in METRICS:
            key = title.lower().replace(" ", "_").replace("/", "_per_")
            row[key] = _metric(data, metric_name, statistic)
        rows.append(row)
    return rows


def _table(
    title: str, field: str, rows: list[dict[str, Any]], higher: bool, precision: int
) -> str:
    by_key = {(row["runtime"], row["offered_qps"]): row for row in rows}
    lines = [f"  === {title} ==="]
    lines.append(f"{'Rate':>10}" + "".join(f"{label:>14}" for _, label in RUNTIMES))
    for rate in RATES:
        values = [float(by_key[(runtime, rate)][field]) for runtime, _ in RUNTIMES]
        best = max(values) if higher else min(values)
        cells = [
            f"{value:.{precision}f}{'*' if value == best else ''}" for value in values
        ]
        lines.append(f"{rate:>10.2f}" + "".join(f"{cell:>14}" for cell in cells))
    return "```text\n" + "\n".join(lines) + "\n```"


def _markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    metadata_path = root / "benchmark_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lines = [
        "# Qwen2.5-VL custom encoder image benchmark",
        "",
        "All cells use 1,000 streaming image requests, exact ISL 515, exact OSL 70, "
        "and one unique 500×500 JPEG per request. `*` marks the best runtime at a rate.",
        "",
        "## Runtime",
        "",
        f"- Model: `{metadata['model']}`",
        f"- Dynamo commit: `{metadata.get('dynamo_commit')}`",
        f"- Container image: `{metadata.get('container_image')}`",
        f"- vLLM version: `{metadata.get('vllm_version')}`",
        f"- Transformers version: `{metadata.get('transformers_version')}`",
        f"- PyTorch version: `{metadata.get('torch_version')}`",
        f"- AIPerf version: `{metadata.get('aiperf_version')}`",
        f"- GPU: `{metadata.get('gpu')}`",
        f"- CUDA_VISIBLE_DEVICES: `{metadata.get('cuda_visible_devices')}`",
        f"- Custom encoder: `{metadata['custom_encoder_class']}`",
        f"- ViT loading: {metadata['custom_encoder_load']}.",
        "- Custom preprocess cache: disabled; queue wait: 1 ms.",
        "",
        "## Results",
        "",
    ]
    for title, metric_name, statistic, higher, precision in METRICS:
        field = title.lower().replace(" ", "_").replace("/", "_per_")
        lines.extend([_table(title, field, rows, higher, precision), ""])

    by_key = {(row["runtime"], row["offered_qps"]): row for row in rows}
    lines.extend(
        [
            "## Artifacts",
            "",
            "| QPS | Runtime | AIPerf JSON | Exact command |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for rate in RATES:
        for runtime, label in RUNTIMES:
            row = by_key[(runtime, rate)]
            lines.append(
                f"| {rate} | {label} | [artifact]({row['artifact']}) | "
                f"[command]({row['command']}) |"
            )
    manifests = [
        root / "workload" / "workload_manifest.json",
        root.parent / "workload" / "workload_manifest.json",
    ]
    manifest = next((path for path in manifests if path.exists()), None)
    if manifest is not None:
        if manifest.is_relative_to(root):
            manifest_link = manifest.relative_to(root)
        else:
            manifest_link = Path("..") / manifest.relative_to(root.parent)
        lines.extend(
            [
                "",
                f"Workload manifest: [workload_manifest.json]({manifest_link})",
            ]
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
    if len(rows) != len(RUNTIMES) * len(RATES):
        raise SystemExit(f"expected nine benchmark cells, found {len(rows)}")

    args.markdown.write_text(_markdown(root, rows), encoding="utf-8")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"markdown={args.markdown}")
    print(f"csv={args.csv}")


if __name__ == "__main__":
    main()
