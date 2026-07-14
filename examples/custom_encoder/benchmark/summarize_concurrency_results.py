# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write Markdown and CSV summaries for the custom-encoder concurrency sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONCURRENCIES = (8, 16, 32)
RUNTIME = "dynamo-custom-encoder"
METRICS = [
    ("TTFT avg (ms)", "time_to_first_token", "avg", 1),
    ("TTFT p50 (ms)", "time_to_first_token", "p50", 1),
    ("TTFT p90 (ms)", "time_to_first_token", "p90", 1),
    ("TTFT p99 (ms)", "time_to_first_token", "p99", 1),
    ("E2E latency avg (ms)", "request_latency", "avg", 1),
    ("E2E latency p50 (ms)", "request_latency", "p50", 1),
    ("E2E latency p90 (ms)", "request_latency", "p90", 1),
    ("E2E latency p99 (ms)", "request_latency", "p99", 1),
    ("Throughput (req/s)", "request_throughput", "avg", 3),
]


def _field(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "_per_")


def _metric(data: dict[str, Any], name: str, statistic: str = "avg") -> float:
    return float(data[name][statistic])


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("profile_export_aiperf.json")):
        if path.parents[1].name != RUNTIME:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        concurrency = int(data["input_config"]["loadgen"]["concurrency"])
        row: dict[str, Any] = {
            "runtime": RUNTIME,
            "concurrency": concurrency,
            "requests": _metric(data, "request_count"),
            "artifact": str(path.relative_to(root)),
            "command": str((path.parent / "command.txt").relative_to(root)),
        }
        for title, metric_name, statistic, _precision in METRICS:
            row[_field(title)] = _metric(data, metric_name, statistic)
        rows.append(row)
    return sorted(rows, key=lambda row: row["concurrency"])


def _dispatch_counts(root: Path) -> dict[int, Counter[tuple[int, int]]]:
    log_path = root / "sweep.log"
    if not log_path.exists():
        return {}
    active: int | None = None
    counts: dict[int, Counter[tuple[int, int]]] = defaultdict(Counter)
    cell_pattern = re.compile(r"Config: .*\bconcurrency=(8|16|32)\b")
    dispatch_pattern = re.compile(
        r"custom_encoder_graph selected_bucket=(\d+) actual_cost=(\d+) batch_size=(\d+)"
    )
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        cell = cell_pattern.search(line)
        if cell:
            active = int(cell.group(1))
        dispatch = dispatch_pattern.search(line)
        if active is not None and dispatch:
            bucket, _cost, batch_size = (int(value) for value in dispatch.groups())
            counts[active][(batch_size, bucket)] += 1
    return dict(counts)


def _table(rows: list[dict[str, Any]]) -> str:
    headers = ["Concurrency"] + [title for title, *_rest in METRICS]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---:"] * len(headers)) + " |")
    for row in rows:
        cells = [str(row["concurrency"])]
        for title, _metric_name, _statistic, precision in METRICS:
            cells.append(f"{float(row[_field(title)]):.{precision}f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    metadata = json.loads(
        (root / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    settings = metadata["settings"]
    lines = [
        "# Qwen2.5 custom-encoder concurrency benchmark",
        "",
        "> **Performance-only adapter:** the complete Qwen2.5-VL-3B vision tower "
        "produces its native 2048-wide output, which is then truncated to the first "
        "1536 columns for the Qwen2.5-1.5B decoder. This is not a trained projection "
        "and the benchmark makes no quality or model-parity claim.",
        "",
        "Each cell uses 1,000 streaming requests, exact ISL 515, exact OSL 75, and "
        "a disjoint pool of unique 500×500 JPEGs between 50 and 60 KiB.",
        "",
        "## Runtime",
        "",
        f"- Decoder: `{metadata['decoder_model']}`",
        f"- Vision encoder: `{metadata['encoder_model']}`",
        f"- Dynamo commit: `{metadata.get('dynamo_commit')}`",
        f"- Container image: `{metadata.get('container_image')}`",
        f"- GPU: `{metadata.get('gpu')}`",
        f"- vLLM: `{metadata.get('vllm_version')}`; Transformers: "
        f"`{metadata.get('transformers_version')}`; PyTorch: "
        f"`{metadata.get('torch_version')}`; AIPerf: "
        f"`{metadata.get('aiperf_version')}`",
        f"- Preprocess concurrency: {settings['preprocess_concurrency']}; maximum "
        f"batch cost: {settings['max_batch_cost']}; queue wait: "
        f"{settings['queue_wait_ms']} ms",
        f"- CUDA graph buckets: `{settings['graph_buckets']}`; captured image shape: "
        f"`{settings['graph_image_sizes']}`",
        "- Preprocessing cache: disabled.",
        "- Bucket 64 is captured for the requested server configuration, but cannot "
        "be selected by this one-image-per-request sweep because client concurrency "
        "never exceeds 32.",
        "",
        "## Results",
        "",
        _table(rows),
        "",
    ]

    dispatch = _dispatch_counts(root)
    lines.extend(["## Observed graph dispatch", ""])
    if dispatch:
        lines.extend(
            [
                "| Concurrency | Maximum batch | Selected buckets | Dispatches by batch→bucket |",
                "| ---: | ---: | --- | --- |",
            ]
        )
        for concurrency in CONCURRENCIES:
            counter = dispatch.get(concurrency, Counter())
            maximum = max((batch for batch, _bucket in counter), default=0)
            buckets = sorted({bucket for _batch, bucket in counter})
            details = ", ".join(
                f"{batch}→{bucket}: {calls}"
                for (batch, bucket), calls in sorted(counter.items())
            )
            lines.append(
                f"| {concurrency} | {maximum} | `{buckets}` | {details or 'none'} |"
            )
    else:
        lines.append("Dispatch records were not available in `sweep.log`.")

    by_concurrency = {int(row["concurrency"]): row for row in rows}
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "| Concurrency | AIPerf JSON | Exact command |",
            "| ---: | --- | --- |",
        ]
    )
    for concurrency in CONCURRENCIES:
        row = by_concurrency[concurrency]
        lines.append(
            f"| {concurrency} | [artifact]({row['artifact']}) | "
            f"[command]({row['command']}) |"
        )
    workload_manifest = root.parent / "workload" / "workload_manifest.json"
    if workload_manifest.exists():
        lines.extend(
            [
                "",
                "Workload manifest: "
                "[workload_manifest.json](../workload/workload_manifest.json)",
            ]
        )
    for name, label in (
        ("graph_verification.log", "CUDA graph verification"),
        ("sweep.log", "Full sweep log"),
    ):
        if (root / name).exists():
            lines.append(f"- [{label}]({name})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = _load_rows(root)
    if len(rows) != len(CONCURRENCIES):
        raise SystemExit(f"expected three benchmark cells, found {len(rows)}")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
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
