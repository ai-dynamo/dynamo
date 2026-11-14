# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, dependency-light decision benchmark reports."""

import argparse
import html
import json
from pathlib import Path

from dynamo_decision_perf.analysis import analyze_run, paired_differences


def _display(value):
    return (
        "unknown"
        if value is None
        else f"{value:.3f}"
        if isinstance(value, float)
        else str(value)
    )


def _cell(value):
    return html.escape(_display(value)).replace("|", "&#124;").replace("\n", " ")


def _chart(runs: list[dict], latency: bool) -> str:
    title = (
        "Request latency (ms)"
        if latency
        else "Offered versus completed HTTP requests/s"
    )
    width, height = 720, max(200, 90 + 55 * len(runs))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{title}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="15" y="24" font-family="sans-serif" font-size="16">{title}</text>',
    ]
    values = []
    for run in runs:
        summary = run["summary"]
        stats = summary["latency_ms"]["all_attempts"]
        values.append(
            (stats["p50"], stats["p95"], stats["p99"])
            if latency
            else (
                summary["rates"]["offered_http_requests_per_second"],
                summary["rates"]["completed_http_requests_per_second"],
                summary["rates"]["valid_http_requests_per_second"],
            )
        )
    maximum = (
        max((value for row in values for value in row if value is not None), default=1)
        or 1
    )
    labels = (
        ("p50", "p95 (n≥100)", "p99 (n≥1000)")
        if latency
        else ("offered", "HTTP completed", "schema valid")
    )
    colors = ("#5c6470", "#247cb0", "#287d50")
    for index, (run, row) in enumerate(zip(runs, values, strict=True)):
        y = 65 + index * 55
        name = html.escape(str(run["summary"]["metadata"].get("run_id", "unknown")))
        state = run["audit"]["status"]
        lines.append(
            f'<text x="15" y="{y}" font-family="sans-serif" font-size="11">{name} [{state}]</text>'
        )
        for offset, (value, color, label) in enumerate(
            zip(row, colors, labels, strict=True)
        ):
            bar = (value or 0) / maximum * 250
            row_y = y - 10 + offset * 13
            lines.append(
                f'<rect x="220" y="{row_y}" width="{bar:.3f}" height="9" fill="{color}"/>'
            )
            lines.append(
                f'<text x="490" y="{row_y + 9}" font-family="sans-serif" font-size="10">{label}: {_display(value)}</text>'
            )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_report(
    runs: list[dict], output_dir: str | Path, comparisons: list[dict] | None = None
) -> list[Path]:
    """Write derived artifacts outside every input raw-artifact directory."""
    output = Path(output_dir).resolve()
    for run in runs:
        raw = Path(run["audit"]["artifact_dir"]).resolve()
        if output == raw or output.is_relative_to(raw):
            raise ValueError("Report output must be separate from raw artifacts")
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decision API performance report",
        "",
        "Metric conformance is not an accuracy measurement. Invalid evidence remains visible and cannot support gain claims. No SLO-based goodput is inferred.",
        "",
        "Rates use the explicit profiling window, including failures and drain. Completed HTTP requests include HTTP errors; valid requests pass the decision schema; completed questions exclude refusals.",
        "",
        "| Run | Audit | Offered/s | HTTP completed/s | Valid/s | Questions/s | p50 ms | p95 ms | p99 ms | Max ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        summary, audit = run["summary"], run["audit"]
        rates, stats = summary["rates"], summary["latency_ms"]["all_attempts"]
        cells = [
            summary["metadata"].get("run_id"),
            audit["status"],
            rates["offered_http_requests_per_second"],
            rates["completed_http_requests_per_second"],
            rates["valid_http_requests_per_second"],
            rates["completed_questions_per_second"],
            stats["p50"],
            stats["p95"],
            stats["p99"],
            stats["max"],
        ]
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    for run in runs:
        summary, audit = run["summary"], run["audit"]
        lines.extend(
            [
                "",
                f"## {_cell(summary['metadata'].get('run_id'))}",
                "",
                f"Cost boundary: {_cell(summary['metadata'].get('cost_boundary'))}. Window: {_cell(summary['measurement_seconds'])} seconds; source: {_cell(summary['metadata'].get('measurement_window_source'))}.",
                "",
                "```json",
                json.dumps(
                    {
                        "counts": summary["counts"],
                        "api_usage": summary["api_usage"],
                        "actual_work": summary["actual_work"],
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ),
                "```",
                "",
            ]
        )
        lines.extend(
            f"- {_cell(item)}" for item in audit["blockers"] + audit["limitations"]
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired observations",
                "",
                "Differences are computed per matched logical request before summarizing. These are end-to-end cost differences, not subtraction of percentiles or isolated frontend overhead.",
                "",
                "```json",
                json.dumps(comparisons, indent=2, sort_keys=True, allow_nan=False),
                "```",
            ]
        )
    content = {
        "report.json": json.dumps(
            {"runs": runs, "comparisons": comparisons or []},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        "report.md": "\n".join(lines) + "\n",
        "latency.svg": _chart(runs, True),
        "saturation.svg": _chart(runs, False),
    }
    paths = []
    for name, text in content.items():
        target = output / name
        if target.is_symlink():
            raise ValueError(f"Refusing symlink report output: {target}")
        target.write_text(text, encoding="utf-8")
        paths.append(target)
    for index, run in enumerate(runs):
        for name in ("audit", "summary"):
            target = output / f"run-{index:04d}-{name}.json"
            if target.is_symlink():
                raise ValueError(f"Refusing symlink report output: {target}")
            target.write_text(
                json.dumps(run[name], indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            paths.append(target)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Audit one or more frozen runs and retain invalid evidence in the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--workload", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len({len(args.artifacts), len(args.metadata), len(args.workload)}) != 1:
        parser.error("Repeat --artifacts, --metadata, and --workload equally")
    runs = [
        analyze_run(
            artifacts,
            json.loads(metadata.read_text()),
            json.loads(workload.read_text()),
        )
        for artifacts, metadata, workload in zip(
            args.artifacts, args.metadata, args.workload, strict=True
        )
    ]
    comparisons = [paired_differences(runs[0], run) for run in runs[1:]]
    paths = write_report(runs, args.output, comparisons)
    print(f"Wrote {len(paths)} derived artifacts to {args.output}")
    return 0 if all(run["audit"]["status"] == "valid" for run in runs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
