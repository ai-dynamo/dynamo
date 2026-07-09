# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import contextlib
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .campaign import sample_p99_ms

def artifact_directory(args: argparse.Namespace) -> Path:
    if args.artifact_dir is not None:
        return args.artifact_dir
    if args.output is not None:
        return args.output.parent / f"{args.output.stem}-artifacts"
    return Path.cwd() / "valkey-module-saturation-artifacts"


def write_samples_csv(samples: Sequence[dict[str, Any]], path: Path) -> None:
    command_names = sorted(
        {
            command
            for sample in samples
            for command in (
                sample.get("latency", {}).keys()
                if isinstance(sample.get("latency"), dict)
                else ()
            )
        }
    )
    fieldnames = [
        "sample_index",
        "repetition",
        "status",
        "mode",
        "connections",
        "pipeline",
        "max_outstanding_commands_per_batch",
        "elapsed_s",
        "iterations",
        "iterations_per_s",
        "commands_per_s",
        "events_per_s",
        "blocks_per_s",
        "queries_per_s",
        "selections_per_s",
        "reservation_cycles_per_s",
        "logical_payload_mib_per_s",
        "total_wire_mib_per_s",
        "server_cpu_percent_of_one_core",
        "client_cpu_percent_of_one_core",
        "p99_ms_max",
        *[f"{command}_p99_ms" for command in command_names],
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, sample in enumerate(samples, start=1):
            telemetry = sample.get("telemetry")
            telemetry = telemetry if isinstance(telemetry, dict) else {}
            row: dict[str, Any] = {
                "sample_index": sample_index,
                "repetition": sample.get("repetition"),
                "status": sample.get("status"),
                "mode": sample.get("mode"),
                "connections": sample.get("connections"),
                "pipeline": sample.get("pipeline"),
                "max_outstanding_commands_per_batch": sample.get(
                    "max_outstanding_commands_per_batch"
                ),
                "elapsed_s": sample.get("elapsed_s"),
                "iterations": sample.get("iterations"),
                "iterations_per_s": sample.get("iterations_per_s"),
                "commands_per_s": sample.get("commands_per_s"),
                "events_per_s": sample.get("events_per_s"),
                "blocks_per_s": sample.get("blocks_per_s"),
                "queries_per_s": sample.get("queries_per_s"),
                "selections_per_s": sample.get("selections_per_s"),
                "reservation_cycles_per_s": sample.get("reservation_cycles_per_s"),
                "logical_payload_mib_per_s": sample.get("logical_payload_mib_per_s"),
                "total_wire_mib_per_s": sample.get("total_wire_mib_per_s"),
                "server_cpu_percent_of_one_core": telemetry.get(
                    "server_cpu_percent_of_one_core"
                ),
                "client_cpu_percent_of_one_core": telemetry.get(
                    "client_cpu_percent_of_one_core"
                ),
                "p99_ms_max": sample_p99_ms(sample),
            }
            latency = sample.get("latency")
            if isinstance(latency, dict):
                for command in command_names:
                    summary = latency.get(command)
                    if isinstance(summary, dict):
                        row[f"{command}_p99_ms"] = summary.get("p99_ms")
            writer.writerow(row)


def render_sweep_plots(summary: dict[str, Any], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = summary["points"]
    by_pipeline: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        by_pipeline.setdefault(int(point["pipeline"]), []).append(point)

    throughput_figure, throughput_axis = plt.subplots(figsize=(8.5, 5.2))
    for pipeline, values in sorted(by_pipeline.items()):
        values.sort(key=lambda value: int(value["connections"]))
        connections = [int(value["connections"]) for value in values]
        medians = [float(value["iterations_per_s_median"]) for value in values]
        minimums = [float(value["iterations_per_s_min"]) for value in values]
        maximums = [float(value["iterations_per_s_max"]) for value in values]
        throughput_axis.plot(
            connections, medians, marker="o", label=f"pipeline {pipeline}"
        )
        throughput_axis.fill_between(connections, minimums, maximums, alpha=0.14)
    throughput_axis.set_xlabel("TCP connections")
    throughput_axis.set_ylabel("Logical iterations/s")
    throughput_axis.set_title("dynkv single-server throughput vs concurrency")
    throughput_axis.grid(True, alpha=0.25)
    throughput_axis.legend()
    throughput_figure.tight_layout()
    throughput_png = output_dir / "throughput-vs-concurrency.png"
    throughput_svg = output_dir / "throughput-vs-concurrency.svg"
    throughput_figure.savefig(throughput_png, dpi=180)
    throughput_figure.savefig(throughput_svg)
    plt.close(throughput_figure)

    latency_figure, latency_axis = plt.subplots(figsize=(8.5, 5.2))
    for point in points:
        p99 = point.get("p99_ms_median_across_commands")
        if not isinstance(p99, int | float):
            continue
        throughput = float(point["iterations_per_s_median"])
        latency_axis.scatter(throughput, float(p99), s=42)
        latency_axis.annotate(
            f"c{point['connections']}/p{point['pipeline']}",
            (throughput, float(p99)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    latency_axis.set_xlabel("Logical iterations/s")
    latency_axis.set_ylabel("Worst command p99 latency (ms)")
    latency_axis.set_title("dynkv p99 latency vs throughput")
    latency_axis.grid(True, alpha=0.25)
    latency_figure.tight_layout()
    latency_png = output_dir / "p99-latency-vs-throughput.png"
    latency_svg = output_dir / "p99-latency-vs-throughput.svg"
    latency_figure.savefig(latency_png, dpi=180)
    latency_figure.savefig(latency_svg)
    plt.close(latency_figure)
    return {
        "throughput_png": str(throughput_png.resolve()),
        "throughput_svg": str(throughput_svg.resolve()),
        "latency_png": str(latency_png.resolve()),
        "latency_svg": str(latency_svg.resolve()),
    }


def emit_campaign_artifacts(
    result: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    samples = result.get("samples")
    if not isinstance(samples, list):
        return result
    output_dir = artifact_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "samples.csv"
    write_samples_csv(samples, csv_path)
    artifact_status: dict[str, Any] = {
        "status": "ok",
        "directory": str(output_dir.resolve()),
        "csv": str(csv_path.resolve()),
        "plots_generated": False,
    }
    for plot_path in (
        output_dir / "throughput-vs-concurrency.png",
        output_dir / "throughput-vs-concurrency.svg",
        output_dir / "p99-latency-vs-throughput.png",
        output_dir / "p99-latency-vs-throughput.svg",
    ):
        with contextlib.suppress(FileNotFoundError):
            plot_path.unlink()
    invalid = [
        index
        for index, sample in enumerate(samples, start=1)
        if not isinstance(sample, dict)
        or sample.get("status") != "ok"
        or not isinstance(sample.get("iterations_per_s"), int | float)
        or sample_p99_ms(sample) is None
    ]
    if invalid:
        artifact_status.update(
            {
                "status": "plots_suppressed",
                "reason": f"invalid or incomplete samples: {invalid}",
            }
        )
    else:
        try:
            artifact_status.update(render_sweep_plots(result["summary"], output_dir))
            artifact_status["plots_generated"] = True
        except ImportError as error:
            artifact_status.update(
                {
                    "status": "plots_suppressed",
                    "reason": f"matplotlib unavailable: {error}",
                }
            )
    result["artifacts"] = artifact_status
    return result


def sample_directory(
    data_root: Path | None, temporary_root: Path, sample_index: int
) -> Path:
    root = data_root or temporary_root
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"sample-{sample_index:03d}"
    directory.mkdir()
    return directory
