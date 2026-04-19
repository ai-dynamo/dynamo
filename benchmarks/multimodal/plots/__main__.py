# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generic aiperf result plotter.
# Usage: python -m benchmarks.multimodal.plots <dataset_dir> [<dataset_dir2> ...]
#
import json
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt

from benchmarks.multimodal.plots.args import PlotArgs, parse_args

JSON_FILENAME = "profile_export_aiperf.json"
PLOTS_DIR_NAME = "plots"
COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]

SUBPLOT_METRICS: list[tuple[str, str, list[str]]] = [
    ("request_latency", "Request Latency", ["avg", "p50", "p90", "p99"]),
    ("time_to_first_token", "TTFT", ["avg", "p50", "p90", "p99"]),
    ("inter_token_latency", "ITL", ["avg", "p50", "p90", "p99"]),
]

SINGLE_METRICS: list[tuple[str, str, str]] = [
    ("request_throughput", "avg", "Request Throughput (req/s)"),
    ("output_token_throughput", "avg", "Output Token Throughput (tok/s)"),
]


def parse_x_value(name: str) -> int:
    """Strip non-numeric prefix and return integer value."""
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse numeric value from '{name}'")


def detect_x_label(dir_names: list[str]) -> str:
    """Auto-detect x-axis label from the common alphabetic prefix of dir names."""
    prefixes: set[str] = set()
    for name in dir_names:
        match = re.match(r"([A-Za-z_]+)", name)
        if match:
            prefixes.add(match.group(1))
    if len(prefixes) == 1:
        return prefixes.pop()
    return "X"


def discover_data(
    dataset_dir: Path,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Scan dataset_dir and return {line_name: {x_value: json_data}}."""
    results: dict[str, dict[int, dict[str, Any]]] = {}
    for subdir in sorted(dataset_dir.iterdir()):
        if not subdir.is_dir() or subdir.name == PLOTS_DIR_NAME:
            continue
        line_data: dict[int, dict[str, Any]] = {}
        for x_dir in sorted(subdir.iterdir()):
            if not x_dir.is_dir():
                continue
            json_file = x_dir / JSON_FILENAME
            if not json_file.exists():
                continue
            x_val = parse_x_value(x_dir.name)
            with open(json_file) as f:
                line_data[x_val] = json.load(f)
        if line_data:
            results[subdir.name] = line_data
    return results


def common_x_values(results: dict[str, dict[int, Any]]) -> list[int]:
    """Return sorted x-values present in ALL lines."""
    sets = [set(data.keys()) for data in results.values()]
    if not sets:
        return []
    return sorted(sets[0].intersection(*sets[1:]))


def get_raw_x_names(dataset_dir: Path, line_name: str) -> list[str]:
    """Get raw directory names for x-axis label detection."""
    line_dir = dataset_dir / line_name
    names: list[str] = []
    for x_dir in sorted(line_dir.iterdir()):
        if x_dir.is_dir() and (x_dir / JSON_FILENAME).exists():
            names.append(x_dir.name)
    return names


def plot_single(
    results: dict[str, dict[int, dict[str, Any]]],
    x_values: list[int],
    key: str,
    stat: str,
    ylabel: str,
    output_path: Path,
    x_label: str,
) -> None:
    """Plot a single-chart metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (line_name, line_data) in enumerate(sorted(results.items())):
        values = [line_data[x].get(key, {}).get(stat) for x in x_values]
        ax.plot(
            x_values,
            values,
            marker="o",
            label=line_name,
            color=COLORS[i % len(COLORS)],
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(ylabel, fontsize=14)
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_subplot_grid(
    results: dict[str, dict[int, dict[str, Any]]],
    x_values: list[int],
    key: str,
    title_prefix: str,
    stats: list[str],
    output_path: Path,
    x_label: str,
) -> None:
    """Plot a 2x2 subplot grid for a metric with multiple stats."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    flat_axes = axes.flatten()

    unit = "req/s" if "throughput" in key else "ms"

    for idx, stat in enumerate(stats):
        ax = flat_axes[idx]
        for i, (line_name, line_data) in enumerate(sorted(results.items())):
            values = [line_data[x].get(key, {}).get(stat) for x in x_values]
            ax.plot(
                x_values,
                values,
                marker="o",
                label=line_name,
                color=COLORS[i % len(COLORS)],
                linewidth=2,
                markersize=6,
            )
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(f"{title_prefix} {stat} ({unit})", fontsize=10)
        ax.set_title(f"{title_prefix} {stat}", fontsize=12)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title_prefix, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {output_path}")


def process_dataset(dataset_dir: Path, output_dir: Optional[Path]) -> None:
    """Process a single dataset directory and generate all plots."""
    print(f"\nDataset: {dataset_dir}")
    results = discover_data(dataset_dir)
    if not results:
        print("  No results found, skipping.")
        return

    x_values = common_x_values(results)
    if not x_values:
        print("  No common x-axis values across lines, skipping.")
        return

    first_line = next(iter(results))
    raw_names = get_raw_x_names(dataset_dir, first_line)
    x_label = detect_x_label(raw_names)

    if output_dir is not None:
        plot_dir = output_dir
    else:
        plot_dir = dataset_dir / PLOTS_DIR_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)

    for key, stat, ylabel in SINGLE_METRICS:
        plot_single(
            results, x_values, key, stat, ylabel, plot_dir / f"{key}.png", x_label
        )

    for key, title_prefix, stats in SUBPLOT_METRICS:
        plot_subplot_grid(
            results,
            x_values,
            key,
            title_prefix,
            stats,
            plot_dir / f"{key}.png",
            x_label,
        )


def is_dataset_dir(path: Path) -> bool:
    """Check if path is a dataset dir (has line_name/x_value/profile_export_aiperf.json)."""
    if not path.is_dir():
        return False
    for subdir in path.iterdir():
        if not subdir.is_dir() or subdir.name == PLOTS_DIR_NAME:
            continue
        for x_dir in subdir.iterdir():
            if x_dir.is_dir() and (x_dir / JSON_FILENAME).exists():
                return True
    return False


def find_dataset_dirs(root: Path) -> list[Path]:
    """Recursively find all dataset dirs under root, skipping plots dirs."""
    found: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name == PLOTS_DIR_NAME:
            continue
        if is_dataset_dir(entry):
            found.append(entry)
        else:
            found.extend(find_dataset_dirs(entry))
    return found


def main() -> None:
    args: PlotArgs = parse_args()

    for input_dir in args.dataset_dirs:
        if not input_dir.is_dir():
            print(f"Warning: '{input_dir}' is not a directory, skipping.")
            continue

        if is_dataset_dir(input_dir):
            dataset_dirs = [input_dir]
        else:
            dataset_dirs = find_dataset_dirs(input_dir)
            if not dataset_dirs:
                print(f"Warning: no dataset dirs found under '{input_dir}'.")

        for dataset_dir in dataset_dirs:
            if args.output_dir is not None:
                try:
                    rel = dataset_dir.relative_to(input_dir)
                except ValueError:
                    rel = Path(dataset_dir.name)
                output_dir = args.output_dir / rel
            else:
                output_dir = None
            process_dataset(dataset_dir, output_dir)


if __name__ == "__main__":
    main()
