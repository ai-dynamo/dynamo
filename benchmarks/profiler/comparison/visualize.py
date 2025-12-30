# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization for profiling method comparison."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.profiler.comparison.metrics import ComparisonResult

logger = logging.getLogger(__name__)


def create_cost_plot(comparison: ComparisonResult, output_path: Path):
    methods = [m.method_name for m in comparison.method_metrics]
    durations = [m.total_duration_seconds / 60 for m in comparison.method_metrics]
    gpu_hours = [m.gpu_hours_consumed for m in comparison.method_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    ax1.bar(methods, durations, color=colors, edgecolor="black")
    ax1.set_ylabel("Duration (minutes)")
    ax1.set_title("Profiling Duration", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(methods, gpu_hours, color=colors, edgecolor="black")
    ax2.set_ylabel("GPU-Hours")
    ax2.set_title("Resource Consumption", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_accuracy_plot(comparison: ComparisonResult, output_path: Path):
    validated = [m for m in comparison.method_metrics if m.validated]
    if not validated:
        return

    methods = [m.method_name for m in validated]
    errors = [abs(m.ttft_error_at_medium) if m.ttft_error_at_medium else 0 for m in validated]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#27ae60" if e < 20 else "#f39c12" if e < 50 else "#e74c3c" for e in errors]
    ax.bar(methods, errors, color=colors, edgecolor="black")
    ax.set_ylabel("TTFT Prediction Error (%)")
    ax.set_title("Predictive Accuracy (Medium Load)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_comparison(results_path: Path, output_dir: Optional[Path] = None):
    comparison = ComparisonResult.load(results_path)
    if output_dir is None:
        output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    create_cost_plot(comparison, output_dir / "profiling_cost.png")
    create_accuracy_plot(comparison, output_dir / "predictive_accuracy.png")
    logger.info(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    visualize_comparison(Path(args.results), Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()
