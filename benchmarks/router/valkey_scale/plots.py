# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import finite_number

def make_plot(
    summaries: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    baseline_count: int,
    concurrency: int,
    isl: int,
    osl: int,
    mocker_processes: int,
) -> tuple[Path, Path]:
    """Create median-RPS error-bar and ideal-linear-scale PNG/SVG artifacts."""

    # Keep import-time dependency optional for --help and post-failure artifact
    # inspection; plotting is required only after a fully valid scale sweep.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts: list[int] = []
    medians: list[float] = []
    lower_error: list[float] = []
    upper_error: list[float] = []
    for summary in summaries:
        count = summary.get("frontend_count")
        median_rps = finite_number(summary.get("request_throughput_rps_median"))
        p25 = finite_number(summary.get("request_throughput_rps_p25"))
        p75 = finite_number(summary.get("request_throughput_rps_p75"))
        if (
            not isinstance(count, int)
            or median_rps is None
            or p25 is None
            or p75 is None
        ):
            raise ValueError("cannot plot an incomplete frontend-count summary")
        counts.append(count)
        medians.append(median_rps)
        lower_error.append(max(0.0, median_rps - p25))
        upper_error.append(max(0.0, p75 - median_rps))

    baseline_index = counts.index(baseline_count)
    baseline_rps = medians[baseline_index]
    ideal = [baseline_rps * count / baseline_count for count in counts]

    figure, (full_axis, zoom_axis) = plt.subplots(
        1, 2, figsize=(13, 5.25), constrained_layout=True
    )

    def plot_measured(axis: Any) -> None:
        axis.errorbar(
            counts,
            medians,
            yerr=[lower_error, upper_error],
            marker="o",
            capsize=5,
            linewidth=2,
            color="#1f77b4",
            label="Valkey HA authoritative (median; IQR bars)",
        )
        axis.set_xticks(counts)
        axis.set_xlabel("Frontend processes")
        axis.set_ylabel("Request throughput (RPS)")
        axis.grid(axis="y", alpha=0.25)

    plot_measured(full_axis)
    full_axis.plot(
        counts,
        ideal,
        linestyle="--",
        linewidth=1.75,
        color="#6c757d",
        label=f"Ideal linear scaling from {baseline_count} frontend(s)",
    )
    full_axis.set_title("Full scale, including ideal linear growth")
    full_axis.legend(loc="best")

    plot_measured(zoom_axis)
    observed_low = min(median - error for median, error in zip(medians, lower_error))
    observed_high = max(median + error for median, error in zip(medians, upper_error))
    observed_span = max(observed_high - observed_low, 1.0)
    zoom_margin = max(5.0, observed_span * 0.3)
    zoom_axis.set_ylim(observed_low - zoom_margin, observed_high + zoom_margin)
    zoom_axis.set_title("Measured range (zoomed)")
    for count, median_rps in zip(counts, medians):
        improvement = (median_rps / baseline_rps - 1.0) * 100.0
        zoom_axis.annotate(
            f"{median_rps:.1f} RPS\n{improvement:+.1f}%",
            (count, median_rps),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    figure.suptitle(
        "Valkey-router frontend scaling\n"
        f"c={concurrency:,}, ISL={isl:,}, OSL={osl:,}; "
        f"four logical mock workers / {mocker_processes} OS process(es)"
    )

    png_path = output_dir / "rps-vs-frontends.png"
    svg_path = output_dir / "rps-vs-frontends.svg"
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    return png_path, svg_path
