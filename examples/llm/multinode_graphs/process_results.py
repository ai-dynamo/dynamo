#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To run this script you need to install:
# pip install pandas matplotlib seaborn

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, MultipleLocator, ScalarFormatter

LOGGER = logging.getLogger(__name__)


_FOOTNOTE_KWARGS = {"fontsize": 7, "ha": "right", "va": "bottom", "color": "gray"}
_FOOTNOTE_BOTTOM_SPACE = 0.15
_FOOTNOTE_X_POSITION = 0.99
_FOOTNOTE_Y_POSITION = 0.01

_LEGEND_KWARGS = {"title": "Legend", "bbox_to_anchor": (1.02, 1), "loc": "upper left"}
_FIGURE_SIZE = (8, 6)


def parse_tp_dp(name):
    """
    Searches the folder name for any occurrence(s) of _tpXdpY,
    sums up X*Y for all matches, and returns that as the total GPU count.
    """
    matches = re.findall(r"_tp(\d+)dp(\d+)", name)
    total_gpus = 0
    for tp_str, dp_str in matches:
        total_gpus += int(tp_str) * int(dp_str)
    return total_gpus


def get_label_from_name(name):
    """
    Parses out a human-friendly label from the directory name.
    For example, 'purevllm_tp1dp1' -> 'purevllm'
    'rustvllm_tp2dp4' -> 'rustvllm'
    'context_tp2dp2' -> 'context' (you could replace 'context' with 'disagg' if desired)
    """
    # If you want to special-case certain strings (e.g. rename 'context' -> 'disagg'),
    # you can do so here:
    base_match = re.match(r"^(.*?)(_tp\d+dp\d+)+$", name)
    if base_match:
        prefix = base_match.group(1)
        # Example: prefix = prefix.replace("context", "disagg")
        return prefix
    else:
        # If we don't match at all, just return the whole name
        # (useful if there's no _tpXdpY in the folder name)
        return name


def get_latest_run_dirs(base_path):
    latest_run_dirs = defaultdict(list)

    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            concurrency_dirs = [
                d for d in os.listdir(subdir_path) if d.startswith("concurrency_")
            ]
            valid_dirs = defaultdict(list)
            for d in concurrency_dirs:
                concurrency = d.split("_")[1]
                json_path = os.path.join(
                    subdir_path, d, "profile_export_genai_perf.json"
                )
                if os.path.exists(json_path):
                    valid_dirs[concurrency].append(d)
            for valid_dir in valid_dirs.values():
                latest_dir = max(
                    valid_dir,
                    key=lambda d: datetime.strptime(
                        d.split("_")[2] + d.split("_")[3], "%Y%m%d%H%M%S"
                    ),
                )
                concurrency = latest_dir.split("_")[1]
                latest_run_dirs[subdir].append(latest_dir)
    return latest_run_dirs


def extract_val_and_concurrency(
    base_path, latest_run_dirs, stat_value="avg", output_tokens_per_request=None
):
    results = []
    for subdir, latest_dirs in latest_run_dirs.items():
        for latest_dir in latest_dirs:
            json_path = os.path.join(
                base_path,
                subdir,
                latest_dir,
                "profile_export_genai_perf.json",
            )
            with open(json_path, "r") as f:
                data = json.load(f)
                # output_token_throughput contains only avg
                output_token_throughput = data.get("output_token_throughput", {}).get(
                    "avg"
                )

                throughput_dictionary = data.get("output_token_throughput_per_user", {})
                # The graph should represent throughput per request but it must relate to latency percentiles not to throughput percentiles
                # The latency percentiles 90 means that throughput percentiles are 10
                # Let's modify statistical value to use such opposite statistical value
                # If stat value starts with p, we need to use 100 - value
                # Otherwise we need to use value
                if stat_value.startswith("p"):
                    int_percentile_value = 100 - int(stat_value.lstrip("p"))
                    adjusted_stat_value = f"p{int_percentile_value}"
                else:
                    adjusted_stat_value = stat_value
                # Replace min with max and max with min
                if stat_value == "min":
                    adjusted_stat_value = "max"
                elif stat_value == "max":
                    adjusted_stat_value = "min"
                # genai-perf produces only percentiles p25, p50, p75, p90, p95, p99 but not p1, p5 and p10 for all variables
                # We need to check if it still the issue in produced data and instead used latency percentiles and estimate throughput per request
                if adjusted_stat_value not in throughput_dictionary:
                    LOGGER.warning(
                        f"Stat value {adjusted_stat_value} not found in throughput_dictionary"
                    )
                    if output_tokens_per_request is None:
                        raise ValueError(
                            f"Stat value {adjusted_stat_value} not found in throughput_dictionary and output_tokens_per_request is not provided"
                        )
                    request_latency_dict = data.get("request_latency", {})
                    # Check if latency is in milliseconds
                    assert request_latency_dict["unit"] == "ms"
                    # Latency stat value don't have to be adjusted because it is already representing latency percentiles
                    request_latency = request_latency_dict.get(stat_value)
                    # Output tokens per request can be only used if genai-perf is executed without standard deviation for this value
                    # If all request have enforced the same output tokens per request, we can use it
                    output_token_throughput_per_request = (
                        output_tokens_per_request / request_latency * 1000
                    )
                else:
                    output_token_throughput_per_request = throughput_dictionary.get(
                        adjusted_stat_value
                    )
                time_to_first_token = data.get("time_to_first_token", {}).get(
                    stat_value
                )
                inter_token_latency = data.get("inter_token_latency", {}).get(
                    stat_value
                )
                # request_throughput contains only avg
                request_throughput = data.get("request_throughput", {}).get("avg")

            concurrency = latest_dir.split("_")[1]
            num_gpus = parse_tp_dp(subdir)

            # Handle the case of num_gpus=0 to avoid division by zero
            if num_gpus > 0 and output_token_throughput is not None:
                output_token_throughput_per_gpu = output_token_throughput / num_gpus
            else:
                output_token_throughput_per_gpu = 0.0

            if num_gpus > 0 and request_throughput is not None:
                request_throughput_per_gpu = request_throughput / num_gpus
            else:
                request_throughput_per_gpu = 0.0

            results.append(
                {
                    "configuration": subdir,
                    "num_gpus": num_gpus,
                    "concurrency": float(concurrency),
                    "output_token_throughput_avg": output_token_throughput,
                    f"output_token_throughput_per_request_{stat_value}": output_token_throughput_per_request,
                    "output_token_throughput_per_gpu_avg": output_token_throughput_per_gpu,
                    f"time_to_first_token_{stat_value}": time_to_first_token,
                    f"inter_token_latency_{stat_value}": inter_token_latency,
                    "request_throughput_per_gpu_avg": request_throughput_per_gpu,
                }
            )
    return results


def create_graph(base_path, results, title, stat_value, footnote):
    # Build data for plotting
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r[f"output_token_throughput_per_request_{stat_value}"],
            "y": r["output_token_throughput_per_gpu_avg"],
        }
        for r in results
        if r.get(f"output_token_throughput_per_request_{stat_value}") is not None
        and r.get("output_token_throughput_per_gpu_avg") is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 8))

    # Plot each label's points
    for label, group in df.groupby("label"):
        # If the label contains 'prefill' (case-insensitive), use a dashed line
        if "prefill" in label.lower():
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="-")
        else:
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="--")

    plt.legend(title="Legend", bbox_to_anchor=(0.75, 1), loc="upper left")

    # Decide what the x-axis label should be
    if stat_value.startswith("p") or stat_value in ["min", "max"]:
        plt.xlabel(f"tokens/s/user for latency {stat_value}")
    else:
        plt.xlabel(f"tokens/s/user {stat_value}")

    # Throughput contains only avg
    plt.ylabel("tokens/s/gpu avg")
    plt.title(f"Throughput vs. tokens per user {title}")

    # Fine-tune ticks
    ax = plt.gca()
    x_interval = 5  # Adjust as needed
    y_interval = 10
    ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    ax.yaxis.set_major_locator(MultipleLocator(y_interval))
    ax.set_ylim(bottom=0)

    # Adjust the bottom margin to make room for the footnote
    # plt.tight_layout()
    plt.subplots_adjust(bottom=_FOOTNOTE_BOTTOM_SPACE)
    if footnote:
        plt.figtext(
            _FOOTNOTE_X_POSITION, _FOOTNOTE_Y_POSITION, footnote, **_FOOTNOTE_KWARGS
        )

    # Save the plot to a file
    if stat_value == "avg":
        file_name = f"{base_path}/plot.png"
    else:
        file_name = f"{base_path}/plot_{stat_value}.png"

    plt.savefig(file_name, dpi=300)
    plt.close()


def create_itl_graph(base_path, results, title, stat_value, footnote):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["concurrency"],
            "y": r[f"inter_token_latency_{stat_value}"],
        }
        for r in results
        if r[f"inter_token_latency_{stat_value}"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)

    # Plot each label's points
    for label, group in df.groupby("label"):
        if "prefill" in label.lower():
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="-")
        else:
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="--")

    plt.legend(**_LEGEND_KWARGS)
    plt.xlabel("concurrency")
    plt.ylabel(f"inter_token_latency {stat_value}")
    plt.title(f"Inter-token latency {title}")

    # Use log base 2 on both axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    # Major ticks at exact powers of 2, e.g. 1, 2, 4, 8, 16, ...
    ax.xaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=10))

    # Minor ticks at subdivisions between powers of 2
    # (you can adjust subs=(0.25, 0.5, 0.75) or use different fractions to get finer/coarser spacing)
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=(0.25, 0.5, 0.75), numticks=50))
    ax.yaxis.set_minor_locator(LogLocator(base=2, subs=(0.25, 0.5, 0.75), numticks=50))

    # Format tick labels as standard numbers (no scientific notation)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    # Enable grid lines for both major and minor ticks
    ax.grid(True, which="both", linestyle="--", alpha=0.7)

    # Adjust the bottom margin to make room for the footnote
    plt.tight_layout()
    plt.subplots_adjust(bottom=_FOOTNOTE_BOTTOM_SPACE)
    if footnote:
        plt.figtext(
            _FOOTNOTE_X_POSITION, _FOOTNOTE_Y_POSITION, footnote, **_FOOTNOTE_KWARGS
        )

    # Save figure
    if stat_value == "avg":
        file_name = f"{base_path}/plot_itl.png"
    else:
        file_name = f"{base_path}/plot_itl_{stat_value}.png"

    plt.savefig(file_name, dpi=300)
    plt.close()


def create_ttft_graph(base_path, results, title, stat_value, footnote):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["concurrency"],
            "y": r[f"time_to_first_token_{stat_value}"],
        }
        for r in results
        if r[f"time_to_first_token_{stat_value}"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 6), constrained_layout=True)

    # Plot each label's points
    for label, group in df.groupby("label"):
        if "prefill" in label.lower():
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="-")
        else:
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="--")

    plt.legend(**_LEGEND_KWARGS)
    plt.xlabel("concurrency")
    plt.ylabel(f"time_to_first_token {stat_value}")
    plt.title(f"Time to first token {title}")

    ax = plt.gca()

    # Use log base 2 on both axes
    ax.set_xscale("log", base=2)

    # Major ticks at exact powers of 2, e.g. 1, 2, 4, 8, 16, ...
    ax.xaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=10))

    # Minor ticks at subdivisions between powers of 2
    # (you can adjust subs=(0.25, 0.5, 0.75) or use different fractions to get finer/coarser spacing)
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=(0.25, 0.5, 0.75), numticks=50))

    # Format tick labels as standard numbers (no scientific notation)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.set_yscale("log")

    # Adjust the bottom margin to make room for the footnote
    plt.tight_layout()
    plt.subplots_adjust(bottom=_FOOTNOTE_BOTTOM_SPACE)
    if footnote:
        plt.figtext(
            _FOOTNOTE_X_POSITION, _FOOTNOTE_Y_POSITION, footnote, **_FOOTNOTE_KWARGS
        )

    if stat_value == "avg":
        file_name = f"{base_path}/plot_ttft.png"
    else:
        file_name = f"{base_path}/plot_ttft_{stat_value}.png"

    plt.savefig(file_name, dpi=300)
    plt.close()


def create_req_graph(base_path, results, title, footnote):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["concurrency"],
            "y": r["request_throughput_per_gpu_avg"],
        }
        for r in results
        if r["request_throughput_per_gpu_avg"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 6), constrained_layout=True)
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)

    # Plot each label's points
    for label, group in df.groupby("label"):
        if "prefill" in label.lower():
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="-")
        else:
            plt.plot(group["x"], group["y"], marker="o", label=label, linestyle="--")

    plt.legend(**_LEGEND_KWARGS)
    plt.xlabel("concurrency")
    # Throughput contains only avg
    plt.ylabel("request_throughput_per_gpu avg")
    plt.title(f"Throughput {title}")

    # Use log base 2 on both axes
    ax.set_xscale("log", base=2)

    # Major ticks at exact powers of 2, e.g. 1, 2, 4, 8, 16, ...
    ax.xaxis.set_major_locator(LogLocator(base=2, subs=(1.0,), numticks=10))

    # Minor ticks at subdivisions between powers of 2
    # (you can adjust subs=(0.25, 0.5, 0.75) or use different fractions to get finer/coarser spacing)
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=(0.25, 0.5, 0.75), numticks=50))

    # Format tick labels as standard numbers (no scientific notation)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # Adjust the bottom margin to make room for the footnote
    plt.tight_layout()
    plt.subplots_adjust(bottom=_FOOTNOTE_BOTTOM_SPACE)
    if footnote:
        plt.figtext(
            _FOOTNOTE_X_POSITION, _FOOTNOTE_Y_POSITION, footnote, **_FOOTNOTE_KWARGS
        )

    file_name = f"{base_path}/plot_req.png"

    plt.savefig(file_name, dpi=300)
    plt.close()


def create_pareto_graph(
    base_path, results, title, stat_value, footnote, show_all_points=False
):
    """
    Creates a Pareto frontier plot. The plot can optionally show all data points if `show_all_points=True`.

    :param base_path: Path where CSV and plot will be saved.
    :param results: List of result dictionaries containing metrics.
    :param title: Title to be displayed on the plot.
    :param stat_value: Statistic to focus on (avg, min, max, pX, etc.).
    :param footnote: Optional text to place at the bottom of the figure.
    :param show_all_points: If True, scatter-plot all data points in addition to the Pareto frontier.
    """

    data_points = [
        {
            "label": result["configuration"].split("_")[0].replace("context", "disagg"),
            "configuration": result["configuration"],
            "concurrency": float(result["concurrency"]),
            f"output_token_throughput_per_request_{stat_value}": result[
                f"output_token_throughput_per_request_{stat_value}"
            ],
            "output_token_throughput_per_gpu_avg": result[
                "output_token_throughput_per_gpu_avg"
            ],
            f"time_to_first_token_{stat_value}": result[
                f"time_to_first_token_{stat_value}"
            ],
            f"inter_token_latency_{stat_value}": result[
                f"inter_token_latency_{stat_value}"
            ],
            "is_pareto_efficient": False,
        }
        for result in results
    ]

    # Load data into a pandas DataFrame
    df = pd.DataFrame(data_points)

    # Function to find Pareto-efficient points
    def pareto_efficient(ids, points):
        points = np.array(points)
        pareto_points = []
        for i, (point_id, point) in enumerate(zip(ids, points)):
            dominated = False
            for j, other_point in enumerate(points):
                if i != j and all(other_point >= point):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append(point)
                df.at[point_id, "is_pareto_efficient"] = True
        return np.array(pareto_points)

    # Create the plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    labels = df["label"].unique()

    for label in labels:
        group = df[df["label"] == label].copy()

        # Determine color and linestyle based on label
        label_lower = label.lower()
        if "prefill" in label_lower:
            # Disaggregated -> dotted line
            linestyle = (0, (1, 1))  # dotted
            if "prefill" in label_lower:
                # NVIDIA green for disaggregation + prefill
                color = "#76B900"
            else:
                color = "black"
        else:
            # Baseline (non-disaggregated) -> solid line
            if "vllm" not in label_lower:
                linestyle = (0, (1, 1))  # dotted
            else:
                linestyle = "solid"
            color = "black"

        # Beautify labels
        label_nice_dict = {
            "prefill": "Dynamo disaggregated",
            "dynamomonolithic": "Dynamo aggregated",
            "vllm": "OSS vLLM aggregated",
        }
        if label in label_nice_dict:
            label_beautified = label_nice_dict[label]
        else:
            label_beautified = label

        # Optionally scatter all points
        if show_all_points:
            plt.scatter(
                group[f"output_token_throughput_per_request_{stat_value}"],
                group["output_token_throughput_per_gpu_avg"],
                color=color,
                alpha=0.6,  # Slight transparency to see overlapping points
                label=f"All Points {label_beautified}",
            )

        # Calculate Pareto frontier
        pareto_points = pareto_efficient(
            group.index,
            group[
                [
                    f"output_token_throughput_per_request_{stat_value}",
                    "output_token_throughput_per_gpu_avg",
                ]
            ].values,
        )
        if len(pareto_points) == 0:
            # If no Pareto points, skip plotting
            continue

        # Sort by x-value before plotting
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]

        # Plot the Pareto frontier
        plt.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            linestyle=linestyle,
            color=color,
            marker="o",
            label=f"Pareto {label_beautified}",
        )

    # Save the data with Pareto indicators
    if stat_value == "avg":
        csv_filename = f"{base_path}/results.csv"
    else:
        csv_filename = f"{base_path}/results_{stat_value}.csv"
    df.to_csv(csv_filename, index=False)

    # Configure axes labels and title
    if stat_value.startswith("p") or stat_value in ("min", "max"):
        plt.xlabel(f"tokens/s/user for latency {stat_value}")
    else:
        plt.xlabel(f"tokens/s/user {stat_value}")
    plt.ylabel("tokens/s/gpu avg")
    plt.title(f"Pareto Efficiency Curves {title}")

    # Legend, grid, layout
    plt.legend(title="Legend", bbox_to_anchor=(0.5, 1), loc="upper left")
    plt.grid(True)

    # Fine-tune ticks
    ax = plt.gca()
    x_interval = 5  # Adjust as needed
    y_interval = 10
    ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    ax.yaxis.set_major_locator(MultipleLocator(y_interval))
    ax.set_ylim(bottom=0)

    # Layout and footnote
    plt.tight_layout()
    plt.subplots_adjust(bottom=_FOOTNOTE_BOTTOM_SPACE)
    if footnote:
        plt.figtext(
            _FOOTNOTE_X_POSITION, _FOOTNOTE_Y_POSITION, footnote, **_FOOTNOTE_KWARGS
        )

    # Save figure
    if stat_value == "avg":
        if show_all_points:
            plot_filename = f"{base_path}/pareto_plot_all_points.png"
        else:
            plot_filename = f"{base_path}/pareto_plot.png"
    else:
        if show_all_points:
            plot_filename = f"{base_path}/pareto_plot_all_points_{stat_value}.png"
        else:
            plot_filename = f"{base_path}/pareto_plot_{stat_value}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()


def main(
    base_path,
    title,
    stat_value="avg",
    footnote="",
    output_tokens_per_request=None,
    show_all_points=False,
):
    latest_run_dirs = get_latest_run_dirs(base_path)
    extracted_values = extract_val_and_concurrency(
        base_path, latest_run_dirs, stat_value, output_tokens_per_request
    )
    print(extracted_values)

    create_graph(base_path, extracted_values, title, stat_value, footnote)
    create_pareto_graph(
        base_path, extracted_values, title, stat_value, footnote, show_all_points
    )
    create_itl_graph(base_path, extracted_values, title, stat_value, footnote)
    create_ttft_graph(base_path, extracted_values, title, stat_value, footnote)
    # Throughput contains only avg
    create_req_graph(base_path, extracted_values, title, footnote)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GAP results")
    parser.add_argument(
        "base_path", type=str, help="Base path to the results directory"
    )
    parser.add_argument("title", type=str, help="Title for all graphs")
    parser.add_argument(
        "stat_value",
        type=str,
        help="Stat value to use for the graphs (avg, p25, p50, p75, p90, p95, p99, min, max)",
    )
    # Addutional optional argument to use output tokens per request
    parser.add_argument(
        "--output-tokens-per-request",
        type=int,
        help="Output tokens per request. This parameter is used only if stat value request is not present in genai-perf output",
    )
    parser.add_argument("--footnote", type=str, help="Footnote for all graphs")
    parser.add_argument(
        "--show-all-points",
        action="store_true",
        help="Show all points in the Pareto plot",
    )
    args = parser.parse_args()
    main(
        args.base_path,
        args.title,
        args.stat_value,
        args.footnote,
        args.output_tokens_per_request,
        args.show_all_points,
    )
