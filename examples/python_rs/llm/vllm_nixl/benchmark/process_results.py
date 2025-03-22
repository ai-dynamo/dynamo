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
from matplotlib.ticker import MultipleLocator

LOGGER = logging.getLogger(__name__)


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
                    subdir_path, d, "my_profile_export_genai_perf.json"
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


def extract_val_and_concurrency(base_path, latest_run_dirs):
    results = []
    for subdir, latest_dirs in latest_run_dirs.items():
        for latest_dir in latest_dirs:
            json_path = os.path.join(
                base_path,
                subdir,
                latest_dir,
                "my_profile_export_genai_perf.json",
            )
            with open(json_path, "r") as f:
                data = json.load(f)
                output_token_throughput = data.get("output_token_throughput", {}).get(
                    "avg"
                )
                output_token_throughput_per_request = data.get(
                    "output_token_throughput_per_request", {}
                ).get("avg")
                time_to_first_token = data.get("time_to_first_token", {}).get("avg")
                inter_token_latency = data.get("inter_token_latency", {}).get("avg")
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
                    "output_token_throughput": output_token_throughput,
                    "output_token_throughput_per_request": output_token_throughput_per_request,
                    "output_token_throughput_per_gpu": output_token_throughput_per_gpu,
                    "time_to_first_token": time_to_first_token,
                    "inter_token_latency": inter_token_latency,
                    "request_throughput_per_gpu": request_throughput_per_gpu,
                }
            )
    return results


def create_graph(base_path, results, title):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["output_token_throughput_per_request"],
            "y": r["output_token_throughput_per_gpu"],
        }
        for r in results
        if r["output_token_throughput_per_request"] is not None
        and r["output_token_throughput_per_gpu"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    plt.legend(title="Legend")
    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title(f"Throughput vs. tokens per user {title}")

    # Save the plot to a file
    plt.savefig(f"{base_path}/plot.png", dpi=300)
    plt.close()


def create_itl_graph(base_path, results, title):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["concurrency"],
            "y": r["inter_token_latency"],
        }
        for r in results
        if r["inter_token_latency"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    plt.legend(title="Legend")
    plt.xlabel("concurrency")
    plt.ylabel("inter_token_latency")
    plt.title(f"Inter-token latency {title}")

    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.savefig(f"{base_path}/plot_itl.png", dpi=300)
    plt.close()


def create_ttft_graph(base_path, results, title):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["concurrency"],
            "y": r["time_to_first_token"],
        }
        for r in results
        if r["time_to_first_token"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    plt.legend(title="Legend")
    plt.xlabel("concurrency")
    plt.ylabel("time_to_first_token")
    plt.title(f"Time to first token ({title})")

    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.savefig(f"{base_path}/plot_ttft.png", dpi=300)
    plt.close()


def create_req_graph(base_path, results, title):
    points = [
        {
            "label": r["configuration"],
            "order": r["concurrency"],
            "x": r["concurrency"],
            "y": r["request_throughput_per_gpu"],
        }
        for r in results
        if r["request_throughput_per_gpu"] is not None
    ]
    df = pd.DataFrame(points).sort_values(by=["label", "order"])

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Plot each label's points
    for label, group in df.groupby("label"):
        plt.plot(group["x"], group["y"], marker="o", label=label)

    plt.legend(title="Legend")
    plt.xlabel("concurrency")
    plt.ylabel("request_throughput_per_gpu")
    plt.title(f"Throughput {title}")

    plt.savefig(f"{base_path}/plot_req.png", dpi=300)
    plt.close()


def create_pareto_graph(base_path, results, title):
    data_points = [
        {
            "label": result["configuration"].split("_")[0].replace("context", "disagg"),
            "configuration": result["configuration"],
            "concurrency": float(result["concurrency"]),
            "output_token_throughput_per_request": result[
                "output_token_throughput_per_request"
            ],
            "output_token_throughput_per_gpu": result[
                "output_token_throughput_per_gpu"
            ],
            "time_to_first_token": result["time_to_first_token"],
            "inter_token_latency": result["inter_token_latency"],
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

    # Plot Pareto frontier for each label
    plt.figure(figsize=(10, 6))
    labels = df["label"].unique()

    for label in labels:
        group = df[df["label"] == label]

        # Plot the points
        plt.scatter(
            group["output_token_throughput_per_request"],
            group["output_token_throughput_per_gpu"],
            label=f"Label {label}",
        )

        # Find and plot Pareto-efficient points
        pareto_points = pareto_efficient(
            group.index,
            group[
                [
                    "output_token_throughput_per_request",
                    "output_token_throughput_per_gpu",
                ]
            ].values,
        )

        # Sort Pareto points by x-axis for plotting
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        plt.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            linestyle="--",
            label=f"Pareto Frontier {label}",
        )

    df.to_csv(f"{base_path}/results.csv")

    # Add labels and legend
    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title("Pareto Efficiency Curves {title}")
    plt.legend()
    plt.grid(True)
    # Get the current axes
    ax = plt.gca()

    # Set the major tick locator for both x and y axes
    x_interval = 5  # Set your desired x-axis interval
    y_interval = 5  # Set your desired y-axis interval
    ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    ax.yaxis.set_major_locator(MultipleLocator(y_interval))
    plt.savefig(
        f"{base_path}/pareto_plot.png", dpi=300
    )  # Save as PNG with high resolution


def main(base_path, title):
    latest_run_dirs = get_latest_run_dirs(base_path)
    extracted_values = extract_val_and_concurrency(base_path, latest_run_dirs)
    LOGGER.info(extracted_values)

    create_graph(base_path, extracted_values, title)
    create_pareto_graph(base_path, extracted_values, title)
    create_itl_graph(base_path, extracted_values, title)
    create_ttft_graph(base_path, extracted_values, title)
    create_req_graph(base_path, extracted_values, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GAP results")
    parser.add_argument(
        "base_path", type=str, help="Base path to the results directory"
    )
    parser.add_argument("title", type=str, help="Title for all graphs")
    args = parser.parse_args()
    main(args.base_path, args.title)
