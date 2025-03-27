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
    """
    Plots a simple 2D Pareto frontier based on
    (tokens/s/user) vs. (tokens/s/gpu).
    """
    data_points = []
    for r in results:
        if (
            r["output_token_throughput_per_request"] is None
            or r["output_token_throughput_per_gpu"] is None
        ):
            continue
        # The 'label' we use can be more generic, or we can do special replacements.
        label_str = get_label_from_name(r["configuration"])
        data_points.append(
            {
                "label": label_str,
                "configuration": r["configuration"],
                "concurrency": r["concurrency"],
                "output_token_throughput_per_request": r[
                    "output_token_throughput_per_request"
                ],
                "output_token_throughput_per_gpu": r["output_token_throughput_per_gpu"],
                "time_to_first_token": r["time_to_first_token"],
                "inter_token_latency": r["inter_token_latency"],
                "is_pareto_efficient": False,  # will get updated below
            }
        )

    df = pd.DataFrame(data_points)

    # -- Pareto frontier finder
    def pareto_efficient(ids, points):
        """Returns the points that are not dominated by others."""
        points = np.array(points)
        for i, (idx, p) in enumerate(zip(ids, points)):
            dominated = False
            for j, q in enumerate(points):
                if i != j and all(q >= p):
                    dominated = True
                    break
            if not dominated:
                df.at[idx, "is_pareto_efficient"] = True

    # We'll group by label and find the frontier in each group
    labels = df["label"].unique()
    plt.figure(figsize=(10, 6))

    for label in labels:
        group = df[df["label"] == label]
        X = group["output_token_throughput_per_request"].values
        Y = group["output_token_throughput_per_gpu"].values

        # Plot raw points
        plt.scatter(X, Y, label=f"{label}")

        # Find & mark Pareto frontier
        pareto_efficient(group.index, np.column_stack((X, Y)))

        # Extract frontier for plotting
        pf_group = group[group["is_pareto_efficient"] == True].copy()  # noqa: E712
        pf_group = pf_group.sort_values(by="output_token_throughput_per_request")
        plt.plot(
            pf_group["output_token_throughput_per_request"],
            pf_group["output_token_throughput_per_gpu"],
            linestyle="--",
        )

    # Store everything to CSV
    df.to_csv(f"{base_path}/results.csv", index=False)

    plt.xlabel("tokens/s/user")
    plt.ylabel("tokens/s/gpu")
    plt.title(f"Pareto Frontier {title}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_path}/pareto_plot.png", dpi=300)
    plt.close()


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
