#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


def get_json_paths(search_paths):
    aiperf_profile_export_json_paths = []
    deployment_config_json_paths = []
    for search_path in search_paths:
        deployment_config_json_path = os.path.join(
            search_path, "deployment_config.json"
        )
        if not os.path.exists(deployment_config_json_path):
            raise Exception(f"deployment_config.json not found in {search_path}")
        for root, _, files in os.walk(search_path):
            for file in files:
                if file == "profile_export_aiperf.json":
                    aiperf_profile_export_json_paths.append(os.path.join(root, file))
                    deployment_config_json_paths.append(deployment_config_json_path)

    return aiperf_profile_export_json_paths, deployment_config_json_paths


# search for -concurrency<number> in the name
def parse_concurrency(name):
    matches = re.findall(r"-concurrency(\d+)", name)
    if len(matches) != 1:
        raise Exception(f"non-unique matches: {matches}")
    concurrency = 0
    for c in matches:
        concurrency += int(c)
    return concurrency


# Get the number of GPUs from the deployment config
def parse_gpus(deployment_config_json_path):
    with open(deployment_config_json_path, "r") as f:
        deployment_config = json.load(f)
    if deployment_config.get("mode") == "aggregated":
        return deployment_config.get("tensor_parallelism") * deployment_config.get(
            "data_parallelism"
        )
    else:
        return deployment_config.get(
            "prefill_tensor_parallelism"
        ) * deployment_config.get("prefill_data_parallelism") + deployment_config.get(
            "decode_tensor_parallelism"
        ) * deployment_config.get(
            "decode_data_parallelism"
        )


def parse_kind_and_mode(deployment_config_json_path):
    with open(deployment_config_json_path, "r") as f:
        deployment_config = json.load(f)
    return deployment_config.get("kind"), deployment_config.get("mode")


def extract_val_and_concurrency(
    aiperf_profile_export_json_paths, deployment_config_json_paths, stat_value="avg"
):
    results = []
    for aiperf_profile_export_json_path, deployment_config_json_path in zip(
        aiperf_profile_export_json_paths, deployment_config_json_paths
    ):
        with open(aiperf_profile_export_json_path, "r") as f:
            data = json.load(f)
            # output_token_throughput contains only avg
            output_token_throughput = data.get("output_token_throughput", {}).get("avg")
            output_token_throughput_per_user = data.get(
                "output_token_throughput_per_user", {}
            ).get(stat_value)
            time_to_first_token = data.get("time_to_first_token", {}).get(stat_value)
            inter_token_latency = data.get("inter_token_latency", {}).get(stat_value)
            # request_throughput contains only avg
            request_throughput = data.get("request_throughput", {}).get("avg")

        concurrency = parse_concurrency(aiperf_profile_export_json_path)
        num_gpus = parse_gpus(deployment_config_json_path)
        kind, mode = parse_kind_and_mode(deployment_config_json_path)

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
                "configuration": aiperf_profile_export_json_path,
                "kind": kind,
                "mode": mode,
                "num_gpus": num_gpus,
                "concurrency": float(concurrency),
                "output_token_throughput_avg": output_token_throughput,
                f"output_token_throughput_per_user_{stat_value}": output_token_throughput_per_user,
                "output_token_throughput_per_gpu_avg": output_token_throughput_per_gpu,
                f"time_to_first_token_{stat_value}": time_to_first_token,
                f"inter_token_latency_{stat_value}": inter_token_latency,
                "request_throughput_per_gpu_avg": request_throughput_per_gpu,
            }
        )
    return results


def create_pareto_graph(results, title="", stat_value="avg"):
    data_points = [
        {
            "label": f"{result['kind']}_{result['mode']}",
            "configuration": result["configuration"],
            "concurrency": float(result["concurrency"]),
            f"output_token_throughput_per_user_{stat_value}": result[
                f"output_token_throughput_per_user_{stat_value}"
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
    df = pd.DataFrame(data_points)

    def pareto_efficient(ids, points):
        """
        Mark Pareto-efficient points.
        A point p is dominated if there's another q
        such that q is >= p in all dimensions.
        """
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

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    labels = df["label"].unique()

    for label in labels:
        group = df[df["label"] == label]
        # Scatter all points
        ax.scatter(
            group[f"output_token_throughput_per_user_{stat_value}"],
            group["output_token_throughput_per_gpu_avg"],
            label=f"Label {label}",
        )

        # Identify and mark Pareto frontier
        pareto_points = pareto_efficient(
            group.index,
            group[
                [
                    f"output_token_throughput_per_user_{stat_value}",
                    "output_token_throughput_per_gpu_avg",
                ]
            ].values,
        )
        # Sort by x-value for a clean line
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            linestyle="--",
            label=f"Pareto Frontier {label}",
        )

    # Save CSV
    if stat_value == "avg":
        df_file_name = "results.csv"
    else:
        df_file_name = f"results_{stat_value}.csv"
    df.to_csv(df_file_name)

    # Axis labels and tick intervals
    ax.set_xlabel(f"tokens/s/user {stat_value}")
    ax.set_ylabel("tokens/s/gpu avg")
    ax.set_title(f"Pareto - {title}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.grid(True)
    x_interval = 5
    y_interval = 5
    ax.xaxis.set_major_locator(MultipleLocator(x_interval))
    ax.yaxis.set_major_locator(MultipleLocator(y_interval))

    if stat_value == "avg":
        file_name = "pareto_plot.png"
    else:
        file_name = f"pareto_plot_{stat_value}.png"
    plt.savefig(file_name, dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser(
        description="Plot Pareto graph from AIPerf artifacts"
    )
    parser.add_argument(
        "--artifacts-root-dir",
        required=True,
        help="Root directory containing artifact directories to search for profile_export_aiperf.json files",
    )
    parser.add_argument(
        "--title",
        default="Single Node",
        help="Title for the Pareto graph",
    )
    args = parser.parse_args()

    # Find all artifacts directories under the root
    artifacts_dirs = glob.glob(os.path.join(args.artifacts_root_dir, "artifacts_*"))
    if not artifacts_dirs:
        raise ValueError(f"No artifacts directories found in {args.artifacts_root_dir}")

    aiperf_profile_export_json_paths, deployment_config_json_paths = get_json_paths(
        artifacts_dirs
    )

    if len(aiperf_profile_export_json_paths) != len(deployment_config_json_paths):
        raise ValueError(
            f"Number of aiperf_profile_export_json_paths ({len(aiperf_profile_export_json_paths)}) does not match number of deployment_config_json_paths ({len(deployment_config_json_paths)})"
        )

    extracted_values = extract_val_and_concurrency(
        aiperf_profile_export_json_paths, deployment_config_json_paths
    )
    create_pareto_graph(extracted_values, title=args.title)
