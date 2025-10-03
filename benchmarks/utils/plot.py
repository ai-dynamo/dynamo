# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_benchmark_results(result_dir: Path) -> List[Tuple[int, Dict]]:
    """
    Parse benchmark results from a deployment directory.

    Args:
        result_dir: Path to the result directory

    Returns:
        List of (concurrency_level, metrics_dict) tuples sorted by concurrency
    """
    results = []

    # Find all concurrency directories (e.g., c1, c2, c5, c10, c50, c100, c250)
    for concurrency_dir in result_dir.iterdir():
        if not concurrency_dir.is_dir() or not concurrency_dir.name.startswith("c"):
            continue

        # Extract concurrency level from directory name
        match = re.match(r"c(\d+)", concurrency_dir.name)
        if not match:
            continue
        concurrency = int(match.group(1))

        # Find the aiperf JSON file
        aiperf_json = None
        for json_file in concurrency_dir.rglob("profile_export_aiperf.json"):
            aiperf_json = json_file
            break

        if aiperf_json and aiperf_json.exists():
            try:
                with open(aiperf_json, "r") as f:
                    metrics = json.load(f)
                results.append((concurrency, metrics))
                print(f"Loaded metrics for concurrency {concurrency}")
            except Exception as e:
                print(f"Error loading {aiperf_json}: {e}")
        else:
            print(f"Warning: No aiperf JSON found for {concurrency_dir}")

    # Sort by concurrency level
    results.sort(key=lambda x: x[0])
    return results


def extract_metric_series(
    results: List[Tuple[int, Dict]], metric_path: str, stat: str = "avg"
) -> Tuple[List[int], List[float]]:
    """
    Extract a time series of a specific metric across concurrency levels.

    Args:
        results: List of (concurrency, metrics) tuples
        metric_path: Dot-separated path to the metric (e.g., 'inter_token_latency')
        stat: Statistic to extract ('avg', 'p50', 'p90', etc.)

    Returns:
        Tuple of (concurrency_levels, metric_values)
    """
    concurrencies = []
    values = []

    path_keys = metric_path.split(".")
    for concurrency, metrics in results:
        try:
            node = metrics
            for k in path_keys:
                node = node[k]
            value = node[stat]
            concurrencies.append(concurrency)
            values.append(float(value))
        except (KeyError, TypeError):
            print(
                f"Warning: {metric_path}.{stat} not found for concurrency {concurrency}"
            )
            continue

    return concurrencies, values


def create_plot(
    title: str,
    xlabel: str,
    ylabel: str,
    data_series: List[Tuple[str, List[int], List[float]]],
    output_path: Path,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
) -> None:
    """
    Create a line plot with multiple series.

    Args:
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        data_series: List of (label, x_values, y_values) tuples
        output_path: Path to save the plot
        log_scale_x: Whether to use log scale for X axis
        log_scale_y: Whether to use log scale for Y axis
    """
    plt.figure(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (label, x_vals, y_vals) in enumerate(data_series):
        if x_vals and y_vals:  # Only plot if we have data
            plt.plot(
                x_vals,
                y_vals,
                marker="o",
                linewidth=2,
                markersize=6,
                color=colors[i % len(colors)],
                label=label,
            )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)

    if log_scale_x:
        plt.xscale("log")
    if log_scale_y:
        plt.yscale("log")

    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def create_efficiency_plot(
    deployment_results: Dict, plots_dir: Path, output_tokens: int = 200
) -> None:
    """
    Create an efficiency plot showing tok/s/gpu vs tok/s/user with concurrency as labeled points.

    Args:
        deployment_results: Dict of deployment_type -> results
        plots_dir: Directory to save plots
        output_tokens: Average output tokens per request (default 200)
    """
    plt.figure(figsize=(12, 8))

    # Support for up to 12 deployments in the plots
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
    ]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+"]

    for deployment_type, results in deployment_results.items():
        tok_s_per_user = []
        tok_s_per_gpu = []
        concurrency_levels = []

        for concurrency, metrics in results:
            try:
                # Get request throughput (requests/sec)
                request_throughput = metrics["request_throughput"]["avg"]

                # Calculate total tokens per second
                total_tok_s = request_throughput * output_tokens

                # Guard against zero concurrency and parameterize GPU count
                if concurrency <= 0:
                    continue
                num_gpus = metrics.get("cluster", {}).get("num_gpus", 1)
                tok_s_user = total_tok_s / concurrency
                tok_s_gpu = total_tok_s / max(1, num_gpus)

                tok_s_per_user.append(tok_s_user)
                tok_s_per_gpu.append(tok_s_gpu)
                concurrency_levels.append(concurrency)

            except KeyError as e:
                print(
                    f"Warning: Missing metric for {deployment_type} concurrency {concurrency}: {e}"
                )
                continue

        if tok_s_per_user and tok_s_per_gpu:
            # Plot points
            color_idx = list(deployment_results.keys()).index(deployment_type)
            color = colors[color_idx % len(colors)]
            marker = markers[color_idx % len(markers)]

            plt.scatter(
                tok_s_per_user,
                tok_s_per_gpu,
                c=color,
                marker=marker,
                s=120,
                alpha=0.8,
                label=deployment_type.title(),
                edgecolors="black",
                linewidth=1.5,
            )

            # Add concurrency labels
            for i, (x, y, c) in enumerate(
                zip(tok_s_per_user, tok_s_per_gpu, concurrency_levels)
            ):
                plt.annotate(
                    f"{c}",
                    (x, y),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    ha="left",
                )

    plt.title("GPU Efficiency vs User Experience", fontsize=14, fontweight="bold")
    plt.xlabel("Tokens/sec per User", fontsize=12)
    plt.ylabel("Tokens/sec per GPU", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add a note about what the numbers represent
    plt.figtext(
        0.02,
        0.02,
        "Note: Numbers on dots indicate concurrency level",
        fontsize=10,
        style="italic",
        alpha=0.7,
    )

    plt.legend()

    plt.tight_layout()
    output_path = plots_dir / "efficiency_tok_s_gpu_vs_user.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved efficiency plot: {output_path}")


def generate_plots(
    base_output_dir: Path, output_dir: Path, benchmark_names: Optional[List[str]] = None
) -> None:
    """
    Generate performance plots from benchmark results.

    Args:
        base_output_dir: Base directory containing benchmark results
        output_dir: Directory to save plots
        benchmark_names: Optional list of specific benchmark names to plot. If None, plots all subdirectories.
    """
    print(f"Generating plots from results in {base_output_dir}")

    if not base_output_dir.exists():
        print(f"Results directory does not exist: {base_output_dir}")
        return

    # Create plots directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse results for each deployment type
    deployment_results = {}

    # Find all subdirectories that contain benchmark results
    names_set = set(benchmark_names) if benchmark_names is not None else None
    for item in base_output_dir.iterdir():
        if item.is_dir() and item.name != "plots":
            deployment_type = item.name

            # If benchmark_names is specified, only process those directories
            if names_set is not None and deployment_type not in names_set:
                print(f"Skipping {deployment_type} (not in specified benchmark names)")
                continue

            results = parse_benchmark_results(item)
            if results:
                deployment_results[deployment_type] = results
                print(f"Found {len(results)} concurrency levels for {deployment_type}")
            else:
                print(f"No valid results found for {deployment_type}")

    if not deployment_results:
        if benchmark_names:
            available = sorted(
                [
                    p.name
                    for p in base_output_dir.iterdir()
                    if p.is_dir() and p.name != "plots"
                ]
            )
            missing = sorted([n for n in benchmark_names if n not in available])
            print(f"No benchmark results found for specified names: {benchmark_names}")
            if missing:
                print(f"Missing (not found under {base_output_dir}): {missing}")
            print(f"Available experiments: {available}")
        else:
            print("No benchmark results found to plot!")

    # 1. P50 Inter-token Latency vs Concurrency
    p50_data = []
    for deployment_type, results in deployment_results.items():
        concurrencies, latencies = extract_metric_series(
            results, "inter_token_latency", "p50"
        )
        if concurrencies:
            p50_data.append((deployment_type.title(), concurrencies, latencies))

    create_plot(
        title="P50 Inter-Token Latency vs Concurrency",
        xlabel="Concurrency Level",
        ylabel="P50 Inter-Token Latency (ms)",
        data_series=p50_data,
        output_path=output_dir / "p50_inter_token_latency_vs_concurrency.png",
        log_scale_x=True,
    )

    # 2. Average Inter-token Latency vs Concurrency
    avg_latency_data = []
    for deployment_type, results in deployment_results.items():
        concurrencies, latencies = extract_metric_series(
            results, "inter_token_latency", "avg"
        )
        if concurrencies:
            avg_latency_data.append((deployment_type.title(), concurrencies, latencies))

    create_plot(
        title="Average Inter-Token Latency vs Concurrency",
        xlabel="Concurrency Level",
        ylabel="Average Inter-Token Latency (ms)",
        data_series=avg_latency_data,
        output_path=output_dir / "avg_inter_token_latency_vs_concurrency.png",
        log_scale_x=True,
    )

    # 3. Request Throughput vs Concurrency
    throughput_data = []
    for deployment_type, results in deployment_results.items():
        concurrencies, throughputs = extract_metric_series(
            results, "request_throughput", "avg"
        )
        if concurrencies:
            throughput_data.append(
                (deployment_type.title(), concurrencies, throughputs)
            )

    create_plot(
        title="Request Throughput vs Concurrency",
        xlabel="Concurrency Level",
        ylabel="Request Throughput (req/s)",
        data_series=throughput_data,
        output_path=output_dir / "request_throughput_vs_concurrency.png",
        log_scale_x=True,
    )

    # 4. Average Time to First Token vs Concurrency
    ttft_data = []
    for deployment_type, results in deployment_results.items():
        concurrencies, ttfts = extract_metric_series(
            results, "time_to_first_token", "avg"
        )
        if concurrencies:
            ttft_data.append((deployment_type.title(), concurrencies, ttfts))

    create_plot(
        title="Average Time to First Token vs Concurrency",
        xlabel="Concurrency Level",
        ylabel="Average Time to First Token (ms)",
        data_series=ttft_data,
        output_path=output_dir / "avg_time_to_first_token_vs_concurrency.png",
        log_scale_x=True,
    )

    # 5. Efficiency plot: tok/s/gpu vs tok/s/user
    create_efficiency_plot(deployment_results, output_dir)

    # Generate summary
    summary_lines = [
        "Benchmark Results Summary",
        "=" * 30,
        "",
        f"Results directory: {base_output_dir}",
        f"Plots generated: {output_dir}",
        "",
        "Deployment Types Found:",
    ]

    for deployment_type, results in deployment_results.items():
        concurrency_levels = [r[0] for r in results]
        summary_lines.append(
            f"  {deployment_type}: {len(results)} concurrency levels ({min(concurrency_levels)}-{max(concurrency_levels)})"
        )

    summary_lines.extend(
        [
            "",
            "Generated Plots:",
            "  - p50_inter_token_latency_vs_concurrency.png",
            "  - avg_inter_token_latency_vs_concurrency.png",
            "  - request_throughput_vs_concurrency.png",
            "  - avg_time_to_first_token_vs_concurrency.png",
            "  - efficiency_tok_s_gpu_vs_user.png",
        ]
    )

    summary_path = output_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(summary_lines))
    print(f"Generated summary: {summary_path}")

    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate performance plots from benchmark results"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for plots (defaults to data-dir/plots)"
    )
    parser.add_argument(
        "--benchmark-name",
        action="append",
        help="Specific benchmark experiment name to plot (can be specified multiple times). If not specified, plots all subdirectories.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    benchmark_names = args.benchmark_name if args.benchmark_name else None

    if args.output_dir:
        # If output dir specified, use it as base and call generate_plots
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_plots(data_dir, output_dir, benchmark_names)
    else:
        # Use data_dir as base output dir
        generate_plots(data_dir, data_dir / "plots", benchmark_names)
