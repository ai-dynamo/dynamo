#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Performance Comparison Plotter

This script takes two JSON files containing performance data and creates a scatter plot
comparing output_token_throughput_per_user vs output_token_throughput_per_gpu.
Points from different files are colored differently, and Pareto lines are added for each dataset.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_plot_data(data: List[Dict]) -> Tuple[List[float], List[float]]:
    """Extract x and y coordinates for plotting from JSON data."""
    x_coords = [entry["output_token_throughput_per_user"] for entry in data]
    y_coords = [entry["output_token_throughput_per_gpu"] for entry in data]
    return x_coords, y_coords


def compute_pareto_frontier(
    x_coords: List[float], y_coords: List[float]
) -> Tuple[List[float], List[float]]:
    """
    Compute the Pareto frontier for a set of points.
    The Pareto frontier connects only the roofline points (actual optimal points from data).
    """
    if not x_coords or not y_coords:
        return [], []

    # Combine coordinates into points
    points = list(zip(x_coords, y_coords))

    # Find the true Pareto optimal points (non-dominated points)
    pareto_points = []

    for i, (x1, y1) in enumerate(points):
        is_dominated = False

        # Check if this point is dominated by any other point
        for j, (x2, y2) in enumerate(points):
            if i != j:
                # Point 2 dominates point 1 if it's better in at least one dimension and not worse in any
                if (x2 >= x1 and y2 > y1) or (x2 > x1 and y2 >= y1):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_points.append((x1, y1))

    # Sort Pareto points by x-coordinate (user throughput)
    pareto_points.sort(key=lambda p: p[0])

    # Unzip the Pareto points
    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        return list(pareto_x), list(pareto_y)
    else:
        return [], []


def find_max_difference_point(
    pareto_x1: List[float],
    pareto_y1: List[float],
    pareto_x2: List[float],
    pareto_y2: List[float],
    user_throughput_threshold: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Find the point with maximum relative difference between the two rooflines.
    If threshold is specified, only considers points above the threshold.
    Returns the user throughput, the two GPU throughputs, and the difference multiplier.
    """
    if not pareto_x1 or not pareto_x2:
        return None, None, None, ""

    # Apply threshold if specified
    if user_throughput_threshold is not None:
        # Filter Pareto points above the threshold
        pareto_points1 = [
            (x, y)
            for x, y in zip(pareto_x1, pareto_y1)
            if x >= user_throughput_threshold
        ]
        pareto_points2 = [
            (x, y)
            for x, y in zip(pareto_x2, pareto_y2)
            if x >= user_throughput_threshold
        ]

        if not pareto_points1 or not pareto_points2:
            return None, None, None, ""

        pareto_x1_filtered: List[float]
        pareto_y1_filtered: List[float]
        pareto_x2_filtered: List[float]
        pareto_y2_filtered: List[float]

        # Unzip the filtered points into separate x and y lists
        pareto_x1_filtered = [point[0] for point in pareto_points1]
        pareto_y1_filtered = [point[1] for point in pareto_points1]
        pareto_x2_filtered = [point[0] for point in pareto_points2]
        pareto_y2_filtered = [point[1] for point in pareto_points2]
    else:
        pareto_x1_filtered, pareto_y1_filtered = pareto_x1, pareto_y1
        pareto_x2_filtered, pareto_y2_filtered = pareto_x2, pareto_y2

    # Find the point with maximum relative difference between rooflines
    max_diff_ratio: float = 0.0
    max_diff_user = None
    max_diff_gpu1 = None
    max_diff_gpu2 = None

    # For each point in the first roofline, find the closest point in the second roofline
    # and calculate the relative difference
    for i, (x1, y1) in enumerate(zip(pareto_x1_filtered, pareto_y1_filtered)):
        # Find the closest point in the second roofline to this x-coordinate
        closest_idx2 = 0
        min_distance = float("inf")
        for j, x2 in enumerate(pareto_x2_filtered):
            distance = abs(x2 - x1)
            if distance < min_distance:
                min_distance = distance
                closest_idx2 = j

        y2 = pareto_y2_filtered[closest_idx2]

        # Calculate the relative difference
        if y2 > 0:  # Avoid division by zero
            if y1 > y2:
                ratio = y1 / y2
            else:
                ratio = y2 / y1

            # Update if this is the maximum ratio found
            if ratio > max_diff_ratio:
                max_diff_ratio = ratio
                max_diff_user = x1
                max_diff_gpu1 = y1
                max_diff_gpu2 = y2

    if max_diff_user is None:
        return None, None, None, ""

    # Create the label
    if max_diff_gpu1 is not None and max_diff_gpu2 is not None:
        label = f"{max_diff_ratio:.1f}x better\nUser: {max_diff_user:.1f}\nGPU1: {max_diff_gpu1:.1f}\nGPU2: {max_diff_gpu2:.1f}"
    else:
        label = "No valid GPU data"

    return max_diff_user, max_diff_gpu1, max_diff_gpu2, label


def plot_performance_comparison(
    file1_path: str,
    file2_path: str,
    output_path: Optional[str] = None,
    user_throughput_threshold: Optional[float] = None,
):
    """Create the performance comparison plot."""

    # Load data from both files
    data1 = load_json_data(file1_path)
    data2 = load_json_data(file2_path)

    # Extract the "kind" field from the data to use as labels
    kind1 = data1[0]["kind"] if data1 and "kind" in data1[0] else file1_path
    kind2 = data2[0]["kind"] if data2 and "kind" in data2[0] else file2_path

    # Extract plotting coordinates
    x1, y1 = extract_plot_data(data1)
    x2, y2 = extract_plot_data(data2)

    # Compute Pareto frontiers
    pareto_x1, pareto_y1 = compute_pareto_frontier(x1, y1)
    pareto_x2, pareto_y2 = compute_pareto_frontier(x2, y2)

    # Find the point where rooflines differ the most
    max_diff_user, max_diff_gpu1, max_diff_gpu2, diff_label = find_max_difference_point(
        pareto_x1, pareto_y1, pareto_x2, pareto_y2, user_throughput_threshold
    )

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot scatter points
    plt.scatter(
        x1, y1, c="blue", alpha=0.6, s=40, label=f"{kind1} ({len(data1)} points)"
    )
    plt.scatter(
        x2, y2, c="red", alpha=0.6, s=40, label=f"{kind2} ({len(data2)} points)"
    )

    # Plot Pareto lines (roofline)
    if pareto_x1 and pareto_y1:
        plt.plot(
            pareto_x1,
            pareto_y1,
            "b-",
            linewidth=3,
            alpha=0.9,
            label=f"{kind1} Roofline ({len(pareto_x1)} points)",
        )
        # Highlight Pareto points
        plt.scatter(
            pareto_x1,
            pareto_y1,
            c="blue",
            s=80,
            alpha=0.9,
            edgecolors="white",
            linewidth=1,
            zorder=5,
        )
    if pareto_x2 and pareto_y2:
        plt.plot(
            pareto_x2,
            pareto_y2,
            "r-",
            linewidth=3,
            alpha=0.9,
            label=f"{kind2} Roofline ({len(pareto_x2)} points)",
        )
        # Highlight Pareto points
        plt.scatter(
            pareto_x2,
            pareto_y2,
            c="red",
            s=80,
            alpha=0.9,
            edgecolors="white",
            linewidth=1,
            zorder=5,
        )

    # Mark the point where rooflines differ the most
    if (
        max_diff_user is not None
        and max_diff_gpu1 is not None
        and max_diff_gpu2 is not None
    ):
        # Plot vertical line at the user throughput where difference is maximum
        plt.axvline(
            x=max_diff_user,
            color="purple",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Max Difference Point",
        )

        # Mark the points on both rooflines
        plt.scatter(
            max_diff_user,
            max_diff_gpu1,
            c="blue",
            s=150,
            alpha=1.0,
            edgecolors="purple",
            linewidth=3,
            zorder=10,
            marker="*",
        )
        plt.scatter(
            max_diff_user,
            max_diff_gpu2,
            c="red",
            s=150,
            alpha=1.0,
            edgecolors="purple",
            linewidth=3,
            zorder=10,
            marker="*",
        )

        # Add annotation with the difference information
        plt.annotate(
            diff_label,
            xy=(max_diff_user, max(max_diff_gpu1, max_diff_gpu2)),
            xytext=(max_diff_user + 10, max(max_diff_gpu1, max_diff_gpu2) + 50),
            arrowprops=dict(arrowstyle="->", color="purple", alpha=0.7),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
            fontsize=10,
            fontweight="bold",
        )

    # Customize the plot
    plt.xlabel("Output Token Throughput per User", fontsize=12)
    plt.ylabel("Output Token Throughput per GPU", fontsize=12)
    plt.title(
        "Performance Comparison: Throughput per GPU vs Throughput per User",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add some statistics as text
    if user_throughput_threshold is not None:
        # Format the statistics with proper conditional handling
        x1_max_str = f"{max(x1):.1f}" if len(x1) > 0 else "N/A"
        y1_max_str = f"{max(y1):.1f}" if len(y1) > 0 else "N/A"
        x2_max_str = f"{max(x2):.1f}" if len(x2) > 0 else "N/A"
        y2_max_str = f"{max(y2):.1f}" if len(y2) > 0 else "N/A"

        stats_text = f"""
Statistics (Max Difference Point: User Throughput ≥ {user_throughput_threshold}):
{kind1}: {len(data1)} points, max per-user: {x1_max_str}, max per-gpu: {y1_max_str}
{kind2}: {len(data2)} points, max per-user: {x2_max_str}, max per-gpu: {y2_max_str}
        """
    else:
        # Format the statistics with proper conditional handling
        x1_max_str = f"{max(x1):.1f}" if len(x1) > 0 else "N/A"
        y1_max_str = f"{max(y1):.1f}" if len(y1) > 0 else "N/A"
        x2_max_str = f"{max(x2):.1f}" if len(x2) > 0 else "N/A"
        y2_max_str = f"{max(y2):.1f}" if len(y2) > 0 else "N/A"

        stats_text = f"""
Statistics:
{kind1}: {len(data1)} points, max per-user: {x1_max_str}, max per-gpu: {y1_max_str}
{kind2}: {len(data2)} points, max per-user: {x2_max_str}, max per-gpu: {y2_max_str}
        """
    plt.text(
        0.02,
        0.02,
        stats_text.strip(),
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Adjust layout and save/show
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance comparison between two JSON files"
    )
    parser.add_argument("file1", help="Path to first JSON file")
    parser.add_argument("file2", help="Path to second JSON file")
    parser.add_argument(
        "--output", "-o", help="Output file path for the plot (optional)"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        help="Minimum user throughput threshold (filters data points below this value)",
    )

    args = parser.parse_args()

    try:
        plot_performance_comparison(args.file1, args.file2, args.output, args.threshold)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
