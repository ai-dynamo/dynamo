#!/usr/bin/env python3
"""
Performance Comparison Plotter

This script takes two JSON files containing performance data and creates a scatter plot
comparing output_token_throughput_per_user vs output_token_throughput_per_gpu.
Points from different files are colored differently, and Pareto lines are added for each dataset.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_plot_data(data: List[Dict]) -> Tuple[List[float], List[float]]:
    """Extract x and y coordinates for plotting from JSON data."""
    x_coords = [entry['output_token_throughput_per_user'] for entry in data]
    y_coords = [entry['output_token_throughput_per_gpu'] for entry in data]
    return x_coords, y_coords

def compute_pareto_frontier(x_coords: List[float], y_coords: List[float]) -> Tuple[List[float], List[float]]:
    """
    Compute the Pareto frontier for a set of points.
    The Pareto frontier connects only the roofline points (actual optimal points from data).
    Includes all leftmost points to ensure complete coverage.
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
    
    # Find the minimum user throughput
    min_user_throughput = min(x_coords)
    
    # Add all points that have the minimum user throughput (leftmost points)
    leftmost_points = [(x, y) for x, y in points if abs(x - min_user_throughput) < 1e-6]
    
    # Add leftmost points that might not be in Pareto set
    for point in leftmost_points:
        if point not in pareto_points:
            pareto_points.append(point)
    
    # Sort Pareto points by x-coordinate (user throughput)
    pareto_points.sort(key=lambda p: p[0])
    
    # Unzip the Pareto points
    if pareto_points:
        pareto_x, pareto_y = zip(*pareto_points)
        return list(pareto_x), list(pareto_y)
    else:
        return [], []

def find_max_difference_point(pareto_x1: List[float], pareto_y1: List[float], 
                            pareto_x2: List[float], pareto_y2: List[float]) -> Tuple[float, float, float, str]:
    """
    Find the maximum difference at the leftmost valid point where both rooflines have data.
    Returns the user throughput, the two GPU throughputs, and the difference multiplier.
    """
    if not pareto_x1 or not pareto_x2:
        return None, None, None, ""
    
    # Find the leftmost valid point where both rooflines have data
    # Use the maximum of the minimum user throughputs from both datasets
    min_user1 = min(pareto_x1)
    min_user2 = min(pareto_x2)
    leftmost_valid_user = max(min_user1, min_user2)
    
    # Find the closest points in both rooflines to this leftmost valid user throughput
    closest_idx1 = 0
    min_distance1 = float('inf')
    for i, x1 in enumerate(pareto_x1):
        distance = abs(x1 - leftmost_valid_user)
        if distance < min_distance1:
            min_distance1 = distance
            closest_idx1 = i
    
    closest_idx2 = 0
    min_distance2 = float('inf')
    for i, x2 in enumerate(pareto_x2):
        distance = abs(x2 - leftmost_valid_user)
        if distance < min_distance2:
            min_distance2 = distance
            closest_idx2 = i
    
    # Get the GPU throughputs at these closest points
    max_diff_user = leftmost_valid_user
    max_diff_gpu1 = pareto_y1[closest_idx1]
    max_diff_gpu2 = pareto_y2[closest_idx2]
    
    # Calculate the performance ratio
    if max_diff_gpu2 > 0:  # Avoid division by zero
        if max_diff_gpu1 > max_diff_gpu2:
            max_diff = max_diff_gpu1 / max_diff_gpu2
            label = f"{max_diff:.1f}x better\nUser: {max_diff_user:.1f}\nGPU1: {max_diff_gpu1:.1f}\nGPU2: {max_diff_gpu2:.1f}"
        else:
            max_diff = max_diff_gpu2 / max_diff_gpu1
            label = f"{max_diff:.1f}x better\nUser: {max_diff_user:.1f}\nGPU1: {max_diff_gpu1:.1f}\nGPU2: {max_diff_gpu2:.1f}"
    else:
        max_diff = 0
        label = f"Invalid comparison\nUser: {max_diff_user:.1f}\nGPU1: {max_diff_gpu1:.1f}\nGPU2: {max_diff_gpu2:.1f}"
    
    return max_diff_user, max_diff_gpu1, max_diff_gpu2, label

def plot_performance_comparison(file1_path: str, file2_path: str, output_path: str = None):
    """Create the performance comparison plot."""
    
    # Load data from both files
    data1 = load_json_data(file1_path)
    data2 = load_json_data(file2_path)
    
    # Extract the "kind" field from the data to use as labels
    kind1 = data1[0]['kind'] if data1 and 'kind' in data1[0] else file1_path
    kind2 = data2[0]['kind'] if data2 and 'kind' in data2[0] else file2_path
    
    # Extract plotting coordinates
    x1, y1 = extract_plot_data(data1)
    x2, y2 = extract_plot_data(data2)
    
    # Compute Pareto frontiers
    pareto_x1, pareto_y1 = compute_pareto_frontier(x1, y1)
    pareto_x2, pareto_y2 = compute_pareto_frontier(x2, y2)
    
    # Find the point where rooflines differ the most
    max_diff_user, max_diff_gpu1, max_diff_gpu2, diff_label = find_max_difference_point(
        pareto_x1, pareto_y1, pareto_x2, pareto_y2
    )
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot scatter points
    plt.scatter(x1, y1, c='blue', alpha=0.6, s=40, label=f'{kind1} ({len(data1)} points)')
    plt.scatter(x2, y2, c='red', alpha=0.6, s=40, label=f'{kind2} ({len(data2)} points)')
    
    # Plot Pareto lines (roofline)
    if pareto_x1 and pareto_y1:
        plt.plot(pareto_x1, pareto_y1, 'b-', linewidth=3, alpha=0.9, label=f'{kind1} Roofline ({len(pareto_x1)} points)')
        # Highlight Pareto points
        plt.scatter(pareto_x1, pareto_y1, c='blue', s=80, alpha=0.9, edgecolors='white', linewidth=1, zorder=5)
    if pareto_x2 and pareto_y2:
        plt.plot(pareto_x2, pareto_y2, 'r-', linewidth=3, alpha=0.9, label=f'{kind2} Roofline ({len(pareto_x2)} points)')
        # Highlight Pareto points
        plt.scatter(pareto_x2, pareto_y2, c='red', s=80, alpha=0.9, edgecolors='white', linewidth=1, zorder=5)
    
    # Mark the point where rooflines differ the most
    if max_diff_user is not None and max_diff_gpu1 is not None and max_diff_gpu2 is not None:
        # Plot vertical line at the user throughput where difference is maximum
        plt.axvline(x=max_diff_user, color='purple', linestyle='--', alpha=0.7, linewidth=2, label='Max Difference Point')
        
        # Mark the points on both rooflines
        plt.scatter(max_diff_user, max_diff_gpu1, c='blue', s=150, alpha=1.0, edgecolors='purple', linewidth=3, zorder=10, marker='*')
        plt.scatter(max_diff_user, max_diff_gpu2, c='red', s=150, alpha=1.0, edgecolors='purple', linewidth=3, zorder=10, marker='*')
        
        # Add annotation with the difference information
        plt.annotate(diff_label, 
                    xy=(max_diff_user, max(max_diff_gpu1, max_diff_gpu2)), 
                    xytext=(max_diff_user + 10, max(max_diff_gpu1, max_diff_gpu2) + 50),
                    arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    fontsize=10, fontweight='bold')
    
    # Customize the plot
    plt.xlabel('Output Token Throughput per User', fontsize=12)
    plt.ylabel('Output Token Throughput per GPU', fontsize=12)
    plt.title('Performance Comparison: Throughput per GPU vs Throughput per User', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    stats_text = f"""
Statistics:
{kind1}: {len(data1)} points, max per-user: {max(x1):.1f}, max per-gpu: {max(y1):.1f}
{kind2}: {len(data2)} points, max per-user: {max(x2):.1f}, max per-gpu: {max(y2):.1f}
    """
    plt.text(0.02, 0.98, stats_text.strip(), transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout and save/show
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot performance comparison between two JSON files')
    parser.add_argument('file1', help='Path to first JSON file')
    parser.add_argument('file2', help='Path to second JSON file')
    parser.add_argument('--output', '-o', help='Output file path for the plot (optional)')
    
    args = parser.parse_args()
    
    try:
        plot_performance_comparison(args.file1, args.file2, args.output)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 