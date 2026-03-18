#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Post-process script for performance sweep results.

This script processes directories containing performance sweep results and extracts:
- Output Token Throughput (tokens/sec)
- Output Token Throughput Per User (tokens/sec/user)
- Deployment configuration (kind, model, total_gpus)
- Concurrency levels

It creates a JSON file for each subdirectory with the pattern ctx*_gen*_*
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_directory_config(dir_name: str) -> Dict[str, str]:
    """
    Parse configuration parameters from directory name

    Args:
        dir_name: Directory name like 'ctx1_gen3_tep4_batch128_eplb0_mtp0'

    Returns:
        Dictionary containing parsed configuration parameters
    """
    config = {}

    # Parse ctx and gen workers
    ctx_match = re.search(r"ctx(\d+)", dir_name)
    if ctx_match:
        config["ctx_workers"] = ctx_match.group(1)

    gen_match = re.search(r"gen(\d+)", dir_name)
    if gen_match:
        config["gen_workers"] = gen_match.group(1)

    # Parse batch size
    batch_match = re.search(r"batch(\d+)", dir_name)
    if batch_match:
        config["batch_size"] = batch_match.group(1)

    # Parse eplb (expert load balancing)
    eplb_match = re.search(r"eplb(\d+)", dir_name)
    if eplb_match:
        config["eplb"] = eplb_match.group(1)

    # Parse mtp mode
    mtp_match = re.search(r"mtp(\d+)", dir_name)
    if mtp_match:
        config["mtp_mode"] = mtp_match.group(1)

    # Parse tep (tensor expert parallel) mode
    tep_match = re.search(r"tep(\d+)", dir_name)
    if tep_match:
        config["tep_mode"] = tep_match.group(1)

    # Parse dep mode
    dep_match = re.search(r"dep(\d+)", dir_name)
    if dep_match:
        config["dep_mode"] = dep_match.group(1)

    return config


def find_ctx_gen_directories(base_path: str) -> List[str]:
    """
    Find all subdirectories that match the pattern ctx*_gen*_*

    Args:
        base_path: Base directory to search in

    Returns:
        List of directory paths matching the pattern
    """
    directories: List[str] = []
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        print(f"Error: Base path {base_path_obj} does not exist")
        return directories

    for item in base_path_obj.iterdir():
        if item.is_dir() and re.match(r"ctx\d+_gen\d+_.*", item.name):
            directories.append(str(item))

    return directories


def parse_deployment_config(config_path: str) -> Dict[str, str]:
    """
    Parse deployment configuration from JSON file

    Args:
        config_path: Path to deployment_config.json

    Returns:
        Dictionary containing kind, model, and total_gpus
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        return {
            "kind": config.get("kind", ""),
            "model": config.get("model", ""),
            "total_gpus": config.get("total_gpus", ""),
        }
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not parse deployment config at {config_path}: {e}")
        return {"kind": "", "model": "", "total_gpus": ""}


def extract_throughput_data(csv_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract throughput data from CSV file

    Args:
        csv_path: Path to profile_export_aiperf.csv

    Returns:
        Tuple of (output_token_throughput, output_token_throughput_per_user)
    """
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)

            output_token_throughput = None
            output_token_throughput_per_user = None

            for row in reader:
                if len(row) >= 2:
                    if row[0] == "Output Token Throughput (tokens/sec)":
                        # Handle comma-separated numbers in quotes
                        value_str = row[1].strip('"').replace(",", "")
                        output_token_throughput = float(value_str)
                    elif row[0] == "Output Token Throughput Per User (tokens/sec/user)":
                        # This metric appears in the first section with percentiles
                        # We need to get the average value (second column)
                        value_str = row[1].strip('"').replace(",", "")
                        output_token_throughput_per_user = float(value_str)

            return output_token_throughput, output_token_throughput_per_user

    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Warning: Could not parse CSV at {csv_path}: {e}")
        return None, None


def extract_concurrency_from_path(dir_path: str) -> Optional[int]:
    """
    Extract concurrency value from directory path

    Args:
        dir_path: Path to directory containing concurrency in name

    Returns:
        Concurrency value as integer, or None if not found
    """
    # Extract the number after 'concurrency'
    match = re.search(r"concurrency(\d+)", dir_path, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def process_directory(dir_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Process a single directory and extract all required data

    Args:
        dir_path: Path to the directory to process

    Returns:
        Dictionary containing extracted data, or None if processing failed
    """
    dir_path_obj = Path(dir_path)
    artifacts_path = dir_path_obj / "aiperf_artifacts"

    if not artifacts_path.exists():
        print(f"Warning: No aiperf_artifacts directory found in {dir_path}")
        return None

    # Parse deployment configuration
    config_path = artifacts_path / "deployment_config.json"
    if not config_path.exists():
        print(f"Warning: No deployment_config.json found in {artifacts_path}")
        return None

    deployment_config = parse_deployment_config(str(config_path))

    # Parse directory configuration
    dir_config = parse_directory_config(dir_path_obj.name)

    # Find CSV files in subdirectories
    csv_files = []
    for item in artifacts_path.iterdir():
        if item.is_dir():
            csv_path = item / "profile_export_aiperf.csv"
            if csv_path.exists():
                csv_files.append(str(csv_path))

    if not csv_files:
        print(f"Warning: No CSV files found in {artifacts_path}")
        return None

    # Extract throughput data from each CSV file
    results = []
    for csv_file in csv_files:
        output_throughput, output_throughput_per_user = extract_throughput_data(
            csv_file
        )
        # Extract concurrency from the CSV file path
        csv_path_obj = Path(csv_file)
        concurrency = extract_concurrency_from_path(csv_path_obj.parent.name)

        if output_throughput is not None and concurrency is not None:
            # Safely validate and convert total_gpus
            total_gpus = 1  # safe default
            try:
                if "total_gpus" not in deployment_config:
                    print(
                        "Warning: 'total_gpus' key missing in deployment config, using default value 1"
                    )
                else:
                    total_gpus = int(deployment_config["total_gpus"])
                    if total_gpus <= 0:
                        print(
                            f"Warning: Invalid total_gpus value '{deployment_config['total_gpus']}', using default value 1"
                        )
                        total_gpus = 1
            except (ValueError, TypeError) as e:
                print(
                    f"Warning: Could not convert total_gpus '{deployment_config.get('total_gpus', 'missing')}' to int: {e}, using default value 1"
                )
                total_gpus = 1

            result = {
                "concurrency": concurrency,
                "output_token_throughput": output_throughput,
                "output_token_throughput_per_user": output_throughput_per_user,
                "output_token_throughput_per_gpu": output_throughput / total_gpus,
                "model": deployment_config["model"],
                "kind": deployment_config["kind"],
                "total_gpus": deployment_config["total_gpus"],
                "ctx_workers": dir_config.get("ctx_workers", ""),
                "gen_workers": dir_config.get("gen_workers", ""),
                "batch_size": dir_config.get("batch_size", ""),
                "eplb": dir_config.get("eplb", ""),
                "mtp_mode": dir_config.get("mtp_mode", ""),
                "tep_mode": dir_config.get("tep_mode", ""),
                "dep_mode": dir_config.get("dep_mode", ""),
            }
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Post-process performance sweep results"
    )
    parser.add_argument(
        "base_path", help="Base directory containing performance sweep results"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for JSON file (default: same as input)"
    )
    parser.add_argument(
        "--output-file",
        default="performance_sweep_results.json",
        help="Output JSON filename (default: performance_sweep_results.json)",
    )

    args = parser.parse_args()

    # Find all ctx*_gen*_* directories
    directories = find_ctx_gen_directories(args.base_path)

    if not directories:
        print(
            f"No directories matching pattern 'ctx*_gen*_*' found in {args.base_path}"
        )
        return

    print(f"Found {len(directories)} directories to process:")
    for dir_path in directories:
        print(f"  - {os.path.basename(dir_path)}")

    # Collect all results from all directories
    all_results: List[Dict[str, Any]] = []
    skipped_directories = []

    # Process each directory
    for dir_path in directories:
        print(f"\nProcessing {os.path.basename(dir_path)}...")

        results = process_directory(dir_path)

        if results is None or not results:
            print(f"  Skipping {os.path.basename(dir_path)} - no valid data found")
            skipped_directories.append(os.path.basename(dir_path))
            continue

        # Add directory name to each result for identification
        for result in results:
            result["directory"] = os.path.basename(dir_path)

        all_results.extend(results)

        # Print summary for this directory
        print(f"  Found {len(results)} results:")
        for result in results:
            print(
                f"    Concurrency {result['concurrency']}: "
                f"{result['output_token_throughput_per_gpu']:.2f} tokens/sec/gpu, "
                f"{result['output_token_throughput_per_user']:.2f} tokens/sec/user"
            )

    if not all_results:
        print("No valid data found in any directory")
        return

    # Create output directory and file
    output_dir = args.output_dir if args.output_dir else args.base_path
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, args.output_file)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(
        f"\nCreated {output_file} with {len(all_results)} total results from {len(directories)} directories"
    )

    # Print summary of skipped directories
    if skipped_directories:
        print(f"\nSkipped directories with no valid data ({len(skipped_directories)}):")
        for skipped_dir in skipped_directories:
            print(f"  - {skipped_dir}")
    else:
        print(f"\nAll {len(directories)} directories had valid data.")


if __name__ == "__main__":
    main()
