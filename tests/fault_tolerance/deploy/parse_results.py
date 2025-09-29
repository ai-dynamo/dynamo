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

"""Parser for AI-Perf results in fault tolerance tests."""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

from tests.fault_tolerance.deploy.scenarios import (
    WORKER_READY_PATTERNS,
    get_all_worker_types,
)


def parse_test_log(
    file_path: str,
) -> Tuple[Optional[float], Optional[datetime], Optional[List[str]]]:
    """
    Parse test log for startup time and fault injection time.

    Args:
        file_path: Path to test.log.txt

    Returns:
        Tuple of (startup_time_seconds, fault_time, start_cmd)
    """
    start_time = None
    ready_time = None
    fault_time = None
    start_cmd: Optional[List[str]] = None

    if not os.path.isfile(file_path):
        return None, None, None

    with open(file_path, "r") as f:
        for line in f:
            # Look for multiprocess creation
            if "Creating ManagedDeployment" in line:
                timestamp = line[26:45]  # Extract timestamp
                start_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

            # Look for "All workers are ready"
            if "All workers are ready" in line:
                timestamp = line[26:45]
                ready_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

            # Look for fault injection
            if "Injecting failure for:" in line:
                timestamp = line[26:45]
                fault_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

                # Extract failure details
                match = re.search(r"Failure\((.*?)\)", line)
                if match:
                    failure_str = match.group(1)
                    parts = failure_str.split(", ")
                    failure_info = {}
                    for part in parts:
                        key_val = part.split("=")
                        if len(key_val) == 2:
                            failure_info[key_val[0]] = key_val[1]

                    # Build command list from failure info
                    if failure_info:
                        start_cmd = [
                            failure_info.get("pod_name", "unknown"),
                            failure_info.get("command", "unknown"),
                        ]

    # Calculate startup time in seconds
    startup_time = None
    if start_time and ready_time:
        startup_time = (ready_time - start_time).total_seconds()

    return startup_time, fault_time, start_cmd


def parse_process_log(file_path: str) -> List[Tuple[datetime, str, str]]:
    """
    Parse process logs for worker events.

    Args:
        file_path: Path to process log file

    Returns:
        List of (timestamp, worker_type, worker_name) tuples
    """
    events = []

    if not os.path.isfile(file_path):
        return events

    with open(file_path, "r") as f:
        for line in f:
            # Check each worker ready pattern
            for worker_type, pattern in WORKER_READY_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    # Extract timestamp
                    timestamp_match = re.search(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line
                    )
                    if timestamp_match:
                        timestamp = datetime.strptime(
                            timestamp_match.group(1), "%Y-%m-%d %H:%M:%S"
                        )

                        # Extract model name if available
                        worker_name = worker_type
                        if (
                            hasattr(match, "groupdict")
                            and "model_name" in match.groupdict()
                        ):
                            model_name = match.group("model_name")
                            if model_name:
                                worker_name = f"{worker_type}[{model_name}]"

                        events.append((timestamp, worker_type, worker_name))
                        break

    return events


def calculate_recovery_time(
    fault_time: Optional[datetime],
    failure_info: Optional[List[str]],
    process_logs_dir: str,
) -> Optional[float]:
    """
    Calculate recovery time after fault injection.

    Args:
        fault_time: Time when fault was injected
        failure_info: List with [pod_name, command] from fault injection
        process_logs_dir: Directory containing process log files

    Returns:
        Recovery time in seconds or None if not found
    """
    if not fault_time or not failure_info:
        return None

    failed_component = failure_info[0]

    # Get all worker types from config
    all_worker_types = get_all_worker_types()

    # Determine what component type failed
    component_type = None
    if "frontend" in failed_component.lower():
        component_type = "Frontend"
    else:
        # Check if it's one of the known worker types
        for worker_type in all_worker_types:
            if worker_type.lower() in failed_component.lower():
                component_type = worker_type
                break

    if not component_type:
        return None

    # Find recovery event
    recovery_time = None
    earliest_recovery = None

    for log_file in os.listdir(process_logs_dir):
        if log_file.endswith(".log.txt"):
            events = parse_process_log(os.path.join(process_logs_dir, log_file))

            for timestamp, worker_type, worker_name in events:
                # Check if this event is after fault injection
                if timestamp > fault_time:
                    # Check if this is the recovery of the failed component
                    if worker_type == component_type:
                        if not earliest_recovery or timestamp < earliest_recovery:
                            earliest_recovery = timestamp

    if earliest_recovery:
        recovery_time = (earliest_recovery - fault_time).total_seconds()

    return recovery_time


def parse_aiperf_client_results(log_dir: str, num_clients: int) -> Dict[str, Any]:
    """
    Parse AI-Perf results from multiple clients.

    Args:
        log_dir: Directory containing client result directories
        num_clients: Number of client processes

    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "latencies": [],
        "ttft": [],  # Time to First Token
        "itl": [],  # Inter-Token Latency
        "throughputs": [],
        "p50_latencies": [],
        "p90_latencies": [],
        "p99_latencies": [],
    }

    for i in range(num_clients):
        client_dir = Path(log_dir) / f"client_{i}"
        profile_json = client_dir / "profile_export.json"

        # Try alternative names
        if not profile_json.exists():
            profile_json = client_dir / "profile_results.json"

        if profile_json.exists():
            try:
                with open(profile_json) as f:
                    client_metrics = json.load(f)

                # Extract metrics
                request_count = client_metrics.get("request_count", 0)
                error_count = client_metrics.get("error_count", 0)

                all_metrics["total_requests"] += request_count
                all_metrics["successful_requests"] += request_count - error_count
                all_metrics["failed_requests"] += error_count

                # Latency metrics
                latencies = client_metrics.get("request_latencies", {})
                if latencies:
                    if "mean" in latencies:
                        all_metrics["latencies"].append(latencies["mean"])
                    if "p50" in latencies:
                        all_metrics["p50_latencies"].append(latencies["p50"])
                    if "p90" in latencies:
                        all_metrics["p90_latencies"].append(latencies["p90"])
                    if "p99" in latencies:
                        all_metrics["p99_latencies"].append(latencies["p99"])

                # Token generation metrics
                ttft_metrics = client_metrics.get("time_to_first_token", {})
                if ttft_metrics and "mean" in ttft_metrics:
                    all_metrics["ttft"].append(ttft_metrics["mean"])

                itl_metrics = client_metrics.get("inter_token_latency", {})
                if itl_metrics and "mean" in itl_metrics:
                    all_metrics["itl"].append(itl_metrics["mean"])

                # Throughput
                throughput_metrics = client_metrics.get("request_throughput", {})
                if throughput_metrics and "value" in throughput_metrics:
                    all_metrics["throughputs"].append(throughput_metrics["value"])

            except Exception as e:
                print(f"Error parsing client_{i} results: {e}")

    return all_metrics


def print_summary_table(
    log_dir: str,
    num_clients: int,
    startup_time: Optional[float],
    recovery_time: Optional[float],
    metrics: Dict[str, Any],
    tablefmt: str = "grid",
    sla: Optional[float] = None,
) -> None:
    """
    Print formatted summary table with AI-Perf metrics.

    Args:
        log_dir: Test directory path
        num_clients: Number of client processes
        startup_time: Time to start deployment (seconds)
        recovery_time: Time to recover from fault (seconds)
        metrics: Aggregated metrics from AI-Perf
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)
    """
    headers = ["Metric", "Value"]
    rows = []

    # Test info
    rows.append(["Test Directory", log_dir])
    rows.append(["Number of Clients", num_clients])
    rows.append(["", ""])

    # Deployment metrics
    rows.append(["=== Deployment Metrics ===", ""])
    if startup_time:
        rows.append(["Startup Time", f"{startup_time:.2f} sec"])
    else:
        rows.append(["Startup Time", "N/A"])

    if recovery_time:
        rows.append(["Recovery Time", f"{recovery_time:.2f} sec"])
    else:
        rows.append(["Recovery Time", "N/A"])
    rows.append(["", ""])

    # Request metrics
    rows.append(["=== Request Metrics ===", ""])
    rows.append(["Total Requests", metrics["total_requests"]])
    rows.append(["Successful Requests", metrics["successful_requests"]])
    rows.append(["Failed Requests", metrics["failed_requests"]])

    if metrics["total_requests"] > 0:
        success_rate = (
            metrics["successful_requests"] / metrics["total_requests"]
        ) * 100
        rows.append(["Success Rate", f"{success_rate:.2f}%"])
    rows.append(["", ""])

    # Latency metrics
    rows.append(["=== Latency Metrics (seconds) ===", ""])

    if metrics["latencies"]:
        mean_latency = np.mean(metrics["latencies"])
        rows.append(["Mean Latency", f"{mean_latency:.3f}"])

        # Check SLA if provided
        if sla is not None:
            sla_status = "✓ PASS" if mean_latency <= sla else "✗ FAIL"
            rows.append(["SLA Status", f"{sla_status} (target: {sla:.3f}s)"])

    if metrics["p50_latencies"]:
        rows.append(["P50 Latency", f"{np.mean(metrics['p50_latencies']):.3f}"])

    if metrics["p90_latencies"]:
        rows.append(["P90 Latency", f"{np.mean(metrics['p90_latencies']):.3f}"])

    if metrics["p99_latencies"]:
        rows.append(["P99 Latency", f"{np.mean(metrics['p99_latencies']):.3f}"])
    rows.append(["", ""])

    # Token generation metrics
    rows.append(["=== Token Generation Metrics ===", ""])

    if metrics["ttft"]:
        rows.append(
            ["Time to First Token (mean)", f"{np.mean(metrics['ttft']):.3f} sec"]
        )

    if metrics["itl"]:
        rows.append(
            ["Inter-Token Latency (mean)", f"{np.mean(metrics['itl']):.4f} sec"]
        )
    rows.append(["", ""])

    # Throughput metrics
    rows.append(["=== Throughput Metrics ===", ""])

    if metrics["throughputs"]:
        total_throughput = sum(metrics["throughputs"])
        rows.append(["Total Throughput", f"{total_throughput:.2f} req/s"])
        rows.append(
            ["Avg Client Throughput", f"{np.mean(metrics['throughputs']):.2f} req/s"]
        )

    # Print table
    print("\n" + "=" * 60)
    print("FAULT TOLERANCE TEST SUMMARY - AI-PERF")
    print("=" * 60)
    print(tabulate(rows, headers=headers, tablefmt=tablefmt))
    print("=" * 60 + "\n")


def process_single_test(
    log_dir: str, tablefmt: str = "grid", sla: Optional[float] = None
) -> Dict[str, Any]:
    """
    Process a single test log directory.

    Args:
        log_dir: Directory containing test results
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)

    Returns:
        Dictionary with test results
    """
    # Parse test configuration
    test_log = os.path.join(log_dir, "test.log.txt")
    startup_time, fault_time, failure_info = parse_test_log(test_log)

    # Calculate recovery time if fault was injected
    recovery_time = None
    if fault_time and failure_info:
        recovery_time = calculate_recovery_time(fault_time, failure_info, log_dir)

    # Count number of clients (client_X directories)
    num_clients = 0
    for item in os.listdir(log_dir):
        if item.startswith("client_") and os.path.isdir(os.path.join(log_dir, item)):
            num_clients += 1

    # Parse AI-Perf results
    metrics = parse_aiperf_client_results(log_dir, num_clients)

    # Print summary
    print_summary_table(
        log_dir, num_clients, startup_time, recovery_time, metrics, tablefmt, sla
    )

    return {
        "log_dir": log_dir,
        "num_clients": num_clients,
        "startup_time": startup_time,
        "recovery_time": recovery_time,
        "metrics": metrics,
    }


def main(
    logs_dir: Optional[str] = None,
    log_paths: Optional[List[str]] = None,
    tablefmt: str = "grid",
    sla: Optional[float] = None,
):
    """
    Main parser entry point with support for multiple log paths.

    Args:
        logs_dir: Base directory for logs (optional)
        log_paths: List of log directories to process
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)

    Returns:
        Combined results from all processed tests
    """
    # Handle different input formats
    if log_paths:
        # Process multiple log paths
        all_results = []
        for log_path in log_paths:
            if logs_dir:
                full_path = os.path.join(logs_dir, log_path)
            else:
                full_path = log_path

            if os.path.isdir(full_path):
                print(f"\nProcessing: {full_path}")
                results = process_single_test(full_path, tablefmt, sla)
                all_results.append(results)
            else:
                print(f"Warning: {full_path} is not a valid directory, skipping...")

        # If multiple tests, also print combined summary
        if len(all_results) > 1:
            print("\n" + "=" * 60)
            print("COMBINED TEST SUMMARY")
            print("=" * 60)

            total_requests = sum(r["metrics"]["total_requests"] for r in all_results)
            total_successful = sum(
                r["metrics"]["successful_requests"] for r in all_results
            )
            total_failed = sum(r["metrics"]["failed_requests"] for r in all_results)

            print(f"Total Tests: {len(all_results)}")
            print(f"Total Requests: {total_requests}")
            print(f"Total Successful: {total_successful}")
            print(f"Total Failed: {total_failed}")

            if total_requests > 0:
                print(
                    f"Overall Success Rate: {(total_successful/total_requests)*100:.2f}%"
                )

            print("=" * 60 + "\n")

        return all_results

    elif logs_dir:
        # Process single directory
        return process_single_test(logs_dir, tablefmt, sla)
    else:
        print("Error: Must provide either logs_dir or log_paths")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse fault tolerance test results")
    parser.add_argument(
        "log_dir", type=str, help="Directory containing test logs and results"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f"Error: {args.log_dir} is not a valid directory")
        exit(1)

    main(args.log_dir)
