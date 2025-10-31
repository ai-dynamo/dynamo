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
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

from tests.fault_tolerance.deploy.scenarios import (
    OVERFLOW_SUFFIX,
    RECOVERY_SUFFIX,
    WORKER_MAP,
    TestPhase,
)


def parse_test_log(
    file_path: str,
) -> Tuple[Optional[float], Optional[List[str]]]:
    """
    Parse test log for startup time and failure info.

    Args:
        file_path: Path to test.log.txt

    Returns:
        Tuple of (startup_time_seconds, failure_info)
    """
    start_time = None
    ready_time = None
    failure_info: Optional[List[str]] = None

    if not os.path.isfile(file_path):
        return None, None

    with open(file_path, "r") as f:
        for line in f:
            # Extract timestamp using regex to handle different log formats
            timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)

            # Look for deployment start
            if "Starting Deployment" in line and timestamp_match:
                timestamp = timestamp_match.group(1)
                start_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

            # Look for deployment ready
            if "Deployment fault-tolerance-test is ready" in line and timestamp_match:
                timestamp = timestamp_match.group(1)
                ready_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

            # Look for fault injection
            if "Injecting failure for:" in line:
                # Extract failure details
                match = re.search(r"Failure\((.*?)\)", line)
                if match:
                    failure_str = match.group(1)
                    parts = failure_str.split(", ")
                    failure_dict = {}
                    for part in parts:
                        key_val = part.split("=")
                        if len(key_val) == 2:
                            failure_dict[key_val[0]] = key_val[1]

                    # Build command list from failure info
                    if failure_dict:
                        failure_info = [
                            failure_dict.get("pod_name", "unknown").strip("'\""),
                            failure_dict.get("command", "unknown").strip("'\""),
                        ]

    # Calculate startup time in seconds
    startup_time = None
    if start_time and ready_time:
        startup_time = (ready_time - start_time).total_seconds()

    return startup_time, failure_info


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Robustly parse timestamp with multiple format attempts.

    Args:
        timestamp_str: Timestamp string to parse

    Returns:
        datetime object or None if parsing fails
    """
    # List of common timestamp formats to try
    timestamp_formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",  # Full format with microseconds and Z
        "%Y-%m-%dT%H:%M:%SZ",  # Without microseconds, with Z
        "%Y-%m-%dT%H:%M:%S.%f",  # With microseconds, no timezone
        "%Y-%m-%dT%H:%M:%S",  # Basic ISO format
        "%Y-%m-%d %H:%M:%S.%f",  # Space separator with microseconds
        "%Y-%m-%d %H:%M:%S",  # Space separator without microseconds
    ]

    for fmt in timestamp_formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    # If no format matches, log the issue
    logging.debug(f"Could not parse timestamp: {timestamp_str}")
    return None


def extract_timestamp_from_log(
    log_path: str, from_end: bool = False, max_lines: int = 10, debug_message: str = ""
) -> Optional[datetime]:
    """
    Extract a timestamp from a log file by parsing JSON lines.

    Args:
        log_path: Path to the log file
        from_end: If True, search from the end of file (for last timestamp)
                  If False, search from beginning (for first timestamp)
        max_lines: Maximum number of lines to check
        debug_message: Debug message to log when timestamp is found

    Returns:
        datetime object or None if no valid timestamp found
    """
    try:
        with open(log_path, "r") as f:
            lines = list(f.readlines())
            if from_end:
                # Read from the end of the file
                lines_to_check = list(reversed(lines))
            else:
                # Read from the beginning of the file
                lines_to_check = lines
            # Limit to max_lines
            lines_to_check = lines_to_check[:max_lines]

            for line in lines_to_check:
                if '"time":"' in line:
                    try:
                        log_entry = json.loads(line)
                        timestamp_str = log_entry.get("time", "")
                        if timestamp_str:
                            parsed_time = parse_timestamp(timestamp_str)
                            if parsed_time:
                                if debug_message:
                                    logging.debug(f"{debug_message}: {timestamp_str}")
                                return parsed_time
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.debug(f"Failed to parse JSON line: {e}")
                        continue
    except IOError as e:
        logging.debug(f"Could not read {log_path}: {e}")

    return None


def extract_test_info_from_dir(
    process_logs_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract backend and deployment type from process_logs_dir.

    Args:
        process_logs_dir: Path like test_fault_scenario[trtllm_agg_token_overflow_2x]

    Returns:
        Tuple of (backend, deploy_type) or (None, None) if not a token overflow test
    """
    test_name = os.path.basename(process_logs_dir)

    # Check if this is a token overflow test
    if "token_overflow" not in test_name:
        return None, None

    # Extract the content between brackets
    match = re.search(r"\[([^\]]+)\]", test_name)
    if not match:
        return None, None

    test_config = match.group(1)

    # Parse backend and deployment type
    # Format: {backend}_{deploy_type}_token_overflow_{multiplier}
    parts = test_config.split("_")

    if len(parts) < 4:
        return None, None

    backend = parts[0]  # vllm, trtllm, sglang
    deploy_type = parts[1]  # agg or disagg

    return backend, deploy_type


def get_decode_worker_dir(backend: str, deploy_type: str) -> Optional[str]:
    """
    Get decode worker directory name from WORKER_MAP.
    Reuses the exact logic from scenarios.py.

    Args:
        backend: Backend type (vllm, trtllm, sglang)
        deploy_type: Deployment type (agg or disagg)

    Returns:
        Worker directory name
    """
    if backend not in WORKER_MAP:
        return None

    # For trtllm agg deployments, use different worker name
    if backend == "trtllm" and deploy_type == "agg":
        return WORKER_MAP[backend]["decode_agg"]  # "TRTLLMWorker"
    else:
        return WORKER_MAP[backend]["decode"]
        # "TRTLLMDecodeWorker", "VllmDecodeWorker", or "decode"


def calculate_recovery_time(
    failure_info: Optional[List[str]],
    process_logs_dir: str,
) -> Optional[float]:
    """
    Calculate recovery time by comparing last timestamp in .previous.log with first in current log.
    This avoids timezone issues between test.log.txt and container logs.

    Args:
        failure_info: List with [pod_name, command] from fault injection
        process_logs_dir: Directory containing process log files

    Returns:
        Recovery time in seconds or None if not found
    """

    if failure_info:
        # Regular test - use failure info
        component_type = failure_info[0].strip("'\"")  # e.g., "Frontend" or "decode"
        component_dir = os.path.join(process_logs_dir, component_type)
    else:
        # Check if this is a mixed token test
        backend, deploy_type = extract_test_info_from_dir(process_logs_dir)
        if not backend or not deploy_type:
            logging.warning(
                f"Could not determine backend or deploy type for {process_logs_dir}"
            )
            return None

        # Mixed token test - get decode worker directory
        decode_worker_dir = get_decode_worker_dir(backend, deploy_type)
        if not decode_worker_dir:
            logging.warning(
                f"Could not determine decode worker for {backend} {deploy_type}"
            )
            return None

        component_dir = os.path.join(process_logs_dir, decode_worker_dir)
        logging.info(
            f"Mixed token test - using decode worker directory: {component_dir}"
        )

    logging.info(f"Component directory: {component_dir}")

    if not os.path.exists(component_dir):
        logging.warning(f"Component directory {component_dir} does not exist")
        return None

    last_timestamp_before = None
    first_timestamp_after = None

    # Find the last timestamp from .previous.log (container before restart)
    for log_file in os.listdir(component_dir):
        if log_file.endswith(".previous.log"):
            log_path = os.path.join(component_dir, log_file)
            logging.info(f"Previous pod log path: {log_path}")
            last_timestamp_before = extract_timestamp_from_log(
                log_path,
                from_end=True,
                max_lines=50,  # Check more lines for better chance of finding timestamp
                debug_message="Last timestamp before failure",
            )
            if last_timestamp_before:
                break

    # Find the first timestamp from current container log
    for log_file in os.listdir(component_dir):
        if log_file.endswith(".log") and not log_file.endswith(
            (".previous.log", ".metrics.log")
        ):
            log_path = os.path.join(component_dir, log_file)
            logging.info(f"Pod log path: {log_path}")
            first_timestamp_after = extract_timestamp_from_log(
                log_path,
                from_end=False,
                max_lines=100,  # May need to skip initial non-JSON output
                debug_message="First timestamp after recovery",
            )
            if first_timestamp_after:
                break

    # Calculate recovery time from container timestamps (both in UTC)
    if last_timestamp_before and first_timestamp_after:
        recovery_time = (first_timestamp_after - last_timestamp_before).total_seconds()
        # Sanity check - recovery should be seconds/minutes, not hours
        if recovery_time > 3600:  # More than 1 hour is likely wrong
            logging.warning(
                f"Recovery time {recovery_time}s seems too large, possible timezone issue"
            )
        return recovery_time

    return None


def parse_aiperf_client_results(log_dir: str) -> Dict[str, Any]:
    """
    Parse AI-Perf results from all client directories.

    Args:
        log_dir: Directory containing client result directories

    Returns:
        Dictionary with aggregated metrics and client count
    """
    all_metrics: Dict[str, Any] = {
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
        "num_clients": 0,
    }

    # Iterate over actual client directories
    for item in sorted(os.listdir(log_dir)):
        if not item.startswith("client_") or not os.path.isdir(
            os.path.join(log_dir, item)
        ):
            continue

        client_dir = Path(log_dir) / item
        all_metrics["num_clients"] += 1

        # Look for AI-Perf results in attempt directories
        profile_json = None

        # Check for attempt directories (attempt_0, attempt_1, etc.)
        for attempt_dir in sorted(client_dir.glob("attempt_*")):
            json_path = attempt_dir / "profile_export_aiperf.json"
            if json_path.exists():
                profile_json = json_path
                break  # Use the first successful attempt

        if not profile_json:
            logging.warning(f"No AI-Perf results found for {item} in {client_dir}")
        else:
            try:
                with open(profile_json) as f:
                    client_metrics = json.load(f)

                # AI-Perf format has "records" dictionary at the top level
                records = client_metrics.get("records", {})

                # Extract successful request count
                request_count_record = records.get("request_count", {})
                successful_count = (
                    int(request_count_record.get("avg", 0))
                    if request_count_record
                    else 0
                )

                # Extract error request count
                error_request_count_record = records.get("error_request_count", {})
                error_request_count = (
                    int(error_request_count_record.get("avg", 0))
                    if error_request_count_record
                    else 0
                )

                # Calculate total requests: successful + errors
                # Note: request_count appears to only track successful requests when errors are present
                request_count = successful_count + error_request_count

                # Fall back to input config if no requests were recorded
                if request_count == 0 and "input_config" in client_metrics:
                    input_config = client_metrics.get("input_config", {})
                    loadgen_config = (
                        input_config.get("loadgen", {}) if input_config else {}
                    )
                    request_count = loadgen_config.get("request_count", 0)

                # Check for errors in error_summary
                error_summary = client_metrics.get("error_summary", [])
                # Sum up actual error counts from each error type
                error_count = sum(error.get("count", 0) for error in error_summary)

                # Check if test was cancelled
                if client_metrics.get("was_cancelled", False):
                    error_count = request_count  # Mark all as failed if cancelled

                # Validate data consistency
                if request_count < error_count:
                    logging.warning(
                        f"Data inconsistency in {item}: error_count ({error_count}) > "
                        f"total_request_count ({request_count}). This may indicate incomplete data from aiperf."
                    )

                all_metrics["total_requests"] += request_count
                all_metrics["successful_requests"] += request_count - error_count
                all_metrics["failed_requests"] += error_count

                # Extract latency metrics
                request_latency = client_metrics.get("request_latency", None)
                if request_latency:
                    all_metrics["latencies"].append(request_latency["avg"] / 1000.0)
                    all_metrics["p50_latencies"].append(request_latency["p50"] / 1000.0)
                    all_metrics["p90_latencies"].append(request_latency["p90"] / 1000.0)
                    all_metrics["p99_latencies"].append(request_latency["p99"] / 1000.0)

                # Time to first token
                ttft_record = client_metrics.get("time_to_first_token", {})
                ttft = ttft_record.get("avg", None) if ttft_record else None
                if ttft:
                    all_metrics["ttft"].append(ttft / 1000.0)  # Convert ms to s

                # Inter-token latency
                itl_record = client_metrics.get("inter_token_latency", {})
                itl = itl_record.get("avg", None) if itl_record else None
                if itl:
                    all_metrics["itl"].append(itl / 1000.0)  # Convert ms to s

                # Throughput from request_throughput record
                throughput_record = client_metrics.get("request_throughput", {})
                req_throughput = (
                    throughput_record.get("avg", 0) if throughput_record else 0
                )
                if req_throughput:
                    all_metrics["throughputs"].append(req_throughput)

            except Exception as e:
                logging.error(f"Error parsing {item} results: {e}")

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
    rows.append(["Number of Clients", str(num_clients)])
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

    # Log table
    logging.info(
        "\n" + "=" * 60 + "\n"
        "FAULT TOLERANCE TEST SUMMARY - AI-PERF\n"
        + "=" * 60
        + "\n"
        + tabulate(rows, headers=headers, tablefmt=tablefmt)
        + "\n"
        + "=" * 60
        + "\n"
    )


def _process_test_phase_results(
    test_phase: TestPhase,
    metrics: Dict[str, Any],
    success_threshold: float,
) -> None:
    """Helper function to process and log results for a specific test phase."""
    if test_phase == TestPhase.OVERFLOW:
        total_reqs = metrics.get("total_requests", 0)
        failed_reqs = metrics.get("failed_requests", 0)
        if total_reqs > 0:
            failure_rate = (failed_reqs / total_reqs) * 100
            logging.info(
                "\n" + "=" * 60 + "\n"
                "Processing OVERFLOW phase - Expecting rejections\n" + "=" * 60 + "\n"
                f"\nOverflow Results: {failed_reqs}/{total_reqs} requests rejected ({failure_rate:.1f}%)"
            )
            if failure_rate < success_threshold:
                logging.warning(
                    f"Expected rejection rate >= {success_threshold}%, got {failure_rate:.1f}%"
                )
            else:
                logging.info("Overflow validation working correctly")
        else:
            logging.warning("No requests to process, total_requests is 0.")

    elif test_phase == TestPhase.RECOVERY:
        total_reqs = metrics.get("total_requests", 0)
        success_reqs = metrics.get("successful_requests", 0)
        if total_reqs > 0:
            success_rate = (success_reqs / total_reqs) * 100
            logging.info(
                "\n" + "=" * 60 + "\n"
                "Processing RECOVERY phase - Expecting success\n" + "=" * 60 + "\n"
                f"\nRecovery Results: {success_reqs}/{total_reqs} requests succeeded ({success_rate:.1f}%)"
            )
            if success_rate < success_threshold:
                logging.warning(
                    f"Expected success rate >= {success_threshold}%, got {success_rate:.1f}%"
                )
            else:
                logging.info("System recovered successfully")
        else:
            logging.warning("No requests to process, total_requests is 0.")
    elif test_phase == TestPhase.STANDARD:
        # Standard test phase doesn't need special processing
        pass
    else:
        raise ValueError(f"Unknown test phase: {test_phase}")


def process_single_test(
    log_dir: str,
    tablefmt: str = "grid",
    sla: Optional[float] = None,
    success_threshold: float = 90.0,
    print_output: bool = True,
) -> Dict[str, Any]:
    """
    Process a single test log directory.

    Args:
        log_dir: Directory containing test results
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)
        success_threshold: Success rate threshold for pass/fail (default: 90.0)
        print_output: If True, print tables and phase headers. If False, only return results.

    Returns:
        Dictionary with test results
    """
    # Detect test phase (overflow or recovery) - check suffix to avoid ambiguity
    test_phase = TestPhase.STANDARD

    if log_dir.endswith(OVERFLOW_SUFFIX):
        test_phase = TestPhase.OVERFLOW
    elif log_dir.endswith(RECOVERY_SUFFIX):
        test_phase = TestPhase.RECOVERY

    # Parse test configuration
    test_log = os.path.join(log_dir, "test.log.txt")
    startup_time, failure_info = parse_test_log(test_log)

    # Calculate recovery time only if fault was injected
    recovery_time = None
    if failure_info:
        recovery_time = calculate_recovery_time(failure_info, log_dir)

    # Parse AI-Perf results (also counts clients)
    metrics = parse_aiperf_client_results(log_dir)

    # Extract client count from metrics
    num_clients = metrics.get("num_clients", 0)

    # Add phase information to metrics (store as string for JSON serialization)
    metrics["test_phase"] = test_phase.name.lower()

    # Process and print phase-specific results
    if print_output:
        _process_test_phase_results(test_phase, metrics, success_threshold)

    # Print summary
    if print_output:
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


def process_overflow_recovery_test(
    overflow_path: str,
    recovery_path: str,
    tablefmt: str = "fancy_grid",
    sla: Optional[float] = None,
    success_threshold: float = 90.0,
) -> Dict[str, Any]:
    """
    Process paired overflow/recovery test and print combined summary.

    Args:
        overflow_path: Path to overflow test directory
        recovery_path: Path to recovery test directory
        tablefmt: Table format for output
        sla: Optional SLA threshold
        success_threshold: Success rate threshold for pass/fail (default: 90.0)

    Returns:
        Combined results dictionary
    """
    overflow_results = process_single_test(
        overflow_path, tablefmt, sla, success_threshold, print_output=False
    )
    recovery_results = process_single_test(
        recovery_path, tablefmt, sla, success_threshold, print_output=False
    )

    combined_metrics = {
        "total_requests": overflow_results["metrics"]["total_requests"]
        + recovery_results["metrics"]["total_requests"],
        "successful_requests": overflow_results["metrics"]["successful_requests"]
        + recovery_results["metrics"]["successful_requests"],
        "failed_requests": overflow_results["metrics"]["failed_requests"]
        + recovery_results["metrics"]["failed_requests"],
        # Performance metrics from recovery phase
        "latencies": recovery_results["metrics"].get("latencies", []),
        "ttft": recovery_results["metrics"].get("ttft", []),
        "itl": recovery_results["metrics"].get("itl", []),
        "throughputs": recovery_results["metrics"].get("throughputs", []),
        "p50_latencies": recovery_results["metrics"].get("p50_latencies", []),
        "p90_latencies": recovery_results["metrics"].get("p90_latencies", []),
        "p99_latencies": recovery_results["metrics"].get("p99_latencies", []),
    }

    base_path = overflow_path
    if overflow_path.endswith(OVERFLOW_SUFFIX):
        base_path = overflow_path[: -len(OVERFLOW_SUFFIX)]
    test_log_path = os.path.join(base_path, "test.log.txt")
    startup_time, _ = parse_test_log(test_log_path)
    recovery_time = calculate_recovery_time(failure_info=[], process_logs_dir=base_path)

    if overflow_results["metrics"]["total_requests"] == 0:
        overflow_rate = 0
    else:
        overflow_rate = (
            overflow_results["metrics"]["failed_requests"]
            / overflow_results["metrics"]["total_requests"]
            * 100
        )

    if recovery_results["metrics"]["total_requests"] == 0:
        recovery_rate = 0
    else:
        recovery_rate = (
            recovery_results["metrics"]["successful_requests"]
            / recovery_results["metrics"]["total_requests"]
            * 100
        )

    logging.info(
        "\n" + "=" * 60 + "\n"
        "SESSION SUMMARY - COMBINED OVERFLOW/RECOVERY TEST\n" + "=" * 60 + "\n"
        "\nPhase Breakdown:\n"
        f"  Overflow: {overflow_results['metrics']['failed_requests']}/"
        f"{overflow_results['metrics']['total_requests']} rejected ({overflow_rate:.1f}%)\n"
        f"  Recovery: {recovery_results['metrics']['successful_requests']}/"
        f"{recovery_results['metrics']['total_requests']} succeeded ({recovery_rate:.1f}%)"
    )

    print_summary_table(
        log_dir=base_path,
        num_clients=overflow_results["num_clients"],
        startup_time=startup_time,
        recovery_time=recovery_time,
        metrics=combined_metrics,
        tablefmt=tablefmt,
        sla=sla,
    )

    return {
        "log_dir": base_path,
        "num_clients": overflow_results["num_clients"],
        "startup_time": startup_time,
        "recovery_time": recovery_time,
        "metrics": combined_metrics,
        "overflow_results": overflow_results,
        "recovery_results": recovery_results,
    }


def main(
    logs_dir: Optional[str] = None,
    log_paths: Optional[List[str]] = None,
    tablefmt: str = "grid",
    sla: Optional[float] = None,
    success_threshold: float = 90.0,
    print_output: bool = True,
):
    """
    Main parser entry point with support for multiple log paths.

    Args:
        logs_dir: Base directory for logs (optional)
        log_paths: List of log directories to process
        tablefmt: Table format for output
        sla: Service level agreement for latency (optional)
        success_threshold: Success rate threshold for pass/fail (default: 90.0)
        print_output: If True, print tables and summaries. If False, only return results.

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
                logging.info(f"\nProcessing: {full_path}")
                results = process_single_test(
                    full_path, tablefmt, sla, success_threshold, print_output
                )
                all_results.append(results)
            else:
                logging.warning(f"{full_path} is not a valid directory, skipping...")

        # If multiple tests, also log combined summary
        if len(all_results) > 1 and print_output:
            total_requests = sum(r["metrics"]["total_requests"] for r in all_results)
            total_successful = sum(
                r["metrics"]["successful_requests"] for r in all_results
            )
            total_failed = sum(r["metrics"]["failed_requests"] for r in all_results)

            # Build summary message
            summary_lines = [
                "\n" + "=" * 60,
                "COMBINED TEST SUMMARY",
                "=" * 60,
                f"Total Tests: {len(all_results)}",
                f"Total Requests: {total_requests}",
                f"Total Successful: {total_successful}",
                f"Total Failed: {total_failed}",
            ]

            if total_requests > 0:
                summary_lines.append(
                    f"Overall Success Rate: {(total_successful/total_requests)*100:.2f}%"
                )

            # Check if this is an overflow/recovery pair and show timing info
            has_overflow = any(
                r["log_dir"].endswith(OVERFLOW_SUFFIX) for r in all_results
            )
            has_recovery = any(
                r["log_dir"].endswith(RECOVERY_SUFFIX) for r in all_results
            )

            if has_overflow and has_recovery:
                # Find startup time from overflow phase
                for r in all_results:
                    if r["log_dir"].endswith(OVERFLOW_SUFFIX) and r.get("startup_time"):
                        summary_lines.append(f"Startup Time: {r['startup_time']}")
                        break

                # Find recovery time stored in recovery phase
                for r in all_results:
                    if r["log_dir"].endswith(RECOVERY_SUFFIX) and r.get(
                        "recovery_time"
                    ):
                        summary_lines.append(
                            f"Recovery Time (gap between phases): {r['recovery_time']}"
                        )
                        break

            summary_lines.append("=" * 60 + "\n")
            logging.info("\n".join(summary_lines))

        return all_results

    elif logs_dir:
        # Process single directory
        return process_single_test(
            logs_dir, tablefmt, sla, success_threshold, print_output
        )
    else:
        logging.error("Must provide either logs_dir or log_paths")
        return None


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Parse fault tolerance test results")
    parser.add_argument(
        "log_dir", type=str, help="Directory containing test logs and results"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        logging.error(f"{args.log_dir} is not a valid directory")
        exit(1)

    main(args.log_dir)
