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

"""Parser for legacy client results (JSONL format) in fault tolerance tests."""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from tabulate import tabulate


def parse_test_log(file_path):
    """Parse test log for startup time and failure info.

    Args:
        file_path: Path to test.log.txt

    Returns:
        Tuple of (startup_time_seconds, fault_time_datetime, start_cmd)
    """
    start_time = None
    ready_time = None
    fault_time = None
    start_cmd: Optional[List[str]] = None

    if not os.path.isfile(file_path):
        return None, None, None

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if "Starting Deployment fault-tolerance-test with spec" in line:
                start_time = datetime.fromisoformat(
                    line.split(" ")[1].replace("T", " ")
                )
                start_cmd = []
            elif "Deployment fault-tolerance-test is ready" in line:
                ready_time = datetime.fromisoformat(
                    line.split(" ")[1].replace("T", " ")
                )
            elif "Injecting failure for:" in line:
                fault_time = datetime.fromisoformat(
                    line.split(" ")[1].replace("T", " ")
                )

    startup_time = (
        (ready_time - start_time).total_seconds() if start_time and ready_time else None
    )
    return startup_time, fault_time, start_cmd


def parse_client_logs(test_dir, expected_length=100):
    """Parse JSONL client logs from legacy client output.

    Args:
        test_dir: Directory containing client_N.log.txt files
        expected_length: Expected output length to validate success

    Returns:
        pandas DataFrame with all client request results
    """
    all_logs = []

    for file in os.listdir(test_dir):
        if file.startswith("client_") and file.endswith(".log.txt"):
            with open(os.path.join(test_dir, file), "r") as f:
                request_number = 0
                for line in f:
                    request_number += 1
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    for result in data["results"]:
                        log_entry = {
                            "time": datetime.fromisoformat(
                                data["time"].replace("T", " ")
                            ),
                            "status": result["status"],
                            "request_elapsed_time": result["request_elapsed_time"],
                            "request_number": request_number - 1,
                            "client": file.split("_")[1].split(".")[0],
                        }

                        # Check if request was successful
                        if (
                            "result" in result
                            and result["result"]
                            and "choices" in result["result"]
                            and result["result"]["choices"]
                        ):
                            log_entry["success"] = True

                            # Extract content from response
                            if "content" in result["result"]["choices"][0]["message"]:
                                content = result["result"]["choices"][0]["message"][
                                    "content"
                                ]
                            elif (
                                "reasoning_content"
                                in result["result"]["choices"][0]["message"]
                            ):
                                content = result["result"]["choices"][0]["message"][
                                    "reasoning_content"
                                ]
                            else:
                                content = ""

                            # Validate content length
                            if not content or len(content) < expected_length:
                                log_entry["success"] = False
                        else:
                            log_entry["success"] = False

                        all_logs.append(log_entry)

    if len(all_logs):
        df = pd.DataFrame(all_logs)
        df.sort_values("time", inplace=True)
        return df

    return None


def calculate_metrics(df, fault_time, sla=None):
    """Calculate metrics before and after fault injection.

    Args:
        df: DataFrame with client request results
        fault_time: Datetime when fault was injected
        sla: Optional SLA threshold for latency violations

    Returns:
        Tuple of metrics (success_before, failure_before, success_after,
        failure_after, avg_before, std_before, avg_after, std_after,
        violations_before, violations_after)
    """
    if fault_time:
        before_fault = df[df["time"] <= fault_time]
        after_fault = df[df["time"] > fault_time]
    else:
        before_fault = df
        after_fault = None

    # Latency metrics (only successful requests)
    successful_before = before_fault[before_fault["success"]]
    avg_before = successful_before["request_elapsed_time"].mean()
    std_before = successful_before["request_elapsed_time"].std()
    success_before_count = before_fault["success"].sum()
    failure_before_count = len(before_fault) - success_before_count

    avg_after, std_after, success_after_count, failure_after_count = (
        None,
        None,
        None,
        None,
    )
    if after_fault is not None and not after_fault.empty:
        successful_after = after_fault[after_fault["success"]]
        avg_after = successful_after["request_elapsed_time"].mean()
        std_after = successful_after["request_elapsed_time"].std()
        success_after_count = after_fault["success"].sum()
        failure_after_count = len(after_fault) - success_after_count

    # SLA violations (only successful requests exceeding the SLA)
    if sla:
        violations_before = (successful_before["request_elapsed_time"] > sla).sum()
        violations_after = (
            (successful_after["request_elapsed_time"] > sla).sum()
            if after_fault is not None and not after_fault.empty
            else None
        )
    else:
        violations_before = None
        violations_after = None

    return (
        success_before_count,
        failure_before_count,
        success_after_count,
        failure_after_count,
        avg_before,
        std_before,
        avg_after,
        std_after,
        violations_before,
        violations_after,
    )


def parse_process_log(log_dir, process_name):
    """Parse process logs to find ready times for recovery calculation.

    Args:
        log_dir: Directory containing process log files
        process_name: Name of process to look for (Frontend, VllmDecodeWorker, etc.)

    Returns:
        Dictionary mapping replica names to list of (timestamp, message, relative_time) tuples
    """
    process_ready_pattern = {
        "Frontend": re.compile(r"added model"),
        "VllmDecodeWorker": re.compile(
            r"VllmWorker for (?P<model_name>.*?) has been initialized"
        ),
        "VllmPrefillWorker": re.compile(
            r"VllmWorker for (?P<model_name>.*?) has been initialized"
        ),
        "decode": re.compile(
            r"Model registration succeeded|Decode worker handler initialized|Worker handler initialized"
        ),
        "prefill": re.compile(
            r"Model registration succeeded|Prefill worker handler initialized|Worker handler initialized"
        ),
        "TRTLLMWorker": re.compile(
            r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
        ),
        "TRTLLMDecodeWorker": re.compile(
            r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
        ),
        "TRTLLMPrefillWorker": re.compile(
            r"TrtllmWorker for (?P<model_name>.*?) has been initialized|Model registration succeeded"
        ),
    }

    if not os.path.isdir(log_dir):
        return {}

    ready_times: Dict[str, List] = {}

    for entry in os.listdir(log_dir):
        if entry.endswith(".log") and "metrics" not in entry:
            replica_number = entry.split(".")[0]

            if replica_number not in ready_times:
                ready_times[replica_number] = []

            process_start_time = None

            with open(os.path.join(log_dir, entry), "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Try to parse as JSONL first
                    try:
                        json_data = json.loads(line)
                        if "time" in json_data:
                            timestamp = datetime.fromisoformat(
                                json_data["time"].replace("Z", "")
                            )
                            log_message = json_data.get("message", "")
                        else:
                            continue
                    except (json.JSONDecodeError, ValueError, KeyError):
                        # Fall back to readable format parsing
                        clean_line = re.sub(
                            r"\x1b\[.*?m", "", line
                        )  # Remove ANSI codes
                        if not clean_line:
                            continue

                        parts = clean_line.split()
                        if len(parts) < 2:
                            continue

                        try:
                            timestamp = datetime.fromisoformat(
                                parts[0].replace("Z", "")
                            )
                        except ValueError:
                            continue

                        log_message = " ".join(parts[1:])

                    if not process_start_time:
                        process_start_time = timestamp

                    relative_time = (timestamp - process_start_time).total_seconds()

                    # Check for process ready patterns
                    if process_name in process_ready_pattern:
                        if process_ready_pattern[process_name].search(log_message):
                            if "previous" in entry:
                                location = 0
                            else:
                                location = -1
                            ready_times[replica_number].insert(
                                location, (timestamp, log_message, relative_time)
                            )

    return ready_times


def calculate_recovery_time(test_dir, failure_type, fault_time):
    """Calculate recovery time after fault injection.

    Args:
        test_dir: Directory containing test logs
        failure_type: Type of failure injected
        fault_time: Datetime when fault was injected

    Returns:
        Recovery time in seconds or None if not applicable
    """
    if not fault_time:
        return None

    processes = [
        "Frontend",
        "VllmDecodeWorker",
        "VllmPrefillWorker",
        "decode",
        "prefill",
        "TRTLLMWorker",
        "TRTLLMDecodeWorker",
        "TRTLLMPrefillWorker",
    ]

    process_start = {}
    start_time = None

    for process in processes:
        starts = parse_process_log(os.path.join(test_dir, process), process)
        if starts:
            process_start[process] = starts

    last_recovery_time = 0
    for process, replicas in process_start.items():
        for replica, container_starts in replicas.items():
            for starts in container_starts:
                start_time = starts[0]
                recovery_time = (start_time - fault_time).total_seconds()
                if recovery_time > last_recovery_time:
                    last_recovery_time = recovery_time

    if last_recovery_time == 0:
        return None
    return last_recovery_time


def process_test_directory(test_dir, sla):
    """Process a single test directory with legacy client results.

    Args:
        test_dir: Path to test directory
        sla: Optional SLA threshold for latency

    Returns:
        Dictionary with test results or None if invalid
    """
    if "test_fault_scenario" not in test_dir:
        return {}

    test_name = test_dir.split("test_fault_scenario[", 1)[1].rstrip("]")
    failure_type = test_name.split("-")[-1]
    test_prefix = "-".join(test_name.split("-")[:-1])

    startup_time, fault_time, start_cmd = parse_test_log(
        os.path.join(test_dir, "test.log.txt")
    )
    df = parse_client_logs(test_dir)

    if df is None or df.empty:
        return None

    (
        success_before,
        failure_before,
        success_after,
        failure_after,
        avg_before,
        std_before,
        avg_after,
        std_after,
        violations_before,
        violations_after,
    ) = calculate_metrics(df, fault_time, sla)

    recovery_time = calculate_recovery_time(test_dir, failure_type, fault_time)

    return {
        "test": test_prefix,
        "cmd": start_cmd,
        "failure": failure_type,
        "start_time": startup_time,
        "success_before_requests": success_before,
        "failed_before_requests": failure_before,
        "success_after_requests": success_after,
        "failed_after_requests": failure_after,
        "avg_latency_before": avg_before,
        "std_latency_before": std_before,
        "avg_latency_after": avg_after,
        "std_latency_after": std_after,
        "violations_before": violations_before,
        "violations_after": violations_after,
        "recovery_time": recovery_time,
    }


def main(logs_dir, tablefmt, log_paths=None, sla=None):
    """Main entry point for parsing legacy client results.

    Args:
        logs_dir: Base directory containing test results
        tablefmt: Table format for output (e.g., "fancy_grid")
        log_paths: Optional list of specific log paths to process
        sla: Optional SLA threshold for latency violations
    """
    results = []

    if log_paths:
        # Process multiple log paths
        for log_path in log_paths:
            result = process_test_directory(log_path, sla)
            if result:
                results.append(result)
    elif logs_dir:
        # Process all test directories in logs_dir
        for entry in os.listdir(logs_dir):
            if entry.startswith("test_fault_scenario[") and os.path.isdir(
                os.path.join(logs_dir, entry)
            ):
                result = process_test_directory(os.path.join(logs_dir, entry), sla)
                if result:
                    results.append(result)

    # Group results by test prefix
    grouped: dict[str, list[dict[str, Any]]] = {}
    commands = {}
    for res in results:
        test_prefix = res["test"]
        if test_prefix not in grouped:
            grouped[test_prefix] = []
            commands[test_prefix] = res["cmd"]
        grouped[test_prefix].append(res)

    # Define order for failure types
    order = [
        "none",
        "frontend",
        "frontend_pod",
        "decode_worker",
        "decode_worker_pod",
        "prefill_worker",
        "prefill_worker_pod",
        "vllm_decode_engine_core",
        "vllm_prefill_engine_core",
        "sglang_decode_scheduler",
        "sglang_decode_detokenizer",
        "sglang_prefill_scheduler",
        "sglang_prefill_detokenizer",
        "trtllm_decode_engine_core",
        "trtllm_prefill_engine_core",
    ]

    # Print grouped tables
    for test_prefix, group in grouped.items():
        # Reorder results by failure type
        new_group = []
        for failure in order:
            for res in group:
                if failure == res["failure"]:
                    new_group.append(res)
        group = new_group

        # Define table headers
        if sla:
            headers = [
                "Failure",
                "Startup",
                "Success\nBefore",
                "Failed\nBefore",
                "Success\nAfter",
                "Failed\nAfter",
                "Latency\nBefore",
                "Latency\nAfter",
                "Violations\nBefore",
                "Violations\nAfter",
                "Recovery",
            ]
        else:
            headers = [
                "Failure",
                "Startup",
                "Success\nBefore",
                "Failed\nBefore",
                "Success\nAfter",
                "Failed\nAfter",
                "Latency\nBefore",
                "Latency\nAfter",
                "Recovery",
            ]

        rows = []
        for res in group:
            if sla:
                row = [
                    res["failure"],
                    res["start_time"],
                    res["success_before_requests"],
                    res["failed_before_requests"],
                    res["success_after_requests"],
                    res["failed_after_requests"],
                    res["avg_latency_before"],
                    res["avg_latency_after"],
                    res["violations_before"],
                    res["violations_after"],
                    res["recovery_time"],
                ]
            else:
                row = [
                    res["failure"],
                    res["start_time"],
                    res["success_before_requests"],
                    res["failed_before_requests"],
                    res["success_after_requests"],
                    res["failed_after_requests"],
                    res["avg_latency_before"],
                    res["avg_latency_after"],
                    res["recovery_time"],
                ]
            rows.append(row)

        print(f"\nTest Group: {test_prefix}")
        print(
            tabulate(
                rows,
                headers,
                tablefmt=tablefmt,
                floatfmt=".2f",
                missingval="N/A",
                numalign="right",
                stralign="center",
            )
        )
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Parse legacy client test results")
    parser.add_argument("--log-dir", default=".", help="Path to the logs directory")
    parser.add_argument(
        "--format", choices=["fancy", "markdown"], default="fancy", help="Table format"
    )
    parser.add_argument(
        "--sla", type=float, default=None, help="SLA threshold for latency"
    )

    args = parser.parse_args()

    # Map format choices to tabulate formats
    tablefmt = "fancy_grid" if args.format == "fancy" else "pipe"

    main(args.log_dir, tablefmt, sla=args.sla)
