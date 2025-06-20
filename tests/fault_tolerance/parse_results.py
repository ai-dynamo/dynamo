import argparse
import json
import os
import re
from datetime import datetime

import pandas as pd
from tabulate import tabulate


def parse_test_log(file_path):
    start_time = None
    ready_time = None
    fault_time = None
    start_cmd = None
    if not os.path.isfile(file_path):
        return None, None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if "Running command: dynamo serve" in line:
                start_time = datetime.fromisoformat(
                    line.split(" ")[1].replace("T", " ")
                )
                start_cmd = line.split("Running command:")[1]
            elif "Deployment Ready" in line:
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


def parse_client_logs(test_dir):
    all_logs = []
    for file in os.listdir(test_dir):
        if file.startswith("client_") and file.endswith(".log.txt"):
            with open(os.path.join(test_dir, file), "r") as f:
                request_number = 0
                for line in f:
                    request_number += 1
                    data = json.loads(line.strip())
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
                        if (
                            "result" in result
                            and result["result"]
                            and "choices" in result["result"]
                            and result["result"]["choices"]
                        ):
                            log_entry["success"] = True
                            content = result["result"]["choices"][0]["message"][
                                "content"
                            ]
                            if not content:
                                log_entry["success"] = False
                        else:
                            log_entry["success"] = False
                        all_logs.append(log_entry)
    if len(all_logs):
        df = pd.DataFrame(all_logs)
        df.sort_values("time", inplace=True)
        return df
    else:
        return None


def calculate_metrics(df, fault_time):
    success = df["success"].sum()
    failure = len(df) - success

    if fault_time:
        before_fault = df[df["time"] < fault_time]
        after_fault = df[df["time"] >= fault_time]
    else:
        before_fault = df
        after_fault = None

    avg_before = before_fault[before_fault["success"]]["request_elapsed_time"].mean()
    std_before = before_fault[before_fault["success"]]["request_elapsed_time"].std()

    if after_fault is None or after_fault.empty:
        avg_after = None
        std_after = None
    else:
        avg_after = after_fault[after_fault["success"]]["request_elapsed_time"].mean()
        std_after = after_fault[after_fault["success"]]["request_elapsed_time"].std()

    return success, failure, avg_before, std_before, avg_after, std_after


def parse_process_log(log_dir, process_name):
    process_start_line = {
        "dynamo_Frontend": "added model",
        "dynamo_VllmWorker": "Starting VllmWorker instance",
        "dynamo_Processor": "Starting Processor instance with all registered endpoints",
        "dynamo_PrefillWorker": "Starting PrefillWorker instance with all registered endpoints",
    }
    process_end_line = {
        "dynamo_Frontend": "SIGTERM received, starting graceful shutdown",
        "dynamo_VllmWorker": "Received shutdown signal, shutting down DistributedRuntime",
        "dynamo_Processor": "Received signal 15, initiating graceful shutdown",
        "dynamo_PrefillWorker": "Shutdown hooks completed successfully",
    }
    process_log_path = os.path.join(log_dir, "error.log")

    if not os.path.isfile(process_log_path):
        return None, None

    process_starts = []
    process_ends = []

    with open(process_log_path, "r") as f:
        for line in f:
            clean_line = re.sub(r"\x1b\[.*?m", "", line.strip())  # Remove ANSI codes
            if not clean_line:
                continue

            parts = clean_line.split()
            if len(parts) < 2:
                continue

            try:
                # Parse timestamp (remove 'Z' for naive datetime)
                timestamp = datetime.fromisoformat(parts[0].replace("Z", ""))
            except ValueError:
                continue

            log_message = " ".join(parts[1:])

            # Check for process start lines
            if process_name in process_start_line:
                if process_start_line[process_name] in log_message:
                    process_starts.append((timestamp, log_message))

            # Check for process end lines
            if process_name in process_end_line:
                if process_end_line[process_name] in log_message:
                    process_ends.append((timestamp, log_message))

    return process_starts, process_ends


def parse_watcher_log(test_dir, fault_time):
    before_requests = []
    after_requests = []
    watcher_log_path = os.path.join(test_dir, "watcher.log.txt")
    if not os.path.isfile(watcher_log_path):
        return None, None
    with open(watcher_log_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if "metrics" not in data:
                continue
            entry_time = datetime.fromisoformat(data["time"].replace("T", " "))
            for metric in data["metrics"]:
                if len(metric) != 2:
                    continue
                _, metric_data = metric
                if (
                    "num_requests_waiting" in metric_data
                    and "request_active_slots" in metric_data
                    and metric_data["request_active_slots"] > 0
                ):
                    if fault_time is None or entry_time < fault_time:
                        before_requests.append(metric_data["num_requests_waiting"])
                    else:
                        after_requests.append(metric_data["num_requests_waiting"])
    avg_before = (
        sum(before_requests) / len(before_requests) if before_requests else None
    )
    avg_after = sum(after_requests) / len(after_requests) if after_requests else None
    return avg_before, avg_after


def calculate_recovery_time(test_dir, failure_type):
    processes = [
        "dynamo_Frontend",
        "dynamo_Processor",
        "dynamo_VllmWorker",
        "dynamo_PrefillWorker",
    ]

    process_start_ends = {}

    for process in processes:
        starts, ends = parse_process_log(os.path.join(test_dir, process), process)
        if starts:
            process_start_ends[process] = (starts, ends)

    if failure_type == "processor":
        end_times = process_start_ends["dynamo_Processor"][1]
        start_time = process_start_ends["dynamo_Processor"][0][-1][0]
        end_time = None
        for x in end_times:
            if x[0] < start_time:
                end_time = x[0]
        if end_time is None:
            return None
        return (start_time - end_time).total_seconds()

    if failure_type == "frontend":
        end_times = process_start_ends["dynamo_Frontend"][1]
        start_time = process_start_ends["dynamo_Frontend"][0][-1][0]
        end_time = None
        for x in end_times:
            if x[0] < start_time:
                end_time = x[0]
        if end_time is None:
            return None
        return (start_time - end_time).total_seconds()

    if failure_type == "decode_worker":
        end_times = [
            x
            for x in process_start_ends["dynamo_VllmWorker"][1]
            if "VllmWorker:1" in x[1]
        ]
        start_times = [
            x
            for x in process_start_ends["dynamo_VllmWorker"][0]
            if "VllmWorker:1" in x[1]
        ]
        if not start_times:
            return None
        start_time = start_times[-1][0]
        end_time = None
        for x in end_times:
            if x[0] < start_time:
                end_time = x[0]
        if end_time is None:
            return None
        return (start_time - end_time).total_seconds()

    if failure_type == "prefill_worker":
        if "dynamo_PrefillWorker" not in process_start_ends:
            return None
        end_times = [
            x
            for x in process_start_ends["dynamo_PrefillWorker"][1]
            if "PrefillWorker:1" in x[1]
        ]
        start_times = [
            x
            for x in process_start_ends["dynamo_PrefillWorker"][0]
            if "PrefillWorker:1" in x[1]
        ]
        start_time = start_times[-1][0]
        end_time = None
        for x in end_times:
            if x[0] < start_time:
                end_time = x[0]
        if end_time is None:
            return None
        return (start_time - end_time).total_seconds()


def process_test_directory(test_dir):
    test_name = test_dir.split("test_worker_failure[", 1)[1].rstrip("]")
    failure_type = test_name.split("-")[-1]
    test_prefix = "-".join(test_name.split("-")[:-1])

    startup_time, fault_time, start_cmd = parse_test_log(
        os.path.join(test_dir, "test.log.txt")
    )
    df = parse_client_logs(test_dir)

    if df is None or df.empty:
        return None
    pending_requests_before, pending_requests_after = parse_watcher_log(
        test_dir, fault_time
    )
    success, failure, avg_before, std_before, avg_after, std_after = calculate_metrics(
        df, fault_time
    )

    recovery_time = calculate_recovery_time(test_dir, failure_type)

    return {
        "test": test_prefix,
        "cmd": start_cmd,
        "failure": failure_type,
        "start_time": startup_time,
        "success_requests": success,
        "failed_requests": failure,
        "avg_latency_before": avg_before,
        "std_latency_before": std_before,
        "avg_latency_after": avg_after if avg_after is not None else "N/A",
        "std_latency_after": std_after if avg_after is not None else "N/A",
        "pending_requests_before": pending_requests_before,
        "pending_requests_after": pending_requests_after,
        "recovery_time": recovery_time,
    }


def main(logs_dir, tablefmt):
    results = []
    for entry in os.listdir(logs_dir):
        if entry.startswith("test_worker_failure[") and os.path.isdir(
            os.path.join(logs_dir, entry)
        ):
            result = process_test_directory(os.path.join(logs_dir, entry))
            if result:
                results.append(result)

    # Group results by test prefix
    grouped = {}
    commands = {}
    for res in results:
        test_prefix = res["test"]
        if test_prefix not in grouped:
            grouped[test_prefix] = []
            commands[test_prefix] = res["cmd"]
        grouped[test_prefix].append(res)

    # Print grouped tables
    for test_prefix, group in grouped.items():
        headers = [
            "Failure",
            "Startup Time",
            "Success",
            "Failed",
            "Latency Before",
            "Latency After",
            "Pending Before",
            "Pending After",
            "Recovery Time",
        ]
        rows = []
        for res in group:
            row = [
                res["failure"],
                f"{res['start_time']:.2f}" if res["start_time"] is not None else "N/A",
                res["success_requests"],
                res["failed_requests"],
                f"{res['avg_latency_before']:.2f} ± {res['std_latency_before']:.2f}"
                if not isinstance(res["avg_latency_before"], str)
                else "N/A",
                f"{res['avg_latency_after']:.2f} ± {res['std_latency_after']:.2f}"
                if not isinstance(res["avg_latency_after"], str)
                else "N/A",
                res["pending_requests_before"],
                res["pending_requests_after"],
                f"{res['recovery_time']:.2f}"
                if res["recovery_time"] is not None
                else "N/A",
            ]
            rows.append(row)

        print(f"\nTest Group: {test_prefix}")
        print(f"\nTest Command: {commands[test_prefix]}")
        print(tabulate(rows, headers, tablefmt=tablefmt))
        print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse test results")
    parser.add_argument("--log-dir", default=".", help="Path to the logs directory")
    parser.add_argument(
        "--format", choices=["fancy", "markdown"], default="fancy", help="Table format"
    )
    args = parser.parse_args()

    # Map format choices to tabulate formats
    tablefmt = (
        "fancy_grid" if args.format == "fancy" else "pipe"
    )  # Using pipe for markdown compatibility

    main(args.log_dir, tablefmt)
