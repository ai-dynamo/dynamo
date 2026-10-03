#!/usr/bin/env python3

import argparse
import collections
import datetime as dt
import json
import pathlib
import re


ANSI = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")
FIELD = lambda name: re.compile(rf'\b{name}=(?:"([^"]+)"|(\d+))')
MARKERS = (
    "MEASUREMENT_STARTED",
    "SENDING_ENDED",
    "MEASUREMENT_ENDED",
    "PROFILING_ENDED",
    "FRONTENDS_STOPPED",
    "COMPLETE",
)


def parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def field(line: str, name: str):
    match = FIELD(name).search(line)
    if not match:
        return None
    return match.group(1) or match.group(2)


def read_markers(run_dir: pathlib.Path):
    parsed = {}
    for name in MARKERS:
        path = run_dir / name
        if path.exists():
            parsed[name] = parse_time(path.read_text().strip())
    return parsed


def warning_phase(timestamp, markers):
    if timestamp is None:
        return "unknown"
    measurement_started = markers.get("MEASUREMENT_STARTED")
    sending_ended = markers.get("SENDING_ENDED")
    profiling_ended = markers.get("PROFILING_ENDED")
    if measurement_started is None or sending_ended is None or profiling_ended is None:
        return "unknown"
    if timestamp < measurement_started:
        return "startup"
    if timestamp < sending_ended:
        return "measurement"
    if timestamp <= profiling_ended:
        return "grace"
    return "teardown"


def analyze_log(path: pathlib.Path, markers):
    warnings = []
    registrations = {}
    first_timestamp = None
    last_timestamp = None
    with path.open(errors="replace") as stream:
        for raw_line in stream:
            line = ANSI.sub("", raw_line)
            timestamp_match = TIMESTAMP.match(line)
            timestamp = parse_time(timestamp_match.group(1)) if timestamp_match else None
            if timestamp is not None:
                first_timestamp = first_timestamp or timestamp
                last_timestamp = timestamp
            publisher_id = field(line, "publisher_id")
            if "EventPublisher registered with discovery" in line and publisher_id:
                registrations[publisher_id] = timestamp.isoformat() if timestamp else None
            if "Direct-ZMQ publisher lane is full" not in line:
                continue
            warnings.append(
                {
                    "timestamp": timestamp,
                    "phase": warning_phase(timestamp, markers),
                    "topic": field(line, "topic") or "unknown",
                    "publisher_id": publisher_id or "unknown",
                    "group_id": field(line, "group_id") or "unknown",
                }
            )

    by_phase = collections.Counter(warning["phase"] for warning in warnings)
    by_topic = collections.Counter(warning["topic"] for warning in warnings)
    by_phase_topic = collections.Counter(
        f'{warning["phase"]}:{warning["topic"]}' for warning in warnings
    )
    by_publisher = collections.Counter(warning["publisher_id"] for warning in warnings)
    by_group = collections.Counter(warning["group_id"] for warning in warnings)
    by_second = collections.Counter(
        warning["timestamp"].replace(microsecond=0).isoformat()
        for warning in warnings
        if warning["timestamp"]
    )
    warning_publishers = set(by_publisher)
    profiling_ended = markers.get("PROFILING_ENDED")
    return (
        {
            "path": str(path),
            "first_log_timestamp": first_timestamp.isoformat()
            if first_timestamp
            else None,
            "last_log_timestamp": last_timestamp.isoformat()
            if last_timestamp
            else None,
            "covers_profiling_end": bool(
                last_timestamp
                and profiling_ended
                and last_timestamp >= profiling_ended
            ),
            "count": len(warnings),
            "first": warnings[0]["timestamp"].isoformat()
            if warnings and warnings[0]["timestamp"]
            else None,
            "last": warnings[-1]["timestamp"].isoformat()
            if warnings and warnings[-1]["timestamp"]
            else None,
            "duration_seconds": (
                (warnings[-1]["timestamp"] - warnings[0]["timestamp"]).total_seconds()
                if len(warnings) > 1
                and warnings[0]["timestamp"]
                and warnings[-1]["timestamp"]
                else 0.0
            ),
            "by_phase": dict(by_phase),
            "by_topic": dict(by_topic),
            "by_phase_topic": dict(by_phase_topic),
            "by_publisher": dict(by_publisher),
            "by_group": dict(by_group),
            "by_second": dict(by_second),
            "warning_publisher_registrations": {
                publisher_id: registrations.get(publisher_id)
                for publisher_id in sorted(warning_publishers)
            },
        },
        warnings,
    )


def metric_value(metrics, name):
    samples = metrics.get(name, [])
    return sum(float(sample.get("value", 0)) for sample in samples)


def analyze_metrics(path: pathlib.Path, warning_times):
    snapshots = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            metrics = record.get("metrics", {})
            if not any("active_sequence_zmq_ingress" in key for key in metrics):
                continue
            names = {
                key for key in metrics if "active_sequence_zmq_ingress" in key
            } | {
                "dynamo_frontend_active_requests",
                "dynamo_frontend_inflight_requests",
                "dynamo_tokio_global_queue_depth",
            }
            timestamp = dt.datetime.fromtimestamp(
                record["timestamp_ns"] / 1e9, dt.timezone.utc
            )
            snapshots.append(
                {
                    "timestamp": timestamp,
                    "endpoint": record.get("endpoint_url"),
                    "values": {name: metric_value(metrics, name) for name in names},
                }
            )

    result = {"snapshot_count": len(snapshots), "endpoints": {}}
    by_endpoint = collections.defaultdict(list)
    for snapshot in snapshots:
        by_endpoint[snapshot["endpoint"]].append(snapshot)
    for endpoint, endpoint_snapshots in by_endpoint.items():
        endpoint_snapshots.sort(key=lambda snapshot: snapshot["timestamp"])
        endpoint_names = sorted(
            set().union(*(snapshot["values"] for snapshot in endpoint_snapshots))
        )
        for snapshot in endpoint_snapshots:
            for name in endpoint_names:
                snapshot["values"].setdefault(name, 0.0)
        first = endpoint_snapshots[0]
        last = endpoint_snapshots[-1]
        closest = []
        for warning_time in warning_times:
            if warning_time is None:
                continue
            snapshot = min(
                endpoint_snapshots,
                key=lambda item: abs((item["timestamp"] - warning_time).total_seconds()),
            )
            closest.append(
                {
                    "warning_time": warning_time.isoformat(),
                    "snapshot_time": snapshot["timestamp"].isoformat(),
                    "values": snapshot["values"],
                }
            )
        result["endpoints"][endpoint] = {
            "first": {
                "timestamp": first["timestamp"].isoformat(),
                "values": first["values"],
            },
            "last": {
                "timestamp": last["timestamp"].isoformat(),
                "values": last["values"],
            },
            "delta": {
                name: last["values"][name] - first["values"][name]
                for name in endpoint_names
            },
            "closest_to_warning": closest,
        }
    return result


def phase_windows(markers):
    result = {}
    for phase, start_name, end_name in (
        ("measurement", "MEASUREMENT_STARTED", "SENDING_ENDED"),
        ("grace", "SENDING_ENDED", "PROFILING_ENDED"),
    ):
        start = markers.get(start_name)
        end = markers.get(end_name)
        if start and end:
            result[phase] = {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_seconds": (end - start).total_seconds(),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args()
    markers = read_markers(args.run_dir)
    analyzed = [
        analyze_log(args.run_dir / f"frontend-numa{index}.log", markers)
        for index in (0, 1)
    ]
    logs = [summary for summary, _ in analyzed]
    warnings = [warning for _, items in analyzed for warning in items]
    by_phase = collections.Counter(warning["phase"] for warning in warnings)
    by_topic = collections.Counter(warning["topic"] for warning in warnings)
    windows = phase_windows(markers)
    result = {
        "markers": {name: timestamp.isoformat() for name, timestamp in markers.items()},
        "phase_windows": windows,
        "coverage": {
            "required_through": markers.get("PROFILING_ENDED").isoformat()
            if markers.get("PROFILING_ENDED")
            else None,
            "frontends_stopped": markers.get("FRONTENDS_STOPPED").isoformat()
            if markers.get("FRONTENDS_STOPPED")
            else None,
            "frontend_log_capture_covers_full_grace": bool(
                markers.get("FRONTENDS_STOPPED")
                and markers.get("PROFILING_ENDED")
                and markers["FRONTENDS_STOPPED"] >= markers["PROFILING_ENDED"]
            ),
            "all_frontend_logs_have_post_grace_timestamp": bool(logs)
            and all(log["covers_profiling_end"] for log in logs),
        },
        "lane_full": {
            "count": len(warnings),
            "measurement_and_grace_count": by_phase["measurement"]
            + by_phase["grace"],
            "by_phase": dict(by_phase),
            "by_topic": dict(by_topic),
            "phase_rates_per_second": {
                phase: by_phase[phase] / window["duration_seconds"]
                if window["duration_seconds"] > 0
                else None
                for phase, window in windows.items()
            },
        },
        "logs": logs,
    }
    warning_times = [
        warning["timestamp"]
        for _, items in analyzed
        for warning in items[:1]
        if warning["timestamp"]
    ]
    metrics_path = args.run_dir / "load_artifacts" / "server_metrics_export.jsonl"
    if metrics_path.exists():
        result["metrics"] = analyze_metrics(metrics_path, warning_times)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
