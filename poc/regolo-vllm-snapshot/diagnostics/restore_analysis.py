#!/usr/bin/env python3
"""Reconstruct restore phases from sealed results, events, and agent logs."""

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys


DURATION_TOKEN = re.compile(r"([0-9]+(?:\.[0-9]+)?)(ms|us|µs|ns|h|m|s)")
DURATION_SCALE = {
    "h": 3600.0,
    "m": 60.0,
    "s": 1.0,
    "ms": 0.001,
    "us": 0.000001,
    "µs": 0.000001,
    "ns": 0.000000001,
}


def parse_go_duration(value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid Go duration: {value!r}")
    matches = list(DURATION_TOKEN.finditer(value))
    if not matches or "".join(match.group(0) for match in matches) != value:
        raise ValueError(f"invalid Go duration: {value!r}")
    return sum(float(match.group(1)) * DURATION_SCALE[match.group(2)] for match in matches)


def parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError(f"invalid timestamp: {value!r}")
    normalized = re.sub(r"(\.\d{6})\d+Z$", r"\1+00:00", value).replace("Z", "+00:00")
    parsed = dt.datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value!r}")
    return parsed.timestamp()


def pod_name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("restore log payload has no pod")
    return value.rsplit("/", 1)[-1]


def parse_agent_log(text):
    starts = {}
    summaries = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        timestamp = line.split(None, 1)[0]
        for message, destination in (
            ("=== Starting external restore ===", starts),
            ("Restore timing summary", summaries),
        ):
            marker = message + "\t"
            if marker not in line:
                continue
            payload = json.loads(line.split(marker, 1)[1])
            run_id = pod_name(payload.get("pod"))
            if run_id in destination:
                kind = "start" if destination is starts else "timing summary"
                raise ValueError(f"duplicate {kind} for {run_id}")
            destination[run_id] = {"timestamp": timestamp, "payload": payload}
    return starts, summaries


def scheduled_timestamp(run_id, events):
    stamps = []
    for event in events.get("items", []):
        if event.get("reason") != "Scheduled":
            continue
        stamp = event.get("eventTime") or event.get("firstTimestamp")
        if stamp:
            stamps.append(stamp)
    if len(stamps) != 1:
        raise ValueError(f"missing Scheduled event for {run_id}")
    return stamps[0]


def distribution(values):
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "nearest_rank_p95": ordered[p95_index],
        "max": ordered[-1],
    }


def pearson(left, right):
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def analyze(results, key, agent_log, events_by_run, expected_restores=10):
    if set(key) != {"A", "B"} or set(key.values()) != {"cold", "restore"}:
        raise ValueError("key must map A/B bijectively to cold/restore")
    run_ids = [row.get("run_id") for row in results]
    if None in run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("results require unique run_id values")

    restore_rows = [row for row in results if key.get(row.get("opaque_arm")) == "restore"]
    if len(restore_rows) != expected_restores:
        raise ValueError(
            f"expected {expected_restores} restore results, found {len(restore_rows)}"
        )
    checkpoint_sizes = {
        row["checkpoint_size_bytes"]
        for row in results
        if isinstance(row.get("checkpoint_size_bytes"), (int, float))
    }
    if len(checkpoint_sizes) != 1:
        raise ValueError("results must contain one unambiguous checkpoint size")
    checkpoint_size = checkpoint_sizes.pop()

    starts, summaries = parse_agent_log(agent_log)
    runs = []
    for ordinal, row in enumerate(restore_rows, 1):
        run_id = row["run_id"]
        if run_id not in starts:
            raise ValueError(f"missing restore start for {run_id}")
        if run_id not in summaries:
            raise ValueError(f"missing timing summary for {run_id}")
        created = parse_timestamp(row["pod_created_at"])
        started = parse_timestamp(starts[run_id]["timestamp"])
        summary = summaries[run_id]
        completed = parse_timestamp(summary["timestamp"])
        restore = summary["payload"].get("restore", {})
        phases = restore.get("phases", {})
        criu = parse_go_duration(phases.get("criu_restore_duration"))
        cuda = parse_go_duration(phases.get("cuda_duration"))
        setup = parse_go_duration(phases.get("nsrestore_setup_duration"))
        inspect = parse_go_duration(phases.get("host_inspect_duration"))
        duration = parse_go_duration(restore.get("duration"))
        phase_total = inspect + setup + criu + cuda
        if not math.isclose(phase_total, duration, rel_tol=0, abs_tol=0.000001):
            raise ValueError(f"phase durations do not sum to restore duration for {run_id}")
        scheduled_at = scheduled_timestamp(run_id, events_by_run.get(run_id, {}))
        scheduled = parse_timestamp(scheduled_at)
        for field in ("ready_s", "http_200_s", "first_token_s"):
            if not isinstance(row.get(field), (int, float)):
                raise ValueError(f"missing {field} for {run_id}")
        first_token = row["first_token_s"]
        runs.append(
            {
                "run_id": run_id,
                "block": row.get("block"),
                "restore_ordinal": ordinal,
                "pod_created_at": row["pod_created_at"],
                "scheduled_at": scheduled_at,
                "restore_start_at": starts[run_id]["timestamp"],
                "restore_summary_at": summary["timestamp"],
                "pod_to_scheduled_s": scheduled - created,
                "pod_to_restore_start_s": started - created,
                "restore_duration_s": duration,
                "host_inspect_s": inspect,
                "nsrestore_setup_s": setup,
                "criu_restore_s": criu,
                "cuda_restore_s": cuda,
                "ready_s": row["ready_s"],
                "http_200_s": row["http_200_s"],
                "first_token_s": first_token,
                "restore_summary_from_pod_s": completed - created,
                "token_after_restore_summary_s": created + first_token - completed,
                "criu_share_of_restore": criu / duration,
                "effective_checkpoint_gb_per_s": checkpoint_size / 1_000_000_000 / criu,
            }
        )

    expected_ids = {row["run_id"] for row in restore_rows}
    unexpected = (set(starts) | set(summaries)) - expected_ids
    if unexpected:
        raise ValueError(f"agent log contains unexpected restore runs: {sorted(unexpected)}")

    metric_names = (
        "pod_to_scheduled_s",
        "pod_to_restore_start_s",
        "restore_duration_s",
        "host_inspect_s",
        "nsrestore_setup_s",
        "criu_restore_s",
        "cuda_restore_s",
        "ready_s",
        "http_200_s",
        "first_token_s",
        "restore_summary_from_pod_s",
        "token_after_restore_summary_s",
        "effective_checkpoint_gb_per_s",
    )
    summary = {name: distribution([row[name] for row in runs]) for name in metric_names}
    tail = max(runs, key=lambda row: row["first_token_s"])
    first_token_excess = tail["first_token_s"] - summary["first_token_s"]["median"]
    criu_excess = tail["criu_restore_s"] - summary["criu_restore_s"]["median"]
    return {
        "schema": "dynamo_snapshot_restore_phase_analysis_v1",
        "source_agent_log_sha256": hashlib.sha256(agent_log.encode()).hexdigest(),
        "checkpoint_size_bytes": checkpoint_size,
        "restore_runs": len(runs),
        "runs": runs,
        "summary": summary,
        "correlations": {
            "criu_vs_first_token_pearson_r": pearson(
                [row["criu_restore_s"] for row in runs],
                [row["first_token_s"] for row in runs],
            ),
            "restore_ordinal_vs_criu_pearson_r": pearson(
                [row["restore_ordinal"] for row in runs],
                [row["criu_restore_s"] for row in runs],
            ),
        },
        "tail": {
            "run_id": tail["run_id"],
            "first_token_excess_over_median_s": first_token_excess,
            "criu_excess_over_median_s": criu_excess,
            "criu_share_of_tail_excess": (
                criu_excess / first_token_excess if first_token_excess > 0 else None
            ),
        },
    }


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_events(directory, results, key):
    events = {}
    for row in results:
        if key.get(row.get("opaque_arm")) != "restore":
            continue
        path = directory / f"{row['run_id']}.json"
        with path.open() as stream:
            events[row["run_id"]] = json.load(stream)
    return events


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument("--key", type=pathlib.Path, required=True)
    parser.add_argument("--agent-log", type=pathlib.Path, required=True)
    parser.add_argument("--events-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expected-restores", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    results = read_jsonl(args.results)
    with args.key.open() as stream:
        key = json.load(stream)
    agent_log = args.agent_log.read_text()
    events = read_events(args.events_dir, results, key)
    report = analyze(results, key, agent_log, events, args.expected_restores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
