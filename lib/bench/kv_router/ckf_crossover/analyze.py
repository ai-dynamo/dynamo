#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BACKENDS = ("crtc", "ckf-native", "ckf-transposed")
CAPACITY_WINDOWS_MS = (24000, 12000, 6000, 3000, 1500, 750)
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("capacity", "iso-check", "final"), required=True
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-threads", type=int, default=8)
    parser.add_argument("--query-concurrency", type=int, default=16)
    return parser.parse_args()


def load_trials(results_dir):
    trials = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict) and "backend" in value and "phase" in value:
            value["_path"] = str(path)
            trials.append(value)
    return trials


def load_publisher_support(results_dir):
    path = results_dir.parent / "corpus_manifest.json"
    if not path.is_file():
        raise SystemExit(f"missing verified corpus manifest: {path}")
    manifest = json.loads(path.read_text())
    publisher = manifest["publisher"]
    timing = manifest["publisher_timing"]
    events = manifest["header"]["totals"]["events"]

    def per_second(value, elapsed_ns):
        if elapsed_ns == 0:
            return 0.0
        return value / (elapsed_ns / 1e9)

    return {
        "relay_publication_bytes": publisher["bytes"],
        "relay_generation_events_s": per_second(events, timing["generation_ns"]),
        "relay_generation_mib_s": per_second(
            publisher["bytes"] / (1024**2), timing["generation_ns"]
        ),
        "relay_full_generation_mib_s": per_second(
            publisher["full_bytes"] / (1024**2), timing["full_generation_ns"]
        ),
        "relay_delta_generation_mib_s": per_second(
            publisher["delta_bytes"] / (1024**2), timing["delta_generation_ns"]
        ),
    }


def mean_ci(values):
    if not values:
        return (0.0, 0.0, 0.0)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return (mean, mean, mean)
    critical = T_CRITICAL_95.get(len(values), 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return (mean, mean - half, mean + half)


def paired_ratio_ci(numerator, denominator):
    by_rep_num = {int(row["repetition"]): row for row in numerator}
    by_rep_den = {int(row["repetition"]): row for row in denominator}
    repetitions = sorted(set(by_rep_num) & set(by_rep_den))
    logs = [
        math.log(
            by_rep_num[rep]["achieved_block_ops_s"]
            / by_rep_den[rep]["achieved_block_ops_s"]
        )
        for rep in repetitions
    ]
    mean, low, high = mean_ci(logs)
    return (math.exp(mean), math.exp(low), math.exp(high), len(logs))


def capacity_groups(trials):
    groups = defaultdict(list)
    for row in trials:
        if row["phase"] != "capacity":
            continue
        groups[(row["backend"], round(float(row["replay_window_ms"])))].append(row)
    return groups


def validate_capacity(groups):
    missing = []
    for backend in BACKENDS:
        for window in CAPACITY_WINDOWS_MS:
            rows = groups.get((backend, window), [])
            reps = sorted(int(row["repetition"]) for row in rows)
            if reps != [1, 2, 3, 4, 5]:
                missing.append(f"{backend}/{window}ms reps={reps}")
    if missing:
        raise SystemExit("capacity matrix incomplete: " + "; ".join(missing))


def summarize_capacity(groups, publisher_support):
    records = []
    for backend in BACKENDS:
        for window in CAPACITY_WINDOWS_MS:
            rows = groups[(backend, window)]
            throughput = mean_ci([row["achieved_block_ops_s"] for row in rows])
            mixed = mean_ci([row["achieved_mixed_ops_s"] for row in rows])
            service_p50 = mean_ci([row["lookup_service"]["p50_us"] for row in rows])
            service_p99 = mean_ci([row["lookup_service"]["p99_us"] for row in rows])
            elapsed_seconds = [row["total_elapsed_ms"] / 1000.0 for row in rows]
            record = {
                "backend": backend,
                "window_ms": window,
                "offered_block_ops_s": statistics.fmean(
                    row["nominal_offered_block_ops_s"] for row in rows
                ),
                "achieved_block_ops_s_mean": throughput[0],
                "achieved_block_ops_s_ci_low": throughput[1],
                "achieved_block_ops_s_ci_high": throughput[2],
                "achieved_mixed_ops_s_mean": mixed[0],
                "lookup_service_p50_us_mean": service_p50[0],
                "lookup_service_p99_us_mean": service_p99[0],
                "query_wait_p99_us_mean": statistics.fmean(
                    row["query_queue_wait"]["p99_us"] for row in rows
                ),
                "issue_lag_p99_us_mean": statistics.fmean(
                    row["issue_lag"]["p99_us"] for row in rows
                ),
                "scheduled_completion_p99_us_mean": statistics.fmean(
                    row["scheduled_to_completion"]["p99_us"] for row in rows
                ),
                "update_apply_p99_us_mean": statistics.fmean(
                    row["update_enqueue_to_applied"]["p99_us"] for row in rows
                ),
                "update_visible_p99_us_mean": statistics.fmean(
                    row["update_scheduled_to_applied"]["p99_us"] for row in rows
                ),
                "all_keep_up": all(row["kept_up"] for row in rows),
                "generator_valid": not any(row["generator_limited"] for row in rows),
                "maximum_update_queue_at_stop": max(
                    row["update_queue"]["at_stop"] for row in rows
                ),
                "maximum_query_queue_at_stop": max(
                    row["query_queue"]["at_stop"] for row in rows
                ),
                "maximum_update_queue": max(
                    row["update_queue"]["maximum_depth"] for row in rows
                ),
                "maximum_query_queue": max(
                    row["query_queue"]["maximum_depth"] for row in rows
                ),
                "maximum_drain_ms": max(
                    max(row["update_queue"]["drain_ms"], row["query_queue"]["drain_ms"])
                    for row in rows
                ),
                "crtc_raw_events": max(row["crtc_raw_events"] for row in rows),
                "crtc_raw_blocks": max(row["crtc_raw_blocks"] for row in rows),
                "ckf_frames": max(row["ckf_frames"] for row in rows),
                "ckf_dirty_buckets": max(row["ckf_dirty_buckets"] for row in rows),
                "ckf_bytes": max(row["ckf_bytes"] for row in rows),
                "ckf_router_ingress_mib_s_mean": statistics.fmean(
                    row["ckf_bytes"] / (1024**2) / elapsed
                    for row, elapsed in zip(rows, elapsed_seconds)
                ),
                "ckf_apply_mib_s_mean": statistics.fmean(
                    row["ckf_apply_mib_s"] for row in rows
                ),
                "ckf_full_apply_mib_s_mean": statistics.fmean(
                    row["ckf_full_apply_mib_s"] for row in rows
                ),
                "ckf_delta_apply_mib_s_mean": statistics.fmean(
                    row["ckf_delta_apply_mib_s"] for row in rows
                ),
                **publisher_support,
                "full_publications": max(row["full_publications"] for row in rows),
                "delta_publications": max(row["delta_publications"] for row in rows),
                "unchanged_publications": max(
                    row["unchanged_publications"] for row in rows
                ),
                "generation_conflicts": max(
                    row["generation_conflicts"] for row in rows
                ),
                "native_fallbacks": max(row["native_fallbacks"] for row in rows),
                "repeated_fallbacks": max(row["repeated_fallbacks"] for row in rows),
                "accuracy_checked": max(
                    row["accuracy"]["checked_results"] for row in rows
                ),
                "accuracy_inflated": max(row["accuracy"]["inflated"] for row in rows),
                "accuracy_maximum_inflation": max(
                    row["accuracy"]["maximum_inflation"] for row in rows
                ),
                "accuracy_under_reported": max(
                    row["accuracy"]["under_reported"] for row in rows
                ),
                "accuracy_full_map_mismatches": max(
                    row["accuracy"]["full_map_mismatches"] for row in rows
                ),
                "accuracy_wrong_best_dc": max(
                    row["accuracy"]["wrong_best_dc"] for row in rows
                ),
                "pipeline_errors": sum(sum(row["errors"].values()) for row in rows),
            }
            records.append(record)
    crtc = {window: groups[("crtc", window)] for window in CAPACITY_WINDOWS_MS}
    for record in records:
        if record["backend"] == "crtc":
            record.update(
                ratio_to_crtc=1.0,
                ratio_ci_low=1.0,
                ratio_ci_high=1.0,
                paired_repetitions=5,
            )
            continue
        ratio = paired_ratio_ci(
            groups[(record["backend"], record["window_ms"])],
            crtc[record["window_ms"]],
        )
        record.update(
            ratio_to_crtc=ratio[0],
            ratio_ci_low=ratio[1],
            ratio_ci_high=ratio[2],
            paired_repetitions=ratio[3],
        )
    crtc_generator_valid = {
        record["window_ms"]: record["generator_valid"]
        for record in records
        if record["backend"] == "crtc"
    }
    for record in records:
        record["comparison_generator_valid"] = (
            record["generator_valid"] and crtc_generator_valid[record["window_ms"]]
        )
    return records


def keepup_ceiling(records, backend):
    candidates = [
        row
        for row in records
        if row["backend"] == backend
        and row["all_keep_up"]
        and row["generator_valid"]
        and row["pipeline_errors"] == 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["offered_block_ops_s"])


def stable_crossover(records, backend):
    rows = sorted(
        (
            row
            for row in records
            if row["backend"] == backend and row["comparison_generator_valid"]
        ),
        key=lambda row: row["offered_block_ops_s"],
    )
    for index, row in enumerate(rows):
        if row["ratio_ci_low"] <= 1.0:
            continue
        if all(candidate["ratio_ci_low"] > 1.0 for candidate in rows[index:]):
            return row["offered_block_ops_s"]
    return None


def write_capacity(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capacity_aggregate.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    with (output_dir / "capacity_aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "| Backend | Window ms | Offered block ops/s | Achieved block ops/s (95% CI) | All 5 keep up | Generator valid | Ratio generator valid | Ratio vs CRTC (95% CI) | Lookup p50 us | Lookup p99 us |",
        "|---|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            "| {backend} | {window_ms} | {offered_block_ops_s:,.0f} | "
            "{achieved_block_ops_s_mean:,.0f} [{achieved_block_ops_s_ci_low:,.0f}, {achieved_block_ops_s_ci_high:,.0f}] | "
            "{all_keep_up} | {generator_valid} | {comparison_generator_valid} | "
            "{ratio_to_crtc:.3f} [{ratio_ci_low:.3f}, {ratio_ci_high:.3f}] | "
            "{lookup_service_p50_us_mean:.2f} | {lookup_service_p99_us_mean:.2f} |".format(
                **row
            )
        )
    (output_dir / "capacity_table.md").write_text("\n".join(lines) + "\n")


def create_iso_plan(records, output_dir):
    crtc = keepup_ceiling(records, "crtc")
    transposed = keepup_ceiling(records, "ckf-transposed")
    if crtc is None or transposed is None:
        raise SystemExit(
            "cannot derive iso load: CRTC or transposed CKF has no keep-up cell"
        )
    rate = 0.5 * min(crtc["offered_block_ops_s"], transposed["offered_block_ops_s"])
    total_blocks = crtc["offered_block_ops_s"] * crtc["window_ms"] / 1000.0
    window_ms = total_blocks / rate * 1000.0
    plan = {
        "crtc_keepup_ceiling_block_ops_s": crtc["offered_block_ops_s"],
        "transposed_keepup_ceiling_block_ops_s": transposed["offered_block_ops_s"],
        "iso_block_ops_s": rate,
        "iso_window_ms": window_ms,
        "native_crossover_block_ops_s": stable_crossover(records, "ckf-native"),
        "transposed_crossover_block_ops_s": stable_crossover(records, "ckf-transposed"),
    }
    (output_dir / "iso_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    headlines = {
        "keepup_ceilings": {
            backend: keepup_ceiling(records, backend) for backend in BACKENDS
        },
        "peak_achieved_cells": {
            backend: max(
                (row for row in records if row["backend"] == backend),
                key=lambda row: row["achieved_block_ops_s_mean"],
            )
            for backend in BACKENDS
        },
        "stable_crossover_block_ops_s": {
            "ckf-native": stable_crossover(records, "ckf-native"),
            "ckf-transposed": stable_crossover(records, "ckf-transposed"),
        },
    }
    (output_dir / "capacity_headlines.json").write_text(
        json.dumps(headlines, indent=2) + "\n"
    )
    return plan


def iso_valid(row, event_threads, query_concurrency):
    return (
        row["achieved_over_offered"] >= 0.999
        and row["update_queue"]["at_stop"] <= event_threads
        and row["query_queue"]["at_stop"] <= query_concurrency
        and max(row["update_queue"]["drain_ms"], row["query_queue"]["drain_ms"])
        <= row["replay_window_ms"] * 0.001
        and not row["generator_limited"]
        and sum(row["errors"].values()) == 0
    )


def iso_check(trials, output_dir, event_threads, query_concurrency):
    rows = [row for row in trials if row["phase"] == "iso"]
    by_backend = defaultdict(list)
    for row in rows:
        by_backend[row["backend"]].append(row)
    missing = [
        backend
        for backend in BACKENDS
        if sorted(int(row["repetition"]) for row in by_backend[backend])
        != [1, 2, 3, 4, 5]
    ]
    if missing:
        raise SystemExit(f"iso matrix incomplete for: {', '.join(missing)}")
    headline_valid = all(
        iso_valid(row, event_threads, query_concurrency)
        for backend in ("crtc", "ckf-transposed")
        for row in by_backend[backend]
    )
    window_ms = rows[0]["replay_window_ms"]
    result = {
        "retry_required": not headline_valid,
        "initial_iso_window_ms": window_ms,
        "retry_iso_window_ms": window_ms * 2.0 if not headline_valid else None,
    }
    (output_dir / "iso_check.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def iso_records(trials, phase, event_threads, query_concurrency, publisher_support):
    records = []
    for backend in BACKENDS:
        rows = [
            row for row in trials if row["phase"] == phase and row["backend"] == backend
        ]
        if len(rows) != 5:
            raise SystemExit(f"{phase} requires 5 {backend} trials, found {len(rows)}")
        service_p50 = mean_ci([row["lookup_service"]["p50_us"] for row in rows])
        service_p99 = mean_ci([row["lookup_service"]["p99_us"] for row in rows])
        wait_p50 = mean_ci([row["query_queue_wait"]["p50_us"] for row in rows])
        wait_p99 = mean_ci([row["query_queue_wait"]["p99_us"] for row in rows])
        completion_p50 = mean_ci(
            [row["scheduled_to_completion"]["p50_us"] for row in rows]
        )
        completion_p99 = mean_ci(
            [row["scheduled_to_completion"]["p99_us"] for row in rows]
        )
        achieved = mean_ci([row["achieved_block_ops_s"] for row in rows])
        valid = all(iso_valid(row, event_threads, query_concurrency) for row in rows)
        records.append(
            {
                "backend": backend,
                "phase": phase,
                "offered_block_ops_s": statistics.fmean(
                    row["nominal_offered_block_ops_s"] for row in rows
                ),
                "achieved_block_ops_s_mean": achieved[0],
                "achieved_block_ops_s_ci_low": achieved[1],
                "achieved_block_ops_s_ci_high": achieved[2],
                "achieved_over_offered_min": min(
                    row["achieved_over_offered"] for row in rows
                ),
                "lookup_service_p50_us_mean": service_p50[0],
                "lookup_service_p50_us_ci_low": service_p50[1],
                "lookup_service_p50_us_ci_high": service_p50[2],
                "lookup_service_p99_us_mean": service_p99[0],
                "lookup_service_p99_us_ci_low": service_p99[1],
                "lookup_service_p99_us_ci_high": service_p99[2],
                "query_wait_p50_us_mean": wait_p50[0],
                "query_wait_p99_us_mean": wait_p99[0],
                "scheduled_completion_p50_us_mean": completion_p50[0],
                "scheduled_completion_p99_us_mean": completion_p99[0],
                "issue_lag_p99_us_mean": statistics.fmean(
                    row["issue_lag"]["p99_us"] for row in rows
                ),
                "update_apply_p99_us_mean": statistics.fmean(
                    row["update_enqueue_to_applied"]["p99_us"] for row in rows
                ),
                "update_visible_p99_us_mean": statistics.fmean(
                    row["update_scheduled_to_applied"]["p99_us"] for row in rows
                ),
                "maximum_queue_at_stop": max(
                    row["update_queue"]["at_stop"] for row in rows
                ),
                "maximum_query_queue_at_stop": max(
                    row["query_queue"]["at_stop"] for row in rows
                ),
                "maximum_update_queue": max(
                    row["update_queue"]["maximum_depth"] for row in rows
                ),
                "maximum_query_queue": max(
                    row["query_queue"]["maximum_depth"] for row in rows
                ),
                "maximum_drain_ms": max(
                    max(row["update_queue"]["drain_ms"], row["query_queue"]["drain_ms"])
                    for row in rows
                ),
                "crtc_raw_events": max(row["crtc_raw_events"] for row in rows),
                "crtc_raw_blocks": max(row["crtc_raw_blocks"] for row in rows),
                "ckf_frames": max(row["ckf_frames"] for row in rows),
                "ckf_dirty_buckets": max(row["ckf_dirty_buckets"] for row in rows),
                "ckf_router_ingress_mib_s_mean": statistics.fmean(
                    row["ckf_bytes"] / (1024**2) / (row["total_elapsed_ms"] / 1000.0)
                    for row in rows
                ),
                "ckf_apply_mib_s_mean": statistics.fmean(
                    row["ckf_apply_mib_s"] for row in rows
                ),
                "ckf_full_apply_mib_s_mean": statistics.fmean(
                    row["ckf_full_apply_mib_s"] for row in rows
                ),
                "ckf_delta_apply_mib_s_mean": statistics.fmean(
                    row["ckf_delta_apply_mib_s"] for row in rows
                ),
                **publisher_support,
                "full_publications": max(row["full_publications"] for row in rows),
                "delta_publications": max(row["delta_publications"] for row in rows),
                "unchanged_publications": max(
                    row["unchanged_publications"] for row in rows
                ),
                "generation_conflicts": max(
                    row["generation_conflicts"] for row in rows
                ),
                "native_fallbacks": max(row["native_fallbacks"] for row in rows),
                "repeated_fallbacks": max(row["repeated_fallbacks"] for row in rows),
                "accuracy_checked": max(
                    row["accuracy"]["checked_results"] for row in rows
                ),
                "accuracy_inflated": max(row["accuracy"]["inflated"] for row in rows),
                "accuracy_maximum_inflation": max(
                    row["accuracy"]["maximum_inflation"] for row in rows
                ),
                "accuracy_under_reported": max(
                    row["accuracy"]["under_reported"] for row in rows
                ),
                "accuracy_full_map_mismatches": max(
                    row["accuracy"]["full_map_mismatches"] for row in rows
                ),
                "accuracy_wrong_best_dc": max(
                    row["accuracy"]["wrong_best_dc"] for row in rows
                ),
                "pipeline_errors": sum(sum(row["errors"].values()) for row in rows),
                "generator_valid": not any(row["generator_limited"] for row in rows),
                "valid_no_backlog": valid,
                "diagnostic_only": backend == "ckf-native" and not valid,
            }
        )
    return records


def load_memory(results_dir):
    rows = []
    for path in sorted(results_dir.glob("memory_*.json")):
        row = json.loads(path.read_text())
        if "mode" in row and "rss_bytes" in row:
            rows.append(row)
    expected = {"crtc", "ckf-native", "ckf-transposed", "relay-producer"}
    actual = {row["mode"] for row in rows}
    if actual != expected:
        raise SystemExit(
            f"memory modes mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    crtc_rss = next(row["rss_bytes"] for row in rows if row["mode"] == "crtc")
    for row in rows:
        row["crtc_rss_over_mode"] = (
            crtc_rss / row["rss_bytes"] if row["rss_bytes"] else None
        )
    return rows


def write_final(trials, results_dir, output_dir, event_threads, query_concurrency):
    capacity = json.loads((output_dir / "capacity_aggregate.json").read_text())
    retry_rows = [row for row in trials if row["phase"] == "iso-retry"]
    phase = "iso-retry" if retry_rows else "iso"
    publisher_support = load_publisher_support(results_dir)
    iso = iso_records(
        trials, phase, event_threads, query_concurrency, publisher_support
    )
    memory = load_memory(results_dir)
    headline = {row["backend"]: row for row in iso}
    if (
        not headline["crtc"]["valid_no_backlog"]
        or not headline["ckf-transposed"]["valid_no_backlog"]
    ):
        raise SystemExit("headline iso load is invalid after the permitted retry")
    (output_dir / "iso_aggregate.json").write_text(json.dumps(iso, indent=2) + "\n")
    with (output_dir / "iso_aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(iso[0]))
        writer.writeheader()
        writer.writerows(iso)
    (output_dir / "memory_aggregate.json").write_text(
        json.dumps(memory, indent=2) + "\n"
    )
    with (output_dir / "memory_aggregate.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(memory[0]))
        writer.writeheader()
        writer.writerows(memory)

    columns = [
        "Phase",
        "Backend",
        "Window ms",
        "Offered block/s",
        "Achieved block/s (95% CI)",
        "Ratio vs CRTC (95% CI)",
        "Valid",
        "Generator valid",
        "Ratio generator valid",
        "Lookup p50 us",
        "Lookup p99 us",
        "Query wait p99 us",
        "Scheduled→done p99 us",
        "Issue lag p99 us",
        "Update apply p99 us",
        "Update visible p99 us",
        "Q stop",
        "Q max",
        "U stop",
        "U max",
        "Max drain ms",
        "CRTC events",
        "CRTC blocks",
        "CKF frames",
        "Dirty buckets",
        "CKF ingress MiB/s",
        "CKF apply MiB/s",
        "Full apply MiB/s",
        "Delta apply MiB/s",
        "Relay build events/s",
        "Relay build MiB/s",
        "Full build MiB/s",
        "Delta build MiB/s",
        "Full",
        "Delta",
        "Unchanged",
        "Gen conflicts",
        "Native fallbacks",
        "Repeated fallbacks",
        "Checked",
        "Inflated",
        "Max inflation",
        "Under",
        "Map mismatch",
        "Wrong best DC",
        "Errors",
        "RSS GiB",
        "PSS GiB",
        "USS GiB",
        "Auth filter GiB",
        "Derived GiB",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in capacity:
        values = [
            "capacity",
            row["backend"],
            str(row["window_ms"]),
            f"{row['offered_block_ops_s']:,.0f}",
            f"{row['achieved_block_ops_s_mean']:,.0f} [{row['achieved_block_ops_s_ci_low']:,.0f}, {row['achieved_block_ops_s_ci_high']:,.0f}]",
            f"{row['ratio_to_crtc']:.3f} [{row['ratio_ci_low']:.3f}, {row['ratio_ci_high']:.3f}]",
            str(row["all_keep_up"]),
            str(row["generator_valid"]),
            str(row["comparison_generator_valid"]),
            f"{row['lookup_service_p50_us_mean']:.2f}",
            f"{row['lookup_service_p99_us_mean']:.2f}",
            f"{row['query_wait_p99_us_mean']:.2f}",
            f"{row['scheduled_completion_p99_us_mean']:.2f}",
            f"{row['issue_lag_p99_us_mean']:.2f}",
            f"{row['update_apply_p99_us_mean']:.2f}",
            f"{row['update_visible_p99_us_mean']:.2f}",
            str(row["maximum_query_queue_at_stop"]),
            str(row["maximum_query_queue"]),
            str(row["maximum_update_queue_at_stop"]),
            str(row["maximum_update_queue"]),
            f"{row['maximum_drain_ms']:.3f}",
            str(row["crtc_raw_events"]),
            str(row["crtc_raw_blocks"]),
            str(row["ckf_frames"]),
            str(row["ckf_dirty_buckets"]),
            f"{row['ckf_router_ingress_mib_s_mean']:.3f}",
            f"{row['ckf_apply_mib_s_mean']:.3f}",
            f"{row['ckf_full_apply_mib_s_mean']:.3f}",
            f"{row['ckf_delta_apply_mib_s_mean']:.3f}",
            f"{row['relay_generation_events_s']:,.0f}",
            f"{row['relay_generation_mib_s']:.3f}",
            f"{row['relay_full_generation_mib_s']:.3f}",
            f"{row['relay_delta_generation_mib_s']:.3f}",
            str(row["full_publications"]),
            str(row["delta_publications"]),
            str(row["unchanged_publications"]),
            str(row["generation_conflicts"]),
            str(row["native_fallbacks"]),
            str(row["repeated_fallbacks"]),
            str(row["accuracy_checked"]),
            str(row["accuracy_inflated"]),
            str(row["accuracy_maximum_inflation"]),
            str(row["accuracy_under_reported"]),
            str(row["accuracy_full_map_mismatches"]),
            str(row["accuracy_wrong_best_dc"]),
            str(row["pipeline_errors"]),
            "—",
            "—",
            "—",
            "—",
            "—",
        ]
        assert len(values) == len(columns)
        lines.append("| " + " | ".join(values) + " |")
    window_ms = next(row["replay_window_ms"] for row in trials if row["phase"] == phase)
    for row in iso:
        values = [
            phase,
            row["backend"],
            f"{window_ms:.3f}",
            f"{row['offered_block_ops_s']:,.0f}",
            f"{row['achieved_block_ops_s_mean']:,.0f} [{row['achieved_block_ops_s_ci_low']:,.0f}, {row['achieved_block_ops_s_ci_high']:,.0f}]",
            "—",
            str(row["valid_no_backlog"]),
            str(row["generator_valid"]),
            "—",
            f"{row['lookup_service_p50_us_mean']:.2f}",
            f"{row['lookup_service_p99_us_mean']:.2f}",
            f"{row['query_wait_p99_us_mean']:.2f}",
            f"{row['scheduled_completion_p99_us_mean']:.2f}",
            f"{row['issue_lag_p99_us_mean']:.2f}",
            f"{row['update_apply_p99_us_mean']:.2f}",
            f"{row['update_visible_p99_us_mean']:.2f}",
            str(row["maximum_query_queue_at_stop"]),
            str(row["maximum_query_queue"]),
            str(row["maximum_queue_at_stop"]),
            str(row["maximum_update_queue"]),
            f"{row['maximum_drain_ms']:.3f}",
            str(row["crtc_raw_events"]),
            str(row["crtc_raw_blocks"]),
            str(row["ckf_frames"]),
            str(row["ckf_dirty_buckets"]),
            f"{row['ckf_router_ingress_mib_s_mean']:.3f}",
            f"{row['ckf_apply_mib_s_mean']:.3f}",
            f"{row['ckf_full_apply_mib_s_mean']:.3f}",
            f"{row['ckf_delta_apply_mib_s_mean']:.3f}",
            f"{row['relay_generation_events_s']:,.0f}",
            f"{row['relay_generation_mib_s']:.3f}",
            f"{row['relay_full_generation_mib_s']:.3f}",
            f"{row['relay_delta_generation_mib_s']:.3f}",
            str(row["full_publications"]),
            str(row["delta_publications"]),
            str(row["unchanged_publications"]),
            str(row["generation_conflicts"]),
            str(row["native_fallbacks"]),
            str(row["repeated_fallbacks"]),
            str(row["accuracy_checked"]),
            str(row["accuracy_inflated"]),
            str(row["accuracy_maximum_inflation"]),
            str(row["accuracy_under_reported"]),
            str(row["accuracy_full_map_mismatches"]),
            str(row["accuracy_wrong_best_dc"]),
            str(row["pipeline_errors"]),
            "—",
            "—",
            "—",
            "—",
            "—",
        ]
        assert len(values) == len(columns)
        lines.append("| " + " | ".join(values) + " |")
    gib = 1024**3
    for row in memory:
        pss = "—" if row["pss_bytes"] is None else f"{row['pss_bytes'] / gib:.3f}"
        uss = "—" if row["uss_bytes"] is None else f"{row['uss_bytes'] / gib:.3f}"
        values = (
            ["memory", row["mode"]]
            + ["—"] * (len(columns) - 7)
            + [
                f"{row['rss_bytes'] / gib:.3f}",
                pss,
                uss,
                f"{row['authoritative_filter_bytes'] / gib:.3f}",
                f"{row['derived_transposed_bytes'] / gib:.3f}",
            ]
        )
        assert len(values) == len(columns)
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "wide_results.md").write_text("\n".join(lines) + "\n")

    summary = {
        "capacity_cells": 90,
        "iso_cells": 15,
        "selected_iso_phase": phase,
        "relay_publisher": publisher_support,
        "capacity": capacity,
        "iso": iso,
        "memory": memory,
    }
    (output_dir / "aggregate_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials = load_trials(args.results_dir)
    if args.stage == "capacity":
        publisher_support = load_publisher_support(args.results_dir)
        groups = capacity_groups(trials)
        validate_capacity(groups)
        records = summarize_capacity(groups, publisher_support)
        write_capacity(records, args.output_dir)
        create_iso_plan(records, args.output_dir)
    elif args.stage == "iso-check":
        iso_check(trials, args.output_dir, args.event_threads, args.query_concurrency)
    else:
        write_final(
            trials,
            args.results_dir,
            args.output_dir,
            args.event_threads,
            args.query_concurrency,
        )


if __name__ == "__main__":
    main()
