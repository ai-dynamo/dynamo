"""Dependency-free protocol primitives shared by the PoC command-line tools."""

import hashlib
import json
import math
import random
import re
import statistics


IDENTITY_FIELDS = {
    "image_digest",
    "model_revision",
    "gpu_product",
    "driver_version",
    "command",
    "args",
    "pod_spec",
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compatibility_hash(identity):
    if set(identity) != IDENTITY_FIELDS:
        missing = sorted(IDENTITY_FIELDS - set(identity))
        extra = sorted(set(identity) - IDENTITY_FIELDS)
        raise ValueError(f"compatibility identity fields mismatch; missing={missing}, extra={extra}")
    if not re.search(r"@sha256:[0-9a-f]{64}$", identity["image_digest"]):
        raise ValueError("image_digest must be an immutable name@sha256:<64 lowercase hex> reference")
    for field in ("model_revision", "gpu_product", "driver_version"):
        if not isinstance(identity[field], str) or not identity[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    for field in ("command", "args"):
        if not isinstance(identity[field], list) or not all(
            isinstance(item, str) for item in identity[field]
        ):
            raise ValueError(f"{field} must be a list of strings")
    if not isinstance(identity["pod_spec"], dict):
        raise ValueError("pod_spec must be an object")
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def checkpoint_id(compatibility_hash_value):
    """Map a full SHA-256 identity to Dynamo's Kubernetes-label-safe locator."""
    if not isinstance(compatibility_hash_value, str) or not re.fullmatch(
        r"[0-9a-f]{64}", compatibility_hash_value
    ):
        raise ValueError("compatibility_hash must be exactly 64 lowercase hexadecimal characters")
    return "h-" + compatibility_hash_value[:61]


def make_run_plan(seed, paired_blocks=10):
    if paired_blocks != 10:
        raise ValueError("V0.1 freezes exactly 10 paired blocks")
    rng = random.Random(seed)
    modes = ["cold", "restore"]
    rng.shuffle(modes)
    key = dict(zip(("A", "B"), modes))
    schedule = []
    for block in range(1, paired_blocks + 1):
        order = ["A", "B"]
        rng.shuffle(order)
        for sequence, arm in enumerate(order, 1):
            schedule.append(
                {
                    "run_id": f"v1-{block:02d}-{sequence:02d}",
                    "block": block,
                    "sequence_in_block": sequence,
                    "opaque_arm": arm,
                }
            )
    return schedule, key


def _validate_and_eligible(records):
    if not isinstance(records, list):
        raise ValueError("records must be a JSONL-derived list")
    run_ids = [row.get("run_id") for row in records]
    if None in run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("run_id values must be present and unique")
    eligible = []
    by_block = {}
    for row in records:
        if row.get("opaque_arm") not in ("A", "B"):
            raise ValueError("opaque_arm must be A or B")
        by_block.setdefault(row.get("block"), []).append(row)
        if row.get("excluded"):
            if not row.get("exclusion_reason") or not row.get("cluster_incident_evidence"):
                raise ValueError("excluded runs require reason and raw cluster-event evidence")
    for block, rows in by_block.items():
        excluded = [row for row in rows if row.get("excluded")]
        active = [row for row in rows if not row.get("excluded")]
        if excluded:
            excluded_arms = [row["opaque_arm"] for row in excluded]
            if excluded_arms.count("A") != excluded_arms.count("B"):
                raise ValueError(f"block {block} has an unpaired exclusion")
        if len(active) != 2 or {row["opaque_arm"] for row in active} != {"A", "B"}:
            raise ValueError(f"block {block} has no single complete eligible A/B pair")
        eligible.extend(active)
    eligible_blocks = {row["block"] for row in eligible}
    if len(eligible_blocks) != 10 or len(eligible) != 20:
        raise ValueError("protocol requires 10 eligible complete blocks; repeat excluded blocks")
    return eligible


def summarize_blinded(records):
    eligible = _validate_and_eligible(records)
    result = {"sealed_before_unblinding": True, "arms": {}}
    for arm in ("A", "B"):
        rows = [row for row in eligible if row["opaque_arm"] == arm]
        def median_or_none(field):
            values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float))]
            return statistics.median(values) if values else None
        result["arms"][arm] = {
            "eligible_runs": len(rows),
            "failed_runs": sum(not bool(row.get("valid_response")) for row in rows),
            "median_ready_s": median_or_none("ready_s"),
            "median_http_200_s": median_or_none("http_200_s"),
            "median_first_token_s": median_or_none("first_token_s"),
            "valid_responses": sum(bool(row.get("valid_response")) for row in rows),
            "median_gpu_memory_mib": median_or_none("gpu_memory_mib"),
        }
    return result


def summarize(records, key):
    if set(key) != {"A", "B"} or set(key.values()) != {"cold", "restore"}:
        raise ValueError("unblinding key must map A/B bijectively to cold/restore")
    eligible = _validate_and_eligible(records)
    by_mode = {
        mode: [row for row in eligible if key[row["opaque_arm"]] == mode]
        for mode in ("cold", "restore")
    }
    cold = by_mode["cold"]
    restore = by_mode["restore"]
    cold_latencies = [row.get("first_token_s") for row in cold]
    restore_latencies = [row.get("first_token_s") for row in restore]
    metrics_complete = all(isinstance(value, (int, float)) for value in cold_latencies + restore_latencies)
    cold_median = statistics.median(cold_latencies) if metrics_complete else None
    restore_median = statistics.median(restore_latencies) if metrics_complete else None
    median_speedup = (
        cold_median / restore_median
        if metrics_complete and restore_median > 0
        else (math.inf if metrics_complete else 0.0)
    )
    cold_by_block = {row["block"]: row for row in cold}
    gpu_deltas = []
    for row in restore:
        baseline = cold_by_block[row["block"]].get("gpu_memory_mib")
        restored = row.get("gpu_memory_mib")
        if not isinstance(baseline, (int, float)) or not isinstance(restored, (int, float)):
            delta = math.inf
        else:
            delta = abs(restored - baseline) / baseline if baseline > 0 else math.inf
        gpu_deltas.append(delta)
    successes = sum(row.get("restore_success") is True for row in restore)
    valid_responses = sum(bool(row.get("valid_response")) for row in eligible)
    checkpoint_durations = [
        row["checkpoint_duration_s"]
        for row in eligible
        if row.get("checkpoint_duration_s") is not None
    ]
    checkpoint_sizes = [
        row["checkpoint_size_bytes"]
        for row in eligible
        if row.get("checkpoint_size_bytes") is not None
    ]
    saved = cold_median - restore_median if metrics_complete else 0
    if checkpoint_durations and saved > 0:
        break_even = statistics.median(checkpoint_durations) / saved
    else:
        break_even = math.inf
    correctness_ok = successes == 10 and valid_responses == 20 and metrics_complete
    memory_ok = max(gpu_deltas) <= 0.05
    if not correctness_ok or not memory_ok:
        decision = "No-Go"
    elif median_speedup >= 3.0:
        decision = "Go"
    else:
        decision = "Optimize"
    return {
        "protocol_version": "V0.1",
        "decision": decision,
        "eligible_runs": 20,
        "cold_median_first_token_s": cold_median,
        "restore_median_first_token_s": restore_median,
        "median_speedup": median_speedup,
        "restore_successes": successes,
        "restore_success_rate": successes / 10,
        "valid_responses": valid_responses,
        "max_paired_gpu_memory_relative_delta": max(gpu_deltas),
        "checkpoint_duration_median_s": (
            statistics.median(checkpoint_durations) if checkpoint_durations else None
        ),
        "checkpoint_size_median_bytes": (
            statistics.median(checkpoint_sizes) if checkpoint_sizes else None
        ),
        "break_even_restores": break_even,
    }
