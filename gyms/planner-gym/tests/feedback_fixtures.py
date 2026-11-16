# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small native-telemetry payloads for online policy contract tests."""

SLO = {"ttft_ms": 100.0, "itl_ms": 20.0, "e2e_ms": None}


def sample(
    at=0,
    *,
    p=1,
    d=1,
    starting_d=0,
    draining_p=0,
    draining_d=0,
    terminal=0,
    completed=0,
    good=0,
    strict_good=None,
    gpu_seconds=0,
    pairs=(),
    slo=None,
):
    return {
        "kind": "baseline" if at == 0 else "periodic",
        "sampled_at_ms": at * 1000,
        "interval_start_ms": max(0, at - 5) * 1000,
        "active_prefill_ids": list(range(p)),
        "starting_prefill_ids": [],
        "draining_prefill_ids": list(range(p, p + draining_p)),
        "active_decode_ids": list(range(d)),
        "starting_decode_ids": list(range(d, d + starting_d)),
        "draining_decode_ids": list(range(d + starting_d, d + starting_d + draining_d)),
        "traffic": {
            "avg_ttft_ms": 80.0,
            "ttft_count": 2 if at else 0,
            "avg_itl_ms": 12.0,
            "itl_count": 4 if at else 0,
        },
        "completion_feedback": {
            "schema_version": 1,
            "sla": slo or SLO,
            "terminal_requests": terminal,
            "completed_requests": completed,
            "good_requests": good,
            "good_requests_strict_itl": good if strict_good is None else strict_good,
            "gpu_seconds": gpu_seconds,
            "latency_samples": list(pairs),
            "latency_population": len(pairs),
        },
    }


def knowline_context(budget=24, *, minimum=2, maximum=16):
    return {
        "slo": SLO,
        "gpu_budget": budget,
        "worker_limits": {"min_total": minimum, "max_total": maximum},
        "engines": {
            "prefill": {"gpus_per_worker": 4, "cold_start_delay_s": 0},
            "decode": {"gpus_per_worker": 8, "cold_start_delay_s": 0},
            "aggregate": {"gpus_per_worker": 8, "cold_start_delay_s": 0},
        },
    }
