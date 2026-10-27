# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace, request and evaluator builders for the retune sweep tests.

Sweeps start their workers with ``spawn`` and pass the evaluator by
reference, so the fake evaluators here are module-level functions that a
worker imports as ``dynamo.planner.tests.retune_sweep_helpers``. This module
imports the sweep, which needs AI Simulate: test modules check for it with
``pytest.importorskip`` before importing this one.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import psutil

from dynamo.llm import KvRouterConfig
from dynamo.mocker.config import normalize_mocker_config
from dynamo.planner.config.planner_config import RetuneEngineSpec
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune.sweep import (
    SweepRequest,
    _read_observations,
    evaluate_candidate,
)
from dynamo.planner.simulation.load_predictor import distinct_predictor_candidates

SCHEMA = "dynamo.request.trace.v1"
ORIGIN_MS = 1_700_000_000_000
# Router temperatures that tell the test evaluators how to behave.
ABORT, HANG, RAISE = 9.0, 8.0, 7.0


def trace_rows(arrivals_ms, *, isl=256, osl=16, total_ms=50.0, prefix="r"):
    return [
        {
            "schema": SCHEMA,
            "event_type": "request_end",
            "event_time_unix_ms": int(arrival + total_ms),
            "request": {
                "request_id": f"{prefix}{index}",
                "request_received_ms": int(arrival),
                "total_time_ms": total_ms,
                "output_tokens": osl,
                "replay": {
                    "trace_block_size": 64,
                    "input_length": isl,
                    "input_sequence_hashes": [index % 3, 100 + index, 200 + index, 7],
                },
            },
        }
        for index, arrival in enumerate(arrivals_ms)
    ]


def write_trace(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def engine_args(**engine):
    return normalize_mocker_config(
        {
            "engine": {
                "block_size": 64,
                "num_gpu_blocks": 4096,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 8192,
                "speedup_ratio": 1000.0,
                **engine,
            }
        }
    )


def disagg_engine():
    return RetuneEngineSpec(
        prefill_engine_args=engine_args(worker_type="prefill"),
        decode_engine_args=engine_args(worker_type="decode"),
        router_config=json.loads(KvRouterConfig().to_json()),
    )


def retune_params(credit=1.0, temperature=0.0, **planner):
    return RetuneParams(
        router={"overlap_score_credit": credit, "router_temperature": temperature},
        planner=planner,
    )


def predictor_candidates(*entries):
    """Pre-sweep ``(label, mapping)`` candidates, as ``retune.planner`` gives them."""

    return tuple(distinct_predictor_candidates(list(entries)))


def sweep_request(trace_paths, candidates, baseline, **overrides):
    values = {
        "window_index": 0,
        "window_start_ms": ORIGIN_MS,
        "window_end_ms": ORIGIN_MS + 10_000,
        "history_start_ms": ORIGIN_MS,
        "as_of_ms": ORIGIN_MS + 20_000,
        "trace_paths": tuple(trace_paths),
        "arrival_speedup_ratio": 1.0,
        "engine": RetuneEngineSpec(),
        "candidates": tuple(sorted(candidates, key=RetuneParams.key)),
        "baseline": baseline,
        "replicas": {"aggregated": 2},
        "base_planner": None,
        "predictor_candidates": None,
        "sla_ttft_ms": 500.0,
        "sla_itl_ms": 30.0,
        "max_parallel": 2,
        "timeout_s": 120.0,
    }
    values.update(overrides)
    return SweepRequest(**values)


def fake_evaluator(
    window_path,
    engine,
    arrival_speedup_ratio,
    params,
    replicas,
    base_planner,
    warmup_path,
    sla_ttft_ms,
    sla_itl_ms,
    window_s,
):
    if params.router.get("router_temperature") == HANG:
        time.sleep(600)
    return 1.0 + params.router.get("overlap_score_credit", 0.0), {
        "planner": dict(params.planner),
        "warmup_trace": (
            None
            if base_planner is None
            else base_planner.get("load_predictor_warmup_trace", "?")
        ),
        # Requests per warmup bucket the inner planner would see.
        "warmup": (
            None
            if warmup_path is None
            else [obs.num_req for obs in _read_observations(warmup_path)]
        ),
        "retune": None if base_planner is None else base_planner.get("retune", "?"),
        "advisory": (
            None if base_planner is None else base_planner.get("advisory", "?")
        ),
        "batch_scheduling": (
            None if base_planner is None else base_planner.get("batch_scheduling", "?")
        ),
        "reports": (
            None
            if base_planner is None
            else (
                base_planner.get("report_interval_hours", "?"),
                base_planner.get("live_dashboard_port", "?"),
            )
        ),
    }


def temp_dir_evaluator(*args):
    """Score as ``fake_evaluator`` after creating a temporary directory.

    cmdstanpy creates one on import and removes it only in an ``atexit``
    hook, which a killed worker never runs.
    """

    path = tempfile.mkdtemp()
    score, metrics = fake_evaluator(*args)
    return score, {**metrics, "temp_dir": path}


def abort_raise_or_evaluate(window_path, engine, arrival_speedup_ratio, params, *rest):
    """Kill or fail the worker on a marker temperature, else replay for real."""

    action = params.router.get("router_temperature")
    if action == ABORT:
        os.abort()
    if action == RAISE:
        raise RuntimeError("replay exploded")
    score, metrics = evaluate_candidate(
        window_path, engine, arrival_speedup_ratio, params, *rest
    )
    return score, {**metrics, "children": len(psutil.Process().children())}
