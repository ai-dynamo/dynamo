# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- optional AISimulate preflight precedes integration imports.

"""Real AgentX G2 qualification through Dynamo's existing replay API."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

pytest.importorskip(
    "aisimulate.sweeper", reason="AISimulate is an optional simulation dependency"
)

from aisimulate.sweeper.provider import AdapterReplaySpec, RuntimeHookSpec
from aisimulate.sweeper.replay import (
    BackendDeploymentSpec,
    ReplayOutputRequirements,
    ReplaySpec,
)

from dynamo.replay.simulation import DynamoReplayRunnerFactory

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.planner,
    pytest.mark.timeout(30),
]


def _trace(tmp_path, trace_format):
    """Authored data: fill G1, evict its prefix, then request that prefix again."""
    turns = (
        ("seed", 0, [1, 2, 3]),
        ("pressure", 100, [11, 12, 13]),
        ("restore", 200, [1, 2, 3]),
        ("finish", 300, [21, 22, 23]),
    )
    if trace_format == "dynamo":
        rows = [
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": timestamp + 2,
                "request": {
                    "request_id": request_id,
                    "request_received_ms": timestamp,
                    "total_time_ms": 2,
                    "output_tokens": 1,
                    "replay": {
                        "trace_block_size": 4,
                        "input_length": 9,
                        "input_sequence_hashes": hashes,
                    },
                },
            }
            for request_id, timestamp, hashes in turns
        ]
    elif trace_format == "weka":
        rows = [
            {
                "id": "host-restore",
                "models": ["target-model"],
                "block_size": 4,
                "hash_id_scope": "local",
                "requests": [
                    {
                        "t": timestamp / 1000,
                        "type": "s",
                        "model": "target-model",
                        "in": 9,
                        "out": 1,
                        "hash_ids": hashes,
                        "api_time": 0.002,
                    }
                    for _, timestamp, hashes in turns
                ],
            }
        ]
    else:
        # Distinct sessions keep Mooncake-delta prompts independent. Explicit
        # hashes provide the same cross-request prefix identity in both formats.
        rows = [
            {
                "request_id": request_id,
                "session_id": request_id,
                "timestamp": timestamp,
                "input_length": 9,
                "output_length": 1,
                "hash_ids": hashes,
            }
            for request_id, timestamp, hashes in turns
        ]
    path = tmp_path / f"{trace_format}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _spec(tmp_path, trace_format="weka", scope="dp_rank_local", *, disagg=False):
    # Fixed timing and explicit bytes avoid models, GPUs and external services.
    args = {
        "engine_type": "vllm",
        "worker_type": "aggregated",
        "dp_size": 1,
        "block_size": 4,
        "num_gpu_blocks": 3,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 16,
        "kv_cache_bytes_per_token": 250000,
        "enable_prefix_caching": True,
        "timing_model": {"type": "fixed", "prefill_ms": 2, "decode_ms": 1},
    }
    if scope is not None:
        args["native_host_offload"] = {
            "scope": scope,
            "num_host_blocks": 8,
            "d2h_bandwidth_gbps": 1,
            "h2d_bandwidth_gbps": 1,
            "kv_layout_id": "dynamo-agentx-g2-fixture-v1",
        }
    roles = ("prefill", "decode") if disagg else ("aggregated",)
    workers = {role: {**copy.deepcopy(args), "worker_type": role} for role in roles}
    deployment = BackendDeploymentSpec(
        deployment_mode="disagg" if disagg else "agg",
        backend="vllm",
        backend_version="test",
        agg_engine_args=workers.get("aggregated"),
        prefill_engine_args=workers.get("prefill"),
        decode_engine_args=workers.get("decode"),
        num_workers=0 if disagg else 1,
        num_prefill_workers=1 if disagg else 0,
        num_decode_workers=1 if disagg else 0,
        performance_model_metadata={
            role: {"config": {"model_path": "target-model"}} for role in roles
        },
    )
    workload = {
        "source_type": "trace",
        "load_type": "trace_timestamps",
        "trace_path": str(_trace(tmp_path, trace_format)),
        "trace_format": trace_format,
        "trace_block_size": 4,
    }
    if trace_format == "weka":
        workload.update(agentic_lanes=1, weka_nested_timestamp_basis="absolute")
    return ReplaySpec(
        backend_deployment=deployment,
        workload=workload,
        goal={"target": "throughput"},
        adapters={
            "dynamo.router": AdapterReplaySpec(
                runtime_hooks=(
                    RuntimeHookSpec(
                        provider="dynamo.router",
                        kind="placement_policy",
                        api_version=1,
                        config={"router_mode": "kv_router", "router_config": {}},
                    ),
                )
            )
        },
    )


@pytest.fixture
def run_replay():
    runner = DynamoReplayRunnerFactory(trace_block_size=4).create(0)
    yield lambda spec: _run(runner, spec)
    runner.close()


def _run(runner, spec):
    report = runner.run(
        spec,
        output_requirements=ReplayOutputRequirements(
            include_raw_report=True, capture_per_request=True
        ),
    )
    native = report.metadata["native_report"]
    # Existing Dynamo report envelope: summary contains native aggregate and
    # lifecycle fields; per_request is a separate capture field.
    return {**native["summary"], **native}


def _host_hits(report):
    # HBM-only reports omit host reuse fields rather than serializing zero.
    return [
        row
        for row in report["per_request"]
        if row.get("first_admission_host_reused_input_tokens", 0) > 0
    ]


@pytest.mark.parametrize(
    "trace_format", ["weka", "dynamo", "mooncake", "mooncake-delta"]
)
@pytest.mark.parametrize("scope", ["dp_rank_local", "cluster_shared"])
def test_existing_api_restores_g2_and_measures_h2d(
    tmp_path, run_replay, trace_format, scope
):
    spec = _spec(tmp_path, trace_format, scope)
    fast = run_replay(spec)
    slow_spec = copy.deepcopy(spec)
    slow_spec.backend_deployment.agg_engine_args["native_host_offload"][
        "h2d_bandwidth_gbps"
    ] = 0.1
    slow = run_replay(slow_spec)
    hbm_spec = copy.deepcopy(spec)
    hbm_spec.backend_deployment.agg_engine_args.pop("native_host_offload")
    hbm = run_replay(hbm_spec)

    for report, wait_ms in ((fast, 2), (slow, 20)):
        assert report["completed_requests"] == 4
        assert report["committed_prefill_tokens"] == 28
        hits = _host_hits(report)
        assert len(hits) == 1
        hit = hits[0]
        assert hit["first_admission_host_reused_input_tokens"] == 8
        assert hit["first_admission_g1_reused_input_tokens"] == 0
        assert hit["admission_history"][0]["host_reused_input_tokens"] == 8
        # Parent G2 routing may score HostPinned residency separately from G1.
        assert hit["routing_history"]
        assert hit["first_admit_ms"] - hit["arrival_time_ms"] == pytest.approx(wait_ms)
        assert hit["ttft_ms"] == pytest.approx(wait_ms + 2)
        if scope == "cluster_shared":
            assert len(report["g2_domains"]) == 1
            assert report["g2_domains"][0]["capacity_blocks"] == 8
    assert hbm["completed_requests"] == 4
    assert not _host_hits(hbm)
    assert hbm["committed_prefill_tokens"] > fast["committed_prefill_tokens"]


def test_1p1d_shared_pool_counts_capacity_and_first_reuse_once(tmp_path, run_replay):
    spec = _spec(tmp_path, scope="cluster_shared", disagg=True)
    report = run_replay(spec)
    assert report["completed_requests"] == 4
    assert len(report["g2_domains"]) == 1
    assert report["g2_domains"][0]["capacity_blocks"] == 8
    hits = _host_hits(report)
    assert len(hits) == 1
    assert hits[0]["first_admission_host_reused_input_tokens"] == 8
    assert {row["pool"] for row in hits[0]["admission_history"]} == {
        "prefill",
        "decode",
    }
    assert (
        sum(
            row["first_admission_host_reused_input_tokens"]
            for row in report["per_request"]
        )
        == 8
    )

    small = copy.deepcopy(spec)
    for args in (
        small.backend_deployment.prefill_engine_args,
        small.backend_deployment.decode_engine_args,
    ):
        args["native_host_offload"]["num_host_blocks"] = 1
    evicted = run_replay(small)
    assert not _host_hits(evicted)
    assert evicted["committed_prefill_tokens"] > report["committed_prefill_tokens"]


@pytest.mark.parametrize("scope", ["dp_rank_local", "cluster_shared"])
def test_parent_resumes_from_g2_after_child_and_join(tmp_path, run_replay, scope):
    spec = _spec(tmp_path, scope=scope)
    path = tmp_path / "weka.jsonl"
    play = json.loads(path.read_text())
    child_request = play["requests"][1]
    play["requests"][1] = {
        "t": 0.1,
        "type": "subagent",
        "agent_id": "pressure",
        "subagent_type": "Explore",
        "duration_ms": 2,
        "status": "completed",
        "models": ["target-model"],
        "requests": [child_request],
    }
    path.write_text(json.dumps(play) + "\n")
    report = run_replay(spec)
    assert report["completed_requests"] == 4
    seed = min(report["per_request"], key=lambda row: row["arrival_time_ms"])
    children = [row for row in report["per_request"] if row["agentic"].get("parent_id")]
    assert len(children) == 1
    child = children[0]
    assert child["agentic"]["parent_id"] == seed["request_id"]
    assert child["agentic"]["root_id"] == seed["request_id"]
    assert child["agentic"]["conversation_id"] != seed["agentic"]["conversation_id"]
    hits = _host_hits(report)
    assert len(hits) == 1
    # Weka may split an independent outer turn into another conversation. The
    # restored turn must specifically resume the original parent's conversation.
    assert hits[0]["agentic"]["conversation_id"] == seed["agentic"]["conversation_id"]
    assert hits[0]["agentic"]["root_id"] == seed["request_id"]
    assert child["terminal_time_ms"] <= hits[0]["arrival_time_ms"]
    assert hits[0]["first_admission_host_reused_input_tokens"] == 8
    assert hits[0]["first_admission_g1_reused_input_tokens"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("kv_layout_id", "incompatible-layout"),
        ("num_host_blocks", 16),
        ("shared_h2d_bandwidth_gbps", 2),
    ],
)
def test_1p1d_shared_pool_rejects_incompatible_participants(
    tmp_path, run_replay, field, value
):
    spec = _spec(tmp_path, scope="cluster_shared", disagg=True)
    spec.backend_deployment.decode_engine_args["native_host_offload"][field] = value
    # The existing binding maps native configuration errors to PyException;
    # pin its diagnostic rather than accepting an arbitrary execution failure.
    with pytest.raises(Exception, match="cluster_shared.*incompatible"):
        run_replay(spec)


@pytest.mark.parametrize("trace_format", ["dynamo", "mooncake", "mooncake-delta"])
def test_non_agentic_g2_keeps_multiple_worker_dp_support(
    tmp_path, run_replay, trace_format
):
    spec = _spec(tmp_path, trace_format, "cluster_shared")
    spec.backend_deployment.agg_engine_args["dp_size"] = 2
    spec = replace(
        spec, backend_deployment=replace(spec.backend_deployment, num_workers=2)
    )
    report = run_replay(spec)
    assert report["completed_requests"] == 4
    assert len(report["g2_domains"]) == 1
    assert report["g2_domains"][0]["capacity_blocks"] == 8
    assert all(row["routing_history"] for row in report["per_request"])


@pytest.mark.parametrize("scope", [None, "dp_rank_local", "cluster_shared"])
def test_snapshot_warmup_waits_for_g2_and_retains_cache(tmp_path, run_replay, scope):
    spec = _spec(tmp_path, scope=scope)
    if scope:
        # Primer D2H lasts 200 ms, longer than the eleven fixed 2 ms passes.
        spec.backend_deployment.agg_engine_args["native_host_offload"][
            "d2h_bandwidth_gbps"
        ] = 0.01
    spec = replace(spec, workload={**spec.workload, "agentic_snapshot": {"seed": 5}})
    cold = run_replay(spec)
    warm = run_replay(replace(spec, workload={**spec.workload, "agentic_warmup": True}))
    assert cold["agentic_snapshots"] == warm["agentic_snapshots"]
    assert 2 < warm["agentic_snapshots"][0]["t_star_ms"] < 100
    assert cold["completed_requests"] == warm["completed_requests"] == 3
    assert cold["committed_prefill_tokens"] == 27
    assert warm["committed_prefill_tokens"] == (19 if scope else 27)
    assert sum(
        row.get("first_admission_host_reused_input_tokens", 0)
        for row in warm["per_request"]
    ) == (8 if scope else 0)
    phases = warm["agentic_phases"]
    assert phases["phase"] == "profile"
    assert phases["failure_reason"] is None
    assert phases["lanes"][0]["warmup_completed"] == 10
    quiescent = max(row["quiescent_at_ms"] for row in phases["requests"])
    assert phases["profile_start_ms"] >= quiescent
    if scope:
        assert phases["profile_start_ms"] > quiescent
        assert phases["profile_start_ms"] == pytest.approx(202)
    assert all(row["agentic_phase"] == "profile" for row in warm["per_request"])


@pytest.mark.parametrize("scope", [None, "dp_rank_local", "cluster_shared"])
@pytest.mark.parametrize("warmup", [False, True])
def test_profile_recycles_cache_identity_and_stops_at_cutoff(
    tmp_path, run_replay, scope, warmup
):
    spec = _spec(tmp_path, scope=scope)
    spec = replace(
        spec,
        workload={
            **spec.workload,
            "agentic_snapshot": {"seed": 5},
            "agentic_warmup": warmup,
            "agentic_profile": {"duration_seconds": 0.7, "response_grace_seconds": 0},
        },
    )
    report = run_replay(spec)
    profile = report["agentic_profile"]
    assert profile["admission_closed"]
    assert profile["admission_cutoff_ms"] - profile[
        "profile_start_ms"
    ] == pytest.approx(700)
    assert profile["finished_at_ms"] == profile["admission_cutoff_ms"]
    assert profile["plays_started"] > 1
    assert profile["client_in_flight_requests"] == 0
    assert profile["unsettled_server_requests"] == 0
    assert not profile["cancel_drain_timed_out"]
    assert all(0 <= row["arrival_time_ms"] < 700 for row in report["per_request"])
    cache_by_play = {}
    for row in report["per_request"]:
        identity = row["agentic"]
        cache_by_play.setdefault(identity["play_id"], set()).add(identity["cache_id"])
    assert len(cache_by_play) > 1
    assert all(len(ids) == 1 for ids in cache_by_play.values())
    assert len({next(iter(ids)) for ids in cache_by_play.values()}) == len(
        cache_by_play
    )
    if warmup:
        assert (
            profile["profile_start_ms"]
            == report["agentic_phases"]["profile_start_ms"]
            > 0
        )
    else:
        assert profile["profile_start_ms"] == 0


@pytest.mark.parametrize("scope", ["dp_rank_local", "cluster_shared"])
def test_profile_cancels_incomplete_host_restore_without_draining_it(
    tmp_path, run_replay, scope
):
    spec = _spec(tmp_path, scope=scope)
    spec.backend_deployment.agg_engine_args["native_host_offload"][
        "h2d_bandwidth_gbps"
    ] = 0.001
    spec = replace(
        spec,
        workload={
            **spec.workload,
            "agentic_snapshot": {"seed": 5},
            "agentic_warmup": True,
            "agentic_profile": {"duration_seconds": 0.25, "response_grace_seconds": 0},
        },
    )
    report = run_replay(spec)
    profile = report["agentic_profile"]
    assert profile["canceled_requests"] == 1
    assert profile["client_in_flight_requests"] == 0
    assert profile["unsettled_server_requests"] == 0
    assert not profile["cancel_drain_timed_out"]
    assert profile["finished_at_ms"] == pytest.approx(profile["profile_start_ms"] + 250)
    canceled = [
        row for row in report["per_request"] if row["terminal_status"] == "canceled"
    ]
    assert len(canceled) == 1
    assert canceled[0]["dispatched_at_ms"] is not None
    assert canceled[0]["first_admit_ms"] is None
    assert canceled[0]["output_length"] == 0
