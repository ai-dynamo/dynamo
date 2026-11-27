# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay router control through the optional ``router_parameters`` scaling decision key.

A Python scaling policy passed to an offline replay may return desired default-policy parameters
keyed by worker type. Rust applies them to the replay's KV routers at the tick's simulated time and
reports each router's revision back in the next snapshot. ``deterministic=True`` seeds the KV
routers, so replays of one trace are comparable.
"""

import json

import pytest

from dynamo._core import run_mocker_synthetic_trace_replay, run_mocker_trace_replay
from dynamo.llm import KvRouterConfig
from dynamo.mocker.config import normalize_mocker_config

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.router,
    pytest.mark.unit,
    pytest.mark.timeout(120),
]

_ABSENT = object()
_TRACE_BLOCK_SIZE = 512


def _engine_payload(**engine):
    return {
        "engine": {
            "block_size": 64,
            "num_gpu_blocks": 4096,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 8192,
            **engine,
        }
    }


def _topology_kwargs(disagg: bool, workers: int = 1) -> dict:
    if disagg:
        return {
            "prefill_engine_args": normalize_mocker_config(
                _engine_payload(worker_type="prefill")
            ),
            "decode_engine_args": normalize_mocker_config(
                _engine_payload(worker_type="decode")
            ),
            "num_prefill_workers": workers,
            "num_decode_workers": workers,
        }
    return {
        "extra_engine_args": normalize_mocker_config(_engine_payload()),
        "num_workers": workers,
    }


def _write_trace(tmp_path, records, name="router_control.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def _small_trace(tmp_path):
    return _write_trace(
        tmp_path,
        [
            {
                "timestamp": float(timestamp),
                "input_length": _TRACE_BLOCK_SIZE * len(hash_ids),
                "output_length": 8,
                "hash_ids": hash_ids,
            }
            for timestamp, hash_ids in [
                (0, [0, 1]),
                (40, [0, 2]),
                (90, [3]),
                (180, [0, 1, 4]),
                (260, [5, 6]),
            ]
        ],
    )


class _RouterPolicy:
    """Ticks at fixed simulated times and returns a scripted ``router_parameters`` value.

    ``_ABSENT`` omits the key, so that tick's decision has only the three mandatory keys.
    """

    def __init__(self, tick_ms, router_parameters, on_tick=None):
        assert len(tick_ms) == len(router_parameters)
        self.tick_ms = list(tick_ms)
        self.router_parameters = list(router_parameters)
        self.snapshots = []
        self._on_tick = on_tick

    def initial_tick_ms(self):
        return self.tick_ms[0]

    def on_tick(self, metrics):
        index = len(self.snapshots)
        self.snapshots.append(metrics)
        if self._on_tick is not None:
            self._on_tick(index, metrics)
        decision = {
            "target_prefill": None,
            "target_decode": None,
            "next_tick_ms": (
                self.tick_ms[index + 1] if index + 1 < len(self.tick_ms) else None
            ),
        }
        value = self.router_parameters[index]
        if value is not _ABSENT:
            decision["router_parameters"] = value
        return decision


def _worker_types(disagg: bool):
    return ("prefill", "decode") if disagg else ("aggregated",)


def _startup_map(config: KvRouterConfig, disagg: bool):
    return {
        worker_type: config.policy_parameters(worker_type)
        for worker_type in _worker_types(disagg)
    }


def _router_policies(snapshot):
    return {
        policy["worker_type"]: (policy["revision"], policy["parameters"])
        for policy in snapshot["router_policies"]
    }


@pytest.mark.parametrize("disagg", [False, True], ids=["agg", "disagg"])
def test_router_parameters_are_level_triggered(tmp_path, disagg):
    config = KvRouterConfig(overlap_score_credit=1.5, router_temperature=0.0)
    startup = _startup_map(config, disagg)
    worker_type = _worker_types(disagg)[0]
    changed = {worker_type: {"prefill_load_scale": 2.5, "router_temperature": 0}}
    # An absent key, None, {} and the startup map are no-ops; a changed map is applied once
    # and repeating it is a no-op.
    script = [_ABSENT, None, {}, startup, changed, changed, _ABSENT]
    policy = _RouterPolicy([20.0 + 30.0 * i for i in range(len(script))], script)
    run_mocker_trace_replay(
        [str(_small_trace(tmp_path))],
        router_config=config,
        router_mode="kv_router",
        trace_block_size=_TRACE_BLOCK_SIZE,
        scaling_policy=policy,
        **_topology_kwargs(disagg),
    )

    # Each snapshot reports the router state before that tick's update.
    policies = [_router_policies(snapshot) for snapshot in policy.snapshots]
    assert [p[worker_type][0] for p in policies] == [0, 0, 0, 0, 0, 1, 1]
    for snapshot in policies[:5]:
        assert set(snapshot) == set(_worker_types(disagg))
        for name, (_revision, parameters) in snapshot.items():
            assert parameters == pytest.approx(startup[name])
    for snapshot in policies[5:]:
        parameters = snapshot[worker_type][1]
        assert parameters["prefill_load_scale"] == pytest.approx(2.5)
        assert parameters["router_temperature"] == 0.0
        assert parameters["overlap_score_credit"] == pytest.approx(1.5)
    if disagg:
        assert [p["decode"][0] for p in policies] == [0] * len(script)


def test_mid_run_credit_change_routes_away_from_a_loaded_worker(tmp_path):
    # Request 1 (two trace blocks) is still decoding when request 2 arrives. Request 2 shares
    # request 1's first trace block and adds seven more. Both workers then project request 2's
    # full prefill, and request 1's worker also carries request 1's unshared decode blocks. With
    # credit 100, the shared prefix cancels that worker's prefill cost, so it wins; with credit 0
    # its extra decode load makes the idle worker strictly cheaper.
    trace = _write_trace(
        tmp_path,
        [
            {
                "timestamp": 0.0,
                "input_length": 2 * _TRACE_BLOCK_SIZE,
                "output_length": 4000,
                "hash_ids": [1, 5],
            },
            {
                "timestamp": 1000.0,
                "input_length": 8 * _TRACE_BLOCK_SIZE,
                "output_length": 2,
                "hash_ids": [1, 9, 10, 11, 12, 13, 14, 15],
            },
        ],
    )
    config = KvRouterConfig(
        overlap_score_credit=100.0,
        overlap_score_credit_decay=0.0,
        router_temperature=0.0,
    )

    def routes(credit):
        policy = _RouterPolicy(
            [250.0], [{"aggregated": {"overlap_score_credit": credit}}]
        )
        report = run_mocker_trace_replay(
            [str(trace)],
            router_config=config,
            router_mode="kv_router",
            trace_block_size=_TRACE_BLOCK_SIZE,
            capture_per_request=True,
            scaling_policy=policy,
            **_topology_kwargs(disagg=False, workers=2),
        )
        first, second = sorted(report.per_request, key=lambda r: r["arrival_time_ms"])
        assert first["terminal_time_ms"] > second["arrival_time_ms"]
        return (
            first["routing_history"][0]["logical_worker_id"],
            second["routing_history"][0]["logical_worker_id"],
        )

    first, second = routes(100.0)
    assert second == first
    first, second = routes(0.0)
    assert second != first


@pytest.mark.parametrize(
    "router_mode, router_parameters, error, match",
    [
        ("kv_router", {"bogus": {"router_temperature": 1.0}}, ValueError, "bogus"),
        ("kv_router", [("aggregated", {})], TypeError, "router_parameters"),
        (
            "kv_router",
            {"aggregated": {"router_temperature": "hot"}},
            TypeError,
            "router_parameters",
        ),
        (
            "round_robin",
            {"aggregated": {"router_temperature": 1.0}},
            ValueError,
            "no aggregated KV router",
        ),
    ],
    ids=["unknown-worker-type", "not-a-dict", "not-a-float", "round-robin"],
)
def test_invalid_router_parameters_fail_the_replay(
    tmp_path, router_mode, router_parameters, error, match
):
    with pytest.raises(error, match=match):
        run_mocker_trace_replay(
            [str(_small_trace(tmp_path))],
            router_mode=router_mode,
            trace_block_size=_TRACE_BLOCK_SIZE,
            scaling_policy=_RouterPolicy([20.0], [router_parameters]),
            **_topology_kwargs(disagg=False),
        )


def test_synthetic_replay_applies_router_parameters():
    policy = _RouterPolicy(
        [1.0, 2.0], [{"aggregated": {"router_temperature": 1.0}}, _ABSENT]
    )
    run_mocker_synthetic_trace_replay(
        input_tokens=64,
        output_tokens=64,
        request_count=8,
        replay_concurrency=2,
        router_mode="kv_router",
        extra_engine_args=normalize_mocker_config(_engine_payload()),
        scaling_policy=policy,
    )
    revisions = [_router_policies(s)["aggregated"][0] for s in policy.snapshots]
    assert revisions == [0, 1]


@pytest.mark.parametrize("disagg", [False, True], ids=["agg", "disagg"])
def test_deterministic_replays_are_identical(tmp_path, disagg):
    # Overlapping requests in three prefix groups, so routing sees ties, overlap and load.
    trace = _write_trace(
        tmp_path,
        [
            {
                "timestamp": 3.0 * index,
                "input_length": 2 * _TRACE_BLOCK_SIZE,
                "output_length": 4 + index % 5,
                "hash_ids": [index % 3, 100 + index],
            }
            for index in range(32)
        ],
    )

    def per_request(**kwargs):
        report = run_mocker_trace_replay(
            [str(trace)],
            router_config=KvRouterConfig(router_temperature=0.5),
            router_mode="kv_router",
            trace_block_size=_TRACE_BLOCK_SIZE,
            capture_per_request=True,
            **_topology_kwargs(disagg, workers=3),
            **kwargs,
        )
        return report.per_request

    first = per_request(deterministic=True)
    assert len(first) == 32
    assert per_request(deterministic=True) == first
    # The default keeps random request IDs.
    assert per_request() != first
    with pytest.raises(ValueError, match="deterministic"):
        run_mocker_trace_replay([str(trace)], replay_mode="online", deterministic=True)


def test_nested_replay_does_not_register_into_the_outer_replay(tmp_path):
    outer_trace = _small_trace(tmp_path)
    inner_trace = _write_trace(
        tmp_path,
        [
            {
                "timestamp": 0.0,
                "input_length": _TRACE_BLOCK_SIZE,
                "output_length": 4,
                "hash_ids": [0],
            }
        ],
        name="inner.jsonl",
    )

    def run_nested_replay(index, _metrics):
        if index == 0:
            # A replay started from a callback runs on the same thread as the outer one.
            run_mocker_trace_replay(
                [str(inner_trace)],
                router_mode="kv_router",
                trace_block_size=_TRACE_BLOCK_SIZE,
                **_topology_kwargs(disagg=False),
            )

    policy = _RouterPolicy([20.0, 120.0], [None, None], on_tick=run_nested_replay)
    run_mocker_trace_replay(
        [str(outer_trace)],
        router_mode="kv_router",
        trace_block_size=_TRACE_BLOCK_SIZE,
        scaling_policy=policy,
        **_topology_kwargs(disagg=False),
    )
    assert [len(s["router_policies"]) for s in policy.snapshots] == [1, 1]
