# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the entrypoint imports.

"""Installed-entrypoint integration for the native AIPerf workload contract.

Deliberately separate from the Sweeper search integration tests so collection
needs only the runner-factory entrypoint, not the sweeper's search dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

if not os.environ.get("AIPERF_ENTRYPOINT_RUN_REQUEST"):
    pytest.importorskip(
        "aisimulate.stack",
        reason="AI Simulate is an optional Dynamo simulation dependency",
    )

from aisimulate.stack import resolve_runner_factory

from dynamo.replay.simulation import DynamoReplayRunnerFactory

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.planner,
]

# Native AIPerf workload through the installed entrypoint. Opt in by setting
# AIPERF_ENTRYPOINT_RUN_REQUEST to a protocol-v2 RunRequestV2 frame file. Once
# set, a missing native binding or function is a failure, never a skip.
_RUN_REQUEST_ENV = "AIPERF_ENTRYPOINT_RUN_REQUEST"
_SEED_HEX = "07" * 32
_DEPLOYMENT_2P2D = {
    "engine": {
        "rank": {
            "backend": "vllm",
            "block_size": 8,
            "enable_prefix_caching": True,
            "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 2.0},
        },
    },
    "prefill_workers": 2,
    "decode_workers": 2,
    "admission_capacity": 64,
}

_requires_aiperf_frame = pytest.mark.skipif(
    not os.environ.get(_RUN_REQUEST_ENV),
    reason=f"set {_RUN_REQUEST_ENV} to a RunRequestV2 frame file to run",
)


def _installed_runner():
    # No injected entry points: resolve through installed importlib metadata.
    return resolve_runner_factory("dynamo").create(0)


def _frame_text() -> str:
    return Path(os.environ[_RUN_REQUEST_ENV]).read_text(encoding="utf-8")


@_requires_aiperf_frame
def test_installed_entrypoint_runs_aiperf_workload_on_2p2d_kv_deployment() -> None:
    assert type(resolve_runner_factory("dynamo")) is DynamoReplayRunnerFactory
    envelope = _installed_runner().run_aiperf_workload(
        _frame_text(),
        json.dumps(_DEPLOYMENT_2P2D),
        identity_seed_hex=_SEED_HEX,
    )

    assert envelope["engine_drained"] is True
    requests = envelope["requests"]
    assert len(requests) == 4
    children = [r for r in requests if r["request"]["parent_correlation_id"] is not None]
    parents = [r for r in requests if r["request"]["parent_correlation_id"] is None]
    assert children and parents, "workload must exercise parent and child requests"
    assert envelope["backend_report"]["completed_requests"] == 4

    records = envelope["backend_per_request"]
    assert len(records) == 4
    for record in records:
        assert record["terminal_status"] == "completed"
        pools = [route["pool"] for route in record["routing_history"]]
        assert pools.count("prefill") == 1, record["routing_history"]
        assert pools.count("decode") == 1, record["routing_history"]


@_requires_aiperf_frame
@pytest.mark.parametrize(
    ("deployment", "seed", "message"),
    [
        ("{not json", _SEED_HEX, "invalid deployment configuration"),
        (
            json.dumps({**_DEPLOYMENT_2P2D, "unknown_field": 1}),
            _SEED_HEX,
            "invalid deployment configuration",
        ),
        (
            json.dumps({**_DEPLOYMENT_2P2D, "decode_workers": 0}),
            _SEED_HEX,
            "prefill and decode worker counts must be positive",
        ),
        (json.dumps(_DEPLOYMENT_2P2D), "07" * 31, "64 hexadecimal digits"),
        (json.dumps(_DEPLOYMENT_2P2D), "zz" * 32, "64 hexadecimal digits"),
    ],
)
def test_real_native_binding_rejects_malformed_config_and_seed(
    deployment: str, seed: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _installed_runner().run_aiperf_workload(
            _frame_text(), deployment, identity_seed_hex=seed
        )
