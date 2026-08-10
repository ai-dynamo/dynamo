# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import WorkerType
from dynamo.vllm.router_hints import enable_router_hint_support

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_enable_router_hint_support_publishes_single_dp_rank_endpoint():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": ["router_hint"],
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    enable_router_hint_support(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY, json.dumps(True)
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps("prefill")
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps({"0": "tcp://127.0.0.1:23280"}),
    )


@pytest.mark.parametrize(
    ("worker_type", "expected_runtime_value"),
    [
        (WorkerType.Aggregated, "aggregated"),
        (WorkerType.Decode, "decode"),
    ],
)
def test_enable_router_hint_support_publishes_worker_type(
    worker_type, expected_runtime_value
):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": ["router_hint"],
                        "control_advertise_host": "worker-a",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    enable_router_hint_support(runtime_config, engine_args, worker_type)

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps(expected_runtime_value)
    )


def test_enable_router_hint_support_publishes_dp_rank_endpoints():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": ["router_hint"],
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "worker-a",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    enable_router_hint_support(
        runtime_config, engine_args, WorkerType.Prefill, dp_range=(4, 2)
    )

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps({"4": "tcp://worker-a:23280", "5": "tcp://worker-a:23281"}),
    )


def test_enable_router_hint_support_brackets_ipv6_endpoint():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": ["router_hint"],
                        "control_advertise_host": "2001:db8::1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    enable_router_hint_support(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps({"0": "tcp://[2001:db8::1]:23280"}),
    )


def test_enable_router_hint_support_rejects_dp_offset_port_overflow():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": ["router_hint"],
                        "control_advertise_host": "worker-a",
                        "control_port": "65535",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="router_hint support requires"):
        enable_router_hint_support(
            runtime_config, engine_args, WorkerType.Prefill, dp_range=(0, 2)
        )

    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize("dp_range", [(-1, 1), (0, 0)])
def test_enable_router_hint_support_rejects_invalid_dp_range(dp_range):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": ["router_hint"],
                        "control_advertise_host": "worker-a",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="router_hint support requires"):
        enable_router_hint_support(
            runtime_config, engine_args, WorkerType.Prefill, dp_range=dp_range
        )

    runtime_config.set_engine_specific.assert_not_called()


def test_enable_router_hint_support_fails_with_multiple_router_hint_tiers():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom-a",
                        "router_capabilities": ["router_hint"],
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    },
                    {
                        "type": "custom-b",
                        "router_capabilities": ["router_hint"],
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23281",
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="exactly one router-hint-capable"):
        enable_router_hint_support(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_not_called()


def test_enable_router_hint_support_skips_for_unsupported_worker_roles():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "kvcc",
                        "router_capabilities": ["router_hint"],
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    enable_router_hint_support(runtime_config, engine_args, WorkerType.Encode)

    runtime_config.set_engine_specific.assert_not_called()


def test_enable_router_hint_support_skips_without_router_hint_capability():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "kvcc",
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    enable_router_hint_support(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_not_called()


def test_enable_router_hint_support_fails_without_advertisable_endpoint():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "kvcc",
                        "router_capabilities": ["router_hint"],
                        "control_host": "0.0.0.0",
                        "control_port": 23280,
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="router_hint support requires"):
        enable_router_hint_support(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_not_called()
