# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.llm import WorkerType
from dynamo.vllm.kv_hints import publish_kv_hint_capabilities

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_publish_kv_hint_capabilities_publishes_transfer_endpoint():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_any_call(
        "kv_hint.transfer.v1", json.dumps(True)
    )
    runtime_config.set_engine_specific.assert_any_call(
        "kv_hint_transfer_worker_type", json.dumps("prefill")
    )
    runtime_config.set_engine_specific.assert_any_call(
        "kv_hint_transfer_source_control_endpoints",
        json.dumps({"0": "tcp://127.0.0.1:23280"}),
    )


@pytest.mark.parametrize(
    ("worker_type", "expected_runtime_value"),
    [
        (WorkerType.Aggregated, "aggregated"),
        (WorkerType.Decode, "decode"),
    ],
)
def test_publish_kv_hint_capabilities_publishes_transfer_worker_type(
    worker_type, expected_runtime_value
):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "worker-a",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    publish_kv_hint_capabilities(runtime_config, engine_args, worker_type)

    runtime_config.set_engine_specific.assert_any_call(
        "kv_hint_transfer_worker_type", json.dumps(expected_runtime_value)
    )


def test_publish_kv_hint_capabilities_publishes_transfer_dp_rank_endpoints():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "worker-a",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    publish_kv_hint_capabilities(
        runtime_config, engine_args, WorkerType.Prefill, dp_range=(4, 2)
    )

    runtime_config.set_engine_specific.assert_any_call(
        "kv_hint_transfer_source_control_endpoints",
        json.dumps({"4": "tcp://worker-a:23280", "5": "tcp://worker-a:23281"}),
    )


def test_publish_kv_hint_capabilities_brackets_transfer_ipv6_endpoint():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "2001:db8::1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_any_call(
        "kv_hint_transfer_source_control_endpoints",
        json.dumps({"0": "tcp://[2001:db8::1]:23280"}),
    )


def test_publish_kv_hint_capabilities_rejects_transfer_port_overflow():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "worker-a",
                        "control_port": "65535",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="TRANSFER hint support requires"):
        publish_kv_hint_capabilities(
            runtime_config, engine_args, WorkerType.Prefill, dp_range=(0, 2)
        )

    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize("dp_range", [(-1, 1), (0, 0)])
def test_publish_kv_hint_capabilities_rejects_invalid_transfer_dp_range(dp_range):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "worker-a",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="TRANSFER hint support requires"):
        publish_kv_hint_capabilities(
            runtime_config, engine_args, WorkerType.Prefill, dp_range=dp_range
        )

    runtime_config.set_engine_specific.assert_not_called()


def test_publish_kv_hint_capabilities_rejects_multiple_transfer_tiers():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom-a",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    },
                    {
                        "type": "custom-b",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23281",
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="exactly one TRANSFER-capable"):
        publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_not_called()


def test_publish_kv_hint_capabilities_skips_transfer_for_unsupported_worker_role():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "kvcc",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_host": "0.0.0.0",
                        "control_advertise_host": "127.0.0.1",
                        "control_port": "23280",
                    }
                ]
            }
        )
    )

    publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Encode)

    runtime_config.set_engine_specific.assert_not_called()


def test_publish_kv_hint_capabilities_skips_unadvertised_transfer():
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

    publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_not_called()


def test_publish_kv_hint_capabilities_rejects_transfer_without_endpoint():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "kvcc",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_host": "0.0.0.0",
                        "control_port": 23280,
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="TRANSFER hint support requires"):
        publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_not_called()
