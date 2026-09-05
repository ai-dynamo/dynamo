# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.llm import WorkerType
from dynamo.vllm.kv_hints import publish_kv_hint_capabilities

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# Success cases: a transfer-hint-capable tier publishes capability, worker role,
# and source-control endpoints keyed by global DP rank.
@pytest.mark.parametrize(
    ("tier", "worker_type", "dp_range", "expected_worker_type", "expected_endpoints"),
    [
        # Single-rank prefill worker publishes rank 0 with the advertise host.
        pytest.param(
            {
                "type": "custom",
                "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                "control_host": "0.0.0.0",
                "control_advertise_host": "127.0.0.1",
                "control_ports": ["23280"],
            },
            WorkerType.Prefill,
            (0, 1),
            "prefill",
            {"0": "tcp://127.0.0.1:23280"},
            id="single-dp-prefill",
        ),
        # Aggregated workers are valid hint participants and publish that role.
        pytest.param(
            {
                "type": "custom",
                "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                "control_advertise_host": "worker-a",
                "control_ports": ["23280"],
            },
            WorkerType.Aggregated,
            (0, 1),
            "aggregated",
            {"0": "tcp://worker-a:23280"},
            id="aggregated-worker-type",
        ),
        # Decode workers are valid hint participants and publish that role.
        pytest.param(
            {
                "type": "custom",
                "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                "control_advertise_host": "worker-a",
                "control_ports": ["23280"],
            },
            WorkerType.Decode,
            (0, 1),
            "decode",
            {"0": "tcp://worker-a:23280"},
            id="decode-worker-type",
        ),
        # A worker managing global DP ranks 4 and 5 uses local ports[0:2].
        pytest.param(
            {
                "type": "custom",
                "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                "control_host": "0.0.0.0",
                "control_advertise_host": "worker-a",
                "control_ports": ["24000", "24001"],
            },
            WorkerType.Prefill,
            (4, 2),
            "prefill",
            {"4": "tcp://worker-a:24000", "5": "tcp://worker-a:24001"},
            id="global-dp-rank-endpoints",
        ),
        # IPv6 advertise hosts are bracketed before publishing tcp:// endpoints.
        pytest.param(
            {
                "type": "custom",
                "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                "control_advertise_host": "2001:db8::1",
                "control_ports": ["23280"],
            },
            WorkerType.Prefill,
            (0, 1),
            "prefill",
            {"0": "tcp://[2001:db8::1]:23280"},
            id="ipv6-advertise-host",
        ),
    ],
)
def test_publish_kv_hint_capabilities_publishes_runtime_metadata(
    tier, worker_type, dp_range, expected_worker_type, expected_endpoints
):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"secondary_tiers": [tier]}
        )
    )

    publish_kv_hint_capabilities(
        runtime_config, engine_args, worker_type, dp_range=dp_range
    )

    runtime_config.set_engine_specific.assert_any_call(
        KV_HINT_TRANSFER_CAPABILITY_KEY, json.dumps(True)
    )
    runtime_config.set_engine_specific.assert_any_call(
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, json.dumps(expected_worker_type)
    )
    runtime_config.set_engine_specific.assert_any_call(
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps(expected_endpoints),
    )


def test_publish_kv_hint_capabilities_keeps_source_endpoint_off_state_agent_worker():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "secondary_tiers": [
                    {
                        "type": "custom",
                        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                        "control_advertise_host": "127.0.0.1",
                        "control_ports": ["23280"],
                    }
                ]
            }
        )
    )

    publish_kv_hint_capabilities(
        runtime_config,
        engine_args,
        WorkerType.Prefill,
        publish_source_endpoints=False,
    )

    published_keys = {
        call.args[0] for call in runtime_config.set_engine_specific.call_args_list
    }
    assert published_keys == {
        KV_HINT_TRANSFER_CAPABILITY_KEY,
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
    }


# Skip cases: without both transfer-hint capability and a supported worker role,
# registration leaves runtime metadata untouched instead of failing.
@pytest.mark.parametrize(
    ("tier", "worker_type"),
    [
        # Tier does not opt in with router_capabilities=[KV_HINT_TRANSFER_CAPABILITY_KEY].
        pytest.param(
            {
                "type": "custom",
                "control_host": "0.0.0.0",
                "control_advertise_host": "127.0.0.1",
                "control_ports": ["23280"],
            },
            WorkerType.Prefill,
            id="without-transfer-hint-capability",
        ),
        # Encode workers do not consume or serve transfer hints today.
        pytest.param(
            {
                "type": "custom",
                "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                "control_host": "0.0.0.0",
                "control_advertise_host": "127.0.0.1",
                "control_ports": ["23280"],
            },
            WorkerType.Encode,
            id="unsupported-encode-worker",
        ),
    ],
)
def test_publish_kv_hint_capabilities_skips_without_supported_hint_participant(
    tier, worker_type
):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"secondary_tiers": [tier]}
        )
    )

    publish_kv_hint_capabilities(runtime_config, engine_args, worker_type)

    runtime_config.set_engine_specific.assert_not_called()


# Fail cases: once a supported worker opts into transfer hints, the advertised
# endpoint metadata must be complete and internally consistent.
@pytest.mark.parametrize(
    ("secondary_tiers", "dp_range", "error_match"),
    [
        # control_host is bind-only; transfer hints require control_advertise_host.
        pytest.param(
            [
                {
                    "type": "custom",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_host": "worker-a",
                    "control_ports": [23280],
                }
            ],
            (0, 1),
            "P2P KV transfer hint support requires",
            id="without-advertise-host",
        ),
        # Every published endpoint port must be in the valid TCP port range.
        pytest.param(
            [
                {
                    "type": "custom",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "worker-a",
                    "control_ports": ["65535", "65536"],
                }
            ],
            (0, 2),
            "P2P KV transfer hint support requires",
            id="invalid-control-port",
        ),
        # The port list is worker-local, so it must match this worker's DP size.
        pytest.param(
            [
                {
                    "type": "custom",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "worker-a",
                    "control_ports": ["24000"],
                }
            ],
            (4, 2),
            "P2P KV transfer hint support requires",
            id="wrong-local-port-count-for-managed-dp-range",
        ),
        # Exactly one secondary tier may provide transfer-hint source endpoints.
        pytest.param(
            [
                {
                    "type": "custom-a",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "127.0.0.1",
                    "control_ports": ["23280"],
                },
                {
                    "type": "custom-b",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "127.0.0.1",
                    "control_ports": ["23281"],
                },
            ],
            (0, 1),
            "P2P KV transfer hint support requires exactly one capable",
            id="multiple-transfer-hint-tiers",
        ),
        # DP start rank must be non-negative.
        pytest.param(
            [
                {
                    "type": "custom",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "worker-a",
                    "control_ports": ["23280"],
                }
            ],
            (-1, 1),
            "P2P KV transfer hint support requires",
            id="negative-dp-start",
        ),
        # DP size must be positive.
        pytest.param(
            [
                {
                    "type": "custom",
                    "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
                    "control_advertise_host": "worker-a",
                    "control_ports": ["23280"],
                }
            ],
            (0, 0),
            "P2P KV transfer hint support requires",
            id="zero-dp-size",
        ),
    ],
)
def test_publish_kv_hint_capabilities_fails_with_invalid_transfer_hint_config(
    secondary_tiers, dp_range, error_match
):
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"secondary_tiers": secondary_tiers}
        )
    )

    with pytest.raises(ValueError, match=error_match):
        publish_kv_hint_capabilities(
            runtime_config, engine_args, WorkerType.Prefill, dp_range=dp_range
        )

    runtime_config.set_engine_specific.assert_not_called()
