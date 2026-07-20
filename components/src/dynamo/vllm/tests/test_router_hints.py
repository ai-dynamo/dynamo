# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINT_RUNTIME_KEY,
)
from dynamo.vllm.router_hints import enable_router_hint_support

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_enable_router_hint_support_extracts_router_hint_endpoint():
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

    enable_router_hint_support(runtime_config, engine_args)

    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY, "true"
    )
    runtime_config.set_engine_specific.assert_any_call(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINT_RUNTIME_KEY,
        json.dumps("tcp://127.0.0.1:23280"),
    )


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
        enable_router_hint_support(runtime_config, engine_args)

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

    enable_router_hint_support(runtime_config, engine_args)

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
        enable_router_hint_support(runtime_config, engine_args)

    runtime_config.set_engine_specific.assert_not_called()
