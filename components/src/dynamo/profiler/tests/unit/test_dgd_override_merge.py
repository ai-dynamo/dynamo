# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for apply_dgd_overrides — verifies router and planner overrides
flow through spec.overrides.dgd correctly (GitHub #10269)."""

import pytest

from dynamo.profiler.utils.config_modifiers.protocol import apply_dgd_overrides

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


def _base_dgd():
    """Minimal generated DGD — simulates what the profiler produces."""
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "generated"},
        "spec": {
            "services": {
                "Frontend": {"envs": [{"name": "EXISTING_VAR", "value": "existing"}]},
                "VllmWorker": {},
                "VllmPrefillWorker": {"extraPodSpec": {"mainContainer": {"args": []}}},
            }
        },
    }


def test_router_mode_env_survives_merge():
    base = _base_dgd()
    override = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "spec": {
            "services": {
                "Frontend": {"envs": [{"name": "DYN_ROUTER_MODE", "value": "kv"}]}
            }
        },
    }

    result = apply_dgd_overrides(base, override)

    envs = result["spec"]["services"]["Frontend"]["envs"]
    router_mode = next(e for e in envs if e["name"] == "DYN_ROUTER_MODE")
    assert router_mode["value"] == "kv"


def test_vllm_prefill_worker_kv_args_survive_merge():
    base = _base_dgd()
    override = {
        "spec": {
            "services": {
                "VllmPrefillWorker": {
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": [
                                "--enable-prefix-caching",
                                "--kv-events-config",
                                '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}',
                            ]
                        }
                    }
                }
            }
        }
    }

    result = apply_dgd_overrides(base, override)

    args = result["spec"]["services"]["VllmPrefillWorker"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert "--enable-prefix-caching" in args
    assert "--kv-events-config" in args
    assert (
        '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
        in args
    )


def test_approximate_kv_mode_envs_survive_merge():
    base = _base_dgd()
    override = {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "spec": {
            "services": {
                "Frontend": {
                    "envs": [
                        {"name": "DYN_ROUTER_MODE", "value": "kv"},
                        {"name": "DYN_ROUTER_USE_KV_EVENTS", "value": "false"},
                    ]
                }
            }
        },
    }

    result = apply_dgd_overrides(base, override)

    envs = result["spec"]["services"]["Frontend"]["envs"]
    router_mode = next(e for e in envs if e["name"] == "DYN_ROUTER_MODE")
    assert router_mode["value"] == "kv"
    kv_events = next(e for e in envs if e["name"] == "DYN_ROUTER_USE_KV_EVENTS")
    assert kv_events["value"] == "false"


def test_envelope_fields_are_stripped():
    base = _base_dgd()
    override = {"apiVersion": "nvidia.com/v1beta1", "kind": "SomethingElse"}

    result = apply_dgd_overrides(base, override)

    assert result["apiVersion"] == "nvidia.com/v1alpha1"
    assert result["kind"] == "DynamoGraphDeployment"
