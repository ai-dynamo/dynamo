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
from dynamo.vllm import kv_hints
from dynamo.vllm.kv_hints import publish_kv_hint_capabilities

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
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


def test_publish_kv_hint_capabilities_reads_tier_inside_multi_connector():
    runtime_config = MagicMock()
    engine_args = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="MultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
                    {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "spec_name": "TieringOffloadingSpec",
                            "secondary_tiers": [
                                {
                                    "type": "kvcr",
                                    "router_capabilities": [
                                        KV_HINT_TRANSFER_CAPABILITY_KEY
                                    ],
                                    "control_advertise_host": "127.0.0.1",
                                    "control_ports": [17771],
                                }
                            ],
                        },
                    },
                ]
            },
        )
    )

    publish_kv_hint_capabilities(runtime_config, engine_args, WorkerType.Prefill)

    runtime_config.set_engine_specific.assert_any_call(
        KV_HINT_TRANSFER_CAPABILITY_KEY, json.dumps(True)
    )
    runtime_config.set_engine_specific.assert_any_call(
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps({"0": "tcp://127.0.0.1:17771"}),
    )


def _kvcr_connector(**tier_overrides):
    tier = {
        "type": "kvcr",
        "router_capabilities": [KV_HINT_TRANSFER_CAPABILITY_KEY],
        "control_advertise_host": "decode-worker",
        "control_ports": [24000, 24001],
        **tier_overrides,
    }
    return {
        "kv_connector": "OffloadingConnector",
        "kv_connector_extra_config": {"secondary_tiers": [tier]},
    }


def _wrapped_engine_args(wrapper, children, **extra_config):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=wrapper,
            kv_connector_extra_config={"connectors": children, **extra_config},
        )
    )


@pytest.mark.parametrize("wrapper", ["MultiConnector", "PdConnector"])
def test_nested_connector_publishes_exact_per_dp_source(wrapper):
    engine_args = _wrapped_engine_args(
        wrapper,
        [
            {
                "kv_connector": "MultiConnector",
                "kv_connector_extra_config": {"connectors": [_kvcr_connector()]},
            },
            {"kv_connector": "NixlConnector"},
        ],
    )
    runtime_config = MagicMock()
    publish_kv_hint_capabilities(
        runtime_config, engine_args, WorkerType.Decode, dp_range=(4, 2)
    )
    calls = runtime_config.set_engine_specific.call_args_list
    assert calls[0].args == (
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps(
            {"4": "tcp://decode-worker:24000", "5": "tcp://decode-worker:24001"}
        ),
    )
    assert calls[-1].args == (KV_HINT_TRANSFER_CAPABILITY_KEY, "true")


def test_nested_connector_rejects_ambiguous_transfer_providers_before_publication():
    engine_args = _wrapped_engine_args(
        "MultiConnector",
        [
            _kvcr_connector(),
            {
                "kv_connector": "PdConnector",
                "kv_connector_extra_config": {"connectors": [_kvcr_connector()]},
            },
        ],
    )
    runtime_config = MagicMock()
    with pytest.raises(ValueError, match="exactly one capable"):
        publish_kv_hint_capabilities(
            runtime_config, engine_args, WorkerType.Decode, (0, 2)
        )
    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize("children", ["OffloadingConnector", ["OffloadingConnector"]])
def test_nested_connector_rejects_malformed_children(children):
    runtime_config = MagicMock()
    with pytest.raises(ValueError, match="list of connector configs"):
        publish_kv_hint_capabilities(
            runtime_config,
            _wrapped_engine_args("PdConnector", children),
            WorkerType.Decode,
        )
    runtime_config.set_engine_specific.assert_not_called()


def test_unknown_wrapper_does_not_claim_child_transfer_capabilities():
    runtime_config = MagicMock()
    publish_kv_hint_capabilities(
        runtime_config,
        _wrapped_engine_args("UncertifiedWrapper", [_kvcr_connector()]),
        WorkerType.Decode,
        (0, 2),
    )
    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "tier_overrides",
    [
        {"control_advertise_host": "0.0.0.0"},
        {"control_advertise_host": "::"},
        {"control_advertise_host": "tcp://decode-worker"},
        {"control_advertise_host": "decode-worker/path"},
        {"control_advertise_host": "decode worker"},
        {"control_advertise_host": "127.1"},
        {"control_advertise_host": "worker.-invalid"},
        {"control_advertise_host": "239.1.1.1"},
        {"control_ports": [24000, 24000]},
        {"control_ports": [24000, True]},
    ],
)
def test_nested_connector_rejects_unusable_source_endpoints(tier_overrides):
    runtime_config = MagicMock()
    with pytest.raises(ValueError, match="P2P KV transfer hint support requires"):
        publish_kv_hint_capabilities(
            runtime_config,
            _wrapped_engine_args("PdConnector", [_kvcr_connector(**tier_overrides)]),
            WorkerType.Decode,
            (0, 2),
        )
    runtime_config.set_engine_specific.assert_not_called()


def test_transfer_domain_is_explicit_and_published_before_capability(monkeypatch):
    from dynamo.common.constants import KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY

    monkeypatch.delenv("DYN_KV_HINT_TRANSFER_DOMAIN", raising=False)
    runtime_config = MagicMock()
    publish_kv_hint_capabilities(
        runtime_config,
        _wrapped_engine_args(
            "PdConnector",
            [_kvcr_connector()],
            router_hint_transfer_domain="certified-v1",
        ),
        WorkerType.Decode,
        (0, 2),
    )
    calls = runtime_config.set_engine_specific.call_args_list
    assert calls[-2].args == (KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY, '"certified-v1"')
    assert calls[-1].args == (KV_HINT_TRANSFER_CAPABILITY_KEY, "true")


def test_transfer_domain_environment_fallback_and_config_precedence(monkeypatch):
    from dynamo.common.constants import KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY

    monkeypatch.setenv("DYN_KV_HINT_TRANSFER_DOMAIN", "deployment-domain")
    for configured_domain, expected in [
        (None, "deployment-domain"),
        ("configured-domain", "configured-domain"),
    ]:
        runtime_config = MagicMock()
        publish_kv_hint_capabilities(
            runtime_config,
            _wrapped_engine_args(
                "PdConnector",
                [_kvcr_connector()],
                router_hint_transfer_domain=configured_domain,
            ),
            WorkerType.Decode,
            (0, 2),
        )
        runtime_config.set_engine_specific.assert_any_call(
            KV_HINT_TRANSFER_DOMAIN_RUNTIME_KEY, json.dumps(expected)
        )


@pytest.mark.parametrize("domain", ["", " ", " padded ", 42])
def test_invalid_transfer_domain_fails_before_any_publication(domain):
    runtime_config = MagicMock()
    with pytest.raises(ValueError, match="nonempty string"):
        publish_kv_hint_capabilities(
            runtime_config,
            _wrapped_engine_args(
                "PdConnector", [_kvcr_connector()], router_hint_transfer_domain=domain
            ),
            WorkerType.Decode,
            (0, 2),
        )
    runtime_config.set_engine_specific.assert_not_called()


def _gpu_connector(**tier_changes):
    tier = {
        "type": "GPUKVCRSecondaryTierManager",
        "router_capabilities": ["router_hint_gpu_direct"],
        "compatibility_domain": "gpu-certified-layout-v1",
        "control_advertise_host": "worker-a",
        "control_ports": [24123],
    }
    tier.update(tier_changes)
    return {
        "kv_connector": "GPUOffloadingConnector",
        "kv_connector_extra_config": {"secondary_tiers": [tier]},
    }


def test_gpu_hint_capability_is_separate_from_cpu_and_keeps_global_rank():
    runtime = MagicMock()
    publish_kv_hint_capabilities(
        runtime,
        _wrapped_engine_args(
            "PdConnector",
            [_gpu_connector()],
            router_hint_transfer_domain="gpu-certified-layout-v1",
        ),
        WorkerType.Decode,
        (3, 1),
    )
    fields = {
        call.args[0]: json.loads(call.args[1])
        for call in runtime.set_engine_specific.call_args_list
    }
    assert fields["router_hint_gpu_direct"] is True
    assert fields["router_hint_gpu_source_control_endpoints"] == {
        "3": "tcp://worker-a:24123"
    }
    assert fields["router_hint_worker_type"] == "decode"
    assert fields["router_hint_transfer_domain"] == "gpu-certified-layout-v1"
    assert KV_HINT_TRANSFER_CAPABILITY_KEY not in fields
    assert KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY not in fields


@pytest.mark.parametrize(
    "tier_changes",
    [
        {"type": "kvcr"},
        {"compatibility_domain": "different-layout"},
        {"control_ports": [True]},
        {"control_advertise_host": "0.0.0.0"},
    ],
)
def test_gpu_hint_rejects_wrong_memory_provider_domain_or_endpoint(tier_changes):
    runtime = MagicMock()
    with pytest.raises(ValueError):
        publish_kv_hint_capabilities(
            runtime,
            _wrapped_engine_args(
                "PdConnector",
                [_gpu_connector(**tier_changes)],
                router_hint_transfer_domain="gpu-certified-layout-v1",
            ),
            WorkerType.Decode,
            (0, 1),
        )
    runtime.set_engine_specific.assert_not_called()


def test_gpu_hint_does_not_advertise_an_unsupported_cache_owner_source():
    runtime = MagicMock()
    with pytest.raises(ValueError, match="CacheOwner"):
        publish_kv_hint_capabilities(
            runtime,
            _wrapped_engine_args(
                "PdConnector",
                [_gpu_connector()],
                router_hint_transfer_domain="gpu-certified-layout-v1",
                dynamo_state_agent={"raw_advertise_host": "worker-a"},
            ),
            WorkerType.Decode,
            (0, 1),
        )
    runtime.set_engine_specific.assert_not_called()


def test_mnnvl_capability_uses_observed_fabric_effective_domain_without_mutating_config(
    monkeypatch,
):
    observed = SimpleNamespace(domain="mnnvl:" + "1" * 32 + ":0")
    helper = SimpleNamespace(
        observe_fabric=MagicMock(return_value=observed),
        effective_domain=MagicMock(return_value="mnnvl-sha256:observed-layout-fabric"),
    )
    loader = MagicMock(return_value=helper)
    monkeypatch.setattr(kv_hints, "importlib", SimpleNamespace(import_module=loader))
    args = _wrapped_engine_args(
        "PdConnector",
        [_gpu_connector(gpu_transport="mnnvl", ucx_tls=["cuda_ipc", "tcp", "self"])],
        router_hint_transfer_domain="gpu-certified-layout-v1",
    )
    runtime = MagicMock()
    publish_kv_hint_capabilities(runtime, args, WorkerType.Decode)
    fields = {
        call.args[0]: json.loads(call.args[1])
        for call in runtime.set_engine_specific.call_args_list
    }
    assert (
        fields["router_hint_transfer_domain"] == "mnnvl-sha256:observed-layout-fabric"
    )
    assert fields["router_hint_gpu_direct"] is True
    assert KV_HINT_TRANSFER_CAPABILITY_KEY not in fields
    loader.assert_called_once_with("kvbm.vllm_integration.gpu_mnnvl")
    helper.effective_domain.assert_called_once_with(
        "gpu-certified-layout-v1", observed.domain
    )
    assert (
        args.kv_transfer_config.kv_connector_extra_config["router_hint_transfer_domain"]
        == "gpu-certified-layout-v1"
    )
    assert (
        args.kv_transfer_config.kv_connector_extra_config["connectors"][0][
            "kv_connector_extra_config"
        ]["secondary_tiers"][0]["compatibility_domain"]
        == "gpu-certified-layout-v1"
    )


def test_default_gpu_profile_does_not_load_mnnvl_or_native_dependencies(monkeypatch):
    loader = MagicMock(
        side_effect=AssertionError("Default profile must not load MNNVL")
    )
    monkeypatch.setattr(kv_hints, "importlib", SimpleNamespace(import_module=loader))
    runtime = MagicMock()
    publish_kv_hint_capabilities(
        runtime,
        _wrapped_engine_args(
            "PdConnector",
            [_gpu_connector()],
            router_hint_transfer_domain="gpu-certified-layout-v1",
        ),
        WorkerType.Decode,
    )
    loader.assert_not_called()
    runtime.set_engine_specific.assert_any_call(
        "router_hint_transfer_domain", '"gpu-certified-layout-v1"'
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"gpu_transport": "unknown"},
        {"gpu_transport": "mnnvl"},
        {"gpu_transport": "mnnvl", "ucx_tls": True},
        {"gpu_transport": "mnnvl", "ucx_tls": ["cuda_ipc", "posix", "self"]},
    ],
)
def test_mnnvl_capability_rejects_unknown_or_nonexplicit_profile_before_native_import(
    monkeypatch, changes
):
    loader = MagicMock(side_effect=AssertionError("Invalid policy must fail first"))
    monkeypatch.setattr(kv_hints, "importlib", SimpleNamespace(import_module=loader))
    runtime = MagicMock()
    with pytest.raises(ValueError):
        publish_kv_hint_capabilities(
            runtime,
            _wrapped_engine_args(
                "PdConnector",
                [_gpu_connector(**changes)],
                router_hint_transfer_domain="gpu-certified-layout-v1",
            ),
            WorkerType.Decode,
        )
    loader.assert_not_called()
    runtime.set_engine_specific.assert_not_called()


def test_mnnvl_and_cpu_hint_providers_cannot_partially_publish_a_shared_domain():
    runtime = MagicMock()
    with pytest.raises(ValueError, match="CPU-hint"):
        publish_kv_hint_capabilities(
            runtime,
            _wrapped_engine_args(
                "PdConnector",
                [
                    _gpu_connector(
                        gpu_transport="mnnvl", ucx_tls=["cuda_ipc", "tcp", "self"]
                    ),
                    _kvcr_connector(
                        control_ports=_gpu_connector()["kv_connector_extra_config"][
                            "secondary_tiers"
                        ][0]["control_ports"]
                    ),
                ],
                router_hint_transfer_domain="gpu-certified-layout-v1",
            ),
            WorkerType.Decode,
        )
    runtime.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"tensor_parallel_size": 4},
        {"data_parallel_size": 2},
        {"distributed_executor_backend": "ray"},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_gpu_tp2_discovery_rejects_unsupported_parallel_topologies(changes):
    args = _wrapped_engine_args(
        "PdConnector",
        [_gpu_connector()],
        router_hint_transfer_domain="gpu-certified-layout-v1",
    )
    args.tensor_parallel_size = 2
    for field, value in changes.items():
        setattr(args, field, value)
    runtime = MagicMock()
    with pytest.raises(ValueError, match="TP1 or TP2|single local MP"):
        publish_kv_hint_capabilities(runtime, args, WorkerType.Decode)
    runtime.set_engine_specific.assert_not_called()
