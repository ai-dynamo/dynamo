# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the inference-gateway (GAIE/EPP) producer.

Covers ``add_inference_gateway_to_config`` and its helpers: the agg-vs-disagg
EndpointPickerConfig, the RoutingProfile -> KV-router env mapping, the
frontend-sidecar conversion (+ standalone Frontend removal), and the
``nvidia.com/inference-gateway-name`` annotation handoff that makes the DGD
controller emit the HTTPRoute.
"""

import pytest

try:
    from dynamo.profiler.utils.dgd_generation import (
        INFERENCE_GATEWAY_NAME_ANNOTATION,
        add_inference_gateway_to_config,
    )
    from dynamo.profiler.utils.dgdr_v1beta1_types import (
        DynamoGraphDeploymentRequestSpec,
        FeaturesSpec,
        InferenceGatewayFeature,
        RoutingProfile,
    )
    from dynamo.profiler.utils.profile_common import (
        derive_epp_image,
        is_inference_gateway_enabled,
    )
except ImportError as e:
    pytest.skip(f"Missing dependency: {e}", allow_module_level=True)


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

_PLANNER_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3"


def _dgdr(
    routing_profile: RoutingProfile = RoutingProfile.Balanced,
    gateway_name: str | None = "my-gateway",
    image: str | None = _PLANNER_IMAGE,
) -> DynamoGraphDeploymentRequestSpec:
    igw = InferenceGatewayFeature(
        enabled=True,
        routingProfile=routing_profile,
        gatewayName=gateway_name,
    )
    return DynamoGraphDeploymentRequestSpec(
        model="Qwen/Qwen3-0.6B",
        image=image,
        features=FeaturesSpec(inferenceGateway=igw),
    )


def _worker(sub_type: str, block_size: int) -> dict:
    return {
        "componentType": "worker",
        "subComponentType": sub_type,
        "envFromSecret": "hf-token-secret",
        "extraPodSpec": {
            "mainContainer": {
                "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag",
                "args": [f"python3 -m dynamo.vllm --model X --block-size {block_size}"],
            }
        },
    }


def _frontend() -> dict:
    return {
        "componentType": "frontend",
        "replicas": 1,
        "extraPodSpec": {
            "mainContainer": {"image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag"}
        },
    }


def _agg_config() -> dict:
    return {
        "apiVersion": "nvidia.com/v1alpha1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": "qwen"},
        "spec": {
            "services": {
                "Frontend": _frontend(),
                "VllmDecodeWorker": _worker("decode", 128),
            }
        },
    }


def _disagg_config() -> dict:
    cfg = _agg_config()
    cfg["spec"]["services"]["VllmDecodeWorker"] = _worker("decode", 16)
    cfg["spec"]["services"]["VllmPrefillWorker"] = _worker("prefill", 16)
    return cfg


def _epp(cfg: dict) -> dict:
    return cfg["spec"]["services"]["Epp"]


def _epp_env(cfg: dict) -> dict:
    return {
        e["name"]: e["value"] for e in _epp(cfg)["extraPodSpec"]["mainContainer"]["env"]
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_is_inference_gateway_enabled():
    assert is_inference_gateway_enabled(_dgdr()) is True
    spec = DynamoGraphDeploymentRequestSpec(model="m")
    assert is_inference_gateway_enabled(spec) is False
    spec = DynamoGraphDeploymentRequestSpec(
        model="m",
        features=FeaturesSpec(inferenceGateway=InferenceGatewayFeature(enabled=False)),
    )
    assert is_inference_gateway_enabled(spec) is False


def test_derive_epp_image_preserves_registry_and_tag():
    assert (
        derive_epp_image("nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.3")
        == "nvcr.io/nvidia/ai-dynamo/epp-image:1.2.3"
    )


# ---------------------------------------------------------------------------
# aggregated
# ---------------------------------------------------------------------------


def test_agg_injects_epp_and_drops_standalone_frontend():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(), cfg)
    services = cfg["spec"]["services"]

    assert "Frontend" not in services, "standalone Frontend should be removed"
    epp = _epp(cfg)
    assert epp["componentType"] == "epp"
    assert epp["replicas"] == 1
    assert (
        epp["extraPodSpec"]["mainContainer"]["image"]
        == "nvcr.io/nvidia/ai-dynamo/epp-image:1.2.3"
    )
    # reuses the worker's pull secret
    assert epp["envFromSecret"] == "hf-token-secret"


def test_agg_endpoint_picker_config_is_decode_only():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(), cfg)
    epp_config = _epp(cfg)["eppConfig"]["config"]

    profiles = [p["name"] for p in epp_config["schedulingProfiles"]]
    assert profiles == ["decode"]

    plugin_names = {p.get("name") for p in epp_config["plugins"]}
    assert "dyn-decode" in plugin_names
    assert "dyn-prefill" not in plugin_names
    assert "prefill-filter" not in plugin_names

    decode_filter = next(
        p for p in epp_config["plugins"] if p.get("name") == "decode-filter"
    )
    # agg has no prefill profile, so unlabeled pods are allowed through
    assert decode_filter["parameters"]["allowsNoLabel"] is True


def test_agg_epp_env_and_block_size_from_worker_args():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(), cfg)
    env = _epp_env(cfg)

    assert env["DYN_MODEL_NAME"] == "Qwen/Qwen3-0.6B"
    assert env["DYN_ENFORCE_DISAGG"] == "false"
    # extracted from the worker's --block-size 128 (not the disagg default)
    assert env["DYN_KV_CACHE_BLOCK_SIZE"] == "128"


def test_workers_get_direct_frontend_sidecar():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(), cfg)
    sidecar = cfg["spec"]["services"]["VllmDecodeWorker"]["frontendSidecar"]

    assert sidecar["args"] == ["-m", "dynamo.frontend", "--router-mode", "direct"]
    assert sidecar["image"] == "nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag"
    assert sidecar["envFromSecret"] == "hf-token-secret"


def test_gateway_name_annotation_set():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(gateway_name="prod-gw"), cfg)
    assert (
        cfg["metadata"]["annotations"][INFERENCE_GATEWAY_NAME_ANNOTATION] == "prod-gw"
    )


def test_no_gateway_name_emits_no_annotation():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(gateway_name=None), cfg)
    # EPP still injected (InferencePool will be created) but no HTTPRoute handoff
    assert "Epp" in cfg["spec"]["services"]
    assert "annotations" not in cfg.get("metadata", {})


# ---------------------------------------------------------------------------
# disaggregated
# ---------------------------------------------------------------------------


def test_disagg_endpoint_picker_config_has_prefill_and_decode():
    cfg = _disagg_config()
    add_inference_gateway_to_config(_dgdr(), cfg)
    epp_config = _epp(cfg)["eppConfig"]["config"]

    profiles = [p["name"] for p in epp_config["schedulingProfiles"]]
    assert profiles == ["prefill", "decode"]

    plugin_names = {p.get("name") for p in epp_config["plugins"]}
    assert {
        "dyn-prefill",
        "dyn-decode",
        "prefill-filter",
        "decode-filter",
    } <= plugin_names

    decode_filter = next(
        p for p in epp_config["plugins"] if p.get("name") == "decode-filter"
    )
    # disagg has an explicit prefill profile, so filters are strict
    assert decode_filter["parameters"]["allowsNoLabel"] is False


def test_disagg_env_and_all_workers_get_sidecar():
    cfg = _disagg_config()
    add_inference_gateway_to_config(_dgdr(), cfg)
    env = _epp_env(cfg)

    assert env["DYN_ENFORCE_DISAGG"] == "true"
    assert env["DYN_KV_CACHE_BLOCK_SIZE"] == "16"

    services = cfg["spec"]["services"]
    assert "frontendSidecar" in services["VllmDecodeWorker"]
    assert "frontendSidecar" in services["VllmPrefillWorker"]


# ---------------------------------------------------------------------------
# routing profile -> router knobs
# ---------------------------------------------------------------------------


def test_routing_profile_throughput_packs_on_cache():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(RoutingProfile.Throughput), cfg)
    env = _epp_env(cfg)
    assert env["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"] == "1.0"
    assert env["DYN_ROUTER_PREFILL_LOAD_SCALE"] == "0.5"


def test_routing_profile_latency_spreads_load():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(RoutingProfile.Latency), cfg)
    env = _epp_env(cfg)
    assert env["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"] == "0.5"
    assert env["DYN_ROUTER_PREFILL_LOAD_SCALE"] == "2.0"


def test_routing_profile_balanced_uses_router_defaults():
    cfg = _agg_config()
    add_inference_gateway_to_config(_dgdr(RoutingProfile.Balanced), cfg)
    env = _epp_env(cfg)
    # balanced leaves the router at its defaults -> no override env emitted
    assert "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT" not in env
    assert "DYN_ROUTER_PREFILL_LOAD_SCALE" not in env
