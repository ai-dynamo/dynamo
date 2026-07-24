# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DGD-owned per-GPU power-limit parsing (Phase 1).

Covers ``Service.get_gpu_power_limit_watts`` and
``resolve_component_power_configs`` — the single authoritative way to read a
component's per-GPU cap and per-replica watts from a DGD dict. No planner loop,
no Kubernetes, no Pod mutation.
"""

import pytest

from dynamo.planner.errors import (
    DuplicateSubComponentError,
    PowerAnnotationInvalidError,
    PowerAnnotationMissingError,
    SubComponentNotFoundError,
)
from dynamo.planner.monitoring.dgd_services import (
    POWER_ANNOTATION_KEY,
    ComponentPowerConfig,
    Service,
    resolve_component_power_configs,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _worker(
    name,
    *,
    comp_type=None,
    watts="350",
    gpus="2",
    node_count=None,
):
    """Build one v1beta1 DGD worker component dict.

    ``watts=None`` omits the annotation entirely; a string value (including
    "" / "abc" / "0" / "-1") is written verbatim so malformed cases can be
    exercised.
    """
    component: dict = {
        "name": name,
        "podTemplate": {
            "spec": {
                "containers": [
                    {"name": "main", "resources": {"limits": {"nvidia.com/gpu": gpus}}}
                ]
            }
        },
    }
    if comp_type is not None:
        component["type"] = comp_type
    if watts is not None:
        component["podTemplate"]["metadata"] = {
            "annotations": {POWER_ANNOTATION_KEY: watts}
        }
    if node_count is not None:
        component["multinode"] = {"nodeCount": node_count}
    return component


def _dgd(*components):
    return {"spec": {"components": list(components)}}


# --------------------------------------------------------------------------- #
# Service.get_gpu_power_limit_watts
# --------------------------------------------------------------------------- #


def test_reads_positive_integer_watts():
    svc = Service(name="P", service=_worker("P", comp_type="prefill", watts="350"))
    assert svc.get_gpu_power_limit_watts() == 350


def test_missing_annotation_raises_missing():
    svc = Service(name="P", service=_worker("P", comp_type="prefill", watts=None))
    with pytest.raises(PowerAnnotationMissingError):
        svc.get_gpu_power_limit_watts()


@pytest.mark.parametrize("bad", ["", "   ", "abc", "3.5", "0", "-1", "-300"])
def test_malformed_or_nonpositive_raises_invalid(bad):
    svc = Service(name="P", service=_worker("P", comp_type="prefill", watts=bad))
    with pytest.raises(PowerAnnotationInvalidError):
        svc.get_gpu_power_limit_watts()


def test_surrounding_whitespace_is_tolerated():
    svc = Service(name="P", service=_worker("P", comp_type="prefill", watts=" 300 "))
    assert svc.get_gpu_power_limit_watts() == 300


# --------------------------------------------------------------------------- #
# ComponentPowerConfig.watts_per_replica
# --------------------------------------------------------------------------- #


def test_watts_per_replica_multiplies_gpus():
    cfg = ComponentPowerConfig(
        component_name="P",
        role="prefill",
        gpu_power_limit_watts=350,
        gpus_per_replica=2,
    )
    assert cfg.watts_per_replica == 700


# --------------------------------------------------------------------------- #
# resolve_component_power_configs — disagg
# --------------------------------------------------------------------------- #


def test_resolves_disagg_prefill_and_decode():
    dgd = _dgd(
        _worker("VllmPrefillWorker", comp_type="prefill", watts="350", gpus="2"),
        _worker("VllmDecodeWorker", comp_type="decode", watts="300", gpus="4"),
    )
    prefill, decode = resolve_component_power_configs(
        dgd, require_prefill=True, require_decode=True
    )
    assert prefill == ComponentPowerConfig(
        component_name="VllmPrefillWorker",
        role="prefill",
        gpu_power_limit_watts=350,
        gpus_per_replica=2,
    )
    assert decode == ComponentPowerConfig(
        component_name="VllmDecodeWorker",
        role="decode",
        gpu_power_limit_watts=300,
        gpus_per_replica=4,
    )
    assert prefill.watts_per_replica == 700
    assert decode.watts_per_replica == 1200


def test_asymmetric_caps_are_independent():
    dgd = _dgd(
        _worker("P", comp_type="prefill", watts="350", gpus="1"),
        _worker("D", comp_type="decode", watts="250", gpus="1"),
    )
    prefill, decode = resolve_component_power_configs(
        dgd, require_prefill=True, require_decode=True
    )
    assert prefill.watts_per_replica == 350
    assert decode.watts_per_replica == 250


# --------------------------------------------------------------------------- #
# resolve_component_power_configs — aggregate
# --------------------------------------------------------------------------- #


def test_agg_generic_worker_resolves_decode_only():
    dgd = _dgd(_worker("VllmWorker", comp_type="worker", watts="300", gpus="4"))
    prefill, decode = resolve_component_power_configs(
        dgd, require_prefill=False, require_decode=True
    )
    assert prefill is None
    assert decode is not None
    assert decode.component_name == "VllmWorker"
    # role reflects the DGD component's actual generic type.
    assert decode.role == "worker"
    assert decode.watts_per_replica == 1200


def test_agg_does_not_manufacture_prefill_config():
    dgd = _dgd(_worker("VllmWorker", comp_type="worker", watts="300", gpus="4"))
    prefill, decode = resolve_component_power_configs(
        dgd, require_prefill=False, require_decode=True
    )
    assert prefill is None


# --------------------------------------------------------------------------- #
# multinode
# --------------------------------------------------------------------------- #


def test_multinode_watts_per_replica_multiplies_node_count():
    # 2 GPUs/pod × nodeCount 3 × 350 W = 2100 W per replica.
    dgd = _dgd(
        _worker("P", comp_type="prefill", watts="350", gpus="2", node_count=3),
        _worker("D", comp_type="decode", watts="300", gpus="1"),
    )
    prefill, _ = resolve_component_power_configs(
        dgd, require_prefill=True, require_decode=True
    )
    assert prefill.gpus_per_replica == 6
    assert prefill.watts_per_replica == 2100


# --------------------------------------------------------------------------- #
# explicit-name vs type resolution
# --------------------------------------------------------------------------- #


def test_explicit_name_resolution_when_type_absent():
    # Component carries no type; the explicit fallback name resolves it.
    dgd = _dgd(_worker("CustomDecode", comp_type=None, watts="300", gpus="2"))
    _, decode = resolve_component_power_configs(
        dgd,
        require_prefill=False,
        require_decode=True,
        decode_name="CustomDecode",
    )
    assert decode is not None
    assert decode.component_name == "CustomDecode"
    assert decode.watts_per_replica == 600


# --------------------------------------------------------------------------- #
# errors surface from the shared resolver
# --------------------------------------------------------------------------- #


def test_missing_role_raises_not_found():
    dgd = _dgd(_worker("D", comp_type="decode", watts="300", gpus="1"))
    with pytest.raises(SubComponentNotFoundError):
        resolve_component_power_configs(dgd, require_prefill=True, require_decode=True)


def test_duplicate_role_raises_duplicate():
    dgd = _dgd(
        _worker("P1", comp_type="prefill", watts="350", gpus="1"),
        _worker("P2", comp_type="prefill", watts="350", gpus="1"),
        _worker("D", comp_type="decode", watts="300", gpus="1"),
    )
    with pytest.raises(DuplicateSubComponentError):
        resolve_component_power_configs(dgd, require_prefill=True, require_decode=True)


def test_missing_annotation_on_required_role_raises():
    dgd = _dgd(
        _worker("P", comp_type="prefill", watts=None, gpus="1"),
        _worker("D", comp_type="decode", watts="300", gpus="1"),
    )
    with pytest.raises(PowerAnnotationMissingError):
        resolve_component_power_configs(dgd, require_prefill=True, require_decode=True)


def test_invalid_gpu_count_raises_value_error():
    dgd = _dgd(_worker("D", comp_type="decode", watts="300", gpus="notanint"))
    with pytest.raises(ValueError):
        resolve_component_power_configs(dgd, require_prefill=False, require_decode=True)


# --------------------------------------------------------------------------- #
# zero / negative GPU topology — must fail loudly, not zero out enforcement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_gpus", ["0", "-1", "-4"])
def test_zero_or_negative_gpu_count_rejected(bad_gpus):
    svc = Service(name="D", service=_worker("D", comp_type="decode", gpus=bad_gpus))
    with pytest.raises(ValueError):
        svc.get_gpu_count()


@pytest.mark.parametrize("bad_nodes", [0, -1, -3])
def test_zero_or_negative_node_count_rejected(bad_nodes):
    svc = Service(
        name="D", service=_worker("D", comp_type="decode", node_count=bad_nodes)
    )
    with pytest.raises(ValueError):
        svc.get_node_count()
    with pytest.raises(ValueError):
        svc.get_total_gpu_count()


@pytest.mark.parametrize("bad_gpus", ["0", "-2"])
def test_zero_or_negative_gpu_count_fails_resolution(bad_gpus):
    # A zero/negative topology must not silently pass through the resolver and
    # produce a watts_per_replica of 0 that disables enforcement for the role.
    dgd = _dgd(_worker("D", comp_type="decode", watts="300", gpus=bad_gpus))
    with pytest.raises(ValueError):
        resolve_component_power_configs(dgd, require_prefill=False, require_decode=True)
