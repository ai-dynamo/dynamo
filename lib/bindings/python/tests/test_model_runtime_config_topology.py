# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import ModelRuntimeConfig


@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
@pytest.mark.pre_merge
def test_model_runtime_config_topology_fields_round_trip():
    runtime_config = ModelRuntimeConfig()

    runtime_config.taints = {"user.taint/example"}
    runtime_config.topology_domains = {"zone": "us-east-1a"}
    runtime_config.kv_transfer_domain = "zone"
    runtime_config.kv_transfer_enforcement = "preferred"
    runtime_config.kv_transfer_preferred_weight = 0.85

    assert runtime_config.taints == {"user.taint/example"}
    assert runtime_config.topology_domains == {"zone": "us-east-1a"}
    assert runtime_config.kv_transfer_domain == "zone"
    assert runtime_config.kv_transfer_enforcement == "preferred"
    assert runtime_config.kv_transfer_preferred_weight == pytest.approx(0.85)

    runtime_config.kv_transfer_enforcement = None
    runtime_config.kv_transfer_preferred_weight = None

    assert runtime_config.kv_transfer_enforcement is None
    assert runtime_config.kv_transfer_preferred_weight is None


@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
@pytest.mark.pre_merge
def test_model_runtime_config_rejects_invalid_topology_policy_values():
    runtime_config = ModelRuntimeConfig()

    with pytest.raises(ValueError, match="kv_transfer_enforcement"):
        runtime_config.kv_transfer_enforcement = "fallback"


@pytest.mark.unit
@pytest.mark.none
@pytest.mark.gpu_0
@pytest.mark.pre_merge
def test_model_runtime_config_topology_domains_setter_is_atomic():
    runtime_config = ModelRuntimeConfig()
    runtime_config.topology_domains = {"zone": "us-east-1a"}

    with pytest.raises(TypeError):
        runtime_config.topology_domains = {"rack": "rack-22", 1: "us-west-2a"}

    assert runtime_config.topology_domains == {"zone": "us-east-1a"}
