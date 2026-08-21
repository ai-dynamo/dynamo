# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.vllm.workflow.components import DynamoVllmStage
from dynamo.workflow import GenerateEndpointBinding
from dynamo.workflow.plan import validate_binding_contract

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
]


def test_dynamo_vllm_stage_matches_generate_endpoint_contract() -> None:
    validate_binding_contract(
        GenerateEndpointBinding("workflows.generator.generate"),
        DynamoVllmStage.request_complete_contract,
    )

    assert set(DynamoVllmStage.request_complete_contract.inputs) == {"request"}
    assert set(DynamoVllmStage.request_complete_contract.outputs) == {"completion"}
