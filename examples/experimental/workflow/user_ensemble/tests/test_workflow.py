# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.experimental.workflow import GenerateEndpointBinding, InlineBinding
from examples.experimental.workflow.user_ensemble.common.stages import (
    EnsembleResponseStage,
)
from examples.experimental.workflow.user_ensemble.workflow.workflow import (
    compile_user_ensemble,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_user_ensemble_compiles_mixed_placement() -> None:
    plan = compile_user_ensemble()

    assert set(plan.bindings) == {
        "encoder",
        "classifier",
        "request_adapter",
        "generator",
        "response",
    }
    assert isinstance(plan.bindings["generator"], GenerateEndpointBinding)
    assert all(
        isinstance(plan.bindings[stage_id], InlineBinding)
        for stage_id in ("encoder", "classifier", "request_adapter", "response")
    )
    assert plan.stage_contracts["generator"].inputs == {"request"}
    assert plan.stage_contracts["generator"].outputs == {"completion"}


async def test_response_stage_preserves_completion_and_adds_scores() -> None:
    result = await EnsembleResponseStage().run(
        {
            "completion": {
                "token_ids": [4, 2],
                "index": 0,
                "finish_reason": "stop",
                "engine_data": {"decoder": "vllm"},
            },
            "scores": {"dummy-positive": 1.0},
        },
        SimpleNamespace(),
    )

    assert result["chunk"]["token_ids"] == [4, 2]
    assert result["chunk"]["engine_data"] == {
        "decoder": "vllm",
        "user_ensemble": {"classifier_scores": {"dummy-positive": 1.0}},
    }
