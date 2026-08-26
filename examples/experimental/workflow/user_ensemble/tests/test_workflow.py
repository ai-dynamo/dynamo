# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from examples.experimental.workflow.user_ensemble.common.stages import (
    EnsembleResponseStage,
)
from examples.experimental.workflow.user_ensemble.workflow.workflow import (
    define_workflow,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_user_ensemble_authors_request_only_generator() -> None:
    workflow_ir = define_workflow().build()
    stages = {stage.id: stage for stage in workflow_ir.stages}

    assert set(stages) == {
        "encoder",
        "classifier",
        "request_adapter",
        "generator",
        "response",
    }
    assert stages["generator"].contract.inputs == {"request"}
    assert stages["generator"].contract.outputs == {"completion"}


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
