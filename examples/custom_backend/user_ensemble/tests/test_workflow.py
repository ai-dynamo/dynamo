# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the workflow example",
)

from dynamo.workflow import ValueRef  # noqa: E402
from examples.custom_backend.user_ensemble.workflow import (  # noqa: E402
    adapt_workflow_result,
    define_workflow,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def test_workflow_is_the_readable_pipeline_and_uses_declared_ports() -> None:
    workflow = define_workflow().build()

    assert [stage.id for stage in workflow.stages] == [
        "encoder",
        "classifier",
        "generator",
    ]
    assert workflow.outputs["scores"] == ValueRef.for_stage_output(
        "classifier", "scores"
    )
    assert workflow.outputs["chunk"] == ValueRef.for_stage_output("generator", "chunk")


def test_shared_result_adapter_does_not_mutate_decoder_chunk() -> None:
    chunk = {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "vllm"}},
    }

    adapted = adapt_workflow_result({"chunk": chunk, "scores": {"category-a": 0.9}})

    assert chunk == {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "vllm"}},
    }
    assert adapted["engine_data"]["ensemble"] == {
        "decoder": "vllm",
        "classifier_scores": {"category-a": 0.9},
    }
