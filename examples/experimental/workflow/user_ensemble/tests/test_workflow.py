# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="the external encoder example requires PyTorch")
pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the vision example",
)

import torch  # noqa: E402

from dynamo.experimental.workflow import StageContext, ValueRef  # noqa: E402
from examples.experimental.workflow.user_ensemble.stages import (  # noqa: E402
    DummyClassifier,
    DummyMetadataClassifier,
    EnsembleResponseStage,
)
from examples.experimental.workflow.user_ensemble.workflow import define_workflow  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _context() -> StageContext:
    return StageContext(
        workflow_name="user-ensemble-test",
        stage_id="test",
        attempt_id="request-1",
    )


def test_workflow_declares_encoder_fanout_to_classifier_and_stock_generator():
    workflow = define_workflow().build()

    assert [stage.id for stage in workflow.stages] == [
        "encoder",
        "classifier",
        "generator",
        "response",
    ]
    assert set(workflow.outputs) == {"chunk"}
    assert workflow.outputs["chunk"] == ValueRef.for_stage_output("response", "chunk")
    encoder = workflow.stages[0]
    classifier = workflow.stages[1]
    generator = workflow.stages[2]
    response = workflow.stages[3]
    assert classifier.inputs["encoder_features"] == ValueRef.for_stage_output(
        encoder.id, "encoder_features"
    )
    assert generator.inputs["encoder_features"] == ValueRef.for_stage_output(
        encoder.id, "encoder_features"
    )
    assert generator.inputs["encoder_metadata"] == ValueRef.for_stage_output(
        encoder.id, "encoder_metadata"
    )
    assert response.inputs["completion"] == ValueRef.for_stage_output(
        generator.id, "completion"
    )
    assert response.inputs["scores"] == ValueRef.for_stage_output(
        classifier.id, "scores"
    )


async def test_classifier_consumes_the_packed_tensor():
    result = await DummyClassifier().run(
        {"encoder_features": torch.tensor([[0.0], [1.0]])},
        _context(),
    )

    assert set(result["scores"]) == {"positive-sample", "negative-sample"}
    assert sum(result["scores"].values()) == pytest.approx(1.0)


async def test_metadata_classifier_does_not_require_the_packed_tensor():
    result = await DummyMetadataClassifier().run(
        {"encoder_metadata": {"row_splits": [0, 1]}},
        _context(),
    )

    assert result == {"scores": {"metadata-control": 1.0}}


async def test_inline_response_stage_preserves_completion_and_attaches_scores():
    completion = {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "stock-vllm"}},
    }

    result = await EnsembleResponseStage().run(
        {"completion": completion, "scores": {"positive-mean": 0.9}},
        _context(),
    )

    assert completion == {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "stock-vllm"}},
    }
    assert result["chunk"]["engine_data"]["ensemble"] == {
        "decoder": "stock-vllm",
        "classifier_scores": {"positive-mean": 0.9},
    }
