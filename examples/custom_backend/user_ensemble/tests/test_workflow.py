# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="the vLLM workflow components require PyTorch")

from dynamo.workflow import ValueRef  # noqa: E402
from examples.custom_backend.user_ensemble.stages import (  # noqa: E402
    DummyClassifier,
    EnsembleResponseStage,
)
from examples.custom_backend.user_ensemble.workflow import define_workflow  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _context() -> SimpleNamespace:
    return SimpleNamespace(raise_if_cancelled=lambda: None)


def test_workflow_fans_request_to_classifier_and_stock_generator() -> None:
    workflow = define_workflow().build()

    assert [stage.id for stage in workflow.stages] == [
        "classifier",
        "generator",
        "response",
    ]
    classifier, generator, response = workflow.stages
    request = ValueRef.for_input("request")
    assert classifier.inputs == {"request": request}
    assert generator.inputs == {"request": request}
    assert response.inputs["completion"] == ValueRef.for_stage_output(
        generator.id, "completion"
    )
    assert response.inputs["scores"] == ValueRef.for_stage_output(
        classifier.id, "scores"
    )
    assert workflow.outputs["chunk"] == ValueRef.for_stage_output(response.id, "chunk")


async def test_classifier_accepts_original_request() -> None:
    request = {"multi_modal_data": {"image_url": [{"Url": "image"}]}}

    result = await DummyClassifier().run({"request": request}, _context())

    assert sum(result["scores"].values()) == pytest.approx(1.0)


async def test_inline_response_preserves_completion_and_attaches_scores() -> None:
    completion = {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "stock-vllm"}},
    }

    result = await EnsembleResponseStage().run(
        {"completion": completion, "scores": {"relevant": 0.75}},
        _context(),
    )

    assert completion == {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "stock-vllm"}},
    }
    assert result["chunk"]["engine_data"]["ensemble"] == {
        "decoder": "stock-vllm",
        "classifier_scores": {"relevant": 0.75},
    }
