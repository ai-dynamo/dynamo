# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

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

from dynamo.workflow import ValueRef  # noqa: E402
from examples.custom_backend.user_ensemble.stages import (  # noqa: E402
    DummyClassifier,
    EncoderStage,
)
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


def _context() -> SimpleNamespace:
    return SimpleNamespace(raise_if_cancelled=lambda: None)


def test_workflow_declares_encoder_fanout_to_classifier_and_stock_generator():
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
    encoder = workflow.stages[0]
    classifier = workflow.stages[1]
    generator = workflow.stages[2]
    assert classifier.inputs["encoder_features"] == ValueRef.for_stage_output(
        encoder.id, "encoder_features"
    )
    assert generator.inputs["encoder_features"] == ValueRef.for_stage_output(
        encoder.id, "encoder_features"
    )
    assert generator.inputs["encoder_metadata"] == ValueRef.for_stage_output(
        encoder.id, "encoder_metadata"
    )


async def test_encoder_stage_packs_dynamic_image_rows_and_metadata():
    first = torch.ones((2, 4), dtype=torch.bfloat16)
    second = torch.full((3, 4), 2, dtype=torch.bfloat16)
    encoder = SimpleNamespace(encode=AsyncMock(return_value=[first, second]))
    stage = EncoderStage(encoder, image_token_id=99)
    request = {
        "multi_modal_data": {
            "image_url": [
                {"Url": "data:image/png;base64,first"},
                {"Url": "data:image/png;base64,second"},
            ]
        }
    }

    result = await stage.run({"request": request}, _context())

    assert result["encoder_metadata"] == {
        "row_splits": [0, 2, 5],
        "image_token_id": 99,
    }
    expected = torch.cat((first, second), dim=0)
    torch.testing.assert_close(result["encoder_features"], expected)
    encoder.encode.assert_awaited_once_with(
        [
            "data:image/png;base64,first",
            "data:image/png;base64,second",
        ]
    )


async def test_classifier_consumes_the_packed_tensor():
    result = await DummyClassifier().run(
        {"encoder_features": torch.tensor([[0.0], [1.0]])},
        _context(),
    )

    assert set(result["scores"]) == {"positive-mean", "negative-mean"}
    assert sum(result["scores"].values()) == pytest.approx(1.0)


def test_result_adapter_preserves_chunk_and_attaches_classifier_scores():
    chunk = {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "stock-vllm"}},
    }

    adapted = adapt_workflow_result({"chunk": chunk, "scores": {"positive-mean": 0.9}})

    assert chunk == {
        "token_ids": [4, 2],
        "engine_data": {"ensemble": {"decoder": "stock-vllm"}},
    }
    assert adapted["engine_data"]["ensemble"] == {
        "decoder": "stock-vllm",
        "classifier_scores": {"positive-mean": 0.9},
    }
