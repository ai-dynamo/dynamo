# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    decode_request_plane_tensor,
)
from dynamo.experimental.workflow import StageContext
from dynamo.experimental.workflow.vllm import (
    DynamoVllmStage,
    EncoderStage,
    ExternalEncoderRequestStage,
)
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder import VisionEncoderBackend
from dynamo.vllm.multimodal_utils.custom_encoder.backend import loader

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _Backend(VisionEncoderBackend):
    image_token_id = 99

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def _context(stage_id: str = "encoder") -> StageContext:
    return StageContext(
        workflow_name="encoder-test",
        stage_id=stage_id,
        attempt_id="request-1",
    )


def test_dynamo_vllm_stage_publishes_request_complete_contract() -> None:
    assert DynamoVllmStage.request_complete_contract.inputs == {"request"}
    assert DynamoVllmStage.request_complete_contract.outputs == {"completion"}


async def test_encoder_stage_packs_dynamic_image_rows_and_metadata() -> None:
    first = torch.ones((2, 4), dtype=torch.bfloat16)
    second = torch.full((3, 4), 2, dtype=torch.bfloat16)
    encoder = SimpleNamespace(
        encode=AsyncMock(return_value=[first, second]),
        shutdown=Mock(),
    )
    stage = EncoderStage(encoder, image_token_id=99)

    result = await stage.run(
        {
            "request": {
                "multi_modal_data": {
                    "image_url": [
                        {"Url": "data:image/png;base64,first"},
                        {"Url": "data:image/png;base64,second"},
                    ]
                }
            }
        },
        _context(),
    )

    assert result["encoder_metadata"] == {
        "row_splits": [0, 2, 5],
        "image_token_id": 99,
    }
    torch.testing.assert_close(
        result["encoder_features"],
        torch.cat((first, second), dim=0),
    )
    encoder.encode.assert_awaited_once_with(
        ["data:image/png;base64,first", "data:image/png;base64,second"]
    )


async def test_external_encoder_request_stage_builds_standard_request() -> None:
    original = {
        "token_ids": [1, 99, 2],
        "multi_modal_data": {"image_url": [{"Url": "unused"}]},
        "mm_processor_kwargs": {"max_pixels": 1},
        "extra_args": {"trace": "keep", "mm_kwargs_nixl": {"drop": True}},
    }
    features = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

    output = await ExternalEncoderRequestStage().run(
        {
            "request": original,
            "encoder_features": features,
            "encoder_metadata": {
                "row_splits": [0, 2],
                "image_token_id": 99,
            },
        },
        _context("prepare-request"),
    )

    request = output["request"]
    assert "encoder_result" not in original
    assert "multi_modal_data" not in request
    assert "mm_processor_kwargs" not in request
    assert request["extra_args"] == {"trace": "keep"}
    parsed = ExternalEncoderResult.from_dict(request["encoder_result"])
    torch.testing.assert_close(
        decode_request_plane_tensor(parsed.features),
        features,
    )


@pytest.mark.parametrize(
    "inputs,message",
    [
        (
            {
                "request": {"encoder_result": {}},
                "encoder_features": torch.ones((1, 2)),
                "encoder_metadata": {"row_splits": [0, 1], "image_token_id": 1},
            },
            "already contains",
        ),
        (
            {
                "request": {"prompt_embeds": "encoded"},
                "encoder_features": torch.ones((1, 2)),
                "encoder_metadata": {"row_splits": [0, 1], "image_token_id": 1},
            },
            "prompt_embeds",
        ),
        (
            {
                "request": {},
                "encoder_features": torch.ones((2, 3)).t(),
                "encoder_metadata": {"row_splits": [0, 3], "image_token_id": 1},
            },
            "contiguous",
        ),
    ],
)
async def test_external_encoder_request_stage_rejects_invalid_inputs(
    inputs: dict,
    message: str,
) -> None:
    with pytest.raises(InvalidArgument, match=message):
        await ExternalEncoderRequestStage().run(inputs, _context("prepare-request"))


@pytest.mark.parametrize(
    "artifacts,message",
    [
        ([], "no image artifacts"),
        ([torch.ones(2)], "must be a 2D"),
        ([torch.empty((1, 0))], "non-zero hidden size"),
        ([torch.ones((1, 2)), torch.ones((1, 3))], "hidden size 2"),
        (
            [torch.ones((1, 2)), torch.ones((1, 2), dtype=torch.float16)],
            "expected torch.float32",
        ),
        ([torch.ones((1, 2), device="meta")], "workflow transfer requires CPU"),
    ],
)
def test_encoder_stage_rejects_invalid_artifacts(
    artifacts: list,
    message: str,
) -> None:
    with pytest.raises(InvalidArgument, match=message):
        EncoderStage._validate_artifacts(artifacts)


@pytest.mark.parametrize(
    "request_value,message",
    [
        ({}, "requires at least one image"),
        (
            {"multi_modal_data": {"audio_url": [{"Url": "audio"}]}},
            "supports image inputs only",
        ),
        (
            {"multi_modal_data": {"image_url": [{}]}},
            "non-empty 'Url' string",
        ),
    ],
)
def test_encoder_stage_rejects_invalid_media(
    request_value: dict,
    message: str,
) -> None:
    with pytest.raises(InvalidArgument, match=message):
        EncoderStage._image_urls(request_value)


def test_encoder_stage_loads_backend_and_closes_once() -> None:
    driver = Mock()
    backend = _Backend()
    with patch(
        "dynamo.experimental.workflow.vllm.stages.AsyncVisionEncoder",
        return_value=driver,
    ) as encoder_type:
        stage = EncoderStage.from_backend(backend, model="org/model", name="encoder")

    encoder_type.assert_called_once_with(backend, name="encoder")
    driver.load.assert_called_once_with("org/model")
    stage.close()
    stage.close()
    driver.shutdown.assert_called_once_with()


def test_encoder_stage_cleans_up_failed_backend_load() -> None:
    driver = Mock()
    driver.load.side_effect = RuntimeError("build failed")
    with patch(
        "dynamo.experimental.workflow.vllm.stages.AsyncVisionEncoder",
        return_value=driver,
    ):
        with pytest.raises(RuntimeError, match="build failed"):
            EncoderStage.from_backend(_Backend(), model="org/model")

    driver.shutdown.assert_called_once_with()


def test_resolve_vision_encoder_backend_class() -> None:
    with patch.object(
        loader.importlib,
        "import_module",
        return_value=SimpleNamespace(Backend=_Backend),
    ):
        assert (
            loader.resolve_vision_encoder_backend_class("author.encoder.Backend")
            is _Backend
        )


def test_resolve_vision_encoder_backend_class_rejects_wrong_type() -> None:
    with patch.object(
        loader.importlib,
        "import_module",
        return_value=SimpleNamespace(Backend=object),
    ):
        with pytest.raises(TypeError, match="VisionEncoderBackend subclass"):
            loader.resolve_vision_encoder_backend_class("author.encoder.Backend")
