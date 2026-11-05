# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media-limit configuration and rejection before raw-media processing."""

import argparse
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm.exceptions import InvalidArgument
from dynamo.sglang.backend_args import DynamoSGLangArgGroup, DynamoSGLangConfig
from dynamo.sglang.media_limits import parse_mm_limits, raise_if_mm_limit_exceeded
from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler
from dynamo.sglang.request_handlers.llm.prefill_handler import PrefillWorkerHandler
from dynamo.sglang.request_handlers.multimodal.encode_worker_handler import (
    MultimodalEncodeWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def test_media_limit_cli_overrides_environment(monkeypatch):
    monkeypatch.delenv("DYN_SGL_LIMIT_MM_PER_PROMPT", raising=False)
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    assert (
        DynamoSGLangConfig.from_cli_args(parser.parse_args([])).limit_mm_per_prompt
        is None
    )

    monkeypatch.setenv("DYN_SGL_LIMIT_MM_PER_PROMPT", '{"image": 2}')
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    assert parser.parse_args([]).limit_mm_per_prompt == {"image": 2}
    assert parser.parse_args(
        ["--limit-mm-per-prompt", '{"image": 1, "video": 0}']
    ).limit_mm_per_prompt == {"image": 1, "video": 0}


@pytest.mark.parametrize(
    "value",
    [
        "",
        "{",
        "null",
        "[]",
        '{"picture": 1}',
        '{"image": -1}',
        '{"image": 1.5}',
        '{"image": true}',
    ],
)
def test_media_limits_reject_invalid_configuration(value, monkeypatch):
    monkeypatch.delenv("DYN_SGL_LIMIT_MM_PER_PROMPT", raising=False)
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--limit-mm-per-prompt", value])

    monkeypatch.setenv("DYN_SGL_LIMIT_MM_PER_PROMPT", value)
    with pytest.raises(ValueError, match="--limit-mm-per-prompt"):
        DynamoSGLangArgGroup().add_arguments(argparse.ArgumentParser())


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
@pytest.mark.parametrize("native", [False, True])
def test_media_limit_boundary_and_zero(modality, native):
    limits = parse_mm_limits(json.dumps({modality: 2}))

    def request(count):
        items = ["data:media;base64,AAAA"] * count
        if native:
            return {"extra_args": {"sglang_tito": {f"{modality}_data": items}}}
        return {"multi_modal_data": {f"{modality}_url": items}}

    raise_if_mm_limit_exceeded(request(2), limits)
    raise_if_mm_limit_exceeded(request(3), None)
    raise_if_mm_limit_exceeded(request(3), {})
    raise_if_mm_limit_exceeded({}, {modality: 0})
    with pytest.raises(InvalidArgument, match=f"At most 2 {modality}"):
        raise_if_mm_limit_exceeded(request(3), limits)
    with pytest.raises(InvalidArgument, match=f"At most 0 {modality}"):
        raise_if_mm_limit_exceeded(request(1), {modality: 0})


def test_media_limits_leave_unlisted_modalities_unlimited():
    raise_if_mm_limit_exceeded(
        {"multi_modal_data": {"image_url": ["a"] * 11}}, {"video": 0}
    )


@pytest.mark.parametrize("items", ["image", ["a", "b"], [["a", "b"]]])
def test_native_media_scalar_and_nested_groups_are_counted(items):
    request = {"extra_args": {"sglang_tito": {"image_data": items}}}
    with pytest.raises(InvalidArgument, match="At most 0 image"):
        raise_if_mm_limit_exceeded(request, {"image": 0})
    if isinstance(items, list):
        with pytest.raises(InvalidArgument, match="At most 1 image"):
            raise_if_mm_limit_exceeded(request, {"image": 1})
    raise_if_mm_limit_exceeded(request, {"image": 2})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["aggregated", "decode", "prefill", "encode", "native"]
)
async def test_media_limit_rejects_before_engine_or_media_loading(route):
    handler_type = {
        "aggregated": DecodeWorkerHandler,
        "decode": DecodeWorkerHandler,
        "native": DecodeWorkerHandler,
        "prefill": PrefillWorkerHandler,
        "encode": MultimodalEncodeWorkerHandler,
    }[route]
    handler = handler_type.__new__(handler_type)
    handler.config = SimpleNamespace(
        dynamo_args=SimpleNamespace(limit_mm_per_prompt={"image": 1})
    )
    handler.serving_mode = (
        DisaggregationMode.DECODE
        if route == "decode"
        else DisaggregationMode.AGGREGATED
    )
    context = Mock()
    context.id.side_effect = AssertionError("request passed the media admission guard")
    handler._extract_media_inputs = Mock(
        side_effect=AssertionError("request reached media extraction")
    )
    handler.engine = SimpleNamespace(async_generate=AsyncMock())
    handler._prepare_image_inputs = AsyncMock()
    request = {"multi_modal_data": {"image_url": ["a", "b"]}}
    if route == "native":
        request = {"extra_args": {"sglang_tito": {"image_data": ["a", "b"]}}}
    elif route == "prefill":
        request = {"request": request}
    elif route == "encode":
        request = json.dumps(request)
    with pytest.raises(InvalidArgument, match="At most 1 image"):
        async for _ in handler.generate(request, context):
            pass
    handler.engine.async_generate.assert_not_called()
    handler._prepare_image_inputs.assert_not_called()
