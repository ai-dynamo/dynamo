# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM Qwen video routing contract."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.vllm.multimodal_utils.models import qwen_video_routing

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
    pytest.mark.timeout(180),
]


def _qwen_vllm_config(multimodal_config):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="qwen3_5",
                architectures=["Qwen3_5ForConditionalGeneration"],
            ),
            multimodal_config=multimodal_config,
        )
    )


@pytest.mark.parametrize(
    ("overrides_video_replacement", "expected"),
    [
        (True, "bare_video_token"),
        (False, "vision_wrapped_video_token"),
    ],
)
def test_publishes_actual_qwen_video_placeholder_target(
    monkeypatch, overrides_video_replacement, expected
):
    class ProcessorMixin:
        def replace_video_token(self):
            pass

    if overrides_video_replacement:

        class Qwen3VLProcessor(ProcessorMixin):
            def replace_video_token(self):
                pass

    else:

        class Qwen3VLProcessor(ProcessorMixin):
            pass

    monkeypatch.setattr(
        qwen_video_routing.transformers, "ProcessorMixin", ProcessorMixin
    )
    monkeypatch.setattr(
        qwen_video_routing.transformers,
        "Qwen3VLProcessor",
        Qwen3VLProcessor,
        raising=False,
    )
    monkeypatch.setattr(
        qwen_video_routing,
        "_resolve_qwen_video_resize_mode",
        lambda: qwen_video_routing.QWEN_VIDEO_RESIZE_LEGACY_CEIL,
    )
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = _qwen_vllm_config(
        SimpleNamespace(
            mm_processor_kwargs=None,
            get_video_pruning_spec=lambda: None,
        )
    )

    qwen_video_routing.publish_vllm_qwen_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_called_once_with(
        qwen_video_routing.VLLM_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        json.dumps(
            {
                "placeholder_target": expected,
                "resize_mode": qwen_video_routing.QWEN_VIDEO_RESIZE_LEGACY_CEIL,
            }
        ),
    )


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(model_type="llama"),
        SimpleNamespace(model_type="qwen3_5", architectures=["Qwen3_5ForCausalLM"]),
    ],
)
def test_skips_video_placeholder_target_for_non_video_models(hf_config):
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))

    qwen_video_routing.publish_vllm_qwen_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_not_called()


def test_skips_video_placeholder_target_for_engine_processor_overrides():
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = _qwen_vllm_config(
        SimpleNamespace(
            mm_processor_kwargs={"max_pixels": 4096},
            get_video_pruning_spec=lambda: None,
        )
    )

    qwen_video_routing.publish_vllm_qwen_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_not_called()


def test_skips_video_placeholder_target_for_video_pruning():
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = _qwen_vllm_config(
        SimpleNamespace(
            mm_processor_kwargs=None,
            get_video_pruning_spec=lambda: ("evs", 0.5),
        )
    )

    qwen_video_routing.publish_vllm_qwen_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_not_called()


def test_skips_video_contract_when_pruning_api_is_unavailable():
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = _qwen_vllm_config(SimpleNamespace(mm_processor_kwargs=None))

    qwen_video_routing.publish_vllm_qwen_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    ("resize_result", "expected"),
    [
        ((1216, 4096), "legacy_ceil"),
        ((1120, 3776), "round_ties_even"),
        ((999, 999), None),
    ],
)
def test_resolve_qwen_video_resize_mode(monkeypatch, resize_result, expected):
    resize = Mock(return_value=resize_result)
    monkeypatch.setattr(qwen_video_routing, "qwen3_smart_resize", resize)

    assert qwen_video_routing._resolve_qwen_video_resize_mode() == expected
    resize.assert_called_once_with(
        num_frames=5,
        height=1120,
        width=3760,
        temporal_factor=2,
        factor=32,
        min_pixels=4096,
        max_pixels=25165824,
    )


def test_resolve_qwen_video_resize_mode_without_qwen_processor(monkeypatch):
    monkeypatch.setattr(qwen_video_routing, "qwen3_smart_resize", None)

    assert qwen_video_routing._resolve_qwen_video_resize_mode() is None


def test_skips_video_contract_without_qwen_processor(monkeypatch):
    monkeypatch.setattr(
        qwen_video_routing.transformers,
        "Qwen3VLProcessor",
        None,
        raising=False,
    )
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = _qwen_vllm_config(
        SimpleNamespace(
            mm_processor_kwargs=None,
            get_video_pruning_spec=lambda: None,
        )
    )

    qwen_video_routing.publish_vllm_qwen_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_not_called()
