# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the vLLM Nemotron video routing contract."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from dynamo.vllm.multimodal_utils.models import nemotron_video_routing

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
    pytest.mark.timeout(180),
]


def _nemotron_vllm_config(multimodal_config):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type=nemotron_video_routing.NEMOTRON_MODEL_TYPE,
                architectures=[nemotron_video_routing.NEMOTRON_MODEL_TYPE],
            ),
            multimodal_config=multimodal_config,
        )
    )


@pytest.mark.parametrize(
    ("pruning_spec", "expected_rate"),
    [(None, 0.0), (("evs", 0.5), 0.5)],
)
def test_publishes_nemotron_video_contract(monkeypatch, pruning_spec, expected_rate):
    monkeypatch.setattr(
        nemotron_video_routing, "_installed_processor_matches_contract", lambda: True
    )
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = _nemotron_vllm_config(
        SimpleNamespace(
            mm_processor_kwargs=None,
            get_video_pruning_spec=lambda: pruning_spec,
        )
    )

    nemotron_video_routing.publish_vllm_nemotron_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_called_once_with(
        nemotron_video_routing.VLLM_NEMOTRON_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY,
        json.dumps({"video_pruning_rate": expected_rate}),
    )


@pytest.mark.parametrize(
    "multimodal_config",
    [
        SimpleNamespace(
            mm_processor_kwargs={"use_audio_in_video": True},
            get_video_pruning_spec=lambda: None,
        ),
        SimpleNamespace(
            mm_processor_kwargs=None,
            get_video_pruning_spec=lambda: ("vidcom2", 0.5),
        ),
        SimpleNamespace(
            mm_processor_kwargs=None,
            get_video_pruning_spec=lambda: ("evs", "not-a-number"),
        ),
        SimpleNamespace(mm_processor_kwargs=None),
    ],
)
def test_skips_unsupported_nemotron_video_contract(monkeypatch, multimodal_config):
    monkeypatch.setattr(
        nemotron_video_routing, "_installed_processor_matches_contract", lambda: True
    )
    runtime_config = SimpleNamespace(set_engine_specific=Mock())

    nemotron_video_routing.publish_vllm_nemotron_video_processor_contract(
        runtime_config, _nemotron_vllm_config(multimodal_config)
    )

    runtime_config.set_engine_specific.assert_not_called()


def test_skips_non_nemotron_model(monkeypatch):
    monkeypatch.setattr(
        nemotron_video_routing, "_installed_processor_matches_contract", lambda: True
    )
    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_vl", architectures=[])
        )
    )

    nemotron_video_routing.publish_vllm_nemotron_video_processor_contract(
        runtime_config, vllm_config
    )

    runtime_config.set_engine_specific.assert_not_called()


def test_runtime_probe_pins_processor_contract(monkeypatch):
    geometry = Mock(return_value=(672, 384, 252))
    retention = Mock(side_effect=[10, 256])

    class Processor:
        @staticmethod
        def get_video_repl(**kwargs):
            encoded = kwargs["tokenizer"](
                nemotron_video_routing._ProbeTokenizer.expected,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            return SimpleNamespace(
                full=[
                    encoded[0][0],
                    19,
                    18,
                    18,
                    20,
                    encoded[1][0],
                    19,
                    20,
                ]
            )

    monkeypatch.setattr(
        nemotron_video_routing,
        "_load_nemotron_processor_contract",
        lambda: (Processor, geometry, retention),
    )

    assert nemotron_video_routing._installed_processor_matches_contract()
    assert retention.call_args_list == [
        call(tokens_per_frame=7, num_frames=3, q=0.5),
        call(tokens_per_frame=256, num_frames=4, q=0.9),
    ]


def test_runtime_probe_fails_closed_for_malformed_replacement(monkeypatch):
    class Processor:
        @staticmethod
        def get_video_repl(**_kwargs):
            return SimpleNamespace()

    monkeypatch.setattr(
        nemotron_video_routing,
        "_load_nemotron_processor_contract",
        lambda: (Processor, lambda **_kwargs: (672, 384, 252), lambda **_kwargs: 0),
    )

    assert not nemotron_video_routing._installed_processor_matches_contract()


def test_runtime_probe_rejects_retention_drift(monkeypatch):
    class Processor:
        @staticmethod
        def get_video_repl(**_kwargs):
            return SimpleNamespace(full=[101, 19, 18, 18, 20, 102, 19, 20])

    monkeypatch.setattr(
        nemotron_video_routing,
        "_load_nemotron_processor_contract",
        lambda: (
            Processor,
            lambda **_kwargs: (672, 384, 252),
            lambda **_kwargs: 11,
        ),
    )

    assert not nemotron_video_routing._installed_processor_matches_contract()


def test_pruning_query_failure_skips_contract(monkeypatch):
    monkeypatch.setattr(
        nemotron_video_routing, "_installed_processor_matches_contract", lambda: True
    )

    def failing_pruning_query():
        raise RuntimeError("unsupported pruning API")

    runtime_config = SimpleNamespace(set_engine_specific=Mock())
    multimodal_config = SimpleNamespace(
        mm_processor_kwargs=None,
        get_video_pruning_spec=failing_pruning_query,
    )

    nemotron_video_routing.publish_vllm_nemotron_video_processor_contract(
        runtime_config, _nemotron_vllm_config(multimodal_config)
    )

    runtime_config.set_engine_specific.assert_not_called()
