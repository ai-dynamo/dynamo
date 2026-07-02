# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``TensorRTLLMEngine._is_unsupported_encoder_arch``.

The check reads only ``architectures`` from ``config.json`` via
``PretrainedConfig.get_config_dict`` -- no model instantiation, no repo code
execution. These tests pin that contract, including that a config declaring a
custom ``auto_map`` is NOT executed.
"""

import json

import pytest

# Importing the engine pulls in tensorrt_llm, which needs libcuda at import
# time. Skip cleanly where the native stack isn't importable rather than
# erroring collection.
try:
    from dynamo.trtllm.engine import TensorRTLLMEngine
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"tensorrt_llm not importable ({exc})", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_check = TensorRTLLMEngine._is_unsupported_encoder_arch


def _model_dir(tmp_path, config, extra=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name, content in (extra or {}).items():
        (tmp_path / name).write_text(content)
    return str(tmp_path)


def test_supported_arch(tmp_path):
    path = _model_dir(tmp_path, {"architectures": ["Qwen3VLForConditionalGeneration"]})
    assert _check(path) is False


def test_unsupported_arch(tmp_path):
    path = _model_dir(tmp_path, {"architectures": ["Llama4ForConditionalGeneration"]})
    assert _check(path) is True


def test_missing_config(tmp_path):
    assert _check(str(tmp_path)) is False


def test_missing_architectures(tmp_path):
    assert _check(_model_dir(tmp_path, {"model_type": "x"})) is False


def test_auto_map_config_not_executed(tmp_path):
    # Regression guard: reading the arch must not execute repo code even when
    # the config declares a custom auto_map. A revert to
    # AutoConfig.from_pretrained(trust_remote_code=True) would run this module.
    marker = tmp_path / "EXECUTED"
    path = _model_dir(
        tmp_path,
        {
            "architectures": ["Llama4ForConditionalGeneration"],
            "auto_map": {"AutoConfig": "configuration_evil.EvilConfig"},
        },
        extra={
            "configuration_evil.py": (
                f"import pathlib; pathlib.Path({str(marker)!r}).write_text('x')\n"
                "from transformers import PretrainedConfig\n"
                "class EvilConfig(PretrainedConfig):\n"
                "    model_type = 'evil'\n"
            )
        },
    )
    result = _check(path)
    assert not marker.exists(), "repo code executed during the arch check"
    assert result is True
