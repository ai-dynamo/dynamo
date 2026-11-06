# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""--image-png-compression-level parsing for TensorRT-LLM image workers."""

import argparse

import pytest
import torch

# Pytest applies marker selection after importing test modules. Importing
# backend_args loads TensorRT-LLM bindings, which require libcuda even though
# these tests do not execute CUDA kernels, so skip collection on CPU runners.
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU not available, but tensorrt_llm import requires CUDA.",
        allow_module_level=True,
    )
pytest.importorskip("tensorrt_llm")

from dynamo.trtllm.backend_args import DynamoTrtllmArgGroup
from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
]

PNG_LEVEL_ENV = "DYN_TRTLLM_IMAGE_PNG_COMPRESSION_LEVEL"


def _parse_png_level(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    DynamoTrtllmArgGroup().add_arguments(parser)
    level = parser.parse_args(argv).image_png_compression_level
    return DiffusionConfig(
        image_png_compression_level=level
    ).image_png_compression_level


@pytest.mark.parametrize(
    ("env_value", "argv", "expected"),
    [
        (None, [], 6),
        ("0", [], 0),
        ("9", [], 9),
        (None, ["--image-png-compression-level", "0"], 0),
        (None, ["--image-png-compression-level", "9"], 9),
        ("9", ["--image-png-compression-level", "0"], 0),
    ],
)
def test_config_from_cli_and_env(monkeypatch, env_value, argv, expected):
    monkeypatch.delenv(PNG_LEVEL_ENV, raising=False)
    if env_value is not None:
        monkeypatch.setenv(PNG_LEVEL_ENV, env_value)

    assert _parse_png_level(argv) == expected


@pytest.mark.parametrize("value", ["-1", "10", "1.5"])
def test_config_rejects_invalid_level_from_env(monkeypatch, value):
    monkeypatch.setenv(PNG_LEVEL_ENV, value)

    with pytest.raises(ValueError):
        _parse_png_level([])
