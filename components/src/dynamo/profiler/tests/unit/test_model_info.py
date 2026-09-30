# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler model-info helpers."""

import json
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]

try:
    from dynamo.profiler.utils.model_info import (
        get_mamba_cache_align_block_size,
        get_model_context_length,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


def test_mamba_cache_align_block_size_from_local_config(tmp_path) -> None:
    """Mamba align floor follows vLLM's Mamba/attention page size."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "mamba_num_heads": 128,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
            }
        )
    )

    assert get_mamba_cache_align_block_size(tmp_path) == 8320


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"max_position_embeddings": 2048}, 2048),
        ({"n_positions": 1024}, 1024),
        ({"text_config": {"max_sequence_length": 4096}}, 4096),
        (
            {
                "max_position_embeddings": 8192,
                "thinker_config": {"text_config": {"max_position_embeddings": 2048}},
            },
            2048,
        ),
        (
            {
                "model_type": "llama",
                "rope_scaling": {"type": "linear", "factor": 4.0},
                "text_config": {"max_position_embeddings": 2048},
            },
            8192,
        ),
        (
            {"max_position_embeddings": 8192, "model_max_length": 131072},
            131072,
        ),
        (
            {
                "max_position_embeddings": 2048,
                "model_type": "llama",
                "rope_scaling": {"type": "linear", "factor": 4.0},
            },
            8192,
        ),
        (
            {
                "max_position_embeddings": 8192,
                "model_type": "llama",
                "rope_scaling": {"rope_type": "llama3", "factor": 8.0},
            },
            8192,
        ),
    ],
)
def test_model_context_length_from_local_config(
    tmp_path, config: dict, expected: int
) -> None:
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert get_model_context_length(tmp_path) == expected


def test_model_context_length_uses_tokenizer_ceiling(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 2048})
    )

    assert get_model_context_length(tmp_path) == 2048


def test_model_context_length_prefers_nested_object_config(monkeypatch) -> None:
    config = SimpleNamespace(
        max_position_embeddings=8192,
        decoder=SimpleNamespace(max_position_embeddings=2048),
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info._load_model_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info._load_tokenizer_config",
        lambda *args, **kwargs: None,
    )

    assert get_model_context_length("test/model") == 2048


def test_model_context_length_prefers_get_text_config(monkeypatch) -> None:
    text_config = SimpleNamespace(max_position_embeddings=2048)
    config = SimpleNamespace(
        max_position_embeddings=8192,
        get_text_config=lambda: text_config,
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info._load_model_config",
        lambda *args, **kwargs: config,
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info._load_tokenizer_config",
        lambda *args, **kwargs: None,
    )

    assert get_model_context_length("test/model") == 2048


def test_model_config_falls_back_when_aic_download_fails(monkeypatch) -> None:
    class DownloadError(Exception):
        pass

    expected = SimpleNamespace(max_position_embeddings=4096)
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.HuggingFaceDownloadError",
        DownloadError,
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info._load_model_config_from_model_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(DownloadError()),
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.AutoConfig.from_pretrained",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info._load_tokenizer_config",
        lambda *args, **kwargs: None,
    )

    assert get_model_context_length("test/model") == 4096
