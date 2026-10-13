# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler model-info helpers."""

import json
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelInfo as HubModelInfo

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
        get_model_weight_size_from_hub,
    )
except ImportError as e:
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)


@pytest.mark.parametrize(
    "weight_files",
    [
        [("model.safetensors", 768148704)],
        [("gptq_model-4bit-128g.safetensors", 768148704)],
        [("pytorch_model.bin", 768148704)],
        [("model.pt", 768148704)],
        [("consolidated.00.pth", 768148704)],
        [
            ("model-00001-of-00002.safetensors", 512 * 1024**2),
            ("model-00002-of-00002.safetensors", 256 * 1024**2),
        ],
    ],
)
def test_hub_weight_size_uses_checkpoint_metadata(monkeypatch, weight_files) -> None:
    def hub_model_info(model_name, *, token=None, files_metadata=False):
        assert model_name == "test/quantized-model"
        assert token == "test-token"
        siblings = [
            {"rfilename": filename, **({"size": size} if files_metadata else {})}
            for filename, size in weight_files
        ]
        siblings.append({"rfilename": "tokenizer.json", "size": 1024**2})
        for filename in (
            "training_args.bin",
            "optimizer.pt",
            "optimizer.bin",
            "scheduler.pt",
            "scaler.pt",
            "rng_state.pth",
            "sentencepiece.bpe.bin",
        ):
            siblings.append({"rfilename": filename, "size": 1024**2})
        # Packed integer parameter counts do not represent checkpoint bytes.
        return HubModelInfo(
            id=model_name,
            siblings=siblings,
            safetensors={"parameters": {"I32": 1000000000}, "total": 1000000000},
        )

    monkeypatch.setattr("dynamo.profiler.utils.model_info.model_info", hub_model_info)

    expected_mib = sum(size for _, size in weight_files) / 1024**2
    assert (
        get_model_weight_size_from_hub("test/quantized-model", token="test-token")
        == expected_mib
    )


@pytest.mark.parametrize(
    "siblings",
    [
        None,
        [{"rfilename": "model.safetensors"}],
        [
            {"rfilename": "model.safetensors"},
            {"rfilename": "optimizer.pt", "size": 1024**2},
        ],
    ],
)
def test_hub_weight_size_falls_back_when_sizes_are_unavailable(
    monkeypatch, siblings
) -> None:
    info = HubModelInfo(
        id="test/model",
        siblings=siblings,
        safetensors={"parameters": {"BF16": 1024**2}, "total": 1024**2},
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.model_info", lambda *args, **kwargs: info
    )

    assert get_model_weight_size_from_hub("test/model") == 2


@pytest.mark.parametrize("extension", [".safetensors", ".bin", ".pt", ".pth"])
def test_hub_weight_size_uses_index_references(
    monkeypatch, tmp_path, extension
) -> None:
    index_name = "checkpoint/custom" + extension + ".index.json"
    index_path = tmp_path / "checkpoint.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer0": "custom-part-1" + extension,
                    "layer1": "custom-part-1" + extension,
                    "layer2": "custom-part-2" + extension,
                }
            }
        )
    )
    info = HubModelInfo(
        id="test/model",
        sha="test-revision",
        siblings=[
            {"rfilename": index_name},
            {"rfilename": "checkpoint/custom-part-1" + extension, "size": 1024**2},
            {"rfilename": "checkpoint/custom-part-2" + extension, "size": 1024**2},
            {"rfilename": "unused.safetensors", "size": 1024**2},
            {"rfilename": "training_args.bin", "size": 1024**2},
            {"rfilename": "optimizer.pt", "size": 1024**2},
        ],
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.model_info", lambda *args, **kwargs: info
    )

    def download_index(*, repo_id, filename, revision, token):
        assert repo_id == "test/model"
        assert filename == index_name
        assert revision == "test-revision"
        assert token == "test-token"
        return index_path

    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.hf_hub_download", download_index
    )

    assert get_model_weight_size_from_hub("test/model", token="test-token") == 2


@pytest.mark.parametrize("index_data", [None, "invalid-json", '{"weight_map": []}'])
def test_hub_weight_size_falls_back_when_index_is_unavailable(
    monkeypatch, tmp_path, index_data
) -> None:
    info = HubModelInfo(
        id="test/model",
        sha="test-revision",
        siblings=[
            {"rfilename": "model.safetensors.index.json"},
            {"rfilename": "model-00001-of-00002.safetensors", "size": 1024**2},
            {"rfilename": "optimizer.pt", "size": 1024**2},
        ],
        safetensors={"parameters": {"BF16": 2 * 1024**2}, "total": 2 * 1024**2},
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.model_info", lambda *args, **kwargs: info
    )
    index_path = tmp_path / "checkpoint.index.json"
    if index_data is not None:
        index_path.write_text(index_data)
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.hf_hub_download",
        lambda **kwargs: index_path,
    )

    assert get_model_weight_size_from_hub("test/model") == 4


def test_hub_weight_size_prefers_one_checkpoint_format(monkeypatch) -> None:
    info = HubModelInfo(
        id="test/model",
        siblings=[
            {"rfilename": "model.safetensors", "size": 1024**2},
            {"rfilename": "pytorch_model.bin", "size": 2 * 1024**2},
        ],
    )
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.model_info", lambda *args, **kwargs: info
    )

    assert get_model_weight_size_from_hub("test/model") == 1


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
