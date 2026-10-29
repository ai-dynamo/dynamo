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
        get_model_weight_size_from_hub,
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


MIB = 1024**2


def _patch_hub(monkeypatch, tmp_path, files, index=None, parameters=None) -> list:
    """Serve `files` from a fake Hub repo, with an optional safetensors index.

    An exception as `index` lists the index but makes downloading it fail.
    """
    siblings = list(files)
    index_path = tmp_path / "model.safetensors.index.json"
    if isinstance(index, Exception):
        siblings.append((index_path.name, 1))
    elif index is not None:
        weight_map = {f"weight{i}": name for i, name in enumerate(index)}
        index_path.write_text(json.dumps({"weight_map": weight_map}))
        siblings.append((index_path.name, index_path.stat().st_size))
    info = SimpleNamespace(
        sha="abc123",
        siblings=[
            SimpleNamespace(rfilename=name, size=size) for name, size in siblings
        ],
        safetensors=(
            SimpleNamespace(parameters=parameters) if parameters is not None else None
        ),
    )
    calls = []

    def fake_model_info(*args, **kwargs):
        calls.append(kwargs)
        return info

    def fake_hf_hub_download(**kwargs):
        assert index is not None, "downloaded an index the repo does not have"
        assert kwargs["revision"] == info.sha
        if isinstance(index, Exception):
            raise index
        return str(index_path)

    monkeypatch.setattr("dynamo.profiler.utils.model_info.model_info", fake_model_info)
    monkeypatch.setattr(
        "dynamo.profiler.utils.model_info.hf_hub_download", fake_hf_hub_download
    )
    return calls


def test_model_weight_size_from_hub_uses_checkpoint_file_sizes(
    monkeypatch, tmp_path
) -> None:
    """Packed GPTQ weights are sized from the file, not two bytes per param."""
    calls = _patch_hub(
        monkeypatch,
        tmp_path,
        [("config.json", 1024), ("model.safetensors", 768_148_704)],
        parameters={"I32": 968_884_224, "BF16": 131_164_160, "F16": 394_240},
    )

    assert get_model_weight_size_from_hub("test/gptq") == 768_148_704 / MIB
    assert calls[0]["files_metadata"] is True


@pytest.mark.parametrize(
    "files, index, expected_mib",
    [
        # gpt2: the same weights in two formats.
        (
            [("model.safetensors", 548 * MIB), ("pytorch_model.bin", 548 * MIB)],
            None,
            548,
        ),
        # Mistral: a consolidated file next to the indexed shards.
        (
            [
                ("consolidated.safetensors", 14 * MIB),
                ("model-00001-of-00002.safetensors", 8 * MIB),
                ("model-00002-of-00002.safetensors", 6 * MIB),
            ],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            14,
        ),
        # Devstral: two shard sets; the model index picks one.
        (
            [
                ("consolidated-00001-of-00002.safetensors", 7 * MIB),
                ("consolidated-00002-of-00002.safetensors", 3 * MIB),
                ("consolidated.safetensors.index.json", 1),
                ("model-00001-of-00002.safetensors", 6 * MIB),
                ("model-00002-of-00002.safetensors", 4 * MIB),
            ],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            10,
        ),
        # A single file left next to the indexed shards of the same name.
        (
            [
                ("model-00001-of-00002.safetensors", 5 * MIB),
                ("model-00002-of-00002.safetensors", 4 * MIB),
                ("model.safetensors", 9 * MIB),
            ],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            9,
        ),
        # An Accelerate checkpoint: optimizer state is larger than the model.
        (
            [
                ("optimizer.bin", 4 * MIB),
                ("pytorch_model.bin", 2 * MIB),
                ("scheduler.bin", 1),
            ],
            None,
            2,
        ),
        # A PEFT adapter next to full PyTorch shards.
        (
            [
                ("adapter_model.safetensors", 1 * MIB),
                ("pytorch_model-00001-of-00002.bin", 9 * MIB),
                ("pytorch_model-00002-of-00002.bin", 3 * MIB),
                ("training_args.bin", 1),
            ],
            None,
            12,
        ),
    ],
)
def test_model_weight_size_from_hub_counts_one_checkpoint_copy(
    monkeypatch, tmp_path, files, index, expected_mib
) -> None:
    _patch_hub(monkeypatch, tmp_path, files, index=index)

    assert get_model_weight_size_from_hub("test/model") == expected_mib


@pytest.mark.parametrize(
    "files, index",
    [
        # Sizes missing for part of the checkpoint.
        (
            [
                ("model-00001-of-00002.safetensors", 4 * MIB),
                ("model-00002-of-00002.safetensors", None),
            ],
            None,
        ),
        # The index lists a shard the repo does not have.
        (
            [("model-00001-of-00002.safetensors", 4 * MIB)],
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        ),
        # A gated repo lists its index but refuses the download.
        (
            [("model-00001-of-00001.safetensors", 4 * MIB)],
            OSError("403 Client Error"),
        ),
        # No weights at the repo root.
        ([("checkpoints/model.safetensors", 4 * MIB)], None),
    ],
)
def test_model_weight_size_from_hub_estimates_without_checkpoint_sizes(
    monkeypatch, tmp_path, files, index
) -> None:
    _patch_hub(monkeypatch, tmp_path, files, index=index, parameters={"BF16": 3 * MIB})

    assert get_model_weight_size_from_hub("test/model") == 6


@pytest.mark.parametrize("error_name", ["ConnectError", "ReadTimeout"])
def test_model_weight_size_from_hub_estimates_on_index_transport_error(
    monkeypatch, tmp_path, error_name
) -> None:
    """huggingface_hub 1.x raises httpx transport errors, which are not OSError."""
    httpx = pytest.importorskip("httpx")
    error = getattr(httpx, error_name)("index download failed")
    _patch_hub(
        monkeypatch,
        tmp_path,
        [("model-00001-of-00001.safetensors", 4 * MIB)],
        index=error,
        parameters={"BF16": 3 * MIB},
    )

    assert get_model_weight_size_from_hub("test/model") == 6
