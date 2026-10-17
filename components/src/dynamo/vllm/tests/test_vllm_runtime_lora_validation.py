# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from dynamo.llm import HttpError
from dynamo.vllm.runtime_lora_validation import validate_snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _safetensors_blob(header=None, payload=bytes(8)) -> bytes:
    if header is None:
        header = {
            "base_model.model.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, 8],
            }
        }
    encoded = json.dumps(header).encode()
    return len(encoded).to_bytes(8, "little") + encoded + payload


def _snapshot(tmp_path, **config_overrides):
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    config = {
        "r": 8,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "base",
        **config_overrides,
    }
    (snapshot / "adapter_config.json").write_text(json.dumps(config))
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob())
    return snapshot


def _assert_invalid(snapshot, cache_root, max_bytes=1024, max_rank=64):
    with pytest.raises(HttpError) as error:
        validate_snapshot(snapshot, cache_root, max_bytes, max_rank, ("base",))
    assert error.value.code == 422
    assert error.value.message == "invalid_lora_adapter"


@pytest.mark.parametrize("relative_path", ["README.md", "nested/metadata.json"])
def test_snapshot_rejects_files_outside_vllm_allowlist(tmp_path, relative_path):
    snapshot = _snapshot(tmp_path)
    unexpected = snapshot / relative_path
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("untrusted")

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_accepts_valid_safetensors_embeddings(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "new_embeddings.safetensors").write_bytes(_safetensors_blob())

    result = validate_snapshot(snapshot, tmp_path, 1024, 64, ("base",))

    assert result == snapshot


@pytest.mark.parametrize("path_kind", ["missing", "file", "outside", "symlink"])
def test_snapshot_requires_real_directory_inside_cache(tmp_path, path_kind):
    snapshot = _snapshot(tmp_path)
    cache_root = tmp_path
    if path_kind == "missing":
        snapshot = tmp_path / "missing"
    elif path_kind == "file":
        snapshot = snapshot / "adapter_config.json"
    elif path_kind == "outside":
        cache_root = tmp_path / "other-cache"
        cache_root.mkdir()
    else:
        link = tmp_path / "linked-adapter"
        link.symlink_to(snapshot, target_is_directory=True)
        snapshot = link

    _assert_invalid(snapshot, cache_root)


@pytest.mark.parametrize("entry_kind", ["symlink", "hardlink", "fifo"])
def test_snapshot_rejects_linked_or_special_files(tmp_path, entry_kind):
    snapshot = _snapshot(tmp_path)
    entry = snapshot / "new_embeddings.safetensors"
    source = snapshot / "adapter_model.safetensors"
    if entry_kind == "symlink":
        entry.symlink_to(source)
    elif entry_kind == "hardlink":
        os.link(source, entry)
    else:
        os.mkfifo(entry)

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_enforces_download_byte_limit(tmp_path):
    snapshot = _snapshot(tmp_path)

    _assert_invalid(snapshot, tmp_path, max_bytes=1)


@pytest.mark.parametrize(
    "missing_name", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_snapshot_requires_config_and_weights(tmp_path, missing_name):
    snapshot = _snapshot(tmp_path)
    (snapshot / missing_name).unlink()

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("content", [b"not json", b"\xff", b"[]"])
def test_snapshot_rejects_invalid_config_json(tmp_path, content):
    snapshot = _snapshot(tmp_path)
    (snapshot / "adapter_config.json").write_bytes(content)

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("rank", [None, True, 0, -1, 1.5, 65])
def test_snapshot_enforces_adapter_rank(tmp_path, rank):
    snapshot = _snapshot(tmp_path, r=rank)

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_accepts_base_alias_and_unbounded_rank(tmp_path):
    snapshot = _snapshot(tmp_path, r=128, base_model_name_or_path="alias")

    assert (
        validate_snapshot(snapshot, tmp_path, 1024, None, (None, "", "base", "alias"))
        == snapshot
    )


def test_snapshot_rejects_other_base_model(tmp_path):
    snapshot = _snapshot(tmp_path, base_model_name_or_path="other")

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize(
    "targets",
    [
        None,
        "",
        "x" * 257,
        "q\nproj",
        "q\x7fproj",
        [],
        [1],
        [""],
        ["x" * 257],
        ["q\nproj"],
        ["missing_proj"],
    ],
)
def test_snapshot_rejects_invalid_or_missing_targets(tmp_path, targets):
    snapshot = _snapshot(tmp_path, target_modules=targets)

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_accepts_regex_targets(tmp_path):
    snapshot = _snapshot(tmp_path, target_modules=".*q_proj")

    assert validate_snapshot(snapshot, tmp_path, 1024, 64, ("base",)) == snapshot


@pytest.mark.parametrize(
    "blob",
    [
        b"short",
        (2).to_bytes(8, "little") + b"{}",
        (17 * 1024 * 1024).to_bytes(8, "little"),
        (100).to_bytes(8, "little") + b"{}",
        (3).to_bytes(8, "little") + b"bad",
        (3).to_bytes(8, "little") + b"\xff\xff\xff",
    ],
)
def test_snapshot_rejects_malformed_weight_headers(tmp_path, blob):
    snapshot = _snapshot(tmp_path)
    (snapshot / "adapter_model.safetensors").write_bytes(blob)

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize(
    "header",
    [
        [1, 2],
        {"__metadata__": "bad"},
        {"__metadata__": {}},
        {"": {}},
        {"q_proj": []},
        {"q_proj": {"dtype": "invalid"}},
    ],
)
def test_snapshot_rejects_invalid_tensor_headers(tmp_path, header):
    snapshot = _snapshot(tmp_path)
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize(
    "shape, offsets",
    [
        (None, [0, 8]),
        ([], [0, 8]),
        ([1] * 9, [0, 8]),
        ([True], [0, 8]),
        ([-1], [0, 8]),
        ([1.5], [0, 8]),
        ([2], None),
        ([2], [0]),
        ([2], [False, 8]),
        ([2], [0, 8.0]),
        ([2], [-1, 7]),
        ([2], [8, 0]),
        ([2], [0, 12]),
        ([2], [0, 4]),
    ],
)
def test_snapshot_rejects_invalid_tensor_shape_or_offsets(tmp_path, shape, offsets):
    snapshot = _snapshot(tmp_path)
    header = {"q_proj": {"dtype": "F32", "shape": shape, "data_offsets": offsets}}
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_rejects_overlapping_tensors(tmp_path):
    snapshot = _snapshot(tmp_path)
    header = {
        name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
        for name in ("q_proj.lora_A", "q_proj.lora_B")
    }
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_rejects_invalid_optional_embeddings(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "new_embeddings.safetensors").write_bytes(b"invalid")

    _assert_invalid(snapshot, tmp_path)
