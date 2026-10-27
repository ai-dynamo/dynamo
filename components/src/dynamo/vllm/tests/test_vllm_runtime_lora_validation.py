# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from math import prod
from pathlib import Path

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


@pytest.mark.parametrize("base", [[], {}, True, 7, None])
def test_snapshot_rejects_non_string_base_metadata(tmp_path, base):
    snapshot = _snapshot(tmp_path, base_model_name_or_path=base)

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("dtype", [[], {}, True, 7, None])
def test_snapshot_rejects_non_string_tensor_dtype(tmp_path, dtype):
    snapshot = _snapshot(tmp_path)
    header = {"q_proj": {"dtype": dtype, "shape": [2], "data_offsets": [0, 8]}}
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("metadata", [{"key": []}, {"key": {}}, {"key": 7}])
def test_snapshot_rejects_non_string_safetensors_metadata(tmp_path, metadata):
    snapshot = _snapshot(tmp_path)
    header = {
        "__metadata__": metadata,
        "q_proj": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize(
    "artifact", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_snapshot_bounds_json_nesting(tmp_path, artifact):
    snapshot = _snapshot(tmp_path)
    nested = b"[" * 65 + b"0" + b"]" * 65
    if artifact.endswith("safetensors"):
        encoded = (
            b'{"q_proj":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"extra":'
            + nested
            + b"}}"
        )
        encoded = len(encoded).to_bytes(8, "little") + encoded
        encoded += bytes(8)
    else:
        encoded = (
            b'{"r":8,"target_modules":["q_proj"],"base_model_name_or_path":"base","extra":'
            + nested
            + b"}"
        )
    (snapshot / artifact).write_bytes(encoded)

    _assert_invalid(snapshot, tmp_path, max_bytes=1 << 20)


def test_snapshot_bounds_config_bytes_independently_of_snapshot_limit(tmp_path):
    snapshot = _snapshot(tmp_path, padding="x" * (1024 * 1024))

    _assert_invalid(snapshot, tmp_path, max_bytes=2 * 1024 * 1024)


def test_snapshot_ignores_json_punctuation_in_strings(tmp_path):
    snapshot = _snapshot(tmp_path, padding='["\\' * 100)

    assert validate_snapshot(snapshot, tmp_path, 2048, 64, ("base",)) == snapshot


def test_snapshot_sanitizes_cache_root_symlink_loop(tmp_path):
    snapshot = _snapshot(tmp_path)
    cache_root = tmp_path / "loop"
    cache_root.symlink_to(cache_root)

    _assert_invalid(snapshot, cache_root)


def test_snapshot_rejects_oversized_json_integer(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "adapter_config.json").write_text('{"r":' + "1" * 5000 + "}")

    _assert_invalid(snapshot, tmp_path, max_bytes=8192)


def test_snapshot_rejects_surrogate_target_metadata(tmp_path):
    snapshot = _snapshot(tmp_path, target_modules=["\ud800"])

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError])
def test_snapshot_sanitizes_file_stat_failures(tmp_path, monkeypatch, failure):
    snapshot = _snapshot(tmp_path)
    original = Path.lstat

    def fail_metadata(path, *args, **kwargs):
        if path.name == "adapter_config.json":
            raise failure(f"private path: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", fail_metadata)

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_sanitizes_directory_scan_failures(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)

    def fail_walk(_path, *, followlinks, onerror=None):
        error = PermissionError(f"private path: {snapshot}")
        if onerror is not None:
            onerror(error)
        return iter(())

    monkeypatch.setattr(os, "walk", fail_walk)

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


@pytest.mark.parametrize(
    "kind", ["trailing", "gap", "leading-gap", "utf16", "utf8-bom"]
)
def test_snapshot_rejects_noncanonical_safetensors(tmp_path, kind):
    snapshot = _snapshot(tmp_path)
    header = {"q_proj": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    payload = bytes(8)
    if kind == "trailing":
        payload = bytes(16)
    elif kind == "gap":
        header = {
            "q_proj": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            "v_proj": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
        }
        payload = bytes(12)
    elif kind == "leading-gap":
        header["q_proj"]["data_offsets"] = [4, 12]
        payload = bytes(12)
    encoded = json.dumps(header).encode("utf-16" if kind == "utf16" else "utf-8")
    if kind == "utf8-bom":
        encoded = b"\xef\xbb\xbf" + encoded
    blob = len(encoded).to_bytes(8, "little") + encoded + payload
    (snapshot / "adapter_model.safetensors").write_bytes(blob)

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_accepts_leading_safetensors_json_whitespace(tmp_path):
    snapshot = _snapshot(tmp_path)
    header = {"q_proj": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    encoded = b" " + json.dumps(header).encode()
    blob = len(encoded).to_bytes(8, "little") + encoded + bytes(8)
    (snapshot / "adapter_model.safetensors").write_bytes(blob)

    assert validate_snapshot(snapshot, tmp_path, 1024, 64, ("base",)) == snapshot


def test_snapshot_requires_distinct_cache_subdirectory(tmp_path):
    snapshot = _snapshot(tmp_path)

    _assert_invalid(snapshot, snapshot)


def test_snapshot_bounds_safetensors_header_before_reading(tmp_path):
    snapshot = _snapshot(tmp_path)
    header_bytes = 17 * 1024 * 1024
    with (snapshot / "adapter_model.safetensors").open("wb") as weights:
        weights.write(header_bytes.to_bytes(8, "little"))
        weights.truncate(8 + header_bytes)

    _assert_invalid(snapshot, tmp_path, max_bytes=32 * 1024 * 1024)


def test_snapshot_rejects_invalid_optional_embeddings(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "new_embeddings.safetensors").write_bytes(b"invalid")

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("constant", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "artifact", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_snapshot_rejects_nonstandard_json_constants(tmp_path, constant, artifact):
    snapshot = _snapshot(tmp_path)
    path = snapshot / artifact
    if artifact.endswith("safetensors"):
        header = {
            "q_proj": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8],
                "extra": constant,
            }
        }
        path.write_bytes(_safetensors_blob(header))
    else:
        config = json.loads(path.read_text())
        path.write_text(json.dumps({**config, "extra": constant}))

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("number", ["1e999", "-1e999"])
@pytest.mark.parametrize(
    "artifact", ["adapter_config.json", "adapter_model.safetensors"]
)
def test_snapshot_rejects_overflowing_json_float(tmp_path, number, artifact):
    snapshot = _snapshot(tmp_path)
    path = snapshot / artifact
    if artifact.endswith("safetensors"):
        header = (
            json.dumps(
                {
                    "q_proj": {
                        "dtype": "F32",
                        "shape": [2],
                        "data_offsets": [0, 8],
                        "extra": 1.0,
                    }
                }
            )
            .replace("1.0", number)
            .encode()
        )
        path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(8))
    else:
        config = json.loads(path.read_text())
        path.write_text(json.dumps({**config, "extra": 1.0}).replace("1.0", number))

    _assert_invalid(snapshot, tmp_path)


@pytest.mark.parametrize("shape", [[2**64, 0], [2**40, 2**40, 0]])
def test_snapshot_rejects_shape_integer_overflow_before_zero(tmp_path, shape):
    snapshot = _snapshot(tmp_path)
    header = {
        "q_proj": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "z_proj": {"dtype": "F32", "shape": shape, "data_offsets": [8, 8]},
    }
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    _assert_invalid(snapshot, tmp_path)


def test_snapshot_accepts_zero_tensor_without_element_product_overflow(tmp_path):
    snapshot = _snapshot(tmp_path)
    header = {
        "q_proj": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "z_proj": {"dtype": "F32", "shape": [2**62, 0], "data_offsets": [8, 8]},
    }
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob(header))

    assert validate_snapshot(snapshot, tmp_path, 1024, 64, ("base",)) == snapshot


@pytest.mark.parametrize(
    "suffix,shape_kind",
    [
        ("lora_A.weight", "matrix-a"),
        ("lora_B.weight", "matrix-b"),
        ("lora_embedding_A", "matrix-a"),
        ("lora_embedding_B", "matrix-b"),
        ("lora_A.weight", "experts-a"),
        ("lora_B.weight", "experts-b"),
    ],
)
@pytest.mark.parametrize(
    "rank,max_rank,accepted", [(64, 64, True), (128, 64, False), (128, None, True)]
)
def test_snapshot_bounds_actual_lora_tensor_rank(
    tmp_path, suffix, shape_kind, rank, max_rank, accepted
):
    snapshot = _snapshot(tmp_path)
    shape = [rank, 2] if shape_kind.endswith("a") else [2, rank]
    if shape_kind.startswith("experts"):
        shape = [2, *shape]
    size = prod(shape) * 4
    header = {
        f"base_model.model.q_proj.{suffix}": {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, size],
        }
    }
    (snapshot / "adapter_model.safetensors").write_bytes(
        _safetensors_blob(header, bytes(size))
    )

    if accepted:
        assert (
            validate_snapshot(snapshot, tmp_path, 16384, max_rank, ("base",))
            == snapshot
        )
    else:
        _assert_invalid(snapshot, tmp_path, max_bytes=16384, max_rank=max_rank)


def test_snapshot_does_not_treat_expert_count_as_lora_rank(tmp_path):
    snapshot = _snapshot(tmp_path)
    shape = [128, 8, 2]
    size = prod(shape) * 4
    header = {
        "base_model.model.q_proj.lora_A.weight": {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [0, size],
        }
    }
    (snapshot / "adapter_model.safetensors").write_bytes(
        _safetensors_blob(header, bytes(size))
    )

    assert validate_snapshot(snapshot, tmp_path, 16384, 64, ("base",)) == snapshot
