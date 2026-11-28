# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile derivation and connector-file handling for hidden-state capture."""

import fcntl
import os
import threading
import time
from types import SimpleNamespace

import pytest

from dynamo.artifacts.vllm.hidden_states import (
    HiddenStateFileError,
    HiddenStateProfile,
    discard_hidden_state_file,
    hidden_state_path,
    hidden_state_profile_from_vllm_config,
    load_hidden_state_file,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def _vllm_config(
    *,
    method="extract_hidden_states",
    connector="ExampleHiddenStatesConnector",
    extra=None,
    layer_ids=(2, 16, 30),
    dtype="torch.bfloat16",
    hidden_size=64,
):
    kv_extra = {"shared_storage_path": "/var/hs"} if extra is None else extra
    if connector == "MultiConnector":
        kv_transfer_config = {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {
                "connectors": [
                    {"kv_connector": "NixlConnector"},
                    {
                        "kv_connector": "ExampleHiddenStatesConnector",
                        "kv_connector_extra_config": kv_extra,
                    },
                ]
            },
        }
    else:
        kv_transfer_config = SimpleNamespace(
            kv_connector=connector, kv_connector_extra_config=kv_extra
        )
    hf_config = SimpleNamespace(eagle_aux_hidden_state_layer_ids=list(layer_ids))
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        ),
        kv_transfer_config=kv_transfer_config,
        model_config=SimpleNamespace(dtype=dtype, get_hidden_size=lambda: hidden_size),
    )


@pytest.mark.parametrize(
    "connector", ["ExampleHiddenStatesConnector", "MultiConnector"]
)
def test_profile_comes_from_engine_config(connector):
    profile = hidden_state_profile_from_vllm_config(_vllm_config(connector=connector))
    assert profile == HiddenStateProfile(
        (2, 16, 30), 64, "bf16", os.path.realpath("/var/hs")
    )
    assert profile.row_bytes == 3 * 64 * 2
    assert profile.runtime_data() == {
        "layer_ids": [2, 16, 30],
        "hidden_size": 64,
        "dtype": "bf16",
    }


def test_profile_defaults_to_connector_storage_root():
    profile = hidden_state_profile_from_vllm_config(_vllm_config(extra={}))
    assert profile.storage_root == os.path.realpath("/tmp")


@pytest.mark.parametrize(
    "config",
    [
        None,
        SimpleNamespace(speculative_config=None, kv_transfer_config=None),
        _vllm_config(method="eagle3"),
        _vllm_config(connector="NixlConnector"),
    ],
)
def test_workers_without_extraction_have_no_profile(config):
    assert hidden_state_profile_from_vllm_config(config) is None


@pytest.mark.parametrize(
    "config,match",
    [
        (_vllm_config(extra={"use_synchronization_lock": False}), "lock"),
        (_vllm_config(extra={"allow_custom_save_path": True}), "custom_save_path"),
        (_vllm_config(layer_ids=()), "layer_ids"),
        (_vllm_config(layer_ids=(2, 2)), "layer_ids"),
        (_vllm_config(dtype="torch.float8_e4m3fn"), "bfloat16"),
        (_vllm_config(hidden_size=0), "hidden size"),
    ],
)
def test_unreadable_extraction_config_fails_at_launch(config, match):
    with pytest.raises(ValueError, match=match):
        hidden_state_profile_from_vllm_config(config)


def _profile(root):
    return HiddenStateProfile((2, 5), 4, "fp16", os.path.realpath(str(root)))


def test_path_is_confined_to_storage_root(tmp_path):
    root = tmp_path / "hs"
    root.mkdir()
    profile = _profile(root)
    inside = root / "req.safetensors"
    assert hidden_state_path({"hidden_states_path": str(inside)}, profile) == str(
        inside
    )
    assert hidden_state_path({"do_remote_decode": False}, profile) is None
    assert hidden_state_path(None, profile) is None
    for path in (str(tmp_path / "other.safetensors"), str(root / ".." / "x"), ""):
        with pytest.raises(HiddenStateFileError):
            hidden_state_path({"hidden_states_path": path}, profile)
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(HiddenStateFileError, match="outside"):
        hidden_state_path({"hidden_states_path": str(root / "escape" / "x")}, profile)


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _write(path, hidden, token_ids, **extra):
    safetensors_torch.save_file(
        {"hidden_states": hidden, "token_ids": torch.tensor(token_ids), **extra},
        str(path),
    )


@pytest.mark.parametrize(
    "dtype,name", [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]
)
def test_load_reads_raw_values_and_deletes_files(tmp_path, dtype, name):
    path = tmp_path / "req.safetensors"
    hidden = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4).to(dtype)
    _write(path, hidden, [10, 11, 12])
    (tmp_path / "req.safetensors.lock").touch()

    loaded = load_hidden_state_file(str(path))

    assert loaded.token_ids == (10, 11, 12)
    assert loaded.dtype == name
    assert loaded.shape == (3, 2, 4)
    expected = hidden.view(torch.int16) if dtype == torch.bfloat16 else hidden
    assert loaded.data == expected.numpy().tobytes()
    assert not path.exists()
    assert not (tmp_path / "req.safetensors.lock").exists()


def test_load_waits_for_the_writer_lock(tmp_path):
    path = tmp_path / "req.safetensors"
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_WRONLY, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    written = threading.Event()

    def writer():
        time.sleep(0.2)
        _write(path, torch.zeros(1, 2, 4, dtype=torch.float16), [10])
        written.set()
        os.close(lock_fd)

    thread = threading.Thread(target=writer)
    thread.start()
    loaded = load_hidden_state_file(str(path))
    thread.join()
    assert written.is_set()
    assert loaded.shape == (1, 2, 4)


@pytest.mark.parametrize(
    "tensors",
    [
        {"hidden_states": torch.zeros(2, 2, 4, dtype=torch.float16)},
        {
            "hidden_states": torch.zeros(2, 8, dtype=torch.float16),
            "token_ids": torch.tensor([1, 2]),
        },
        {
            "hidden_states": torch.zeros(2, 2, 4, dtype=torch.float16),
            "token_ids": torch.tensor([1]),
        },
        {
            "hidden_states": torch.zeros(1, 2, 4, dtype=torch.float64),
            "token_ids": torch.tensor([1]),
        },
        {
            "hidden_states": torch.zeros(1, 2, 4, dtype=torch.float16),
            "token_ids": torch.tensor([1]),
            "extra": torch.zeros(1),
        },
    ],
)
def test_malformed_files_are_rejected_and_deleted(tmp_path, tensors):
    path = tmp_path / "req.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    with pytest.raises(HiddenStateFileError):
        load_hidden_state_file(str(path))
    assert not path.exists()


def test_missing_or_corrupt_file_is_a_file_error(tmp_path):
    path = tmp_path / "req.safetensors"
    with pytest.raises(HiddenStateFileError):
        load_hidden_state_file(str(path))
    path.write_bytes(b"not safetensors")
    with pytest.raises(HiddenStateFileError):
        load_hidden_state_file(str(path))
    assert not path.exists()


def test_discard_waits_for_writer_before_deleting(tmp_path):
    path = tmp_path / "req.safetensors"
    lock_fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_WRONLY, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    discard_hidden_state_file(str(path))
    time.sleep(0.1)
    # The writer creates the file after the discard was requested.
    _write(path, torch.zeros(1, 2, 4, dtype=torch.float16), [10])
    os.close(lock_fd)
    deadline = time.monotonic() + 5
    while path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not path.exists()
    assert not (tmp_path / "req.safetensors.lock").exists()
