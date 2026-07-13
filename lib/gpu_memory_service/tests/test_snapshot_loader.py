# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot loader CLI."""

import json
import threading

import pytest

try:
    from gpu_memory_service.cli.snapshot import loader
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_list_checkpoint_devices_requires_exact_visible_device_match(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "device-2").mkdir()
    (tmp_path / "device-0").mkdir()
    (tmp_path / "device-0-copy").mkdir()
    (tmp_path / "not-a-device").mkdir()
    (tmp_path / "device-1").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(loader.cuda_utils, "list_devices", lambda: [0, 2])

    assert loader._list_checkpoint_devices(str(tmp_path)) == [0, 2]


@pytest.mark.parametrize(
    ("visible_devices", "checkpoint_dirs", "expected"),
    [
        ([0, 1], ["device-0"], "missing=1"),
        ([0], ["device-0", "device-1"], "extra=1"),
        ([7], [], "missing=7"),
        ([2], ["device-02"], "missing=2"),
    ],
)
def test_list_checkpoint_devices_rejects_mismatched_checkpoints(
    tmp_path,
    monkeypatch,
    visible_devices,
    checkpoint_dirs,
    expected,
):
    for dirname in checkpoint_dirs:
        (tmp_path / dirname).mkdir()
    monkeypatch.setattr(loader.cuda_utils, "list_devices", lambda: visible_devices)

    with pytest.raises(RuntimeError, match=expected):
        loader._list_checkpoint_devices(str(tmp_path))


def test_load_device_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []

    class FakeStorageClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def load_to_gms(self, input_dir, *, max_workers, clear_existing):
            calls.append(
                (
                    "load_to_gms",
                    {
                        "input_dir": input_dir,
                        "max_workers": max_workers,
                        "clear_existing": clear_existing,
                    },
                )
            )

    monkeypatch.setattr(loader, "get_socket_path", lambda device: f"/tmp/gms-{device}")
    monkeypatch.setattr(loader, "GMSStorageClient", FakeStorageClient)
    monkeypatch.setattr(
        loader.cuda_utils,
        "cuda_runtime_set_device",
        lambda device: calls.append(("set_device", device)),
    )

    loader._load_device(
        "/checkpoints/run/versions/1",
        3,
        16,
        "nixl",
        [],
        2,
    )

    assert calls[0] == ("set_device", 3)
    assert calls[1][0] == "init"
    assert calls[1][1]["socket_path"] == "/tmp/gms-3"
    assert calls[1][1]["device"] == 3
    assert calls[2] == (
        "load_to_gms",
        {
            "input_dir": "/checkpoints/run/versions/1/device-3",
            "max_workers": 16,
            "clear_existing": True,
        },
    )


def test_file_transfer_gate_waits_for_all_devices_and_release(tmp_path):
    ready_file = tmp_path / "transfer-ready"
    release_file = tmp_path / "transfer-release"
    gate = loader._FileTransferGate(
        ready_file=str(ready_file),
        release_file=str(release_file),
        participants=[0, 1],
        timeout_s=1.0,
    )
    completed: list[int] = []

    threads = [
        threading.Thread(
            target=lambda device=device: (
                gate.wait(device),
                completed.append(device),
            )
        )
        for device in [0, 1]
    ]
    threads[0].start()
    assert not ready_file.exists()
    threads[1].start()

    for _ in range(100):
        if ready_file.exists():
            break
        threading.Event().wait(0.01)
    assert json.loads(ready_file.read_text(encoding="utf-8"))["participants"] == [
        0,
        1,
    ]
    assert completed == []

    release_file.write_text("release\n", encoding="utf-8")
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    assert sorted(completed) == [0, 1]


def test_file_transfer_gate_times_out_without_release(tmp_path):
    gate = loader._FileTransferGate(
        ready_file=str(tmp_path / "transfer-ready"),
        release_file=str(tmp_path / "transfer-release"),
        participants=[0],
        timeout_s=0.01,
    )

    with pytest.raises(TimeoutError, match="transfer release marker"):
        gate.wait(0)


def test_loader_rejects_transfer_gate_for_non_posix_backend(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "device-0").mkdir()
    monkeypatch.setattr(loader.cuda_utils, "list_devices", lambda: [0])

    with pytest.raises(SystemExit):
        loader.main(
            [
                "--checkpoint-dir",
                str(tmp_path),
                "--transfer-backend",
                "nixl-gds",
                "--experiment-transfer-ready-file",
                str(tmp_path / "ready"),
                "--experiment-transfer-release-file",
                str(tmp_path / "release"),
            ]
        )
