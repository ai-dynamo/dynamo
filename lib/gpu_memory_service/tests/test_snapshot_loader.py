# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot loader CLI."""

from types import SimpleNamespace

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from _fake_vmm import FakeVMM

try:
    from gpu_memory_service.cli import snapshot as snapshot_cli
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
    monkeypatch.setattr(loader, "get_vmm", lambda: FakeVMM(devices=[0, 2]))

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
    monkeypatch.setattr(loader, "get_vmm", lambda: FakeVMM(devices=visible_devices))

    with pytest.raises(RuntimeError, match=expected):
        loader._list_checkpoint_devices(str(tmp_path))


def test_load_device_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []
    fake_vmm = FakeVMM(devices=[3])
    fake_vmm.calls = calls  # share the calls list

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
    monkeypatch.setattr(loader, "get_vmm", lambda: fake_vmm)

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


class _ExitedProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.pid = 1

    def poll(self) -> int:
        return 0

    def terminate(self) -> None:
        return None

    def wait(self) -> int:
        return 0


def test_v1_loader_defaults_to_all_visible_devices(monkeypatch):
    started: list[list[str]] = []
    monkeypatch.setattr(snapshot_cli, "init_vmm", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        snapshot_cli, "get_vmm", lambda: SimpleNamespace(list_devices=lambda: [0, 2])
    )
    monkeypatch.setattr(
        snapshot_cli.subprocess,
        "Popen",
        lambda command: started.append(command) or _ExitedProcess(command),
    )

    loader.main(
        ["--use-v1", "--checkpoint-dir", "/ckpt", "--transfer-backend", "nixl-gds"]
    )

    assert [command[-2:] for command in started] == [
        ["--device", "0"],
        ["--device", "2"],
    ]
    assert all(
        "gpu_memory_service.v1.snapshot.loader" in command for command in started
    )


def test_v1_loader_device_flag_stays_rank_local(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda name: SimpleNamespace(main=lambda argv: calls.append((name, argv))),
    )

    loader.main(
        [
            "--use-v1",
            "--checkpoint-dir",
            "/ckpt",
            "--device",
            "1",
            "--transfer-backend",
            "nixl-gds",
        ]
    )

    assert calls == [
        (
            "gpu_memory_service.v1.snapshot.loader",
            [
                "--checkpoint-dir",
                "/ckpt",
                "--device",
                "1",
                "--transfer-backend",
                "nixl-gds",
            ],
        )
    ]
