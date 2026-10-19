# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot loader CLI."""

import sys

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from _fake_vmm import FakeVMM

try:
    import gpu_memory_service.cli.snapshot as snapshot_cli
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


def test_list_checkpoint_devices_can_scope_to_one_device(tmp_path, monkeypatch):
    (tmp_path / "device-0").mkdir()
    (tmp_path / "device-1").mkdir()
    monkeypatch.setattr(loader, "get_vmm", lambda: FakeVMM(devices=[0, 1]))

    assert loader._list_checkpoint_devices(str(tmp_path), device=0) == [0]


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


class _KeepaliveReached(Exception):
    pass


def _stop_at_keepalive(monkeypatch):
    def fake_sleep(seconds):
        raise _KeepaliveReached

    monkeypatch.setattr(loader.time, "sleep", fake_sleep)


def test_main_forwards_process_argv_to_v1_helper(monkeypatch):
    forwarded = []

    monkeypatch.setenv("DYN_GMS_USE_V1", "true")
    monkeypatch.setattr(
        sys,
        "argv",
        ["loader.py", "--checkpoint-dir", "/tmp/cp", "--max-workers", "1"],
    )
    monkeypatch.setattr(
        loader,
        "run_per_device",
        lambda module, argv: forwarded.append((module, argv)),
    )
    _stop_at_keepalive(monkeypatch)

    with pytest.raises(_KeepaliveReached):
        loader.main()

    # These base arguments are forwarded to each per-device child, so an empty
    # list here would make each child fail on a missing --checkpoint-dir.
    assert forwarded == [
        (
            "gpu_memory_service.v1.snapshot.loader",
            ["--checkpoint-dir", "/tmp/cp", "--max-workers", "1"],
        )
    ]


def test_main_forwards_explicit_argv_unchanged(monkeypatch):
    forwarded = []

    monkeypatch.setenv("DYN_GMS_USE_V1", "true")
    monkeypatch.setattr(sys, "argv", ["loader.py", "--max-workers", "4"])
    monkeypatch.setattr(
        loader,
        "run_per_device",
        lambda module, argv: forwarded.append((module, argv)),
    )
    monkeypatch.setattr(loader, "_wait_for_termination", lambda: None)

    loader.main(argv=["--checkpoint-dir", "/tmp/cp"])

    assert forwarded == [
        ("gpu_memory_service.v1.snapshot.loader", ["--checkpoint-dir", "/tmp/cp"])
    ]


@pytest.mark.parametrize("help_flag", ["-h", "--help", "--hel", "--h"])
def test_main_v1_help_exits_without_keepalive(monkeypatch, help_flag):
    # --help runs one child that prints usage and exits 0; main() must return.
    forwarded = []

    def fake_sleep(seconds):
        raise AssertionError("keepalive must not run for --help")

    monkeypatch.setenv("DYN_GMS_USE_V1", "true")
    monkeypatch.setattr(
        loader,
        "run_per_device",
        lambda module, argv: forwarded.append((module, argv)),
    )
    monkeypatch.setattr(loader.time, "sleep", fake_sleep)

    loader.main(argv=[help_flag])

    assert forwarded == [("gpu_memory_service.v1.snapshot.loader", [help_flag])]


@pytest.mark.parametrize("help_flag", ["-h", "--help", "--hel", "--h"])
def test_run_per_device_help_starts_one_child_without_cuda(monkeypatch, help_flag):
    # Help must not initialize CUDA or print usage once per GPU.
    started = []

    class FakeProcess:
        pid = 1

        def poll(self):
            return 0

        def wait(self):
            return 0

    def fake_popen(command):
        started.append(command)
        return FakeProcess()

    def fail_init_vmm(device_type):
        raise AssertionError("help must not initialize CUDA")

    monkeypatch.setattr(snapshot_cli, "init_vmm", fail_init_vmm)
    monkeypatch.setattr(snapshot_cli.subprocess, "Popen", fake_popen)

    snapshot_cli.run_per_device("gpu_memory_service.v1.snapshot.loader", [help_flag])

    assert started == [
        [sys.executable, "-m", "gpu_memory_service.v1.snapshot.loader", help_flag]
    ]
