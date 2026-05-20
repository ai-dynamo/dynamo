# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS snapshot loader CLI."""

import pytest

try:
    from gpu_memory_service.cli.snapshot import loader
    from gpu_memory_service.common.vmm import VMMDeviceType
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


class _FakeVMM:
    """Minimal VMMDevice mock for unit tests."""

    def __init__(self, devices):
        self._devices = devices
        self.calls = []

    def ensure_initialized(self):
        pass

    def list_devices(self):
        return self._devices

    def runtime_set_device(self, device):
        self.calls.append(("set_device", device))


def test_list_checkpoint_devices_requires_exact_visible_device_match(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "device-2").mkdir()
    (tmp_path / "device-0").mkdir()
    (tmp_path / "device-0-copy").mkdir()
    (tmp_path / "not-a-device").mkdir()
    (tmp_path / "device-1").write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(loader, "get_vmm_device", lambda _kind: _FakeVMM([0, 2]))

    assert loader._list_checkpoint_devices(str(tmp_path), VMMDeviceType.CUDA) == [0, 2]


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
    monkeypatch.setattr(
        loader, "get_vmm_device", lambda _kind: _FakeVMM(visible_devices)
    )

    with pytest.raises(RuntimeError, match=expected):
        loader._list_checkpoint_devices(str(tmp_path), VMMDeviceType.CUDA)


def test_load_device_sets_cuda_context_before_storage_client(monkeypatch):
    calls = []
    fake_vmm = _FakeVMM([3])
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
    monkeypatch.setattr(loader, "get_vmm_device", lambda _kind: fake_vmm)

    loader._load_device(
        "/checkpoints/run/versions/1",
        3,
        16,
        "nixl",
        [],
        2,
        VMMDeviceType.CUDA,
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
