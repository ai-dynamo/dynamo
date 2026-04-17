# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS checkpoint saver sentinel."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]

_ROOT = Path(__file__).resolve().parent.parent
_UTILS_PATH = _ROOT / "common" / "utils.py"
_SAVER_PATH = _ROOT / "cli" / "snapshot" / "saver.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_saver(monkeypatch: pytest.MonkeyPatch, devices: list[int]):
    utils_module = _load_module("gpu_memory_service.common.utils", _UTILS_PATH)

    gpu_memory_service_package = types.ModuleType("gpu_memory_service")
    gpu_memory_service_package.__path__ = []
    common_package = types.ModuleType("gpu_memory_service.common")
    common_package.__path__ = []
    cli_package = types.ModuleType("gpu_memory_service.cli")
    cli_package.__path__ = []
    cli_snapshot_package = types.ModuleType("gpu_memory_service.cli.snapshot")
    cli_snapshot_package.__path__ = []
    snapshot_package = types.ModuleType("gpu_memory_service.snapshot")
    snapshot_package.__path__ = []

    cuda_utils_module = types.ModuleType("gpu_memory_service.common.cuda_utils")
    cuda_utils_module.list_devices = lambda: list(devices)

    storage_client_module = types.ModuleType(
        "gpu_memory_service.snapshot.storage_client"
    )

    class FakeStorageClient:
        saves: list[tuple[str, str, int, int]] = []

        def __init__(self, output_dir: str, socket_path: str, device: int):
            self.output_dir = output_dir
            self.socket_path = socket_path
            self.device = device

        def save(self, max_workers: int) -> None:
            self.saves.append(
                (self.output_dir, self.socket_path, self.device, max_workers)
            )

    storage_client_module.GMSStorageClient = FakeStorageClient

    modules = {
        "gpu_memory_service": gpu_memory_service_package,
        "gpu_memory_service.common": common_package,
        "gpu_memory_service.common.cuda_utils": cuda_utils_module,
        "gpu_memory_service.common.utils": utils_module,
        "gpu_memory_service.cli": cli_package,
        "gpu_memory_service.cli.snapshot": cli_snapshot_package,
        "gpu_memory_service.snapshot": snapshot_package,
        "gpu_memory_service.snapshot.storage_client": storage_client_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    common_package.cuda_utils = cuda_utils_module
    common_package.utils = utils_module
    cli_package.snapshot = cli_snapshot_package
    snapshot_package.storage_client = storage_client_module
    gpu_memory_service_package.common = common_package
    gpu_memory_service_package.cli = cli_package
    gpu_memory_service_package.snapshot = snapshot_package

    saver_module = _load_module("gpu_memory_service.cli.snapshot.saver", _SAVER_PATH)
    monkeypatch.setattr(saver_module, "wait_for_weights_socket", lambda _device: None)
    monkeypatch.setattr(
        saver_module, "get_socket_path", lambda device: f"/tmp/fake-gms-{device}.sock"
    )
    return saver_module, utils_module, FakeStorageClient


def test_get_checkpoint_save_complete_path_uses_socket_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    utils_module = _load_module("gms_test_utils", _UTILS_PATH)

    monkeypatch.delenv("GMS_SOCKET_DIR", raising=False)
    assert utils_module.get_checkpoint_save_complete_path() is None

    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    assert utils_module.get_checkpoint_save_complete_path() == str(
        tmp_path / "gms-checkpoint-save-complete"
    )


def test_saver_marks_checkpoint_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    saver_module, utils_module, fake_storage_client = _load_saver(monkeypatch, [0])
    checkpoint_dir = tmp_path / "checkpoint"
    socket_dir = tmp_path / "shared"
    checkpoint_dir.mkdir()
    socket_dir.mkdir()

    monkeypatch.setenv("GMS_CHECKPOINT_DIR", str(checkpoint_dir))
    monkeypatch.setenv("GMS_SOCKET_DIR", str(socket_dir))
    monkeypatch.setenv("GMS_SAVE_WORKERS", "3")
    monkeypatch.setattr(saver_module.signal, "signal", lambda *_args: None)

    def _stop_sleep(_seconds: float) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(saver_module.time, "sleep", _stop_sleep)

    with pytest.raises(SystemExit):
        saver_module.main()

    assert fake_storage_client.saves == [
        (
            str(checkpoint_dir / "device-0"),
            "/tmp/fake-gms-0.sock",
            0,
            3,
        )
    ]

    ready_path = Path(utils_module.get_checkpoint_save_complete_path())
    assert ready_path.read_text(encoding="utf-8") == "ready"
