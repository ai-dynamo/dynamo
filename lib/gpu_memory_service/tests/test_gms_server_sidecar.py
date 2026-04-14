# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_ready_file_written_when_sockets_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))

    # Create mock socket files for 1 device, 2 tags
    for tag in ("weights", "kv_cache"):
        socket_file = tmp_path / f"gms_GPU-0000_0000_{tag}.sock"
        socket_file.touch()

    ready_file = tmp_path / "gms-ready"
    assert not ready_file.exists()

    mock_nvml = MagicMock()
    mock_nvml.nvmlInit.return_value = None
    mock_nvml.nvmlShutdown.return_value = None
    mock_nvml.nvmlDeviceGetCount.return_value = 1
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle"
    mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-0000_0000"

    with (
        patch.dict("sys.modules", {"pynvml": mock_nvml}),
        patch("subprocess.Popen") as mock_popen,
    ):
        proc = mock_popen.return_value
        proc.poll.return_value = 0  # exit immediately

        import gpu_memory_service.common.utils as utils_mod
        import gpu_memory_service.cli.server as server_mod

        importlib.reload(utils_mod)
        importlib.reload(server_mod)

        try:
            server_mod.main()
        except SystemExit:
            pass

    assert ready_file.exists()

