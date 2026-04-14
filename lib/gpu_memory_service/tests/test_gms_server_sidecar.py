# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_ready_file_written_when_sockets_exist(tmp_path):
    os.environ["GMS_SOCKET_DIR"] = str(tmp_path)

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

        # Reimport to pick up patched modules and env
        import gpu_memory_service.cli.gms_sidecar_common as common_mod
        import gpu_memory_service.common.utils as utils_mod
        import gpu_memory_service.cli.gms_server_sidecar as sidecar_mod

        importlib.reload(utils_mod)
        importlib.reload(common_mod)
        importlib.reload(sidecar_mod)

        try:
            sidecar_mod.main()
        except SystemExit:
            pass

    assert ready_file.exists()


def test_ready_file_path_uses_socket_dir(tmp_path):
    os.environ["GMS_SOCKET_DIR"] = str(tmp_path)

    import gpu_memory_service.cli.gms_sidecar_common as mod
    importlib.reload(mod)

    assert mod.ready_file_path() == tmp_path / "gms-ready"
