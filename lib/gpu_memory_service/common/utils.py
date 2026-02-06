# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import pynvml


def get_socket_path(device: int) -> str:
    """Get GMS socket path for the given CUDA device.

    The socket path is based on GPU UUID, making it stable across different
    CUDA_VISIBLE_DEVICES configurations.

    Args:
        device: CUDA device index.

    Returns:
        Socket path (e.g., "/tmp/gms_GPU-12345678-1234-1234-1234-123456789abc.sock").
    """
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
    finally:
        pynvml.nvmlShutdown()
    return f"/tmp/gms_{uuid}.sock"
