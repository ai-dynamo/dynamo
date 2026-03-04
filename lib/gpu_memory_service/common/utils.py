# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for GPU Memory Service."""

import os
import tempfile

import pynvml


def get_socket_path(device: int) -> str:
    """Get GMS socket path for the given CUDA device.

    The socket path is based on GPU UUID.
    If CUDA_VISIBLE_DEVICES is set, ``device`` is interpreted as the CUDA-local
    ordinal in that visibility scope.

    Args:
        device: CUDA device index.

    Returns:
        Socket path (e.g., "<tempdir>/gms_GPU-12345678-1234-1234-1234-123456789abc.sock").
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_token = None
    if visible_devices:
        tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
        if 0 <= device < len(tokens):
            visible_token = tokens[device]

    pynvml.nvmlInit()
    try:
        if visible_token is None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        elif visible_token.startswith("GPU-") or visible_token.startswith("MIG-"):
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible_token)
            except TypeError:
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible_token.encode("utf-8"))
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible_token))

        uuid = pynvml.nvmlDeviceGetUUID(handle)
    finally:
        pynvml.nvmlShutdown()

    if isinstance(uuid, bytes):
        uuid = uuid.decode("utf-8")

    return os.path.join(tempfile.gettempdir(), f"gms_{uuid}.sock")
