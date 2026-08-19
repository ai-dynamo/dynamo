# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def resolve_sglang_disaggregated_devices(
    cuda_visible_devices: str | None,
) -> tuple[str, str]:
    """Return the decode and prefill tokens for a two-GPU SGLang deployment."""
    if cuda_visible_devices is None:
        devices = ("0", "1")
    else:
        devices = tuple(
            device.strip()
            for device in cuda_visible_devices.split(",")
            if device.strip()
        )

    if len(devices) < 2:
        configured = cuda_visible_devices if cuda_visible_devices is not None else "0,1"
        raise ValueError(
            "SGLang disaggregated cancellation requires at least two entries in "
            f"CUDA_VISIBLE_DEVICES; got {configured!r}"
        )

    return devices[0], devices[1]
