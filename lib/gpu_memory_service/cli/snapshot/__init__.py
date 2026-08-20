# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot saver/loader CLI helpers."""

from __future__ import annotations


def has_device_argument(argv: list[str]) -> bool:
    """Return True when argv names a single ``--device``.

    ``--device-type`` is not a device selector.
    """
    return any(
        argument == "--device" or argument.startswith("--device=") for argument in argv
    )


def should_fan_out_v1(argv: list[str]) -> bool:
    """Fan out one V1 child per visible GPU unless the caller named a device."""
    if any(argument in {"-h", "--help"} for argument in argv):
        return False
    return not has_device_argument(argv)
