# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for the multi-architecture Power Agent image."""

from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


def test_dcgm_library_copy_uses_debian_multiarch_directory():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "/usr/lib/*-linux-gnu/libdcgm.so*" in dockerfile
    assert "/usr/lib/x86_64-linux-gnu/libdcgm.so*" not in dockerfile
    assert "/usr/lib/aarch64-linux-gnu/libdcgm.so*" not in dockerfile


def test_transactional_runtime_modules_ship_in_image():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    for module in (
        "pod_report.py",
        "podresources_identity.py",
        "podresources_api.py",
        "podresources_api_grpc.py",
    ):
        assert module in dockerfile
