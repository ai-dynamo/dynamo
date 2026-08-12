# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Marker categories every test must declare one of.

Shared by tests/conftest.py, which defaults them onto unmarked tests, and
tests/report_pytest_markers.py, which fails when a test declares none.
Keeping one definition means the default and the gate cannot drift apart.
"""

REQUIRED_CATEGORIES: dict[str, frozenset[str]] = {
    "Lifecycle": frozenset({"pre_merge", "post_merge", "nightly", "weekly", "release"}),
    "Test Type": frozenset(
        {
            "unit",
            "integration",
            "e2e",
            "benchmark",
            "stress",
            "multimodal",
            "performance",
        }
    ),
    "Hardware": frozenset(
        {
            "gpu_0",
            "gpu_1",
            "gpu_2",
            "gpu_4",
            "gpu_8",
            "h100",
            "k8s",
            "xpu_1",
            "xpu_2",
        }
    ),
}
