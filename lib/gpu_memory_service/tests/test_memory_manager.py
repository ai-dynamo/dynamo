# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

from gpu_memory_service.client import memory_manager


def test_memory_manager_initializes_cuda_before_selecting_device() -> None:
    calls: list[object] = []

    with mock.patch.object(
        memory_manager,
        "cuda_ensure_initialized",
        side_effect=lambda: calls.append("init"),
    ):
        with mock.patch.object(
            memory_manager,
            "cuda_set_current_device",
            side_effect=lambda device: calls.append(("set", device)),
        ):
            with mock.patch.object(
                memory_manager,
                "cumem_get_allocation_granularity",
                side_effect=lambda device: calls.append(("granularity", device))
                or 2097152,
            ):
                manager = memory_manager.GMSClientMemoryManager(
                    "/tmp/gms.sock",
                    device=3,
                )

    assert manager.granularity == 2097152
    assert calls == ["init", ("set", 3), ("granularity", 3)]
