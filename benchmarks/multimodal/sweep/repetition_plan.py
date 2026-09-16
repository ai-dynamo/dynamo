# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def balanced_config_orders(
    configs: Sequence[dict[str, Any]], repetitions: int
) -> list[list[dict[str, Any]]]:
    """Return deterministic rotations followed by their reverse orders."""
    if not configs:
        raise ValueError("At least one benchmark config is required")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    labels = [config["label"] for config in configs]
    if len(labels) != len(set(labels)):
        raise ValueError("Benchmark config labels must be unique")

    rotations = [
        list(configs[index:]) + list(configs[:index]) for index in range(len(configs))
    ]
    balanced = rotations + [list(reversed(order)) for order in rotations]
    return [balanced[index % len(balanced)] for index in range(repetitions)]
