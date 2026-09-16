# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import random
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


def randomized_config_orders(
    configs: Sequence[dict[str, Any]], repetitions: int, seed: int
) -> list[list[dict[str, Any]]]:
    """Return seeded random orders, avoiding repeats until all orders are used."""
    if not configs:
        raise ValueError("At least one benchmark config is required")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    labels = [config["label"] for config in configs]
    if len(labels) != len(set(labels)):
        raise ValueError("Benchmark config labels must be unique")

    rng = random.Random(seed)
    # Sweep configs are intentionally small. Enumerating permutations gives a
    # reproducible sample without replacement for each complete cycle.
    permutations = list(itertools.permutations(configs))
    orders: list[list[dict[str, Any]]] = []
    while len(orders) < repetitions:
        rng.shuffle(permutations)
        orders.extend(list(order) for order in permutations)
    return orders[:repetitions]
