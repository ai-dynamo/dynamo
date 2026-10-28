# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Media-neutral sampling of per-request slots from a fixed pool."""

import random


def sample_pool_slots(
    py_rng: random.Random,
    pool: list[str],
    num_requests: int,
    items_per_request: int,
) -> list[str]:
    """Sample slots from a fixed pool, no duplicates within each request.

    Every item in the pool is guaranteed to appear at least once. Callers must
    validate first that ``len(pool) >= items_per_request`` and that
    ``num_requests * items_per_request >= len(pool)``.
    """
    # Round-robin every pool item into requests so each appears at least once
    shuffled = list(pool)
    py_rng.shuffle(shuffled)
    requests: list[list[str]] = [[] for _ in range(num_requests)]
    for i, item in enumerate(shuffled):
        requests[i % num_requests].append(item)

    # Fill remaining slots with random pool samples (no intra-request duplicates)
    for req in requests:
        remaining = items_per_request - len(req)
        if remaining > 0:
            used = set(req)
            available = [item for item in pool if item not in used]
            req.extend(py_rng.sample(available, remaining))
        py_rng.shuffle(req)

    return [item for req in requests for item in req]
