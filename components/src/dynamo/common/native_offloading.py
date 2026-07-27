# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime metadata for backend-native KV offloading capacity."""

from __future__ import annotations

from dynamo.common.token_capacity import (
    get_capacity_tokens,
    token_capacity_payload,
)

NATIVE_OFFLOADING_CAPACITY_RUNTIME_KEY = "native_offloading_capacity"


def native_offloading_capacity(total_tokens: object) -> dict[str, int] | None:
    """Build runtime metadata from an authoritative backend token capacity."""
    return token_capacity_payload(total_tokens)


def get_native_offloading_capacity_tokens(runtime_data: object) -> int | None:
    """Read native offloading capacity from a worker's runtime metadata."""
    return get_capacity_tokens(
        runtime_data,
        NATIVE_OFFLOADING_CAPACITY_RUNTIME_KEY,
    )
