# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.sglang.capacity import (
    kv_event_block_sizes,
    scale_kv_block_capacity,
    scale_kv_block_usage,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_kv_event_block_sizes_preserve_physical_page_size():
    server_args = SimpleNamespace(page_size=1)
    dynamo_args = SimpleNamespace(kv_event_coalescing_block_size=16)

    assert kv_event_block_sizes(server_args, dynamo_args) == (1, 16)
    assert server_args.page_size == 1


def test_kv_event_block_size_must_be_source_multiple():
    with pytest.raises(ValueError, match="must be a multiple"):
        kv_event_block_sizes(
            SimpleNamespace(page_size=4),
            SimpleNamespace(kv_event_coalescing_block_size=10),
        )


def test_kv_block_units_use_floor_capacity_and_ceiling_usage():
    assert scale_kv_block_capacity(33, 1, 16) == 2
    assert scale_kv_block_usage(33, 1, 16) == 3
