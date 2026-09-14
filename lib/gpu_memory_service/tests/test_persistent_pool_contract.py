# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gpu_memory_service.common.persistent_pool import (
    PersistentPoolAllocation,
    PersistentPoolKey,
)


@pytest.mark.parametrize("engine_id,tag", [("", "kv_pool"), ("engine-a", "")])
def test_persistent_pool_key_requires_both_identity_parts(engine_id, tag):
    with pytest.raises(ValueError):
        PersistentPoolKey(engine_id, tag)


def test_allocation_exposes_legacy_inventory_identity():
    allocation = PersistentPoolAllocation(
        key=PersistentPoolKey("engine-a", "kv_pool"),
        allocation_id="allocation-1",
        size=6000,
        aligned_size=8192,
    )

    assert allocation.engine_id == "engine-a"
    assert allocation.tag == "kv_pool"
