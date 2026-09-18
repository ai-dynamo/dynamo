# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.gpu_memory_service.conftest import (
    _descendant_identities,
    _live_seen_identities,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_leak_guard_keeps_reparented_descendant_identity():
    initial = {
        10: (1, 100),
        20: (10, 200),
        30: (20, 300),
        40: (1, 400),
    }
    seen = _descendant_identities(10, initial)
    assert seen == {10: 100, 20: 200, 30: 300}

    # The launcher and intermediate child exited. PID 30 is now adopted by
    # init, but its birth tick proves it is the same leaked process.
    final = {30: (1, 300), 40: (1, 400)}
    assert _live_seen_identities(seen, final) == {30}


def test_leak_guard_ignores_reused_pid():
    seen = {30: 300}
    assert _live_seen_identities(seen, {30: (1, 999)}) == set()
