# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from gpu_memory_service.common import persistent_pool

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_backoff_is_capped_to_remaining_budget(monkeypatch):
    now = 0.0
    sleeps = []
    attempts = []

    def sleep(delay):
        nonlocal now
        sleeps.append(delay)
        now += delay

    def busy():
        attempts.append(now)
        raise RuntimeError("busy")

    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", "0.12")
    monkeypatch.setattr(persistent_pool.time, "monotonic", lambda: now)
    monkeypatch.setattr(persistent_pool.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="busy"):
        persistent_pool.retry_persistent_claim(busy, lambda _: True)
    assert sleeps == pytest.approx([0.05, 0.07])
    assert attempts == pytest.approx([0, 0.05, 0.12])


@pytest.mark.parametrize("budget", ["nan", "inf", "-1", "invalid"])
def test_invalid_retry_budget_fails_before_rpc(monkeypatch, budget):
    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", budget)

    def unexpected():
        pytest.fail("must validate configuration before claiming")

    with pytest.raises(ValueError):
        persistent_pool.retry_persistent_claim(unexpected, lambda _: True)


@pytest.mark.parametrize("budget,is_busy", [("0", True), ("2", False)])
def test_zero_budget_and_permanent_errors_do_not_sleep(monkeypatch, budget, is_busy):
    monkeypatch.setenv("GMS_PERSISTENT_CLAIM_RETRY_SECS", budget)
    monkeypatch.setattr(
        persistent_pool.time, "sleep", lambda _: pytest.fail("no retry")
    )

    def fail():
        raise RuntimeError("refused")

    with pytest.raises(RuntimeError, match="refused"):
        persistent_pool.retry_persistent_claim(fail, lambda _: is_busy)
