# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.kv_lease_client import KVLease
from gpu_memory_service.integrations.sglang import install_kv_leases as hooks


@pytest.mark.parametrize("pages", [[2, 1], [], [1, 1], [0], [3]])
def test_retention_uses_exact_cpu_page_leases(monkeypatch, pages):
    allocator = SimpleNamespace()
    sealed = []
    retained = set()
    leases = {1: KVLease(1, 4), 2: KVLease(2, 7)}
    monkeypatch.setitem(
        hooks._STATE,
        id(allocator),
        {
            "leases_by_page": leases,
            "retained_pages": retained,
            "client": SimpleNamespace(seal=lambda values: sealed.extend(values)),
        },
    )
    result = hooks.retain_hbm_pages(allocator, pages)
    if pages == [2, 1]:
        assert result == [leases[2], leases[1]]
        assert sealed == result
        assert retained == {1, 2}
    else:
        assert result == []
        assert sealed == []
        assert retained == set()


def test_failed_seal_does_not_mark_pages_retained(monkeypatch):
    allocator = SimpleNamespace()

    def fail(_leases):
        raise RuntimeError("generation changed")

    retained = set()
    monkeypatch.setitem(
        hooks._STATE,
        id(allocator),
        {
            "leases_by_page": {1: KVLease(1, 4)},
            "retained_pages": retained,
            "client": SimpleNamespace(seal=fail),
        },
    )
    with pytest.raises(RuntimeError, match="generation changed"):
        hooks.retain_hbm_pages(allocator, [1])
    assert retained == set()
