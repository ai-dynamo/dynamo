# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prefix cache's host tier copies on a GPU: pinned rows and batched
GPU<->host block copies (``cuMemcpyBatchAsync``, or one copy per block)."""

import pytest

from dynamo.deepinfer.kv_cache.host import HostBlockPool

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.mark.parametrize("batched", [True, False], ids=["batch", "per-block"])
def test_blocks_round_trip_through_host_rows(batched):
    # DeepSeek-V4-Flash's packed KV block row.
    gpu = torch.randint(0, 256, (64, 114048), dtype=torch.uint8, device="cuda")
    original = gpu.clone()
    pool = HostBlockPool(gpu, 16)
    if not batched:
        pool._batch = None
    elif pool._batch is None:
        pytest.skip("cuMemcpyBatchAsync unavailable")
    pool.store([3, 7, 9, 40], [0, 5, 2, 15])
    gpu.zero_()
    pool.load([0, 5, 2, 15], [10, 11, 12, 13])
    torch.cuda.synchronize()
    for src, dst in zip([3, 7, 9, 40], [10, 11, 12, 13]):
        assert torch.equal(gpu[dst], original[src])
    assert not gpu[:10].any()


def test_pinned_rows_hold_snapshots_across_chunks(monkeypatch):
    from dynamo.deepinfer.kv_cache import host

    monkeypatch.setattr(host, "CHUNK_BYTES", 2 * 4096)
    states = torch.randint(0, 256, (4, 8192), dtype=torch.uint8, device="cuda")
    rows = host.PinnedRows(states[:1, :4096], 5)  # 3 chunks of up to 2 rows
    assert rows.count == 5 and len(rows.chunks) == 3
    assert all(chunk.is_pinned() for chunk in rows.chunks)
    for slot in range(5):
        rows.row(slot).copy_(states[slot % 4, :4096], non_blocking=True)
    torch.cuda.synchronize()
    for slot in range(5):
        assert torch.equal(rows.row(slot).cuda(), states[slot % 4, :4096])
