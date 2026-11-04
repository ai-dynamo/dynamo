# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local retained GPU banks; native transfer qualification is separate."""

from types import SimpleNamespace

import pytest
import torch
from kvbm.vllm_integration import gpu_memory
from kvbm.vllm_integration.gpu_memory import GPUCacheCopyHandler, GPUCacheWorker

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.kvbm,
]


@pytest.mark.pre_merge
@pytest.mark.gpu_0
def test_rank_local_copy_still_rejects_cross_gpu_payload():
    original = SimpleNamespace(
        is_cuda=True, dtype=torch.int8, device=torch.device("cuda", 0), shape=(4, 8)
    )
    cached = SimpleNamespace(
        is_cuda=True, dtype=torch.int8, device=torch.device("cuda", 1), shape=(4, 8)
    )
    with pytest.raises(ValueError, match="one GPU"):
        GPUCacheCopyHandler([original], [cached], 1, [], True)


@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_worker_passes_rank_identity_and_uses_only_local_row_stride(
    monkeypatch, world_size
):
    opened, views, handlers = [], [], []

    class ByteTensor:
        def view(self, *args):
            return self

    def view(shape, strides, offset):
        views.append((shape, strides, offset))
        return object()

    def open_selected(*args, **kwargs):
        opened.append((args, kwargs))
        return SimpleNamespace(as_strided=view)

    def handler(*args):
        handlers.append(args)
        return object()

    monkeypatch.setattr(gpu_memory, "open_region", open_selected)
    monkeypatch.setattr(gpu_memory, "GPUCacheCopyHandler", handler)
    groups = [[SimpleNamespace(tensor_idx=0, page_size_bytes=3)]]
    caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(tensor=ByteTensor(), page_size_bytes=3),
            SimpleNamespace(tensor=ByteTensor(), page_size_bytes=5),
        ],
        group_data_refs=groups,
    )
    spec = SimpleNamespace(
        config=SimpleNamespace(
            engine_id="worker-mapping",
            parallel=SimpleNamespace(rank=world_size - 1, world_size=world_size),
        ),
        gpu_transport="cuda_ipc",
        num_chunks=7,
        row_bytes=16,
        logical_row_bytes=16 * world_size,
        blocks_per_chunk=2,
    )
    worker = GPUCacheWorker(spec, caches)
    worker._initialize()
    assert opened == [
        (
            ("worker-mapping", (7, 16)),
            {"rank": world_size - 1, "world_size": world_size},
        )
    ]
    assert views == [((7, 6), (16, 1), 0), ((7, 10), (16, 1), 6)]
    assert len(handlers) == 2
    assert all(arguments[3] is groups for arguments in handlers)
    assert [arguments[4] for arguments in handlers] == [True, False]
    spec.row_bytes = 18
    with pytest.raises(ValueError, match="does not cover"):
        GPUCacheWorker(spec, caches)._initialize()
