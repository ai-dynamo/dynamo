# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._internal.aic import AicSession

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


_GIB = 1 << 30


class FakeBackend:
    def __init__(self, memory):
        self._memory = memory

    def _get_memory_usage(self, *_args, **_kwargs):
        return self._memory


class FakeModel:
    def get_kvcache_bytes_per_sequence(self, seq_len):
        return seq_len * _GIB


class FakeDatabase:
    system_spec = {"gpu": {"mem_capacity": 1000 * _GIB}}


def make_session(backend_name, memory):
    session = object.__new__(AicSession)
    session._backend_name = backend_name
    session._backend = FakeBackend(memory)
    session._database = FakeDatabase()
    session._model = FakeModel()
    return session


def test_estimate_num_gpu_blocks_uses_vllm_total_memory_fraction():
    session = make_session(
        "vllm",
        {
            "total": 400.0,
            "kvcache": 50.0,
            "weights": 300.0,
            "activations": 40.0,
            "nccl": 5.0,
            "others": 5.0,
        },
    )

    blocks = session.estimate_num_gpu_blocks(
        block_size=10,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.8,
    )

    assert blocks == 45


def test_estimate_num_gpu_blocks_uses_sglang_static_memory_fraction():
    session = make_session(
        "sglang",
        {
            "total": 900.0,
            "kvcache": 0.0,
            "weights": 300.0,
            "activations": 500.0,
            "nccl": 10.0,
            "others": 20.0,
        },
    )

    blocks = session.estimate_num_gpu_blocks(
        block_size=10,
        max_num_batched_tokens=128,
        mem_fraction_static=0.7,
    )

    assert blocks == 37
