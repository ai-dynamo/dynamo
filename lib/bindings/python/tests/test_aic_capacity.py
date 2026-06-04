# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._internal.aic import (
    _DEFAULT_NEXTN_ACCEPT_RATES,
    _NEXTN_ACCEPT_RATES_LEN,
    AicSession,
    _pad_nextn_accept_rates,
    resolve_backend_version,
)

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
    def __init__(self):
        self.system_spec = {"gpu": {"mem_capacity": 1000 * _GIB}}


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


def test_trtllm_version_resolution():
    assert resolve_backend_version("trtllm", "0.20.0") == "0.20.0"


def test_estimate_num_gpu_blocks_uses_trtllm_free_memory_fraction():
    # TRT-LLM now supports KV capacity estimation (it previously raised). It
    # applies free_gpu_memory_fraction to the memory left *after* the model
    # footprint, unlike vLLM's gpu_memory_utilization (a fraction of total):
    #   non_kv = total - kvcache       = 400 - 50   = 350 GiB
    #   free   = capacity - non_kv     = 1000 - 350  = 650 GiB
    #   blocks = floor(free * fraction / block_bytes)
    session = make_session(
        "trtllm",
        {
            "total": 400.0,
            "kvcache": 50.0,
            "weights": 300.0,
            "activations": 40.0,
            "nccl": 5.0,
            "others": 5.0,
        },
    )

    # Default free_gpu_memory_fraction (0.9): floor(650 * 0.9 / 10) = 58.
    # gpu_memory_utilization is accepted for signature parity but ignored here.
    assert (
        session.estimate_num_gpu_blocks(
            block_size=10,
            max_num_batched_tokens=128,
            gpu_memory_utilization=0.8,
        )
        == 58
    )

    # Explicit free_gpu_memory_fraction drives the budget: floor(650 * 0.8 / 10) = 52.
    assert (
        session.estimate_num_gpu_blocks(
            block_size=10,
            max_num_batched_tokens=128,
            free_gpu_memory_fraction=0.8,
        )
        == 52
    )


def test_pad_nextn_accept_rates_defaults_when_omitted():
    # Omitted/empty input falls back to AIC's CLI default, not all zeros.
    assert _pad_nextn_accept_rates(None) == _DEFAULT_NEXTN_ACCEPT_RATES
    assert _pad_nextn_accept_rates("") == _DEFAULT_NEXTN_ACCEPT_RATES
    assert _pad_nextn_accept_rates([]) == _DEFAULT_NEXTN_ACCEPT_RATES


def test_pad_nextn_accept_rates_pads_and_truncates():
    assert _pad_nextn_accept_rates([0.9, 0.4]) == [0.9, 0.4, 0.0, 0.0, 0.0]
    assert _pad_nextn_accept_rates("0.9,0.4") == [0.9, 0.4, 0.0, 0.0, 0.0]
    assert _pad_nextn_accept_rates([0.1] * 7) == [0.1] * _NEXTN_ACCEPT_RATES_LEN


@pytest.mark.parametrize(
    "bad",
    ["0.85,abc,0", [1.5, 0.0], [-0.1, 0.0], [float("nan")], [float("inf")]],
)
def test_pad_nextn_accept_rates_rejects_invalid(bad):
    with pytest.raises(ValueError):
        _pad_nextn_accept_rates(bad)
