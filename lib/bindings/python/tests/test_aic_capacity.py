# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._internal.aic import (
    _DEFAULT_NEXTN_ACCEPT_RATES,
    _NEXTN_ACCEPT_RATES_LEN,
    DEFAULT_FREE_GPU_MEMORY_FRACTION,
    DEFAULT_MEM_FRACTION_STATIC,
    _pad_nextn_accept_rates,
    estimate_num_gpu_blocks,
    resolve_backend_version,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _patch_memory(monkeypatch, return_value=123):
    """Patch aiconfigurator's unified estimator and record forwarded kwargs.

    ``estimate_num_gpu_blocks`` now delegates the budget math to
    ``aiconfigurator.sdk.memory.estimate_num_gpu_blocks`` (the single source of
    truth), so these tests assert the dynamo->AIC mapping rather than recompute
    the math themselves.
    """
    memory = pytest.importorskip("aiconfigurator.sdk.memory")
    calls = []

    def fake(model_path, system, backend, **kwargs):
        calls.append(
            {"model_path": model_path, "system": system, "backend": backend, **kwargs}
        )
        return return_value

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", fake)
    return calls


def test_estimate_num_gpu_blocks_maps_vllm_to_total_fraction(monkeypatch):
    calls = _patch_memory(monkeypatch, 45)

    blocks = estimate_num_gpu_blocks(
        backend_name="vllm",
        system="h200_sxm",
        model_path="some/model",
        tp_size=1,
        block_size=10,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.8,
    )

    assert blocks == 45
    kw = calls[0]
    assert kw["backend"] == "vllm"
    assert kw["memory_fraction_kind"] == "of_total"
    assert kw["memory_fraction_value"] == 0.8
    assert kw["scheduler_block_size"] == 10
    assert kw["max_num_tokens"] == 128
    assert kw["tp_size"] == 1


def test_estimate_num_gpu_blocks_maps_sglang_to_static_fraction(monkeypatch):
    calls = _patch_memory(monkeypatch, 37)

    blocks = estimate_num_gpu_blocks(
        backend_name="sglang",
        system="h200_sxm",
        model_path="some/model",
        tp_size=1,
        block_size=10,
        max_num_batched_tokens=128,
        mem_fraction_static=0.7,
    )

    assert blocks == 37
    assert calls[0]["memory_fraction_kind"] == "of_total"
    assert calls[0]["memory_fraction_value"] == 0.7


def test_estimate_num_gpu_blocks_sglang_defaults_static_fraction(monkeypatch):
    calls = _patch_memory(monkeypatch)

    estimate_num_gpu_blocks(
        backend_name="sglang",
        system="h200_sxm",
        model_path="some/model",
        tp_size=1,
        block_size=10,
        max_num_batched_tokens=128,
    )

    assert calls[0]["memory_fraction_value"] == DEFAULT_MEM_FRACTION_STATIC


def test_estimate_num_gpu_blocks_maps_trtllm_to_free_fraction(monkeypatch):
    calls = _patch_memory(monkeypatch, 58)

    # gpu_memory_utilization is accepted for signature parity but ignored for
    # trtllm; the default free_gpu_memory_fraction drives the budget.
    estimate_num_gpu_blocks(
        backend_name="trtllm",
        system="h200_sxm",
        model_path="some/model",
        tp_size=1,
        block_size=10,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.8,
    )
    assert calls[0]["memory_fraction_kind"] == "of_free"
    assert calls[0]["memory_fraction_value"] == DEFAULT_FREE_GPU_MEMORY_FRACTION

    calls.clear()
    estimate_num_gpu_blocks(
        backend_name="trtllm",
        system="h200_sxm",
        model_path="some/model",
        tp_size=1,
        block_size=10,
        max_num_batched_tokens=128,
        free_gpu_memory_fraction=0.8,
    )
    assert calls[0]["memory_fraction_value"] == 0.8


def test_estimate_num_gpu_blocks_forwards_parallel_mapping_and_version(monkeypatch):
    calls = _patch_memory(monkeypatch)

    estimate_num_gpu_blocks(
        backend_name="vllm",
        system="h200_sxm",
        model_path="some/model",
        tp_size=2,
        block_size=10,
        max_num_batched_tokens=128,
        backend_version="0.99.0",
        moe_tp_size=4,
        moe_ep_size=8,
        attention_dp_size=2,
    )

    kw = calls[0]
    assert kw["backend_version"] == "0.99.0"
    assert kw["tp_size"] == 2
    assert kw["moe_tp_size"] == 4
    assert kw["moe_ep_size"] == 8
    assert kw["attention_dp_size"] == 2


def test_estimate_num_gpu_blocks_rejects_unsupported_backend():
    # Validated before aiconfigurator is imported, so no patching is needed.
    with pytest.raises(ValueError, match="does not support backend"):
        estimate_num_gpu_blocks(
            backend_name="not-a-backend",
            system="h200_sxm",
            model_path="some/model",
            tp_size=1,
            block_size=10,
            max_num_batched_tokens=128,
        )


def test_estimate_num_gpu_blocks_errors_clearly_when_aic_missing(monkeypatch):
    # Estimation is opt-in; if it is reached without aiconfigurator installed,
    # fail loudly with an actionable message (not a raw ModuleNotFoundError, and
    # not a silent fallback to the default block count).
    import sys

    monkeypatch.setitem(sys.modules, "aiconfigurator.sdk", None)
    with pytest.raises(RuntimeError, match="aiconfigurator is required"):
        estimate_num_gpu_blocks(
            backend_name="vllm",
            system="h200_sxm",
            model_path="some/model",
            tp_size=1,
            block_size=10,
            max_num_batched_tokens=128,
        )


def test_trtllm_version_resolution():
    assert resolve_backend_version("trtllm", "0.20.0") == "0.20.0"


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
