# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import builtins
import types

import pytest

from dynamo._internal.aic import (
    _DEFAULT_NEXTN_ACCEPT_RATES,
    _NEXTN_ACCEPT_RATES_LEN,
    DEFAULT_FREE_GPU_MEMORY_FRACTION,
    DEFAULT_MEM_FRACTION_STATIC,
    AicMemoryEstimatorUnavailableError,
    _normalize_aic_quant_mode,
    _pad_nextn_accept_rates,
    _resolve_quant_mode,
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
    """Patch aiconfigurator-core's unified estimator and record forwarded kwargs.

    ``estimate_num_gpu_blocks`` now delegates the budget math to
    ``aiconfigurator_core.sdk.memory.estimate_num_gpu_blocks`` (the single source of
    truth), so these tests assert the dynamo->AIC mapping rather than recompute
    the math themselves.
    """
    memory = pytest.importorskip("aiconfigurator_core.sdk.memory")
    calls = []

    def fake(model_path, system, backend, **kwargs):
        calls.append(
            {"model_path": model_path, "system": system, "backend": backend, **kwargs}
        )
        return return_value

    monkeypatch.setattr(memory, "estimate_num_gpu_blocks", fake)
    return calls


def test_runtime_loader_does_not_import_upper_aiconfigurator(monkeypatch):
    """The mocker/runtime path must remain usable with only the core wheel."""
    pytest.importorskip("aiconfigurator_core")
    import dynamo._internal.aic as aic_mod

    real_import = builtins.__import__

    def reject_upper_package(name, *args, **kwargs):
        """Reject accidental imports of the upper AIC distribution."""
        if name == "aiconfigurator" or name.startswith("aiconfigurator."):
            raise AssertionError(f"runtime imported upper package: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_upper_package)
    loaded = aic_mod._load_aiconfigurator()

    assert set(loaded) == {
        "config",
        "get_backend",
        "get_model",
        "get_database",
        "get_supported_databases",
    }


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


def test_estimate_num_gpu_blocks_reports_unavailable_estimator(monkeypatch):
    # The low-level helper reports a typed, actionable error so mocker can apply
    # its fallback without masking unrelated estimator failures.
    def missing_memory(module_name):
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr("dynamo._internal.aic.importlib.import_module", missing_memory)

    with pytest.raises(
        AicMemoryEstimatorUnavailableError,
        match=r"aiconfigurator\.sdk\.memory is required",
    ):
        estimate_num_gpu_blocks(
            backend_name="vllm",
            system="h200_sxm",
            model_path="some/model",
            tp_size=1,
            block_size=10,
            max_num_batched_tokens=128,
        )


def test_estimate_num_gpu_blocks_propagates_transitive_import_error(monkeypatch):
    missing_dependency = ModuleNotFoundError(name="aiconfigurator_core")

    def broken_memory_module(_module_name):
        raise missing_dependency

    monkeypatch.setattr(
        "dynamo._internal.aic.importlib.import_module", broken_memory_module
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        estimate_num_gpu_blocks(
            backend_name="vllm",
            system="h200_sxm",
            model_path="some/model",
            tp_size=1,
            block_size=10,
            max_num_batched_tokens=128,
        )

    assert exc_info.value is missing_dependency


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


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("auto", None),
        ("None", None),
        ("NULL", None),
        ("  AUTO  ", None),
        # `int4` is the user-facing alias; AIC's quant-mode name is `int4_wo`.
        ("int4", "int4_wo"),
        # Already-canonical names pass through (only stripped).
        ("fp8", "fp8"),
        ("fp8_block", "fp8_block"),
        ("w4a16_mxfp4", "w4a16_mxfp4"),
        ("  fp8  ", "fp8"),
    ],
)
def test_normalize_aic_quant_mode(value, expected):
    # Single source of truth for the AIC quant-mode vocabulary: shared by the
    # KV-block estimator (`estimate_num_gpu_blocks`), the replay arg resolver,
    # and the Rust latency-engine build path.
    assert _normalize_aic_quant_mode(value) == expected


def test_estimate_num_gpu_blocks_forwards_normalized_quant_modes(monkeypatch):
    # The dynamo dtype overrides must reach AIC normalized: `int4` -> `int4_wo`
    # and `auto` -> None (AIC's "use the model default" sentinel).
    calls = _patch_memory(monkeypatch)

    estimate_num_gpu_blocks(
        backend_name="vllm",
        system="h200_sxm",
        model_path="some/model",
        tp_size=1,
        block_size=10,
        max_num_batched_tokens=128,
        gemm_dtype="int4",
        moe_dtype="w4a16_mxfp4",
        fmha_dtype="auto",
        kv_cache_dtype="fp8",
        comm_dtype="fp8",
    )

    kw = calls[0]
    assert kw["gemm_quant_mode"] == "int4_wo"
    assert kw["moe_quant_mode"] == "w4a16_mxfp4"
    assert kw["fmha_quant_mode"] is None
    assert kw["kvcache_quant_mode"] == "fp8"
    assert kw["comm_quant_mode"] == "fp8"


def test_resolve_quant_mode_per_field():
    common = pytest.importorskip("aiconfigurator_core.sdk.common")

    assert _resolve_quant_mode("gemm", "int4") == common.GEMMQuantMode.int4_wo
    assert _resolve_quant_mode("gemm", "fp8") == common.GEMMQuantMode.fp8
    assert _resolve_quant_mode("moe", "w4a16_mxfp4") == common.MoEQuantMode.w4a16_mxfp4
    assert _resolve_quant_mode("fmha", "bfloat16") == common.FMHAQuantMode.bfloat16
    assert _resolve_quant_mode("kvcache", "fp8") == common.KVCacheQuantMode.fp8
    assert _resolve_quant_mode("comm", "fp8") == common.CommQuantMode.fp8
    # "auto"/None mean "use the model default".
    assert _resolve_quant_mode("gemm", "auto") is None
    assert _resolve_quant_mode("kvcache", None) is None


def test_resolve_quant_mode_rejects_unsupported_per_field():
    pytest.importorskip("aiconfigurator_core.sdk.common")

    # `int4` -> `int4_wo` is valid for GEMM/MoE but not for KV cache or FMHA,
    # which have narrower vocabularies. The error must name the field and the
    # allowed values rather than surfacing a bare KeyError from aiconfigurator.
    with pytest.raises(ValueError, match="kvcache quant mode"):
        _resolve_quant_mode("kvcache", "int4")
    with pytest.raises(ValueError, match="fmha quant mode"):
        _resolve_quant_mode("fmha", "int4")
    with pytest.raises(ValueError, match="comm quant mode"):
        _resolve_quant_mode("comm", "int4")
    with pytest.raises(ValueError, match="supported values"):
        _resolve_quant_mode("gemm", "not-a-dtype")


def test_aic_session_forwards_quant_modes_to_model_config(monkeypatch):
    common = pytest.importorskip("aiconfigurator_core.sdk.common")
    import dynamo._internal.aic as aic_mod

    captured: dict = {}

    class FakeModelConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_model = types.SimpleNamespace(
        model_name="m", context_ops=[], generation_ops=[], _nextn=0
    )
    fake = {
        "config": types.SimpleNamespace(ModelConfig=FakeModelConfig),
        "get_database": lambda system, backend, version: object(),
        "get_supported_databases": lambda: {},
        "get_model": lambda model_path, model_config, backend_name: fake_model,
        "get_backend": lambda backend_name: object(),
    }
    monkeypatch.setattr(aic_mod, "_load_aiconfigurator", lambda: fake)
    # Skip the optional compiled-engine build (it would import aiconfigurator's
    # AIC-core Rust engine step); we only care about the ModelConfig wiring here.
    monkeypatch.setenv("DYNAMO_AIC_DISABLE_COMPILED_ENGINE", "1")

    aic_mod.AicSession(
        backend_name="vllm",
        system="h200_sxm",
        model_path="m",
        tp_size=1,
        gemm_dtype="int4",
        moe_dtype="w4a16_mxfp4",
        fmha_dtype="fp8",
        kv_cache_dtype="auto",  # -> omitted, ModelConfig keeps its default
        comm_dtype="fp8",
    )

    assert captured["gemm_quant_mode"] == common.GEMMQuantMode.int4_wo
    assert captured["moe_quant_mode"] == common.MoEQuantMode.w4a16_mxfp4
    assert captured["fmha_quant_mode"] == common.FMHAQuantMode.fp8
    assert "kvcache_quant_mode" not in captured
    assert captured["comm_quant_mode"] == common.CommQuantMode.fp8
