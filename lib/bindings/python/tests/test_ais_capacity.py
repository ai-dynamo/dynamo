# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo._internal.ais import (
    _DEFAULT_NEXTN_ACCEPT_RATES,
    _NEXTN_ACCEPT_RATES_LEN,
    _normalize_quant_mode,
    _pad_nextn_accept_rates,
    _resolve_quant_mode,
    resolve_backend_version,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_default_version_uses_queryable_slot(backend):
    assert resolve_backend_version(backend, None) == "current"


def test_trtllm_version_resolution():
    assert resolve_backend_version("trtllm", "0.20.0") == "0.20.0"


def test_pad_nextn_accept_rates_defaults_when_omitted():
    # Omitted/empty input preserves Dynamo's historical default, not all zeros.
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
        # `int4` is the user-facing alias; AIS's quant-mode name is `int4_wo`.
        ("int4", "int4_wo"),
        # Already-canonical names pass through (only stripped).
        ("fp8", "fp8"),
        ("fp8_block", "fp8_block"),
        ("w4a16_mxfp4", "w4a16_mxfp4"),
        ("  fp8  ", "fp8"),
    ],
)
def test_normalize_quant_mode(value, expected):
    assert _normalize_quant_mode(value) == expected


def test_resolve_quant_mode_per_field():
    common = pytest.importorskip("aisimulate_core.sdk.common")

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
    pytest.importorskip("aisimulate_core.sdk.common")

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


class _CanonicalEstimator:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def diagnostics(self):
        return {"readiness": "ready", "provenance": {"config": self.config}}

    def estimate_forward_pass_time_ms(self, fpm):
        self.calls.append(fpm["scheduled_requests"])
        return 3.0


@pytest.fixture
def canonical_estimator(monkeypatch):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    created = []

    def create(config):
        model = _CanonicalEstimator(config)
        created.append(model)
        return model

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", create)
    return created


def test_session_passes_full_canonical_config_and_cached_prefix(canonical_estimator):
    from dynamo._internal.ais import AisSession

    config = {
        "model": "model",
        "system": "gpu",
        "backend": "vllm",
        "worker_type": "aggregated",
        "systems_paths": ["first", "second"],
        "estimator_config": {"correction": {"enabled": False}},
    }
    session = AisSession(config=config)
    assert canonical_estimator[0].config == config
    assert session.predict_prefill(2, 128, 512) == 3
    assert canonical_estimator[0].calls == [
        {
            "num_prefill_requests": 2,
            "sum_prefill_tokens": 256,
            "sum_prefill_kv_tokens": 1024,
        }
    ]


@pytest.mark.parametrize(
    "cost",
    [
        {"nextn": 2},
        {
            "nextn": 0,
            "speculation": {"kind": "ngram", "params": {"num_speculative_tokens": 2}},
        },
    ],
)
def test_session_decode_preserves_stride_kv_and_verification_width(
    canonical_estimator, cost
):
    from dynamo._internal.ais import AisSession

    session = AisSession(config={"model": "model", **cost})
    assert session.predict_decode(2, 100, 35) == 3 * 34
    assert canonical_estimator[0].calls == [
        {"num_decode_requests": 6, "sum_decode_kv_tokens": 6 * 101},
        {"num_decode_requests": 6, "sum_decode_kv_tokens": 6 * 133},
    ]


def test_session_rejects_cold_regression(monkeypatch):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    from dynamo._internal.ais import AisSession

    class Cold(_CanonicalEstimator):
        def diagnostics(self):
            return {"readiness": "insufficient_data"}

    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", Cold)
    with pytest.raises(ValueError, match="cold fpm_regression"):
        AisSession(config={"model": "model", "estimation_mode": "fpm_regression"})


@pytest.mark.parametrize("nextn,attention_dp", [(0, 1), (2, 1), (0, 2), (2, 2)])
def test_canonical_session_matches_native_static_query_oracle(nextn, attention_dp):
    # Only packaged model metadata/performance tables are loaded, no weights or GPU.
    from aisimulate_core.sdk.engine import EngineHandle

    from dynamo._internal.ais import AisSession

    payload = {
        "model": "Qwen/Qwen3-32B",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "current",
        "worker_type": "aggregated",
        "estimation_mode": "op_level",
        "nextn": nextn,
        "attention_dp": attention_dp,
    }
    session = AisSession(config=payload)
    oracle = EngineHandle.compile(
        model_path=payload["model"],
        system=payload["system"],
        backend=payload["backend"],
        backend_version="current",
        nextn=nextn,
        attention_dp_size=attention_dp,
    )
    for batch, new_tokens, prefix in [(1, 128, 0), (2, 128, 256)]:
        assert session.predict_prefill(batch, new_tokens, prefix) == pytest.approx(
            oracle.predict_prefill_latency(batch, new_tokens + prefix, prefix)
        )
    for batch, context, output in [(4, 1024, 2), (2, 512, 35)]:
        assert session.predict_decode(batch, context, output) == pytest.approx(
            oracle.predict_decode_latency(batch, context, output)
        )


def test_canonical_capacity_forwards_scheduler_options(monkeypatch):
    import aisimulate.aic

    from dynamo._internal.ais import estimate_canonical_num_gpu_blocks

    calls = []

    def materialize(raw):
        calls.append(raw)
        return {**raw, "num_gpu_blocks": 123}

    monkeypatch.setattr(aisimulate.aic, "materialize_aic_num_gpu_blocks", materialize)
    config = {
        "model": "model",
        "system": "h200_sxm",
        "backend": "vllm",
        "worker_type": "aggregated",
        "systems_paths": ["first", "second"],
    }
    assert (
        estimate_canonical_num_gpu_blocks(config, block_size=64, max_num_seqs=128)
        == 123
    )
    assert calls == [
        {
            "block_size": 64,
            "max_num_seqs": 128,
            "timing_model": {
                "type": "external",
                "provider": "aic",
                "config": {
                    **config,
                    "estimation_mode": "auto",
                    "fallback_policy": "deny",
                },
            },
        }
    ]
