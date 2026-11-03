# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.strict_rl_config import (
    validate_strict_rl_model,
    validate_strict_rl_profile,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.timeout(20),
]


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_moe", True),
        ("is_quantized", True),
        ("quantization", "fp8"),
        ("dtype", torch.float16),
    ],
)
def test_strict_model_rejects_unsupported_profiles(field, value):
    model = SimpleNamespace(
        is_moe=False, is_quantized=False, quantization=None, dtype=torch.bfloat16
    )
    setattr(model, field, value)
    with pytest.raises(ValueError, match="dense BF16"):
        validate_strict_rl_model(model)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("enable_lora", True),
        ("speculative_config", {}),
        ("weight_transfer_config", None),
        ("weight_transfer_config", SimpleNamespace(backend="unknown")),
        ("generation_config", "auto"),
        ("override_generation_config", {"temperature": 0.5}),
        ("quantization", "fp8"),
        ("kv_cache_dtype", "fp8"),
        ("kv_cache_dtype", "float16"),
        ("load_format", "gms"),
    ],
)
def test_strict_profile_rejects_unsupported_engine_configuration(
    supported_strict_profile, field, value
):
    config, engine = supported_strict_profile
    setattr(engine, field, value)
    with pytest.raises(ValueError, match="strict RL"):
        validate_strict_rl_profile(config, engine)


@pytest.mark.parametrize(
    "role",
    [
        "embedding_worker",
        "classify_worker",
        "headless",
        "use_vllm_tokenizer",
        "enable_multimodal",
        "gms_shadow_mode",
    ],
)
def test_strict_profile_rejects_unsupported_worker_configuration(
    supported_strict_profile, role
):
    config, engine = supported_strict_profile
    setattr(config, role, True)
    with pytest.raises(ValueError, match="strict RL"):
        validate_strict_rl_profile(config, engine)


@pytest.fixture
def supported_strict_profile():
    config = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.AGGREGATED,
        embedding_worker=False,
        classify_worker=False,
        headless=False,
        use_vllm_tokenizer=False,
        enable_multimodal=False,
        gms_shadow_mode=False,
        benchmark_mode=None,
    )
    engine = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        enable_lora=False,
        speculative_config=None,
        quantization=None,
        weight_transfer_config=SimpleNamespace(backend="ipc"),
        generation_config="vllm",
        override_generation_config={},
        kv_cache_dtype="auto",
        load_format="auto",
    )
    return config, engine


def test_strict_profile_rejects_benchmark_mode(supported_strict_profile):
    config, engine = supported_strict_profile
    config.benchmark_mode = "decode"
    with pytest.raises(ValueError, match="strict RL.*benchmark"):
        validate_strict_rl_profile(config, engine)


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
def test_strict_profile_accepts_supported_configuration(
    monkeypatch, supported_strict_profile, cache_dtype
):
    monkeypatch.delenv("DYN_GMS_USE_V1", raising=False)
    config, engine = supported_strict_profile
    engine.kv_cache_dtype = cache_dtype
    validate_strict_rl_profile(config, engine)


def test_strict_profile_rejects_gms_v1(monkeypatch, supported_strict_profile):
    monkeypatch.setenv("DYN_GMS_USE_V1", "true")
    with pytest.raises(ValueError, match="strict RL"):
        validate_strict_rl_profile(*supported_strict_profile)
