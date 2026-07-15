# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for selecting the legacy vLLM engine-client topology."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.usage.usage_lib")

from dynamo.vllm.backend_args import DisaggregationMode, DynamoVllmConfig  # noqa: E402
from dynamo.vllm.llm_engine import VllmLLMEngine  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def make_config() -> DynamoVllmConfig:
    config = DynamoVllmConfig()
    config.engine_client_mode = "sync-inproc"
    config.disaggregation_mode = DisaggregationMode.AGGREGATED
    config.route_to_encoder = False
    config.multimodal_worker = False
    config.embedding_worker = False
    config.enable_rl = False
    config.headless = False
    config.gms_shadow_mode = False
    config.benchmark_mode = None
    return config


def test_sync_inproc_accepts_pure_text_aggregated_serving() -> None:
    make_config()._validate_engine_client_mode()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("disaggregation_mode", DisaggregationMode.DECODE, "disaggregated"),
        ("route_to_encoder", True, "route-to-encoder"),
        ("multimodal_worker", True, "multimodal-worker"),
        ("embedding_worker", True, "embedding-worker"),
        ("enable_rl", True, "enable-rl"),
        ("headless", True, "headless"),
        ("gms_shadow_mode", True, "gms-shadow-mode"),
        ("benchmark_mode", "agg", "benchmark-mode"),
    ],
)
def test_sync_inproc_rejects_unsupported_modes(
    field: str,
    value: object,
    message: str,
) -> None:
    config = make_config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        config._validate_engine_client_mode()


@pytest.mark.asyncio
async def test_unified_entrypoint_rejects_sync_inproc() -> None:
    config = SimpleNamespace(engine_client_mode="sync-inproc")
    with pytest.raises(NotImplementedError, match="legacy"):
        await VllmLLMEngine.from_args(config=config)  # type: ignore[arg-type]
