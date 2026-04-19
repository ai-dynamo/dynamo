# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `build_kv_transfer_config`.

Phase 4 updates:
- Both versions now point at the canonical `kvbm.v{N}.vllm.connector`
  façades (1↔2 char mirror; existing `vllm_integration` paths preserved
  as backcompat).
- v2 builder takes an `onboard_mode` parameter (`intra` / `inter`) and
  injects `leader.onboard.mode` accordingly.
"""

import pytest

from .server import KvbmModelConfig, build_kv_transfer_config

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


@pytest.fixture
def model_config() -> KvbmModelConfig:
    return KvbmModelConfig(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        block_size=16,
        attention_backend="FLASH_ATTN",
    )


def test_v1_payload_uses_canonical_v1_facade(model_config: KvbmModelConfig) -> None:
    cfg = build_kv_transfer_config("v1", model_config)
    assert cfg == {
        "kv_connector": "DynamoConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": "kvbm.v1.vllm.connector",
    }


@pytest.mark.parametrize("onboard_mode", ["intra", "inter"])
def test_v2_payload_uses_canonical_v2_facade(
    model_config: KvbmModelConfig, onboard_mode: str
) -> None:
    cfg = build_kv_transfer_config("v2", model_config, onboard_mode=onboard_mode)
    assert cfg["kv_connector_module_path"] == "kvbm.v2.vllm.connector"
    assert cfg["kv_connector"] == "DynamoConnector"
    assert cfg["kv_role"] == "kv_both"


def test_v2_payload_omits_leader_nova(model_config: KvbmModelConfig) -> None:
    """v2 agg deliberately omits the discovery block (lib/kvbm-config/src/messenger.rs:43)."""
    cfg = build_kv_transfer_config("v2", model_config)
    leader = cfg["kv_connector_extra_config"]["leader"]
    assert "nova" not in leader
    assert "velo" not in leader  # rename caveat: serde key is still 'nova'


@pytest.mark.parametrize("onboard_mode", ["intra", "inter"])
def test_v2_payload_has_required_leader_blocks(
    model_config: KvbmModelConfig, onboard_mode: str
) -> None:
    cfg = build_kv_transfer_config(
        "v2", model_config, onboard_mode=onboard_mode, cpu_blocks=2000
    )
    leader = cfg["kv_connector_extra_config"]["leader"]
    assert leader["cache"]["host"] == {"num_blocks": 2000}
    assert leader["tokio"]["worker_threads"] == 2
    assert leader["onboard"] == {"mode": onboard_mode}


def test_v2_payload_omits_cache_host_when_cpu_blocks_none(
    model_config: KvbmModelConfig,
) -> None:
    """When cpu_blocks is None, the v2 leader config must NOT contain a
    cache.host block — the Rust leader will then fail hard on startup per
    the phase-5 mandatory-tier contract."""
    cfg = build_kv_transfer_config("v2", model_config, cpu_blocks=None)
    leader = cfg["kv_connector_extra_config"]["leader"]
    assert "cache" not in leader


def test_v2_payload_default_onboard_mode_is_intra(
    model_config: KvbmModelConfig,
) -> None:
    cfg = build_kv_transfer_config("v2", model_config)
    assert cfg["kv_connector_extra_config"]["leader"]["onboard"]["mode"] == "intra"


def test_v2_payload_has_required_worker_blocks(model_config: KvbmModelConfig) -> None:
    cfg = build_kv_transfer_config("v2", model_config)
    worker = cfg["kv_connector_extra_config"]["worker"]
    assert "UCX" in worker["nixl"]["backends"]
    assert "POSIX" in worker["nixl"]["backends"]
    assert worker["tokio"]["worker_threads"] == 2


def test_unknown_version_raises(model_config: KvbmModelConfig) -> None:
    with pytest.raises(ValueError, match="unknown kvbm_version"):
        build_kv_transfer_config("v3", model_config)  # type: ignore[arg-type]


def test_unknown_onboard_mode_raises(model_config: KvbmModelConfig) -> None:
    with pytest.raises(ValueError, match="unknown onboard_mode"):
        build_kv_transfer_config("v2", model_config, onboard_mode="bogus")
