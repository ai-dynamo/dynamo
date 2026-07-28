# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM backend components."""

import importlib
import json
import logging
import os
import re
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

import dynamo.llm as dynamo_llm
from dynamo.vllm import envs
from dynamo.vllm.args import (
    _connector_to_kv_transfer_json,
    _is_routable,
    _uses_dynamo_connector,
    _uses_nixl_connector,
    configure_rl_logprobs_mode,
    ensure_side_channel_host,
    get_host_ip,
    parse_args,
    update_engine_config_with_dynamo,
)
from dynamo.vllm.constants import DisaggregationMode
from dynamo.vllm.headless import build_headless_namespace
from dynamo.vllm.tests.conftest import make_cli_args_fixture

# Get path relative to this test file
REPO_ROOT = Path(__file__).resolve().parents[5]
TEST_DIR = REPO_ROOT / "tests"
# Now construct the full path to the shared test fixture
JINJA_TEMPLATE_PATH = str(
    REPO_ROOT / "tests" / "serve" / "fixtures" / "custom_template.jinja"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    # gpu_1 not gpu_0: vLLM DeviceConfig(device='auto') fails on CPU-only arm64
    # runners with "Failed to infer device type" even for mock tests.
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.pre_merge,
]

# Create vLLM-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_vllm_cli = make_cli_args_fixture("dynamo.vllm")


def _load_vllm_main() -> ModuleType:
    """Load the entrypoint only in tests that need it.

    The lightweight pre-commit collection environment intentionally omits
    uvloop, which ``dynamo.vllm.main`` imports at module scope. Eagerly loading
    it here would prevent collection of every otherwise dependency-light test
    in this module.
    """
    return importlib.import_module("dynamo.vllm.main")


@pytest.mark.parametrize(
    "enable_lora, model_type, expected",
    [
        (True, dynamo_llm.ModelType.Prefill, 3),
        (True, dynamo_llm.ModelType.Chat, 3),
        (True, dynamo_llm.ModelType.Embedding, None),
        (False, dynamo_llm.ModelType.Prefill, None),
    ],
)
def test_base_model_lora_capacity(enable_lora, model_type, expected):
    config = SimpleNamespace(
        engine_args=SimpleNamespace(enable_lora=enable_lora, max_loras=3)
    )

    assert _load_vllm_main()._base_model_lora_capacity(config, model_type) == expected


def test_kv_event_block_size_prefers_cached_main_attention_value():
    """LoRA MDC registration relies on this: the cached main-attention block
    size (configured at engine setup) wins over cache_config.block_size, so
    adapter cards carry the same block size as the base-model card on
    hybrid-attention models where vLLM inflates the attention block size."""
    from dynamo.vllm.cache_info import (
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        get_configured_kv_event_block_size,
    )

    vllm_config = SimpleNamespace(
        additional_config={DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: 1056},
        cache_config=SimpleNamespace(block_size=16),
    )
    assert get_configured_kv_event_block_size(vllm_config) == 1056


def test_kv_event_block_size_falls_back_to_cache_config():
    from dynamo.vllm.cache_info import get_configured_kv_event_block_size

    vllm_config = SimpleNamespace(
        additional_config=None,
        cache_config=SimpleNamespace(block_size=16),
    )
    assert get_configured_kv_event_block_size(vllm_config) == 16


def test_custom_jinja_template_invalid_path(mock_vllm_cli):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"

    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--custom-jinja-template", invalid_path)

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        parse_args()


def test_custom_jinja_template_valid_path(mock_vllm_cli):
    """Test that valid absolute path is stored correctly."""
    mock_vllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)

    config = parse_args()

    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )


def test_custom_jinja_template_env_var_expansion(monkeypatch, mock_vllm_cli):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    mock_vllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=cli_path)

    config = parse_args()

    assert "$JINJA_DIR" not in config.custom_jinja_template
    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )


# --endpoint flag tests


def test_endpoint_overrides_defaults(mock_vllm_cli):
    """Test that --endpoint overrides default namespace/component/endpoint."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "dyn://mynamespace.mycomponent.myendpoint",
    )
    config = parse_args()
    assert config.namespace == "mynamespace"
    assert config.component == "mycomponent"
    assert config.endpoint == "myendpoint"


def test_endpoint_not_provided_preserves_defaults(mock_vllm_cli):
    """Test that without --endpoint, defaults are preserved."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.namespace == "dynamo"
    assert config.component == "backend"
    assert config.endpoint == "generate"


def test_endpoint_overrides_with_prefill_worker(mock_vllm_cli):
    """Test that --endpoint overrides even with --disaggregation-mode prefill."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "dyn://custom.worker.serve",
        "--disaggregation-mode",
        "prefill",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    )
    config = parse_args()
    assert config.namespace == "custom"
    assert config.component == "worker"
    assert config.endpoint == "serve"


def test_endpoint_invalid_format_raises(mock_vllm_cli):
    """Test that invalid --endpoint format raises ValueError."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "invalid-endpoint",
    )
    with pytest.raises(ValueError, match="Invalid endpoint format"):
        parse_args()


# --connector removal tests


def test_connector_nixl_raises_error_with_migration_hint(mock_vllm_cli):
    """Test that --connector nixl raises ValueError with --kv-transfer-config hint."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--connector", "nixl")
    with pytest.raises(ValueError, match="--connector is no longer supported"):
        parse_args()


def test_connector_none_raises_error(mock_vllm_cli):
    """Test that --connector none raises ValueError telling user it's no longer needed."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--connector", "none")
    with pytest.raises(ValueError, match="no longer needed"):
        parse_args()


def test_env_var_dyn_connector_raises_error(monkeypatch, mock_vllm_cli):
    """Test that DYN_CONNECTOR env var raises error for vLLM backend."""
    monkeypatch.setenv("DYN_CONNECTOR", "nixl")
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    with pytest.raises(ValueError, match="no longer supported"):
        parse_args()


def test_model_express_url_is_accepted_for_compatibility(mock_vllm_cli):
    """Test that legacy ModelExpress manifests still parse."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--model-express-url",
        "http://model-express:8080",
    )

    config = parse_args()

    assert config.model_express_url == "http://model-express:8080"


def test_model_express_url_env_is_accepted_for_compatibility(
    monkeypatch, mock_vllm_cli
):
    """Test that legacy MODEL_EXPRESS_URL still maps to config."""
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://model-express:8080")
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")

    config = parse_args()

    assert config.model_express_url == "http://model-express:8080"


def test_prefill_worker_without_kv_transfer_config_raises(mock_vllm_cli):
    """Test that --disaggregation-mode prefill without --kv-transfer-config raises ValueError."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "prefill")
    with pytest.raises(ValueError, match="--kv-transfer-config"):
        parse_args()


def test_connector_to_kv_transfer_json_single():
    """Test _connector_to_kv_transfer_json returns valid JSON for a single connector."""
    result = json.loads(_connector_to_kv_transfer_json(["nixl"]))
    assert result == {"kv_connector": "NixlConnector", "kv_role": "kv_both"}


def test_connector_to_kv_transfer_json_multi():
    """Test _connector_to_kv_transfer_json wraps multiple connectors in PdConnector."""
    result = json.loads(_connector_to_kv_transfer_json(["kvbm", "nixl"]))
    assert result["kv_connector"] == "PdConnector"
    nested = result["kv_connector_extra_config"]["connectors"]
    nested_names = [c["kv_connector"] for c in nested]
    assert "DynamoConnector" in nested_names
    assert "NixlConnector" in nested_names


# _uses_nixl_connector / _uses_dynamo_connector tests


def _make_engine_cfg(kv_connector=None, extra_config=None):
    """Build a minimal fake engine config for connector detection tests."""
    if kv_connector is None:
        return SimpleNamespace(kv_transfer_config=None)
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector=kv_connector,
            kv_connector_extra_config=extra_config,
        )
    )


_PD_KVBM_NIXL = {
    "connectors": [
        {"kv_connector": "DynamoConnector", "kv_role": "kv_both"},
        {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
    ]
}


def test_uses_nixl_connector_direct_and_nested():
    """Test _uses_nixl_connector for direct, nested-in-PdConnector, and absent cases."""
    assert _uses_nixl_connector(_make_engine_cfg("NixlConnector")) is True
    assert _uses_nixl_connector(_make_engine_cfg("PdConnector", _PD_KVBM_NIXL)) is True
    assert _uses_nixl_connector(_make_engine_cfg("LMCacheConnectorV1")) is False
    assert _uses_nixl_connector(_make_engine_cfg("LMCacheMPConnector")) is False
    assert _uses_nixl_connector(_make_engine_cfg("FlexKVConnectorV1")) is False
    assert _uses_nixl_connector(_make_engine_cfg()) is False


def test_uses_dynamo_connector_direct_and_nested():
    """Test _uses_dynamo_connector for direct, nested-in-PdConnector, and absent cases."""
    assert _uses_dynamo_connector(_make_engine_cfg("DynamoConnector")) is True
    assert (
        _uses_dynamo_connector(_make_engine_cfg("PdConnector", _PD_KVBM_NIXL)) is True
    )
    assert _uses_dynamo_connector(_make_engine_cfg("NixlConnector")) is False
    assert _uses_dynamo_connector(_make_engine_cfg()) is False


def test_headless_namespace_has_required_fields(mock_vllm_cli):
    """Test that build_headless_namespace produces a Namespace with fields
    required by vLLM's run_headless(), including the api_server_count fallback."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--headless",
    )
    config = parse_args()
    assert config.headless is True

    ns = build_headless_namespace(config)

    # Required by run_headless()
    assert hasattr(ns, "api_server_count")
    assert ns.api_server_count == 0
    # Core engine fields must survive the round-trip
    assert hasattr(ns, "model")
    assert hasattr(ns, "tensor_parallel_size")


def test_rl_logprobs_force_converts_raw_mode():
    config = SimpleNamespace(
        enable_rl=True,
        engine_args=SimpleNamespace(logprobs_mode="raw_logprobs"),
    )

    configure_rl_logprobs_mode(config)

    assert config.engine_args.logprobs_mode == "processed_logprobs"


def test_rl_logprobs_keeps_processed_mode():
    config = SimpleNamespace(
        enable_rl=True,
        engine_args=SimpleNamespace(logprobs_mode="processed_logprobs"),
    )

    configure_rl_logprobs_mode(config)

    assert config.engine_args.logprobs_mode == "processed_logprobs"


def test_rl_logprobs_rejects_logits_modes():
    config = SimpleNamespace(
        enable_rl=True,
        engine_args=SimpleNamespace(logprobs_mode="raw_logits"),
    )

    with pytest.raises(ValueError, match="processed_logprobs"):
        configure_rl_logprobs_mode(config)


def test_parse_args_does_not_track_logprobs_mode_presence(mock_vllm_cli):
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert not hasattr(config, "logprobs_mode_explicitly_set")


def test_parse_args_splits_served_model_name_into_aliases(mock_vllm_cli):
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "primary",
        "alias-one",
        "alias-two",
    )
    config = parse_args()
    assert config.served_model_name == "primary"
    assert config.served_model_aliases == ["alias-one", "alias-two"]


def test_parse_args_splits_comma_packed_served_model_name(mock_vllm_cli):
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "primary,alias-one",
        "alias-two",
    )
    config = parse_args()
    assert config.served_model_name == "primary"
    assert config.served_model_aliases == ["alias-one", "alias-two"]


def test_parse_args_without_served_model_name_has_no_aliases(mock_vllm_cli):
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.served_model_aliases == []


def test_should_prefetch_model_for_default_load_format():
    from dynamo.vllm.main import should_prefetch_model

    config = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        engine_args=SimpleNamespace(load_format="auto"),
    )

    assert should_prefetch_model(config) is True


@pytest.mark.parametrize("load_format", ["modelexpress", "mx"])
def test_should_not_prefetch_model_for_modelexpress_load_formats(load_format):
    from dynamo.vllm.main import (
        should_prefetch_model,
        should_register_model_ignore_weights,
        uses_modelexpress_load_format,
    )

    config = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        engine_args=SimpleNamespace(load_format=load_format),
    )

    assert uses_modelexpress_load_format(config) is True
    assert should_prefetch_model(config) is False
    assert should_register_model_ignore_weights(config) is True


def test_should_not_prefetch_existing_local_model(tmp_path):
    from dynamo.vllm.main import should_prefetch_model

    config = SimpleNamespace(
        model=str(tmp_path),
        engine_args=SimpleNamespace(load_format="auto"),
    )

    assert should_prefetch_model(config) is False


def test_should_register_model_fetch_weights_for_default_load_format():
    from dynamo.vllm.main import should_register_model_ignore_weights

    config = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        engine_args=SimpleNamespace(load_format="auto"),
    )

    assert should_register_model_ignore_weights(config) is False


def test_setup_vllm_engine_reuses_engine_config_model_config(monkeypatch):
    from dynamo.vllm import main as vllm_main

    class FakeModelConfig:
        def get_diff_sampling_param(self):
            return {"temperature": 0.7}

    vllm_config = SimpleNamespace(
        additional_config={},
        cache_config=SimpleNamespace(block_size=None),
        model_config=FakeModelConfig(),
    )

    class FakeEngineArgs:
        enable_log_requests = False
        enable_lora = False
        disable_log_stats = True
        load_format = "modelexpress"

        def create_model_config(self):
            raise AssertionError("setup_vllm_engine must not create ModelConfig twice")

        def create_engine_config(self, usage_context):
            return vllm_config

    engine_client = SimpleNamespace(vllm_config=vllm_config)

    class FakeAsyncLLM:
        @staticmethod
        def from_vllm_config(**_kwargs):
            return engine_client

    class FakeMetrics:
        def __init__(self, **_kwargs):
            pass

        def set_model_load_time(self, _load_time):
            pass

    monkeypatch.setattr(vllm_main, "setup_multiprocess_prometheus", lambda: None)
    monkeypatch.setattr(vllm_main, "LLMBackendMetrics", FakeMetrics)
    monkeypatch.setattr(vllm_main, "_uses_dynamo_connector", lambda _args: False)
    monkeypatch.setattr(vllm_main, "AsyncLLM", FakeAsyncLLM)
    monkeypatch.setattr(
        vllm_main,
        "get_engine_cache_info",
        lambda _engine: {"block_size": 16},
    )

    config = SimpleNamespace(
        component="backend",
        namespace="dynamo",
        engine_args=FakeEngineArgs(),
        gms_shadow_mode=False,
        multimodal_embedding_cache_capacity_gb=0,
        route_to_encoder=False,
        served_model_name="Qwen/Qwen3-0.6B",
    )

    _, _, default_sampling_params, _, _ = vllm_main.setup_vllm_engine(config)

    assert default_sampling_params == {"temperature": 0.7}


# --disaggregation-mode tests


def test_disaggregation_mode_default(mock_vllm_cli):
    """Test that default disaggregation mode is AGGREGATED."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED


def test_kv_events_disabled_by_default_without_explicit_config(mock_vllm_cli):
    """Test that vLLM no longer auto-creates kv_events_config."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.engine_args.kv_events_config is None
    assert config.use_kv_events is False


def test_disaggregation_mode_prefill(mock_vllm_cli):
    """Test --disaggregation-mode prefill sets correct state."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disaggregation-mode",
        "prefill",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    )
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.PREFILL
    assert config.component == "prefill"


def test_disaggregation_mode_decode(mock_vllm_cli):
    """Test --disaggregation-mode decode sets correct state."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "decode")
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.DECODE


# --- _is_routable tests (pure logic, no mocking) ---


class TestIsRoutable:
    def test_accepts_private_ipv4(self):
        assert _is_routable("10.0.0.5") is True
        assert _is_routable("192.168.1.1") is True

    def test_accepts_private_ipv6(self):
        assert _is_routable("fd00::1") is True

    def test_rejects_loopback_v4(self):
        assert _is_routable("127.0.0.1") is False

    def test_rejects_loopback_v6(self):
        assert _is_routable("::1") is False

    def test_rejects_link_local_v4(self):
        assert _is_routable("169.254.1.1") is False

    def test_rejects_link_local_v6(self):
        assert _is_routable("fe80::1") is False

    def test_rejects_unspecified(self):
        assert _is_routable("0.0.0.0") is False
        assert _is_routable("::") is False

    def test_rejects_multicast(self):
        assert _is_routable("224.0.0.1") is False

    def test_rejects_invalid(self):
        assert _is_routable("not-an-ip") is False


# --- get_host_ip tests (mock socket module functions) ---


class TestGetHostIp:
    def test_hostname_resolution_success(self):
        """getaddrinfo returns routable IPv4 → returns it."""
        with patch(
            "dynamo.vllm.args._try_hostname_resolution", return_value="10.0.0.5"
        ):
            result = get_host_ip()
        assert result == "10.0.0.5"

    def test_hostname_loopback_falls_through_to_udp(self):
        """getaddrinfo returns 127.0.0.1, UDP returns 10.0.0.5 → returns 10.0.0.5."""
        with (
            patch(
                "dynamo.vllm.args._try_hostname_resolution", return_value="127.0.0.1"
            ),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            mock_udp.side_effect = lambda family, target: (
                "10.0.0.5" if family == socket.AF_INET else None
            )
            result = get_host_ip()
        assert result == "10.0.0.5"

    def test_hostname_link_local_falls_through_to_udp(self):
        """getaddrinfo returns 169.254.1.1, UDP returns 10.0.0.5 → returns 10.0.0.5."""
        with (
            patch(
                "dynamo.vllm.args._try_hostname_resolution", return_value="169.254.1.1"
            ),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            mock_udp.side_effect = lambda family, target: (
                "10.0.0.5" if family == socket.AF_INET else None
            )
            result = get_host_ip()
        assert result == "10.0.0.5"

    def test_ipv6_fallback(self):
        """IPv4 strategies fail, IPv6 UDP returns fd00::1 → returns fd00::1."""
        with (
            patch("dynamo.vllm.args._try_hostname_resolution", return_value=None),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            mock_udp.side_effect = lambda family, target: (
                "fd00::1" if family == socket.AF_INET6 else None
            )
            result = get_host_ip()
        assert result == "fd00::1"

    def test_all_fail_raises_runtime_error(self):
        """All strategies fail → RuntimeError with VLLM_NIXL_SIDE_CHANNEL_HOST in message."""
        with (
            patch("dynamo.vllm.args._try_hostname_resolution", return_value=None),
            patch("dynamo.vllm.args._try_udp_connect", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="VLLM_NIXL_SIDE_CHANNEL_HOST"):
                get_host_ip()


# --- ensure_side_channel_host tests ---


class TestEnsureSideChannelHost:
    def test_preserves_existing_env_var(self, monkeypatch):
        """Pre-set env var → verify not overwritten."""
        monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "192.168.99.99")
        with patch("dynamo.vllm.args.get_host_ip") as mock_get:
            ensure_side_channel_host()
            mock_get.assert_not_called()
        import os

        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.99.99"

    def test_sets_env_var_on_successful_detection(self, monkeypatch):
        """No env var set, successful detection populates the side-channel host."""
        monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
        with patch("dynamo.vllm.args.get_host_ip", return_value="10.0.0.5"):
            ensure_side_channel_host()

        import os

        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.5"

    def test_raises_when_detection_fails_and_no_env(self, monkeypatch):
        """All strategies fail, no env var → RuntimeError."""
        monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
        with patch(
            "dynamo.vllm.args.get_host_ip",
            side_effect=RuntimeError("Unable to determine"),
        ):
            with pytest.raises(RuntimeError, match="Unable to determine"):
                ensure_side_channel_host()


# --- vllm_omni optional dependency tests ---


class TestVllmOmniOptionalDependency:
    def test_dynamo_vllm_main_importable_without_vllm_omni(self):
        """dynamo.vllm.main must import cleanly even when vllm_omni is absent.

        Setting sys.modules["vllm_omni"] = None blocks ALL imports from the
        vllm_omni package — Python always resolves the top-level package first,
        so a None sentinel at the root raises ImportError for any submodule import.
        """
        # Save and evict any already-cached vllm_omni and dynamo.vllm.omni modules
        saved = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k == "vllm_omni"
            or k.startswith("vllm_omni.")
            or k == "dynamo.vllm.main"
            or k.startswith("dynamo.vllm.omni")
        }
        # Explicitly block the top-level vllm_omni package regardless of prior imports
        sys.modules["vllm_omni"] = None  # type: ignore[assignment]

        try:
            import dynamo.vllm.main  # noqa: F401
        except ImportError as e:
            pytest.fail(f"dynamo.vllm.main has a hard dependency on vllm_omni: {e}")
        finally:
            sys.modules.pop("vllm_omni", None)
            # Remove any modules imported during this test
            for mod in list(sys.modules):
                if mod == "dynamo.vllm.main" or mod.startswith("dynamo.vllm.omni"):
                    sys.modules.pop(mod, None)
            # Restore original state
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# Benchmark mode unit tests
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass and grid generation."""

    def test_benchmark_config_defaults(self):
        from dynamo.vllm.instrumented_scheduler import BenchmarkConfig

        cfg = BenchmarkConfig()
        assert cfg.mode == "agg"
        assert cfg.warmup_iterations == 5
        assert cfg.output_path == "/tmp/benchmark_results.json"

    def test_benchmark_config_from_dict(self):
        from dynamo.vllm.instrumented_scheduler import BenchmarkConfig

        cfg = BenchmarkConfig(
            mode="decode",
            warmup_iterations=2,
            output_path="/tmp/test.json",
        )
        assert cfg.mode == "decode"
        assert cfg.warmup_iterations == 2
        assert cfg.output_path == "/tmp/test.json"

    def test_benchmark_config_kwargs_unpack(self):
        from dynamo.vllm.instrumented_scheduler import BenchmarkConfig

        d = {"mode": "prefill", "warmup_iterations": 1}
        cfg = BenchmarkConfig(**d)
        assert cfg.mode == "prefill"
        assert cfg.warmup_iterations == 1

    def test_benchmark_operational_controls_reach_scheduler_config(
        self, mock_vllm_cli, tmp_path
    ):
        output = tmp_path / "benchmark.json"
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "prefill",
            "--benchmark-warmup-iterations",
            "2",
            "--benchmark-output-path",
            str(output),
        )

        config = parse_args()

        assert config._benchmark_additional_config == {
            "mode": "prefill",
            "warmup_iterations": 2,
            "output_path": str(output),
            "timeout": 900,
            "prefill_max_new_token_samples": 64,
            "prefill_max_kv_read_token_samples": 16,
            "decode_max_kv_read_token_samples": 128,
            "decode_max_batch_size_samples": 128,
            "prefix_max_batch_size_samples": 3,
        }

    def test_benchmark_points_file_is_embedded_in_benchmark_config(
        self, mock_vllm_cli, tmp_path
    ):
        points_path = tmp_path / "points.json"
        points = {
            "schema_version": 1,
            "prefill": [
                {
                    "total_prefill_tokens": 8,
                    "total_kv_read_tokens": 0,
                    "batch_size": 1,
                }
            ],
            "decode": [{"total_kv_read_tokens": 32, "batch_size": 2}],
        }
        points_path.write_text(json.dumps(points))
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "agg",
            "--benchmark-points-file",
            str(points_path),
        )

        config = parse_args()

        bench = config._benchmark_additional_config
        assert bench["points"] == points
        assert "benchmark_points_file" not in bench
        assert not any(key.endswith("_samples") for key in bench)

    def test_benchmark_sampling_controls_reach_scheduler_config(self, mock_vllm_cli):
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "agg",
            "--prefill-max-new-token-samples",
            "12",
            "--prefill-max-kv-read-token-samples",
            "6",
            "--decode-max-kv-read-token-samples",
            "24",
            "--decode-max-batch-size-samples",
            "20",
            "--prefix-max-batch-size-samples",
            "2",
        )

        config = parse_args()

        assert (
            config._benchmark_additional_config["prefill_max_new_token_samples"] == 12
        )
        assert (
            config._benchmark_additional_config["prefill_max_kv_read_token_samples"]
            == 6
        )
        assert (
            config._benchmark_additional_config["decode_max_kv_read_token_samples"]
            == 24
        )
        assert (
            config._benchmark_additional_config["decode_max_batch_size_samples"] == 20
        )
        assert config._benchmark_additional_config["prefix_max_batch_size_samples"] == 2

    @pytest.mark.parametrize(
        ("legacy_flag", "replacement_name", "legacy_value", "expected"),
        [
            ("--benchmark-prefill-granularity", "prefill_max_new_token_samples", 1, 2),
            (
                "--benchmark-prefill-kv-read-granularity",
                "prefill_max_kv_read_token_samples",
                6,
                6,
            ),
            (
                "--benchmark-prefill-batch-granularity",
                "prefix_max_batch_size_samples",
                1,
                1,
            ),
            (
                "--benchmark-decode-length-granularity",
                "decode_max_kv_read_token_samples",
                1,
                2,
            ),
            (
                "--benchmark-decode-batch-granularity",
                "decode_max_batch_size_samples",
                7,
                7,
            ),
        ],
    )
    def test_legacy_benchmark_sampling_flags_are_mapped(
        self,
        mock_vllm_cli,
        legacy_flag,
        replacement_name,
        legacy_value,
        expected,
    ):
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "agg",
            legacy_flag,
            str(legacy_value),
        )

        with pytest.warns(DeprecationWarning):
            config = parse_args()

        assert getattr(config, replacement_name) == expected
        assert config._benchmark_additional_config[replacement_name] == expected

    @pytest.mark.parametrize(
        ("legacy_env", "replacement_name", "expected"),
        [
            ("DYN_BENCHMARK_PREFILL_GRANULARITY", "prefill_max_new_token_samples", 2),
            (
                "DYN_BENCHMARK_PREFILL_KV_READ_GRANULARITY",
                "prefill_max_kv_read_token_samples",
                5,
            ),
            (
                "DYN_BENCHMARK_PREFILL_BATCH_GRANULARITY",
                "prefix_max_batch_size_samples",
                1,
            ),
            (
                "DYN_BENCHMARK_DECODE_LENGTH_GRANULARITY",
                "decode_max_kv_read_token_samples",
                2,
            ),
            (
                "DYN_BENCHMARK_DECODE_BATCH_GRANULARITY",
                "decode_max_batch_size_samples",
                9,
            ),
        ],
    )
    def test_legacy_benchmark_sampling_env_vars_are_mapped(
        self,
        mock_vllm_cli,
        monkeypatch,
        legacy_env,
        replacement_name,
        expected,
    ):
        monkeypatch.setenv(legacy_env, str(expected))
        mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--benchmark-mode", "agg")

        with pytest.warns(DeprecationWarning):
            config = parse_args()

        assert getattr(config, replacement_name) == expected
        assert isinstance(getattr(config, replacement_name), int)

    def test_legacy_and_new_benchmark_sampling_conflict(self, mock_vllm_cli):
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "prefill",
            "--benchmark-prefill-granularity",
            "12",
            "--prefill-max-new-token-samples",
            "20",
        )

        with pytest.raises(ValueError, match="cannot combine"):
            parse_args()

    def test_legacy_conflicts_with_explicit_new_default(self, mock_vllm_cli):
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "prefill",
            "--benchmark-prefill-granularity",
            "12",
            "--prefill-max-new-token-samples",
            "64",
        )

        with pytest.raises(ValueError, match="cannot combine"):
            parse_args()

    def test_legacy_env_conflicts_with_explicit_new_default_env(
        self, mock_vllm_cli, monkeypatch
    ):
        monkeypatch.setenv("DYN_BENCHMARK_PREFILL_GRANULARITY", "12")
        monkeypatch.setenv("DYN_PREFILL_MAX_NEW_TOKEN_SAMPLES", "64")
        mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--benchmark-mode", "prefill")

        with pytest.raises(ValueError, match="cannot combine"):
            parse_args()

    @pytest.mark.parametrize("value", [0, 1025])
    def test_legacy_benchmark_sampling_range_is_validated(self, mock_vllm_cli, value):
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "prefill",
            "--benchmark-prefill-granularity",
            str(value),
        )

        with pytest.raises(ValueError, match="must be between 1 and 1024"):
            parse_args()

    def test_prefill_without_prefix_caching_is_allowed(self, mock_vllm_cli):
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--benchmark-mode",
            "prefill",
            "--no-enable-prefix-caching",
        )

        assert parse_args().benchmark_mode == "prefill"


class TestBenchmarkGrid:
    """Tests for benchmark grid generation logic (no GPU required)."""

    def _make_grid_helper(self):
        """Return (prefill_grid_fn, decode_grid_fn) that operate on plain params."""
        import numpy as np

        def generate_prefill_grid(max_num_scheduled_tokens, granularity):
            isls = np.unique(
                np.linspace(10, max_num_scheduled_tokens, granularity, dtype=int)
            )
            return [int(x) for x in isls]

        def generate_decode_grid(
            block_size,
            max_model_len,
            max_num_running_reqs,
            num_gpu_blocks,
            length_granularity,
            batch_granularity,
        ):
            total_kv_tokens = num_gpu_blocks * block_size
            ctx_lens = np.unique(
                np.linspace(block_size, max_model_len, length_granularity, dtype=int)
            )
            points = []
            for ctx_len in ctx_lens:
                ctx_len = int(ctx_len)
                max_batch = min(max_num_running_reqs, total_kv_tokens // ctx_len)
                if max_batch < 1:
                    continue
                batch_sizes = np.unique(
                    np.linspace(1, max_batch, batch_granularity, dtype=int)
                )
                for bs in batch_sizes:
                    points.append((ctx_len, int(bs)))
            return points

        return generate_prefill_grid, generate_decode_grid

    def test_prefill_grid_count(self):
        gen_prefill, _ = self._make_grid_helper()
        isls = gen_prefill(max_num_scheduled_tokens=8192, granularity=10)
        assert len(isls) == 10
        assert isls[0] == 10
        assert isls[-1] == 8192

    def test_prefill_grid_dedup(self):
        gen_prefill, _ = self._make_grid_helper()
        isls = gen_prefill(max_num_scheduled_tokens=20, granularity=100)
        assert len(isls) == len(set(isls))

    def test_decode_grid_batch_capped(self):
        _, gen_decode = self._make_grid_helper()
        points = gen_decode(
            block_size=16,
            max_model_len=4096,
            max_num_running_reqs=64,
            num_gpu_blocks=256,
            length_granularity=3,
            batch_granularity=3,
        )
        total_kv = 256 * 16
        for ctx_len, bs in points:
            assert bs <= min(64, total_kv // ctx_len)
            assert bs >= 1

    def test_decode_grid_skips_large_ctx(self):
        _, gen_decode = self._make_grid_helper()
        points = gen_decode(
            block_size=16,
            max_model_len=100000,
            max_num_running_reqs=64,
            num_gpu_blocks=100,
            length_granularity=5,
            batch_granularity=3,
        )
        total_kv = 100 * 16
        for ctx_len, bs in points:
            assert ctx_len <= total_kv


def test_build_sampling_params_maps_max_thinking_tokens():
    from dynamo.vllm.handlers import build_sampling_params

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_thinking_tokens": 1024},
        "output_options": {},
    }
    sp = build_sampling_params(request, default_sampling_params={})
    assert sp.thinking_token_budget == 1024


@pytest.mark.parametrize(
    ("constraint_name", "constraint_value"),
    [
        pytest.param(
            "json",
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            id="json-object",
        ),
        pytest.param(
            "json",
            json.dumps(
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                }
            ),
            id="json-string",
        ),
        pytest.param("regex", r"(red|blue|green)", id="regex"),
        pytest.param(
            "grammar",
            'root ::= "red" | "blue" | "green"',
            id="grammar",
        ),
        pytest.param("choice", ["red", "blue", "green"], id="choice"),
    ],
)
def test_build_sampling_params_maps_guided_decoding(constraint_name, constraint_value):
    from vllm.sampling_params import StructuredOutputsParams

    from dynamo.vllm.handlers import build_sampling_params

    request = {
        "token_ids": [1, 2, 3],
        "sampling_options": {
            "guided_decoding": {constraint_name: constraint_value},
        },
        "stop_conditions": {},
        "output_options": {},
    }

    sp = build_sampling_params(request, default_sampling_params={})

    assert isinstance(sp.structured_outputs, StructuredOutputsParams)
    for field in ("json", "regex", "grammar", "choice"):
        expected = constraint_value if field == constraint_name else None
        assert getattr(sp.structured_outputs, field) == expected


def test_build_sampling_params_caps_omitted_max_tokens_to_generation_default():
    from dynamo.vllm.handlers import build_sampling_params

    model_max_len = 100
    defaults = {"max_tokens": 32}

    def make(stop_conditions):
        return {
            "token_ids": [1, 2, 3],
            "sampling_options": {},
            "stop_conditions": stop_conditions,
            "output_options": {},
        }

    remaining = model_max_len - 3

    # omitted max_tokens → capped to configured default
    sp = build_sampling_params(make({}), defaults, model_max_len)
    assert sp.max_tokens == 32

    # explicit max_tokens == remaining context → treated as explicit, not capped
    sp = build_sampling_params(make({"max_tokens": remaining}), defaults, model_max_len)
    assert sp.max_tokens == remaining

    sp = build_sampling_params(make({"max_tokens": 10}), defaults, model_max_len)
    assert sp.max_tokens == 10

    sp = build_sampling_params(make({"max_tokens": 64}), defaults, model_max_len)
    assert sp.max_tokens == 64

    sp = build_sampling_params(make({}), {}, model_max_len)
    assert sp.max_tokens == remaining


def _make_dynamo_config(**overrides):
    """Build a minimal fake DynamoConfig for update_engine_config_with_dynamo tests."""
    defaults = {
        "disaggregation_mode": DisaggregationMode.AGGREGATED,
        "use_kv_events": False,
        "enable_local_indexer": True,
        "embedding_worker": False,
        "headless": False,
        "multimodal_worker": False,
        "multimodal_decode_worker": False,
        "fpm_trace": False,
        "benchmark_mode": None,
        "benchmark_warmup_iterations": 5,
        "benchmark_output_path": "/tmp/benchmark_results.json",
        "benchmark_timeout": 900,
        "prefill_max_new_token_samples": 64,
        "prefill_max_kv_read_token_samples": 16,
        "decode_max_kv_read_token_samples": 128,
        "decode_max_batch_size_samples": 128,
        "prefix_max_batch_size_samples": 3,
        "_benchmark_points": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_engine_config_with_runner(runner="auto", **overrides):
    """Build a fake engine config with runner and other fields used by defaults loop."""
    defaults = {
        "runner": runner,
        "enable_prefix_caching": True,
        "block_size": 16,
        "skip_tokenizer_init": True,
        "enable_log_requests": True,
        "disable_log_stats": True,
        "kv_events_config": None,
        "kv_transfer_config": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestRunnerPreservation:
    """update_engine_config_with_dynamo must not overwrite a user-set --runner."""

    def test_runner_auto_is_preserved(self):
        """When user passes --runner auto (also the vLLM default),
        Dynamo must leave it alone so vLLM's own auto-detection runs."""
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner(runner="auto")

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert engine_cfg.runner == "auto"

    def test_runner_pooling_preserved(self):
        """When user passes --runner pooling (for embedding models),
        Dynamo must NOT overwrite it."""
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner(runner="pooling")

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert engine_cfg.runner == "pooling"

    def test_runner_generate_explicit_preserved(self):
        """When user explicitly passes --runner generate, it should still be 'generate'."""
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner(runner="generate")

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert engine_cfg.runner == "generate"

    def test_runner_draft_preserved(self):
        """When user passes --runner draft, Dynamo must NOT overwrite it."""
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner(runner="draft")

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert engine_cfg.runner == "draft"

    def test_no_runner_attr_skipped_gracefully(self):
        """If engine_config lacks a 'runner' attr (older vLLM), no error is raised."""
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner()
        del engine_cfg.runner  # simulate older vLLM without runner

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert not hasattr(engine_cfg, "runner")


class TestForwardPassMetricsActivation:
    """FPM tracing should activate vLLM's existing FPM instrumentation."""

    def test_cli_flag_enables_trace_and_exports_env(self, monkeypatch, mock_vllm_cli):
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        monkeypatch.delenv("DYN_FPM_TRACE", raising=False)
        mock_vllm_cli("--fpm-trace", "--model", "Qwen/Qwen3-0.6B")

        config = parse_args()

        assert config.fpm_trace is True
        assert os.environ["DYN_FPM_TRACE"] == "1"
        assert (
            config.engine_args.scheduler_cls
            == "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
        )

    def test_trace_enables_instrumented_scheduler_with_default_port(self, monkeypatch):
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        dynamo_cfg = _make_dynamo_config(fpm_trace=True)
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert (
            engine_cfg.scheduler_cls
            == "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
        )
        assert envs.DYN_FORWARDPASS_METRIC_PORT == 20380

    def test_explicit_port_wins_when_trace_is_enabled(self, monkeypatch):
        monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
        dynamo_cfg = _make_dynamo_config(fpm_trace=True)
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert (
            engine_cfg.scheduler_cls
            == "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
        )
        assert envs.DYN_FORWARDPASS_METRIC_PORT == 23456

    def test_false_trace_does_not_enable_fpm(self, monkeypatch):
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert engine_cfg.scheduler_cls is None

    def test_disabled_trace_does_not_start_relay(self, monkeypatch):
        vllm_main = _load_vllm_main()
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        dynamo_cfg = _make_dynamo_config()
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)
        assert (
            vllm_main.setup_fpm_relay(dynamo_cfg, SimpleNamespace(), SimpleNamespace())
            is None
        )

        assert engine_cfg.scheduler_cls is None

    def test_custom_scheduler_warns_and_serving_configuration_continues(
        self, monkeypatch, caplog
    ):
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        dynamo_cfg = _make_dynamo_config(fpm_trace=True)
        engine_cfg = _make_engine_config_with_runner(
            scheduler_cls="example.CustomScheduler"
        )

        with caplog.at_level(logging.WARNING, logger="dynamo.vllm.args"):
            update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert engine_cfg.scheduler_cls == "example.CustomScheduler"
        assert "InstrumentedScheduler will NOT be injected" in caplog.text

    def test_trace_only_starts_relay_on_default_port(self, monkeypatch):
        vllm_main = _load_vllm_main()
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        monkeypatch.setattr(
            vllm_main, "get_dp_range_for_worker", lambda _config: (0, 1)
        )

        constructed = []

        class FakeRelay:
            def __init__(self, **kwargs):
                constructed.append(kwargs)

        monkeypatch.setattr(dynamo_llm, "FpmEventRelay", FakeRelay, raising=False)
        dynamo_cfg = _make_dynamo_config(fpm_trace=True)
        endpoint = SimpleNamespace()

        relays = vllm_main.setup_fpm_relay(dynamo_cfg, endpoint, SimpleNamespace())

        assert relays is not None
        assert len(relays) == 1
        assert constructed == [
            {
                "endpoint": endpoint,
                "zmq_endpoint": "tcp://127.0.0.1:20380",
            }
        ]

    @pytest.mark.parametrize(
        ("overrides", "role", "fpm_trace_relay_supported"),
        [
            ({}, "unified backend", False),
            ({"embedding_worker": True}, "embedding", True),
            ({"headless": True}, "headless", True),
            (
                {"disaggregation_mode": DisaggregationMode.ENCODE},
                "multimodal encode",
                True,
            ),
        ],
    )
    def test_trace_does_not_inject_scheduler_without_relay(
        self,
        monkeypatch,
        caplog,
        overrides,
        role,
        fpm_trace_relay_supported,
    ):
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        dynamo_cfg = _make_dynamo_config(fpm_trace=True, **overrides)
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        with caplog.at_level(logging.WARNING, logger="dynamo.vllm.args"):
            update_engine_config_with_dynamo(
                dynamo_cfg,
                engine_cfg,
                fpm_trace_relay_supported=fpm_trace_relay_supported,
            )

        assert engine_cfg.scheduler_cls is None
        assert f"vLLM {role} workers do not create a Dynamo FPM relay" in caplog.text

    def test_explicit_port_preserves_legacy_activation_for_unsupported_role(
        self, monkeypatch
    ):
        monkeypatch.setenv("DYN_FORWARDPASS_METRIC_PORT", "23456")
        dynamo_cfg = _make_dynamo_config(embedding_worker=True)
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        update_engine_config_with_dynamo(
            dynamo_cfg,
            engine_cfg,
            fpm_trace_relay_supported=False,
        )

        assert (
            engine_cfg.scheduler_cls
            == "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
        )

    def test_benchmark_does_not_reapply_trace_scheduler(
        self, monkeypatch, caplog, tmp_path
    ):
        monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
        dynamo_cfg = _make_dynamo_config(
            fpm_trace=True,
            benchmark_mode="agg",
            benchmark_warmup_iterations=5,
            benchmark_output_path=str(tmp_path / "benchmark_results.json"),
            benchmark_timeout=300,
        )
        engine_cfg = _make_engine_config_with_runner(scheduler_cls=None)

        with caplog.at_level(logging.INFO, logger="dynamo.vllm.args"):
            update_engine_config_with_dynamo(dynamo_cfg, engine_cfg)

        assert (
            engine_cfg.scheduler_cls
            == "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler"
        )
        assert "Forward pass metrics enabled" in caplog.text
        assert "Benchmark mode: auto-enabling InstrumentedScheduler" not in caplog.text


class TestEmbeddingWorkerFlag:
    """Parsing + validation for --embedding-worker."""

    def test_default_false(self, mock_vllm_cli):
        """Without --embedding-worker, the flag is False."""
        mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
        config = parse_args()
        assert config.embedding_worker is False

    def test_flag_sets_true(self, mock_vllm_cli):
        """--embedding-worker on its own with default agg mode parses cleanly."""
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--embedding-worker",
            "--runner",
            "pooling",
        )
        config = parse_args()
        assert config.embedding_worker is True

    def test_rejects_prefill_disagg(self, mock_vllm_cli):
        """--embedding-worker combined with --disaggregation-mode prefill is rejected."""
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--embedding-worker",
            "--runner",
            "pooling",
            "--disaggregation-mode",
            "prefill",
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
        )
        with pytest.raises(ValueError, match="--embedding-worker is only valid"):
            parse_args()

    def test_rejects_decode_disagg(self, mock_vllm_cli):
        """--embedding-worker combined with --disaggregation-mode decode is rejected."""
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--embedding-worker",
            "--runner",
            "pooling",
            "--disaggregation-mode",
            "decode",
        )
        with pytest.raises(ValueError, match="--embedding-worker is only valid"):
            parse_args()

    def test_rejects_multimodal_combo(self, mock_vllm_cli):
        """--embedding-worker combined with multimodal flags is rejected."""
        mock_vllm_cli(
            "--model",
            "Qwen/Qwen3-0.6B",
            "--embedding-worker",
            "--runner",
            "pooling",
            "--enable-multimodal",
        )
        with pytest.raises(
            ValueError, match="--embedding-worker cannot be combined with multimodal"
        ):
            parse_args()


def test_build_sampling_params_openai_maps_max_thinking_tokens():
    from dynamo.vllm.handlers import build_sampling_params_openai

    request = {
        "model": "test-model",
        "prompt": "Solve: 1+1.",
        "max_tokens": 32,
        "nvext": {"max_thinking_tokens": 1024},
    }
    sp = build_sampling_params_openai(request, default_sampling_params={})
    assert sp.thinking_token_budget == 1024


@pytest.mark.asyncio
async def test_generate_text_mode_applies_nvext_cache_salt():
    from dynamo.vllm.handlers import DecodeWorkerHandler

    captured = {}

    class InputParams:
        def get_input_param(self, request, use_tokenizer):
            assert use_tokenizer is True
            return [1, 2, 3]

    class EngineClient:
        def generate(self, prompt, *args, **kwargs):
            captured["prompt"] = prompt

            async def gen():
                output = SimpleNamespace(index=0, text="ok", finish_reason=None)
                yield SimpleNamespace(outputs=[output])

            return gen()

    @asynccontextmanager
    async def abort_monitor(*args, **kwargs):
        yield

    handler = SimpleNamespace(
        input_param_manager=InputParams(),
        default_sampling_params={},
        config=SimpleNamespace(disaggregation_mode=DisaggregationMode.AGGREGATED),
        engine_client=EngineClient(),
        _deferred_aborts={},
        _shutdown_on_engine_dead=lambda exc: None,
        _abort_monitor=abort_monitor,
        _to_local_dp_rank=lambda rank: None,
    )
    context = SimpleNamespace(trace_headers=lambda: {})
    request = {
        "model": "test-model",
        "prompt": "ignored after tokenization",
        "nvext": {"cache_salt": "tenant-a"},
    }

    chunks = [
        chunk
        async for chunk in DecodeWorkerHandler._generate_text_mode(
            handler, request, context, "req-1"
        )
    ]

    assert chunks
    assert captured["prompt"]["cache_salt"] == "dynamo-cache-salt:tenant-a"
