# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM backend components."""

import json
import re
import socket
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dynamo.vllm.args import (
    _connector_to_kv_transfer_json,
    _is_routable,
    _uses_dynamo_connector,
    _uses_nixl_connector,
    ensure_side_channel_host,
    get_host_ip,
    parse_args,
)
from dynamo.vllm.constants import DisaggregationMode
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
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]

# Create vLLM-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_vllm_cli = make_cli_args_fixture("dynamo.vllm")


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


@pytest.mark.parametrize("load_format", ["mx-source", "mx-target"])
def test_model_express_url_from_cli_arg(mock_vllm_cli, load_format):
    """Test that --model-express-url is stored when load format is mx-source/mx-target."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--load-format",
        load_format,
        "--model-express-url",
        "http://mx-server:8080",
    )
    config = parse_args()
    assert config.model_express_url == "http://mx-server:8080"


@pytest.mark.parametrize("load_format", ["mx-source", "mx-target"])
def test_model_express_url_from_env_var(monkeypatch, mock_vllm_cli, load_format):
    """Test that MODEL_EXPRESS_URL env var is used as fallback."""
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://env-mx:9090")
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--load-format",
        load_format,
    )
    config = parse_args()
    assert config.model_express_url == "http://env-mx:9090"


@pytest.mark.parametrize("load_format", ["mx-source", "mx-target"])
def test_model_express_url_cli_overrides_env(monkeypatch, mock_vllm_cli, load_format):
    """Test that --model-express-url takes precedence over MODEL_EXPRESS_URL."""
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://env-mx:9090")
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--load-format",
        load_format,
        "--model-express-url",
        "http://cli-mx:8080",
    )
    config = parse_args()
    assert config.model_express_url == "http://cli-mx:8080"


@pytest.mark.parametrize("load_format", ["mx-source", "mx-target"])
def test_model_express_url_missing_raises(monkeypatch, mock_vllm_cli, load_format):
    """Test that missing server URL raises ValueError for mx load formats."""
    monkeypatch.delenv("MODEL_EXPRESS_URL", raising=False)
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--load-format",
        load_format,
    )
    with pytest.raises(
        ValueError,
        match=re.escape(f"--load-format={load_format}"),
    ):
        parse_args()


def test_model_express_url_none_for_default_load_format(mock_vllm_cli):
    """Test that model_express_url is None when load format is not mx-*."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.model_express_url is None


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

    from dynamo.vllm.main import build_headless_namespace

    ns = build_headless_namespace(config)

    # Required by run_headless()
    assert hasattr(ns, "api_server_count")
    assert ns.api_server_count == 0
    # Core engine fields must survive the round-trip
    assert hasattr(ns, "model")
    assert hasattr(ns, "tensor_parallel_size")


# --disaggregation-mode tests


def test_disaggregation_mode_default(mock_vllm_cli):
    """Test that default disaggregation mode is AGGREGATED."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED
    assert config.is_prefill_worker is False
    assert config.is_decode_worker is False


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
    assert config.is_prefill_worker is True
    assert config.is_decode_worker is False
    assert config.component == "prefill"


def test_disaggregation_mode_decode(mock_vllm_cli):
    """Test --disaggregation-mode decode sets correct state."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "decode")
    config = parse_args()
    assert config.disaggregation_mode == DisaggregationMode.DECODE
    assert config.is_prefill_worker is False
    assert config.is_decode_worker is True


def test_legacy_is_prefill_worker_emits_deprecation(mock_vllm_cli):
    """Test that --is-prefill-worker still works but emits DeprecationWarning."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--is-prefill-worker",
        "--kv-transfer-config",
        '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = parse_args()
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()
    assert config.disaggregation_mode == DisaggregationMode.PREFILL
    assert config.is_prefill_worker is True


def test_legacy_is_decode_worker_emits_deprecation(mock_vllm_cli):
    """Test that --is-decode-worker still works but emits DeprecationWarning."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--is-decode-worker")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = parse_args()
    deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(deprecation_warnings) >= 1
    assert "deprecated" in str(deprecation_warnings[0].message).lower()
    assert config.disaggregation_mode == DisaggregationMode.DECODE
    assert config.is_decode_worker is True


def test_conflicting_legacy_and_new_flags_raises(mock_vllm_cli):
    """Test that combining legacy flags with explicit --disaggregation-mode raises ValueError."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disaggregation-mode",
        "prefill",
        "--is-decode-worker",
    )
    with pytest.raises(ValueError, match="Cannot combine"):
        parse_args()


def test_explicit_default_mode_with_legacy_flag_raises(mock_vllm_cli):
    """Test that --disaggregation-mode agg --is-decode-worker raises ValueError."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disaggregation-mode",
        "agg",
        "--is-decode-worker",
    )
    with pytest.raises(ValueError, match="Cannot combine"):
        parse_args()


# --- _is_routable tests (pure logic, no mocking) ---


class TestIsRoutable:
    def test_accepts_routable_addresses(self):
        """Private/routable IPs (v4 and v6) must be accepted."""
        assert _is_routable("10.0.0.5") is True
        assert _is_routable("192.168.1.1") is True
        assert _is_routable("fd00::1") is True

    def test_rejects_non_routable_and_invalid(self):
        """Loopback, link-local, unspecified, multicast, and garbage must be rejected."""
        assert _is_routable("127.0.0.1") is False
        assert _is_routable("::1") is False
        assert _is_routable("169.254.1.1") is False
        assert _is_routable("fe80::1") is False
        assert _is_routable("0.0.0.0") is False
        assert _is_routable("not-an-ip") is False


# --- get_host_ip tests (mock socket module functions) ---


class TestGetHostIp:
    def test_hostname_resolution_success(self):
        """getaddrinfo returns routable IPv4 → returns it, skipping UDP."""
        with (
            patch("dynamo.vllm.args._try_hostname_resolution", return_value="10.0.0.5"),
            patch("dynamo.vllm.args._try_udp_connect") as mock_udp,
        ):
            result = get_host_ip()
            mock_udp.assert_not_called()
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
        """Pre-set env var → verify not overwritten and get_host_ip not called."""
        monkeypatch.setenv("VLLM_NIXL_SIDE_CHANNEL_HOST", "192.168.99.99")
        with patch("dynamo.vllm.args.get_host_ip") as mock_get:
            ensure_side_channel_host()
            mock_get.assert_not_called()
        import os

        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.99.99"

    def test_sets_env_var_from_auto_detection(self, monkeypatch):
        """No env var set → auto-detect writes detected IP into env."""
        import os

        monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
        with patch("dynamo.vllm.args.get_host_ip", return_value="10.1.2.3"):
            ensure_side_channel_host()
        assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.1.2.3"

    def test_no_fallback_on_detection_failure(self, monkeypatch):
        """When detection fails, env var must NOT be set to a localhost fallback."""
        import os

        monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
        with patch(
            "dynamo.vllm.args.get_host_ip",
            side_effect=RuntimeError("Unable to determine"),
        ):
            with pytest.raises(RuntimeError, match="Unable to determine"):
                ensure_side_channel_host()
        assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in os.environ
