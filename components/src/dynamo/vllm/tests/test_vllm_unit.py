# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM backend components."""

import re
from pathlib import Path

import pytest

from dynamo.vllm.args import _connector_to_kv_transfer_json, parse_args
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
    """Test that --endpoint overrides even with --is-prefill-worker."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--endpoint",
        "dyn://custom.worker.serve",
        "--is-prefill-worker",
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
    """Test that --is-prefill-worker without --kv-transfer-config raises ValueError."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--is-prefill-worker")
    with pytest.raises(ValueError, match="--kv-transfer-config"):
        parse_args()


def test_connector_to_kv_transfer_json_single():
    """Test _connector_to_kv_transfer_json for single connector."""
    result = _connector_to_kv_transfer_json(["nixl"])
    assert '"NixlConnector"' in result
    assert '"kv_both"' in result


def test_connector_to_kv_transfer_json_multi():
    """Test _connector_to_kv_transfer_json for multiple connectors."""
    result = _connector_to_kv_transfer_json(["kvbm", "nixl"])
    assert '"PdConnector"' in result
    assert '"DynamoConnector"' in result
    assert '"NixlConnector"' in result


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
