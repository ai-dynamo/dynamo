# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM backend components."""

import json
import re
import warnings
from pathlib import Path

import pytest

from dynamo.vllm.args import parse_args
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


# --connector deprecation tests


def test_connector_nixl_emits_deprecation_warning(mock_vllm_cli):
    """Test that --connector nixl emits a DeprecationWarning with exact JSON."""
    mock_vllm_cli(
        "--model", "Qwen/Qwen3-0.6B", "--connector", "nixl", "--enforce-eager"
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_args()

    dep_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
    assert len(dep_warnings) >= 1, "Expected at least one DeprecationWarning"

    messages = [str(x.message) for x in dep_warnings]
    json_warning = [m for m in messages if "--kv-transfer-config" in m]
    assert json_warning, f"Expected --kv-transfer-config in warning, got: {messages}"
    # Should contain the exact NixlConnector JSON
    assert '"kv_connector":"NixlConnector"' in json_warning[0]


def test_connector_none_no_warning(mock_vllm_cli):
    """Test that --connector none does NOT warn (it's the only way to opt out of the default)."""
    mock_vllm_cli(
        "--model", "Qwen/Qwen3-0.6B", "--connector", "none", "--enforce-eager"
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_args()

    connector_warnings = [
        x
        for x in w
        if issubclass(x.category, FutureWarning) and "--connector" in str(x.message)
    ]
    assert len(connector_warnings) == 0, (
        f"Expected no --connector warnings for --connector none, got: "
        f"{[str(x.message) for x in connector_warnings]}"
    )


def test_implicit_default_connector_emits_warning(mock_vllm_cli):
    """Test that the implicit default --connector nixl emits a FutureWarning about the upcoming change."""
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--enforce-eager")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_args()

    dep_warnings = [
        x
        for x in w
        if issubclass(x.category, FutureWarning)
        and "default --connector nixl" in str(x.message)
    ]
    assert len(dep_warnings) == 1, (
        f"Expected one implicit-default FutureWarning, got: "
        f"{[str(x.message) for x in dep_warnings]}"
    )
    # Should suggest the replacement JSON
    assert "--kv-transfer-config" in str(dep_warnings[0].message)


def test_env_var_dyn_connector_emits_deprecation_warning(monkeypatch, mock_vllm_cli):
    """Test that DYN_CONNECTOR env var also triggers deprecation warning."""
    monkeypatch.setenv("DYN_CONNECTOR", "nixl")
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B", "--enforce-eager")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parse_args()

    dep_warnings = [x for x in w if issubclass(x.category, FutureWarning)]
    assert (
        len(dep_warnings) >= 1
    ), "Expected DeprecationWarning for DYN_CONNECTOR env var"

    messages = [str(x.message) for x in dep_warnings]
    json_warning = [m for m in messages if "--kv-transfer-config" in m]
    assert json_warning, f"Expected --kv-transfer-config in warning, got: {messages}"


def test_connector_to_kv_transfer_json_single_nixl():
    """Test _connector_to_kv_transfer_json for single nixl connector."""
    from dynamo.vllm.args import _connector_to_kv_transfer_json

    result = _connector_to_kv_transfer_json(["nixl"])
    parsed = json.loads(result)
    assert parsed["kv_connector"] == "NixlConnector"
    assert parsed["kv_role"] == "kv_both"


def test_connector_to_kv_transfer_json_single_lmcache():
    """Test _connector_to_kv_transfer_json for single lmcache connector."""
    from dynamo.vllm.args import _connector_to_kv_transfer_json

    result = _connector_to_kv_transfer_json(["lmcache"])
    parsed = json.loads(result)
    assert parsed["kv_connector"] == "LMCacheConnectorV1"


def test_connector_to_kv_transfer_json_single_kvbm():
    """Test _connector_to_kv_transfer_json for single kvbm connector."""
    from dynamo.vllm.args import _connector_to_kv_transfer_json

    result = _connector_to_kv_transfer_json(["kvbm"])
    parsed = json.loads(result)
    assert parsed["kv_connector"] == "DynamoConnector"
    assert parsed["kv_connector_module_path"] == "kvbm.vllm_integration.connector"


def test_connector_to_kv_transfer_json_multi():
    """Test _connector_to_kv_transfer_json for multiple connectors uses PdConnector."""
    from dynamo.vllm.args import _connector_to_kv_transfer_json

    result = _connector_to_kv_transfer_json(["kvbm", "nixl"])
    parsed = json.loads(result)
    assert parsed["kv_connector"] == "PdConnector"
    assert "connectors" in parsed["kv_connector_extra_config"]
