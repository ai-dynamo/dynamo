# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM backend components."""

import re
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
