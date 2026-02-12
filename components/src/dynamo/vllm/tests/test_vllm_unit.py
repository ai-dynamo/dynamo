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
    pytest.mark.gpu_0,
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


def test_default_endpoint_from_model_name(mock_vllm_cli):
    """Test that endpoint is auto-generated from model name."""
    mock_vllm_cli("--model", "Qwen/Qwen2.5-7B-Instruct")

    config = parse_args()

    # Should generate unique endpoint from model name
    assert config.endpoint.startswith("generate_qwen_qwen2_5-7b-instruct_")
    assert config.namespace == "dynamo"
    assert config.component == "backend"


def test_explicit_endpoint_override(mock_vllm_cli):
    """Test that --endpoint flag overrides default."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--endpoint",
        "dyn://custom.backend.myendpoint",
    )

    config = parse_args()

    # Should use explicit endpoint
    assert config.namespace == "custom"
    assert config.component == "backend"
    assert config.endpoint == "myendpoint"


def test_different_models_different_endpoints(mock_vllm_cli):
    """Test that different models get different endpoints."""
    # Model 1
    mock_vllm_cli("--model", "Qwen/Qwen2.5-7B-Instruct")
    config1 = parse_args()
    ep1 = config1.endpoint

    # Model 2
    mock_vllm_cli("--model", "Qwen/Qwen3-0.6B")
    config2 = parse_args()
    ep2 = config2.endpoint

    # Endpoints should be different
    assert ep1 != ep2


def test_prefill_worker_keeps_generate_endpoint(mock_vllm_cli):
    """Test that prefill workers keep hardcoded 'generate' endpoint."""
    mock_vllm_cli("--model", "Qwen/Qwen2.5-7B-Instruct", "--is-prefill-worker")

    config = parse_args()

    # Prefill workers should keep "generate" endpoint
    assert config.endpoint == "generate"
    assert config.component == "prefill"
    assert config.namespace == "dynamo"


def test_multimodal_processor_keeps_generate_endpoint(mock_vllm_cli):
    """Test that multimodal processor workers keep hardcoded 'generate' endpoint."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--enable-multimodal",
        "--multimodal-processor",
    )

    config = parse_args()

    # Multimodal processor should keep "generate" endpoint
    assert config.endpoint == "generate"
    assert config.component == "processor"
    assert config.namespace == "dynamo"


def test_encoder_worker_keeps_generate_endpoint(mock_vllm_cli):
    """Test that encoder workers keep hardcoded 'generate' endpoint."""
    mock_vllm_cli(
        "--model",
        "Qwen/Qwen2.5-7B-Instruct",
        "--enable-multimodal",
        "--multimodal-encode-worker",
    )

    config = parse_args()

    # Encoder worker should keep "generate" endpoint
    assert config.endpoint == "generate"
    assert config.component == "encoder"
    assert config.namespace == "dynamo"


def test_decode_worker_keeps_generate_endpoint(mock_vllm_cli):
    """Test that decode workers keep hardcoded 'generate' endpoint for planner compatibility."""
    mock_vllm_cli("--model", "Qwen/Qwen2.5-7B-Instruct", "--is-decode-worker")

    config = parse_args()

    # Decode workers should keep "generate" endpoint for planner compatibility
    assert config.endpoint == "generate"
    assert config.component == "backend"
    assert config.namespace == "dynamo"
