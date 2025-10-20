# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRTLLM backend components."""

import re
from pathlib import Path

import pytest

from dynamo.trtllm.utils.trtllm_utils import cmd_line_args
from tests.unit.conftest import make_cli_args_fixture

# Get path relative to this test file
TEST_DIR = Path(__file__).parent.parent
JINJA_TEMPLATE_PATH = str(TEST_DIR / "serve" / "fixtures" / "custom_template.jinja")


pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm_marker,
    pytest.mark.gpu_1,
]


# Create TRTLLM-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_trtllm_cli = make_cli_args_fixture("dynamo.trtllm")


def test_custom_jinja_template_invalid_path(mock_trtllm_cli):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"
    mock_trtllm_cli(
        "--model", "Qwen/Qwen3-0.6B", "--custom-jinja-template", invalid_path
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        cmd_line_args()  # This will read in from argv


def test_custom_jinja_template_valid_path(mock_trtllm_cli):
    """Test that valid absolute path is stored correctly."""
    mock_trtllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)
    config = cmd_line_args()

    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )


def test_custom_jinja_template_env_var_expansion(monkeypatch, mock_trtllm_cli):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    mock_trtllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=cli_path)

    config = cmd_line_args()

    assert "$JINJA_DIR" not in config.custom_jinja_template
    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )
