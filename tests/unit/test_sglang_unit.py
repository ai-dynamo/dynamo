# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import re
import sys
from pathlib import Path

import pytest

from dynamo.sglang.args import parse_args

# Get path relative to this test file
TEST_DIR = Path(__file__).parent.parent
JINJA_TEMPLATE_PATH = str(TEST_DIR / "serve" / "fixtures" / "custom_template.jinja")


pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
]


def test_custom_jinja_template_invalid_path(monkeypatch):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.sglang",
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--custom-jinja-template",
            invalid_path,
        ],
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        parse_args(sys.argv[1:])


def test_custom_jinja_template_valid_path(monkeypatch):
    """Test that valid absolute path is stored correctly."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.sglang",
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--custom-jinja-template",
            JINJA_TEMPLATE_PATH,
        ],
    )

    config = parse_args(sys.argv[1:])

    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


def test_custom_jinja_template_env_var_expansion(monkeypatch):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.sglang",
            "--model-path",
            "Qwen/Qwen3-0.6B",
            "--custom-jinja-template",
            cli_path,
        ],
    )

    config = parse_args(sys.argv[1:])

    assert "$JINJA_DIR" not in config.dynamo_args.custom_jinja_template
    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )
