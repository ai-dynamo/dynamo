# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import re
import sys
from pathlib import Path

import pytest
import yaml

from dynamo.sglang.args import parse_args
from dynamo.sglang.tests.conftest import make_cli_args_fixture

# Get path relative to this test file
REPO_ROOT = Path(__file__).resolve().parents[5]
TEST_DIR = REPO_ROOT / "tests"
# Now construct the full path to the shared test fixture
JINJA_TEMPLATE_PATH = str(
    REPO_ROOT / "tests" / "serve" / "fixtures" / "custom_template.jinja"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]
# Create SGLang-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


@pytest.mark.asyncio
async def test_custom_jinja_template_invalid_path(mock_sglang_cli):
    """Test that invalid file path raises FileNotFoundError."""
    invalid_path = "/nonexistent/path/to/template.jinja"
    mock_sglang_cli(
        "--model", "Qwen/Qwen3-0.6B", "--custom-jinja-template", invalid_path
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(f"Custom Jinja template file not found: {invalid_path}"),
    ):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_custom_jinja_template_valid_path(mock_sglang_cli):
    """Test that valid absolute path is stored correctly."""
    mock_sglang_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


@pytest.mark.asyncio
async def test_custom_jinja_template_env_var_expansion(monkeypatch, mock_sglang_cli):
    """Test that environment variables in paths are expanded by Python code."""
    jinja_dir = str(TEST_DIR / "serve" / "fixtures")
    monkeypatch.setenv("JINJA_DIR", jinja_dir)

    cli_path = "$JINJA_DIR/custom_template.jinja"
    mock_sglang_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=cli_path)

    config = await parse_args(sys.argv[1:])

    assert "$JINJA_DIR" not in config.dynamo_args.custom_jinja_template
    assert config.dynamo_args.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.dynamo_args.custom_jinja_template}"
    )


# --- Tool Call Parser Validation Tests ---


@pytest.mark.asyncio
async def test_tool_call_parser_valid_with_dynamo_tokenizer(mock_sglang_cli):
    """Valid parser name works when using Dynamo's tokenizer."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-tool-call-parser",
        "hermes",  # supported by Dynamo
    )

    config = await parse_args(sys.argv[1:])

    assert config.dynamo_args.dyn_tool_call_parser == "hermes"


@pytest.mark.asyncio
async def test_tool_call_parser_invalid_with_dynamo_tokenizer(mock_sglang_cli):
    """Invalid parser name exits when using Dynamo's tokenizer."""
    mock_sglang_cli(
        "--model", "Qwen/Qwen3-0.6B", "--dyn-tool-call-parser", "nonexistent_parser"
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_tool_call_parser_both_flags_error(mock_sglang_cli):
    """Setting both --dyn-tool-call-parser and --tool-call-parser exits with error."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-tool-call-parser",
        "hermes",
        "--tool-call-parser",
        "qwen25",
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_namespace_flag_drives_default_endpoint_namespace(mock_sglang_cli):
    """CLI namespace should be used for auto-derived endpoint."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--namespace",
        "custom-ns",
    )

    config = await parse_args(sys.argv[1:])
    assert config.dynamo_args.namespace == "custom-ns"


@pytest.mark.asyncio
async def test_obsolete_dyn_endpoint_types_flag_is_supported(mock_sglang_cli):
    """Obsolete --dyn-endpoint-types alias should map to endpoint_types."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dyn-endpoint-types",
        "completions",
    )

    config = await parse_args(sys.argv[1:])
    assert config.dynamo_args.endpoint_types == "completions"


@pytest.mark.asyncio
async def test_disagg_config_requires_disagg_config_key(mock_sglang_cli):
    """--disagg-config and --disagg-config-key must be provided together."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        "/tmp/nonexistent.yaml",
    )

    with pytest.raises(ValueError, match="disagg_config.*disagg_config_key.*together"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_key_requires_disagg_config(mock_sglang_cli):
    """--disagg-config-key alone should fail."""
    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(ValueError, match="disagg_config.*disagg_config_key.*together"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_key_not_found_error(tmp_path, mock_sglang_cli):
    """Missing disagg section key should raise a clear ValueError."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"tensor_parallel_size": 1}}), encoding="utf-8"
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "decode",
    )

    with pytest.raises(ValueError, match="Disagg config key 'decode' not found"):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_section_must_be_dict(tmp_path, mock_sglang_cli):
    """Selected disagg section must be a dictionary."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(yaml.safe_dump({"prefill": "not-a-dict"}), encoding="utf-8")

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(
        ValueError, match="Disagg config section 'prefill' must be a dictionary"
    ):
        await parse_args(sys.argv[1:])


@pytest.mark.asyncio
async def test_disagg_config_preserves_bootstrap_port(tmp_path, mock_sglang_cli):
    """Bootstrap port from disagg section should not be overridden by auto-port logic."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"disaggregation-bootstrap-port": 42345}}),
        encoding="utf-8",
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    config = await parse_args(sys.argv[1:])
    assert config.server_args.disaggregation_bootstrap_port == 42345


@pytest.mark.asyncio
async def test_disagg_config_rejects_dynamo_keys(tmp_path, mock_sglang_cli, capfd):
    """Disagg config should only accept SGLang-native keys."""
    config_path = tmp_path / "disagg.yaml"
    config_path.write_text(
        yaml.safe_dump({"prefill": {"store-kv": "mem"}}), encoding="utf-8"
    )

    mock_sglang_cli(
        "--model",
        "Qwen/Qwen3-0.6B",
        "--disagg-config",
        str(config_path),
        "--disagg-config-key",
        "prefill",
    )

    with pytest.raises(SystemExit):
        await parse_args(sys.argv[1:])

    out, err = capfd.readouterr()
    assert "unrecognized arguments: --store-kv mem" in err
