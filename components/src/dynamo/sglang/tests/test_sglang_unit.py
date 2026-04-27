# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import re
import sys
from pathlib import Path

import pytest
import yaml
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

import dynamo.sglang._compat as sglang_compat
from dynamo.sglang._compat import (
    ensure_sglang_top_level_exports,
    filter_supported_async_generate_kwargs,
)
from dynamo.sglang.args import parse_args
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
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
    pytest.mark.gpu_1,  # needs sglang & GPU packages installed but does not actually use GPU
    pytest.mark.profiled_vram_gib(0),  # These unit tests do not actually use GPU VRAM
    pytest.mark.pre_merge,
]
# Create SGLang-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


def test_compat_restores_sglang_top_level_exports():
    """Dynamo supports SGLang builds that omit top-level Engine/ServerArgs."""
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    missing = object()
    original_engine = getattr(sgl, "Engine", missing)
    original_server_args = getattr(sgl, "ServerArgs", missing)

    try:
        if hasattr(sgl, "Engine"):
            delattr(sgl, "Engine")
        if hasattr(sgl, "ServerArgs"):
            delattr(sgl, "ServerArgs")

        ensure_sglang_top_level_exports()

        assert sgl.Engine is Engine
        assert sgl.ServerArgs is ServerArgs
    finally:
        if original_engine is missing:
            if hasattr(sgl, "Engine"):
                delattr(sgl, "Engine")
        else:
            sgl.Engine = original_engine

        if original_server_args is missing:
            if hasattr(sgl, "ServerArgs"):
                delattr(sgl, "ServerArgs")
        else:
            sgl.ServerArgs = original_server_args


def test_compat_filters_async_generate_kwargs_for_older_engines():
    class OldEngine:
        async def async_generate(self, input_ids=None, sampling_params=None):
            return None

    kwargs = {
        "input_ids": [1, 2, 3],
        "return_routed_experts": True,
    }

    assert filter_supported_async_generate_kwargs(OldEngine(), kwargs) == {
        "input_ids": [1, 2, 3]
    }


def test_compat_keeps_async_generate_kwargs_for_newer_engines():
    class NewEngine:
        async def async_generate(self, return_routed_experts=False):
            return None

    kwargs = {"return_routed_experts": True}

    assert filter_supported_async_generate_kwargs(NewEngine(), kwargs) == kwargs


def test_compat_keeps_async_generate_kwargs_for_variadic_engines():
    class VariadicEngine:
        async def async_generate(self, **kwargs):
            return None

    kwargs = {"return_routed_experts": True}

    assert filter_supported_async_generate_kwargs(VariadicEngine(), kwargs) == kwargs


def test_compat_caches_async_generate_signature_inspection(monkeypatch):
    class CachedEngine:
        async def async_generate(self, return_routed_experts=False):
            return None

    sglang_compat._get_async_generate_supported_kwarg_names.cache_clear()
    calls = 0
    original_signature = sglang_compat.inspect.signature

    def counting_signature(obj):
        nonlocal calls
        calls += 1
        return original_signature(obj)

    monkeypatch.setattr(sglang_compat.inspect, "signature", counting_signature)

    kwargs = {"return_routed_experts": True}
    assert filter_supported_async_generate_kwargs(CachedEngine(), kwargs) == kwargs
    assert filter_supported_async_generate_kwargs(CachedEngine(), kwargs) == kwargs
    assert calls == 1

    sglang_compat._get_async_generate_supported_kwarg_names.cache_clear()


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


def test_disagg_health_check_payload_includes_bootstrap_info():
    payload = SglangDisaggHealthCheckPayload().to_dict()

    assert payload["bootstrap_info"]["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert payload["bootstrap_info"]["bootstrap_port"] == 0
    assert payload["bootstrap_info"]["bootstrap_room"] == 0
    assert payload["token_ids"] == [1]


def test_prefill_health_check_payload_is_disagg_compatible_alias():
    payload = SglangPrefillHealthCheckPayload().to_dict()

    assert "request" not in payload
    assert payload["bootstrap_info"]["bootstrap_host"] == FAKE_BOOTSTRAP_HOST
    assert payload["stop_conditions"]["max_tokens"] == 1
