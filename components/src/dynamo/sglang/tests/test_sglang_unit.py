# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import re
import sys
from pathlib import Path

import pytest
import yaml
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

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


# ---------------------------------------------------------------------------
# LoRA registration model_type gate
# ---------------------------------------------------------------------------
# Pins the serving_mode → model_type selection in LoraMixin.load_lora so a
# refactor that flips prefill back to Chat|Completions cannot silently land:
# it would re-introduce the disagg hang where the frontend routes
# /v1/chat/completions directly to the prefill worker.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "serving_mode, endpoint_types, expected_model_type_str",
    [
        ("prefill", "chat,completions", "prefill"),
        ("decode", "chat,completions", "chat,completions"),
        ("agg", "chat,completions", "chat,completions"),
        ("decode", "completions", "completions"),
        ("agg", "chat", "chat"),
    ],
)
async def test_lora_registration_model_type_gate(
    monkeypatch, serving_mode, endpoint_types, expected_model_type_str
):
    """LoraMixin.load_lora must select model_type based on serving_mode.

    PREFILL → ModelType.Prefill (so the prefill router activates and the frontend
    does not route chat completions directly to prefill).
    Otherwise → parse_endpoint_types(endpoint_types) (mirrors base-model
    registration so --endpoint-types overrides are honored).
    """
    from unittest.mock import AsyncMock, MagicMock

    from dynamo.common.constants import DisaggregationMode
    from dynamo.sglang.request_handlers import handler_base
    from dynamo.sglang.request_handlers.handler_base import LoraMixin

    # Capture the kwargs passed to register_llm.
    captured: dict = {}

    async def fake_register_llm(**kw):
        captured.update(kw)

    # Fake LoRA manager that returns a successful download.
    fake_lora_manager = MagicMock()
    fake_lora_manager.download_lora = AsyncMock(
        return_value={"status": "success", "local_path": "/tmp/fake_lora"}
    )

    monkeypatch.setattr(handler_base, "register_llm", fake_register_llm)
    monkeypatch.setattr(handler_base, "get_lora_manager", lambda: fake_lora_manager)
    monkeypatch.setattr(handler_base, "lora_name_to_id", lambda name: 12345)

    # Fake SGLang engine — only the LoRA load path is exercised.
    fake_load_result = SimpleNamespace(success=True, error_message=None)
    fake_engine = MagicMock()
    fake_engine.tokenizer_manager = MagicMock()
    fake_engine.tokenizer_manager.load_lora_adapter = AsyncMock(
        return_value=fake_load_result
    )

    # Exercise the mixin in isolation — avoids needing a concrete subclass
    # with abstract methods, real publisher, runtime, etc. The mixin only
    # touches engine, config, generate_endpoint, and its own LoRA tracking.
    class _Host(LoraMixin):
        pass

    handler = _Host()
    handler.engine = fake_engine
    handler.generate_endpoint = MagicMock()

    config = MagicMock()
    config.serving_mode = DisaggregationMode(serving_mode)
    config.server_args.model_path = "/models/base"
    config.server_args.page_size = 16
    config.dynamo_args.endpoint_types = endpoint_types
    handler.config = config

    handler._init_lora_tracking()

    # Drain the async generator.
    results = [
        chunk
        async for chunk in handler.load_lora(
            {"lora_name": "test_lora", "source": {"uri": "s3://x/y"}}
        )
    ]

    assert results and results[-1]["status"] == "success", results
    assert captured, "register_llm was not invoked"
    assert (
        str(captured["model_type"]) == expected_model_type_str
    ), f"model_type {captured['model_type']} != expected {expected_model_type_str}"
    assert captured["lora_name"] == "test_lora"
