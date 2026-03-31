# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SGLang backend components."""

import importlib
import os
import re
import sys
import types
from pathlib import Path

import pytest
import sglang.srt.distributed as distributed_module
import yaml
import sglang.srt.distributed.device_communicators as device_communicators
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

from dynamo.sglang.args import parse_args
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangPrefillHealthCheckPayload,
)
from dynamo.sglang import patches as sglang_patches
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
    pytest.mark.gpu_1,  # needs sglang installed (GPU node) but uses no GPU
    pytest.mark.max_vram_gib(0),
    pytest.mark.pre_merge,
]
# Create SGLang-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_sglang_cli = make_cli_args_fixture("dynamo.sglang")


def test_snapshot_patches_force_file_dist_init_and_pynccl(monkeypatch):
    """Checkpoint mode should avoid TCPStore listeners and keep PyNccl enabled."""

    class FakePyNcclCommunicator:
        def __init__(self, *args, **kwargs):
            self.available = kwargs.get("available", True)
            self.world_size = kwargs.get("world_size", 2)
            self.disabled = True

    class FakeGroup:
        def __init__(self):
            self.world_size = 2
            self.pynccl_comm = FakePyNcclCommunicator(world_size=2, available=True)

    def fake_init_model_parallel_group(*args, **kwargs):
        return FakeGroup()

    scheduler_calls = []

    def fake_run_scheduler_process(*args, **kwargs):
        scheduler_calls.append((args, kwargs))

    class FakeEngine:
        run_scheduler_process_func = staticmethod(fake_run_scheduler_process)

    fake_pynccl_module = types.SimpleNamespace(PyNcclCommunicator=FakePyNcclCommunicator)
    fake_parallel_state_module = types.SimpleNamespace(
        init_model_parallel_group=fake_init_model_parallel_group
    )
    fake_engine_module = types.ModuleType("sglang.srt.entrypoints.engine")
    fake_engine_module.Engine = FakeEngine
    stale_store_path = Path("/tmp/dynamo-sglang-dist-init-test-pod")
    stale_store_path.write_text("stale", encoding="utf-8")

    monkeypatch.setenv("POD_UID", "test-pod")
    monkeypatch.setenv("SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE", "tcp://127.0.0.1:1234")
    monkeypatch.setattr(device_communicators, "pynccl", fake_pynccl_module, raising=False)
    monkeypatch.setattr(
        distributed_module,
        "parallel_state",
        fake_parallel_state_module,
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.device_communicators.pynccl",
        fake_pynccl_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.distributed.parallel_state",
        fake_parallel_state_module,
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.engine", fake_engine_module)

    patches = importlib.reload(sglang_patches)
    patches.apply_snapshot_patches()

    assert os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] == (
        "file:///tmp/dynamo-sglang-dist-init-test-pod"
    )
    assert not stale_store_path.exists()

    communicator = fake_pynccl_module.PyNcclCommunicator(world_size=2, available=True)
    assert communicator.disabled is False

    group = fake_parallel_state_module.init_model_parallel_group([], 0, "gloo")
    assert group.pynccl_comm.disabled is False

    FakeEngine.run_scheduler_process_func("tp0", rank=0)
    assert scheduler_calls == [(("tp0",), {"rank": 0})]

    single_rank = fake_pynccl_module.PyNcclCommunicator(world_size=1, available=True)
    assert single_rank.disabled is True


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
