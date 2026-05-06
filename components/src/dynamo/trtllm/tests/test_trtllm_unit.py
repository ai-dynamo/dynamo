# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRTLLM backend components."""

import asyncio
import re
from pathlib import Path
from unittest import mock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )

from dynamo.trtllm.args import Config, parse_args
from dynamo.trtllm.constants import Modality
from dynamo.trtllm.tests.conftest import make_cli_args_fixture
from dynamo.trtllm.utils.trtllm_utils import deep_update, warn_override_collisions
from dynamo.trtllm.workers.llm_worker import init_llm_worker

# Get path relative to this test file
REPO_ROOT = Path(__file__).resolve().parents[5]
TEST_DIR = REPO_ROOT / "tests"
# Now construct the full path to the shared test fixture
JINJA_TEMPLATE_PATH = str(
    REPO_ROOT / "tests" / "serve" / "fixtures" / "custom_template.jinja"
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
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
        parse_args()  # Reads from argv set by fixture


def test_custom_jinja_template_valid_path(mock_trtllm_cli):
    """Test that valid absolute path is stored correctly."""
    mock_trtllm_cli(model="Qwen/Qwen3-0.6B", custom_jinja_template=JINJA_TEMPLATE_PATH)
    config = parse_args()

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

    config = parse_args()

    assert "$JINJA_DIR" not in config.custom_jinja_template
    assert config.custom_jinja_template == JINJA_TEMPLATE_PATH, (
        f"Expected custom_jinja_template value to be {JINJA_TEMPLATE_PATH}, "
        f"got {config.custom_jinja_template}"
    )


# ---- Tests for trtllm/args.py (Config, parse_args) ----


def test_parse_args_returns_config_with_expected_attrs(monkeypatch):
    """parse_args returns a Config instance with model, component, and endpoint set."""
    monkeypatch.delenv("DYN_NAMESPACE", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_MODEL", raising=False)
    config = parse_args(["--namespace", "testns", "--model-path", "Qwen/Qwen3-0.6B"])
    assert isinstance(config, Config)
    assert config.model == "Qwen/Qwen3-0.6B"
    assert config.namespace == "testns"
    assert config.component == "tensorrt_llm"
    assert config.endpoint == "generate"


def test_config_use_kv_events_derived_from_publish_events(monkeypatch):
    """Config.validate sets use_kv_events from publish_events_and_metrics."""
    monkeypatch.delenv("DYN_TRTLLM_PUBLISH_EVENTS", raising=False)
    config = parse_args(["--publish-events"])
    assert config.publish_events_and_metrics is True
    assert config.use_kv_events is True

    config_off = parse_args(["--no-publish-events"])
    assert config_off.publish_events_and_metrics is False
    assert config_off.use_kv_events is False


def test_config_has_connector(monkeypatch):
    """Config.has_connector returns True only for the single configured connector."""
    monkeypatch.delenv("DYN_CONNECTOR", raising=False)
    config_none = parse_args(["--connector", "none"])
    assert config_none.has_connector("none") is True
    assert config_none.has_connector("kvbm") is False

    config_kvbm = parse_args(["--connector", "kvbm"])
    assert config_kvbm.has_connector("kvbm") is True
    assert config_kvbm.has_connector("none") is False


def test_config_multiple_connectors_fails(monkeypatch):
    """Config.validate fails if multiple connectors are provided."""
    monkeypatch.delenv("DYN_CONNECTOR", raising=False)
    with pytest.raises(
        ValueError,
        match="TRT-LLM supports at most one connector entry. Use `--connector none` or `--connector kvbm`.",
    ):
        parse_args(["--connector", "none", "kvbm"])


# ---- Tests for trtllm_utils.deep_update ----


def test_deep_update_nested_merge():
    """deep_update merges nested dicts without removing existing keys."""
    target = {"a": 1, "b": {"x": 10, "y": 20}}
    source = {"b": {"y": 21, "z": 30}}
    deep_update(target, source)
    assert target == {"a": 1, "b": {"x": 10, "y": 21, "z": 30}}


def test_deep_update_overwrites_scalar_with_value():
    """deep_update overwrites a key with a non-dict value."""
    target = {"a": 1, "b": {"x": 10}}
    source = {"a": 2, "b": 99}
    deep_update(target, source)
    assert target == {"a": 2, "b": 99}


def test_deep_update_empty_source_unchanged():
    """deep_update with empty source leaves target unchanged."""
    target = {"a": 1, "b": {"x": 10}}
    deep_update(target, {})
    assert target == {"a": 1, "b": {"x": 10}}


def test_deep_update_adds_new_keys():
    """deep_update adds new keys from source that are not in target."""
    target = {"a": 1}
    source = {"b": 2, "c": {"nested": 3}}
    deep_update(target, source)
    assert target == {"a": 1, "b": 2, "c": {"nested": 3}}


# ---- Tests for trtllm_utils.warn_override_collisions ----


def test_warn_override_collisions_logs_replaced_scalar(caplog):
    """A scalar override that changes an existing value emits a warning."""
    target = {"max_seq_len": 1024}
    source = {"max_seq_len": 2048}
    with caplog.at_level("WARNING"):
        warn_override_collisions(target, source)
    assert any(
        "max_seq_len" in r.message and "1024" in r.message and "2048" in r.message
        for r in caplog.records
    )


def test_warn_override_collisions_recurses_into_nested_dicts(caplog):
    """Nested-dict overrides report the full dotted path."""
    target = {"kv_cache_config": {"max_tokens": 1000, "free_gpu_memory_fraction": 0.85}}
    source = {"kv_cache_config": {"max_tokens": 2592}}
    with caplog.at_level("WARNING"):
        warn_override_collisions(target, source)
    assert any("kv_cache_config.max_tokens" in r.message for r in caplog.records)
    # free_gpu_memory_fraction wasn't in source — should not warn.
    assert not any("free_gpu_memory_fraction" in r.message for r in caplog.records)


def test_warn_override_collisions_skips_new_keys(caplog):
    """Keys not present in target are additions, not collisions — no warning."""
    target = {"max_seq_len": 1024}
    source = {"max_batch_size": 32}
    with caplog.at_level("WARNING"):
        warn_override_collisions(target, source)
    assert caplog.records == []


def test_warn_override_collisions_skips_identical_values(caplog):
    """Override with the same value as target is a no-op — no warning."""
    target = {"max_seq_len": 1024}
    source = {"max_seq_len": 1024}
    with caplog.at_level("WARNING"):
        warn_override_collisions(target, source)
    assert caplog.records == []


# ---- Tests for engine_args resolution with extra/override engine args ----


class EngineArgsCaptured(Exception):
    """Raised by mocked get_llm_engine to capture engine_args and stop execution."""

    def __init__(self, engine_args):
        self.engine_args = engine_args


def _mock_get_llm_engine(engine_args, *args, **kwargs):
    """Mock for get_llm_engine that captures engine_args and short-circuits."""
    raise EngineArgsCaptured(engine_args)


@pytest.mark.asyncio
async def test_init_llm_worker_engine_args_without_overrides(monkeypatch):
    """Without overrides, engine_args passed to get_llm_engine use CLI defaults."""
    monkeypatch.delenv("DYN_TRTLLM_MAX_NUM_TOKENS", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_MAX_BATCH_SIZE", raising=False)

    config = parse_args(["--model", "fake-model"])

    with (
        mock.patch("dynamo.trtllm.workers.llm_worker.tokenizer_factory"),
        mock.patch("dynamo.trtllm.workers.llm_worker.nixl_connect.Connector"),
        mock.patch("dynamo.trtllm.workers.llm_worker.dump_config"),
        mock.patch("dynamo.trtllm.workers.llm_worker.LLMBackendMetrics"),
        mock.patch(
            "dynamo.trtllm.workers.llm_worker.get_llm_engine",
            side_effect=_mock_get_llm_engine,
        ),
    ):
        with pytest.raises(EngineArgsCaptured) as exc_info:
            await init_llm_worker(
                runtime=mock.MagicMock(),
                config=config,
                shutdown_event=asyncio.Event(),
            )

        engine_args = exc_info.value.engine_args
        assert engine_args["max_num_tokens"] == config.max_num_tokens
        assert engine_args["max_batch_size"] == config.max_batch_size


@pytest.mark.asyncio
async def test_init_llm_worker_engine_args_with_extra_engine_args(
    tmp_path, monkeypatch
):
    """--extra-engine-args YAML overrides are reflected in engine_args passed to get_llm_engine."""
    monkeypatch.delenv("DYN_TRTLLM_MAX_NUM_TOKENS", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_MAX_BATCH_SIZE", raising=False)

    yaml_file = tmp_path / "engine_config.yaml"
    yaml_file.write_text("max_num_tokens: 32768\nmax_batch_size: 512\n")

    config = parse_args(
        [
            "--model",
            "fake-model",
            "--extra-engine-args",
            str(yaml_file),
        ]
    )
    # CLI config should NOT reflect the YAML values
    assert config.max_num_tokens != 32768
    assert config.max_batch_size != 512

    with (
        mock.patch("dynamo.trtllm.workers.llm_worker.tokenizer_factory"),
        mock.patch("dynamo.trtllm.workers.llm_worker.nixl_connect.Connector"),
        mock.patch("dynamo.trtllm.workers.llm_worker.dump_config"),
        mock.patch("dynamo.trtllm.workers.llm_worker.LLMBackendMetrics"),
        mock.patch(
            "dynamo.trtllm.workers.llm_worker.get_llm_engine",
            side_effect=_mock_get_llm_engine,
        ),
    ):
        with pytest.raises(EngineArgsCaptured) as exc_info:
            await init_llm_worker(
                runtime=mock.MagicMock(),
                config=config,
                shutdown_event=asyncio.Event(),
            )

        engine_args = exc_info.value.engine_args
        assert engine_args["max_num_tokens"] == 32768, (
            f"Expected max_num_tokens=32768 from YAML override, "
            f"got {engine_args['max_num_tokens']}"
        )
        assert engine_args["max_batch_size"] == 512, (
            f"Expected max_batch_size=512 from YAML override, "
            f"got {engine_args['max_batch_size']}"
        )


class MultimodalProcessorInstantiated(Exception):
    """Custom exception for testing MultimodalRequestProcessor."""


@pytest.mark.asyncio
async def test_init_llm_worker_creates_multimodal_processor():
    config = parse_args(["--model", "fake-model", "--modality", "multimodal"])
    assert config.modality == Modality.MULTIMODAL

    # Mock everything init_llm_worker touches before MultimodalRequestProcessor.
    with (
        mock.patch("dynamo.trtllm.workers.llm_worker.tokenizer_factory"),
        mock.patch(
            "dynamo.trtllm.workers.llm_worker.AutoConfig.from_pretrained",
        ),
        mock.patch(
            "dynamo.trtllm.workers.llm_worker.MultimodalRequestProcessor",
            side_effect=MultimodalProcessorInstantiated,
        ),
    ):
        with pytest.raises(MultimodalProcessorInstantiated):
            await init_llm_worker(
                runtime=mock.MagicMock(),
                config=config,
                shutdown_event=asyncio.Event(),
            )
