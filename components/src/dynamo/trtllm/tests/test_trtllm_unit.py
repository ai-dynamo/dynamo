# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TRTLLM backend components."""

import asyncio
import os
import re
import warnings
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
from dynamo.trtllm.constants import DisaggregationMode, Modality
from dynamo.trtllm.tests.conftest import make_cli_args_fixture
from dynamo.trtllm.utils.trtllm_utils import deep_update, warn_override_collisions
from dynamo.trtllm.workers.llm_worker import (
    _populate_kv_cache_capacity,
    init_llm_worker,
)

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
    pytest.mark.profiled_vram_gib(0),
]


# Create TRTLLM-specific CLI args fixture
# This will use monkeypatch to write to argv
mock_trtllm_cli = make_cli_args_fixture("dynamo.trtllm")


def test_populate_kv_cache_capacity_publishes_engine_values():
    runtime_config = mock.Mock()
    engine = mock.Mock()
    engine.get_kv_cache_capacity.return_value = {
        "maxNumBlocks": 123,
        "tokensPerBlock": 64,
        "maxNumTokens": 7872,
    }

    block_size = _populate_kv_cache_capacity(runtime_config, engine, 32)

    assert runtime_config.total_kv_blocks == 123
    assert block_size == 64


def test_populate_kv_cache_capacity_falls_back_when_unavailable(caplog):
    runtime_config = mock.Mock()
    engine = mock.Mock()
    engine.get_kv_cache_capacity.return_value = {}

    with caplog.at_level("WARNING"):
        block_size = _populate_kv_cache_capacity(runtime_config, engine, 32)

    assert block_size == 32
    assert "Planner KV-rate scaling will remain unavailable" in caplog.text


def test_populate_kv_cache_capacity_rejects_invalid_values():
    runtime_config = mock.Mock()
    engine = mock.Mock()
    engine.get_kv_cache_capacity.return_value = {
        "maxNumBlocks": 0,
        "tokensPerBlock": 64,
        "maxNumTokens": 0,
    }

    with pytest.raises(ValueError, match="Invalid TRT-LLM KV-cache capacity"):
        _populate_kv_cache_capacity(runtime_config, engine, 32)


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
    assert config.component == "backend"
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


def test_deprecated_publish_events_flag_alias_maps_and_logs(monkeypatch, caplog):
    """The deprecated --publish-events-and-metrics alias must still map to
    publish_events_and_metrics AND surface its deprecation notice on the log
    stream, which is visible under CPython's default warning filters (a bare
    warnings.warn(DeprecationWarning) from library code is not)."""
    monkeypatch.delenv("DYN_TRTLLM_PUBLISH_KV_EVENTS", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS", raising=False)
    with caplog.at_level("WARNING"), pytest.warns(
        DeprecationWarning, match="--publish-events-and-metrics is deprecated"
    ):
        config = parse_args(["--publish-events-and-metrics"])
    assert config.publish_events_and_metrics is True
    assert config.use_kv_events is True
    assert any(
        "--publish-events-and-metrics is deprecated" in r.message
        for r in caplog.records
    )


def test_deprecated_publish_events_env_alias_maps_and_logs(monkeypatch, caplog):
    """The deprecated DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS env var must still
    map to the new env var AND surface its deprecation notice on the log
    stream."""
    # parse_args copies the deprecated env var into the new one via a direct
    # os.environ write. Swap in a throwaway copy so monkeypatch restores the real
    # environment on teardown and that write does not leak into later tests.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("DYN_TRTLLM_PUBLISH_KV_EVENTS", raising=False)
    monkeypatch.setenv("DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS", "true")
    with caplog.at_level("WARNING"), pytest.warns(
        DeprecationWarning, match="DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS is deprecated"
    ):
        config = parse_args([])
    assert config.publish_events_and_metrics is True
    assert config.use_kv_events is True
    assert os.environ["DYN_TRTLLM_PUBLISH_KV_EVENTS"] == "true"
    assert any(
        "DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS is deprecated" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_init_llm_worker_rejects_invalid_kv_cache_config_override(monkeypatch):
    monkeypatch.delenv("DYN_TRTLLM_OVERRIDE_ENGINE_ARGS", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_PUBLISH_KV_EVENTS", raising=False)
    config = parse_args(
        [
            "--model",
            "fake-model",
            "--publish-kv-events",
            "--override-engine-args",
            '{"kv_cache_config": []}',
        ]
    )

    with pytest.raises(
        TypeError, match="kv_cache_config must be a dict or KvCacheConfig, got list"
    ):
        await init_llm_worker(
            runtime=mock.MagicMock(),
            config=config,
            shutdown_event=asyncio.Event(),
        )


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


def test_enable_multimodal_maps_to_multimodal_modality():
    config = parse_args(["--model", "fake-model", "--enable-multimodal"])

    assert config.enable_multimodal is True
    assert config.modality == Modality.MULTIMODAL


def test_modality_multimodal_alias_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = parse_args(["--model", "fake-model", "--modality", "multimodal"])

    assert config.enable_multimodal is True
    assert config.modality == Modality.MULTIMODAL
    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, DeprecationWarning)
    ]


def test_disaggregation_mode_accepts_canonical_agg():
    config = parse_args(["--model", "fake-model", "--disaggregation-mode", "agg"])

    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED
    assert config.component == "backend"


def test_disaggregation_mode_accepts_pd_alias():
    config = parse_args(["--model", "fake-model", "--disaggregation-mode", "pd"])

    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED
    assert config.component == "backend"


def test_disaggregation_mode_legacy_aggregated_value_warns():
    with pytest.warns(DeprecationWarning, match="prefill_and_decode"):
        config = parse_args(
            ["--model", "fake-model", "--disaggregation-mode", "prefill_and_decode"]
        )

    assert config.disaggregation_mode == DisaggregationMode.AGGREGATED


def test_conversation_affinity_cli_flag(monkeypatch):
    """--conversation-affinity sets conversation_affinity=True in Config."""
    monkeypatch.delenv("DYN_ENGINE_CONV_AFFINITY", raising=False)
    config = parse_args(["--model", "fake-model", "--conversation-affinity"])
    assert config.conversation_affinity is True


def test_conversation_affinity_env_var(monkeypatch):
    """DYN_ENGINE_CONV_AFFINITY=true is read and sets conversation_affinity=True."""
    monkeypatch.setenv("DYN_ENGINE_CONV_AFFINITY", "true")
    config = parse_args(["--model", "fake-model"])
    assert config.conversation_affinity is True


def test_conversation_affinity_defaults_false(monkeypatch):
    """conversation_affinity defaults to False when neither flag nor env var is set."""
    monkeypatch.delenv("DYN_ENGINE_CONV_AFFINITY", raising=False)
    config = parse_args(["--model", "fake-model"])
    assert config.conversation_affinity is False


def test_enable_multimodal_rejects_diffusion_modality():
    with pytest.raises(ValueError, match="--enable-multimodal cannot be combined"):
        parse_args(
            [
                "--model",
                "fake-model",
                "--enable-multimodal",
                "--modality",
                "video_diffusion",
            ]
        )


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
    """--extra-engine-args YAML overrides are reflected in engine_args and MDC-visible config."""
    monkeypatch.delenv("DYN_TRTLLM_MAX_NUM_TOKENS", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_MAX_BATCH_SIZE", raising=False)
    monkeypatch.delenv("DYN_TRTLLM_MAX_SEQ_LEN", raising=False)

    yaml_file = tmp_path / "engine_config.yaml"
    yaml_file.write_text(
        "max_seq_len: 32768\nmax_num_tokens: 32768\nmax_batch_size: 512\n"
    )

    config = parse_args(
        [
            "--model",
            "fake-model",
            "--max-seq-len",
            "131072",
            "--extra-engine-args",
            str(yaml_file),
        ]
    )
    # CLI config should NOT reflect the YAML values
    assert config.max_seq_len != 32768
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
        assert engine_args["max_seq_len"] == 32768, (
            f"Expected max_seq_len=32768 from YAML override, "
            f"got {engine_args['max_seq_len']}"
        )
        assert engine_args["max_num_tokens"] == 32768, (
            f"Expected max_num_tokens=32768 from YAML override, "
            f"got {engine_args['max_num_tokens']}"
        )
        assert engine_args["max_batch_size"] == 512, (
            f"Expected max_batch_size=512 from YAML override, "
            f"got {engine_args['max_batch_size']}"
        )
        # MDC registration reads config.max_seq_len, so keep it in sync with
        # the final engine args.
        assert config.max_seq_len == 32768
        assert config.max_num_tokens == 32768
        assert config.max_batch_size == 512


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
