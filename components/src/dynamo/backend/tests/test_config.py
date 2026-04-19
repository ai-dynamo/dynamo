# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoRuntimeConfig and DynamoBackendConfig.

Tests cover configuration validation, field defaults, and the three-tier
config hierarchy.
"""

import argparse

import pytest

from dynamo.backend.args import (
    DynamoBackendArgGroup,
    DynamoBackendConfig,
    DynamoRuntimeConfig,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

# ── Shared test defaults ────────────────────────────────────

_RUNTIME_DEFAULTS = dict(
    namespace="test-ns",
    discovery_backend="mem",
    request_plane="tcp",
    event_plane="nats",
    connector=[],
    enable_local_indexer=True,
    durable_kv_events=False,
    endpoint_types="chat,completions",
    multimodal_embedding_cache_capacity_gb=0,
    output_modalities=["text"],
)


def _make_config(**overrides) -> DynamoRuntimeConfig:
    """Create a DynamoRuntimeConfig via from_cli_args with test defaults."""
    defaults = dict(_RUNTIME_DEFAULTS)
    defaults.update(overrides)
    return DynamoRuntimeConfig.from_cli_args(argparse.Namespace(**defaults))


def _make_backend_config(**overrides) -> DynamoBackendConfig:
    """Create a DynamoBackendConfig via from_cli_args with test defaults."""
    defaults = dict(_RUNTIME_DEFAULTS)
    defaults.update(overrides)
    return DynamoBackendConfig.from_cli_args(argparse.Namespace(**defaults))


class TestDynamoRuntimeConfigValidate:
    """Tests for DynamoRuntimeConfig.validate()."""

    def test_sets_enable_local_indexer_from_durable_kv(self):
        """enable_local_indexer should be inverse of durable_kv_events."""
        config = _make_config(durable_kv_events=True, enable_local_indexer=True)
        config.validate()

        assert config.enable_local_indexer is False

    def test_local_indexer_enabled_when_not_durable(self):
        """enable_local_indexer should be True when durable_kv_events is False."""
        config = _make_config(durable_kv_events=False)
        config.validate()

        assert config.enable_local_indexer is True

    def test_no_template_validation_when_none(self):
        """Should not raise when custom_jinja_template is None."""
        config = _make_config(custom_jinja_template=None)
        config.validate()  # should not raise

    def test_default_endpoint_types(self):
        """Should default to chat,completions."""
        config = _make_config()
        assert config.endpoint_types == "chat,completions"

    def test_discovery_backend_field(self):
        """Should store the discovery_backend value."""
        config = _make_config(discovery_backend="etcd")
        assert config.discovery_backend == "etcd"


class TestDynamoBackendConfig:
    """Tests for DynamoBackendConfig field defaults and methods."""

    def test_default_model_is_none(self):
        """model should default to None when not provided via CLI."""
        config = _make_backend_config()
        assert config.model is None

    def test_model_from_cli(self):
        """model should pick up the CLI-provided value."""
        config = _make_backend_config(model="my-model")
        assert config.model == "my-model"

    def test_default_disaggregation_mode(self):
        """disaggregation_mode should default to 'aggregated'."""
        config = _make_backend_config()
        assert config.disaggregation_mode == "aggregated"

    def test_disaggregation_mode_from_cli(self):
        """disaggregation_mode should accept CLI value."""
        config = _make_backend_config(disaggregation_mode="prefill")
        assert config.disaggregation_mode == "prefill"

    def test_default_component(self):
        """component should default to 'backend'."""
        config = _make_backend_config()
        assert config.component == "backend"

    def test_component_from_cli(self):
        """component should accept CLI value."""
        config = _make_backend_config(component="decoder")
        assert config.component == "decoder"

    def test_default_served_model_name(self):
        """served_model_name should default to None."""
        config = _make_backend_config()
        assert config.served_model_name is None

    def test_default_use_kv_events(self):
        """use_kv_events should default to False."""
        config = _make_backend_config()
        assert config.use_kv_events is False

    def test_get_model_name_returns_served_name(self):
        """get_model_name() should prefer served_model_name over model."""
        config = _make_backend_config(
            model="base-model", served_model_name="display-name"
        )
        assert config.get_model_name() == "display-name"

    def test_get_model_name_falls_back_to_model(self):
        """get_model_name() should fall back to model when served_model_name is None."""
        config = _make_backend_config(model="my-model")
        assert config.get_model_name() == "my-model"

    def test_get_model_name_returns_unknown_when_both_none(self):
        """get_model_name() should return 'unknown' when both model and served_model_name are None."""
        config = _make_backend_config()
        assert config.get_model_name() == "unknown"

    def test_inherits_runtime_fields(self):
        """DynamoBackendConfig should inherit DynamoRuntimeConfig fields."""
        config = _make_backend_config(discovery_backend="etcd", model="test")
        assert config.discovery_backend == "etcd"
        assert config.model == "test"

    def test_validate_works(self):
        """validate() should work on DynamoBackendConfig (inherited from DynamoRuntimeConfig)."""
        config = _make_backend_config(durable_kv_events=True, enable_local_indexer=True)
        config.validate()
        assert config.enable_local_indexer is False


class TestDynamoBackendConfigSubclass:
    """Tests for subclassing DynamoBackendConfig with config.extra."""

    def test_subclass_with_extra(self):
        """A subclass can store engine-specific config on extra."""

        class MyExtraConfig:
            def __init__(self, gpu_mem=0.9):
                self.gpu_mem = gpu_mem

        class MyConfig(DynamoBackendConfig):
            extra = None

        config = MyConfig.from_cli_args(argparse.Namespace(**_RUNTIME_DEFAULTS))
        config.extra = MyExtraConfig(gpu_mem=0.85)

        assert config.extra.gpu_mem == 0.85
        assert config.model is None  # base default

    def test_post_parse_model_default(self):
        """Subclasses should set model defaults post-parse when CLI omits --model."""

        class MyConfig(DynamoBackendConfig):
            pass

        config = MyConfig.from_cli_args(argparse.Namespace(**_RUNTIME_DEFAULTS))
        if not config.model:
            config.model = "my-default-model"

        assert config.model == "my-default-model"

    def test_cli_model_overrides_post_parse_default(self):
        """CLI-provided model should take precedence over post-parse defaults."""

        class MyConfig(DynamoBackendConfig):
            pass

        args = dict(_RUNTIME_DEFAULTS)
        args["model"] = "cli-model"
        config = MyConfig.from_cli_args(argparse.Namespace(**args))
        if not config.model:
            config.model = "my-default-model"

        assert config.model == "cli-model"


class TestDynamoBackendArgGroup:
    """Tests for DynamoBackendArgGroup CLI argument registration."""

    def _parse(self, *cli_args):
        """Parse CLI args through DynamoBackendArgGroup."""
        parser = argparse.ArgumentParser()
        DynamoBackendArgGroup().add_arguments(parser)
        return parser.parse_args(list(cli_args))

    def test_model_flag(self):
        args = self._parse("--model", "my-model")
        assert args.model == "my-model"

    def test_model_default_is_none(self):
        args = self._parse()
        assert args.model is None

    def test_served_model_name_flag(self):
        args = self._parse("--served-model-name", "display")
        assert args.served_model_name == "display"

    def test_disaggregation_mode_flag(self):
        args = self._parse("--disaggregation-mode", "prefill")
        assert args.disaggregation_mode == "prefill"

    def test_disaggregation_mode_default(self):
        args = self._parse()
        assert args.disaggregation_mode == "aggregated"

    def test_disaggregation_mode_rejects_invalid(self):
        with pytest.raises(SystemExit):
            self._parse("--disaggregation-mode", "invalid")

    def test_component_flag(self):
        args = self._parse("--component", "decoder")
        assert args.component == "decoder"

    def test_component_default(self):
        args = self._parse()
        assert args.component == "backend"
