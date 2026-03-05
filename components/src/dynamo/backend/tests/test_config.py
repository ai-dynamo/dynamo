# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoRuntimeConfig.

Tests cover configuration validation and model name resolution.
"""

import argparse
import os
import tempfile

import pytest

from dynamo.backend.args import DynamoRuntimeConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def _make_config(**overrides) -> DynamoRuntimeConfig:
    """Create a DynamoRuntimeConfig via from_cli_args with test defaults."""
    defaults = dict(
        namespace="test-ns",
        component="backend",
        endpoint="generate",
        model="test-model",
        store_kv="mem",
        request_plane="tcp",
        event_plane="nats",
        connector=["nixl"],
        enable_local_indexer=True,
        durable_kv_events=False,
    )
    defaults.update(overrides)
    return DynamoRuntimeConfig.from_cli_args(argparse.Namespace(**defaults))


class TestDynamoRuntimeConfigValidate:
    """Tests for DynamoRuntimeConfig.validate()."""

    def test_sets_enable_local_indexer_from_durable_kv(self):
        """enable_local_indexer should be inverse of durable_kv_events."""
        config = _make_config(durable_kv_events=True, enable_local_indexer=True)
        config.validate()

        assert config.enable_local_indexer is False
        assert config.use_kv_events is True

    def test_disables_kv_events_when_not_durable(self):
        """use_kv_events should be False when durable_kv_events is False."""
        config = _make_config(durable_kv_events=False)
        config.validate()

        assert config.enable_local_indexer is True
        assert config.use_kv_events is False

    def test_validates_existing_jinja_template(self):
        """Should accept a valid file path for custom_jinja_template."""
        with tempfile.NamedTemporaryFile(suffix=".jinja", delete=False) as f:
            f.write(b"{{ content }}")
            path = f.name

        try:
            config = _make_config(custom_jinja_template=path)
            config.validate()
            assert config.custom_jinja_template == path
        finally:
            os.unlink(path)

    def test_raises_on_missing_jinja_template(self):
        """Should raise FileNotFoundError for a non-existent template path."""
        config = _make_config(custom_jinja_template="/no/such/file.jinja")

        with pytest.raises(FileNotFoundError, match="Custom Jinja template"):
            config.validate()

    def test_no_template_validation_when_none(self):
        """Should not raise when custom_jinja_template is None."""
        config = _make_config(custom_jinja_template=None)
        config.validate()  # should not raise


class TestDynamoRuntimeConfigGetModelName:
    """Tests for DynamoRuntimeConfig.get_model_name()."""

    def test_returns_served_model_name_when_set(self):
        """Should prefer served_model_name over model."""
        config = _make_config(model="base-model", served_model_name="display-name")
        assert config.get_model_name() == "display-name"

    def test_falls_back_to_model(self):
        """Should return model when served_model_name is None."""
        config = _make_config(model="base-model", served_model_name=None)
        assert config.get_model_name() == "base-model"
