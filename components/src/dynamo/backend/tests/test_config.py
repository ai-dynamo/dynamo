# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoRuntimeConfig.

Tests cover configuration validation and field defaults.
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
    defaults.update(overrides)
    return DynamoRuntimeConfig.from_cli_args(argparse.Namespace(**defaults))


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
