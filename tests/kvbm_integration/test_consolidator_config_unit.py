# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for consolidator_config.should_enable_consolidator.

Covers both raw dict and typed object paths for kv_connector_config,
and asserts that the production code path (build_kv_connector_config)
produces a type that the consolidator check handles correctly.
"""

import os
from unittest.mock import patch

import pytest

pytest.importorskip(
    "kvbm.trtllm_integration.consolidator_config",
    reason="kvbm trtllm_integration not available",
)
from kvbm.trtllm_integration.consolidator_config import (  # noqa: E402
    should_enable_consolidator,
)

KVBM_MODULE = "kvbm.trtllm_integration.connector"


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.kvbm
@pytest.mark.gpu_0
class TestShouldEnableConsolidatorDict:
    """Tests that only need kvbm (no GPU, no trtllm)."""

    def test_dict_config_with_kvbm(self):
        """Raw dict path (--extra-engine-args YAML, used in tests)."""
        arg_map = {"kv_connector_config": {"connector_module": KVBM_MODULE}}
        assert should_enable_consolidator(arg_map) is True

    def test_non_kvbm_connector(self):
        arg_map = {"kv_connector_config": {"connector_module": "other.connector"}}
        assert should_enable_consolidator(arg_map) is False

    def test_no_connector_config(self):
        assert should_enable_consolidator({"backend": "pytorch"}) is False

    def test_env_var_disables(self):
        arg_map = {"kv_connector_config": {"connector_module": KVBM_MODULE}}
        with patch.dict(
            os.environ, {"DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR": "false"}
        ):
            assert should_enable_consolidator(arg_map) is False


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.kvbm
@pytest.mark.trtllm
@pytest.mark.gpu_1
class TestShouldEnableConsolidatorTyped:
    """Tests that need trtllm (requires GPU for import)."""

    def test_typed_config_with_kvbm(self):
        """Typed object path (DYN_CONNECTOR=kvbm, used in production)."""
        from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

        config = KvCacheConnectorConfig(
            connector_module=KVBM_MODULE,
            connector_scheduler_class="DynamoKVBMConnectorLeader",
            connector_worker_class="DynamoKVBMConnectorWorker",
        )
        arg_map = {"kv_connector_config": config}
        assert should_enable_consolidator(arg_map) is True

    def test_production_build_kv_connector_config(self):
        """Assert build_kv_connector_config output works with consolidator check.

        This is the key regression test: if llm_worker.py changes the type
        returned by build_kv_connector_config, this test will catch it.
        """
        from dynamo.trtllm.workers.llm_worker import build_kv_connector_config

        class FakeConfig:
            connector = ("kvbm",)

        config = build_kv_connector_config(FakeConfig())
        assert config is not None
        assert hasattr(config, "connector_module")
        assert KVBM_MODULE in config.connector_module

        # The consolidator check must work with this exact object
        arg_map = {"kv_connector_config": config}
        assert should_enable_consolidator(arg_map) is True
