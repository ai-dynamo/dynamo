# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for create_kv_transfer_config and _merge_user_kv_config."""

from unittest.mock import MagicMock

import pytest
from vllm.config import KVTransferConfig

from dynamo.vllm.args import _merge_user_kv_config, create_kv_transfer_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


def _make_configs(connector_list, user_kv_transfer_config=None):
    """Build minimal mock dynamo_config / engine_config for testing."""
    dynamo_config = MagicMock()
    dynamo_config.connector = connector_list

    engine_config = MagicMock()
    engine_config.kv_transfer_config = user_kv_transfer_config
    return dynamo_config, engine_config


# ---------------------------------------------------------------------------
# _merge_user_kv_config
# ---------------------------------------------------------------------------


class TestMergeUserKvConfig:
    def test_matching_connector_merges_extra_config(self):
        """kv_connector_extra_config is merged when kv_connector matches."""
        cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        _merge_user_kv_config(cfg, user)

        assert cfg["kv_connector_extra_config"] == {"backends": ["LIBFABRIC"]}

    def test_non_matching_connector_skips_merge(self):
        """Nothing is merged when kv_connector does not match."""
        cfg = {"kv_connector": "DynamoConnector", "kv_role": "kv_both"}
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        _merge_user_kv_config(cfg, user)

        assert "kv_connector_extra_config" not in cfg

    def test_none_kv_connector_skips_merge(self):
        """When user sets kv_connector=None, merge is skipped."""
        cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        user = KVTransferConfig(
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        _merge_user_kv_config(cfg, user)

        assert "kv_connector_extra_config" not in cfg

    def test_preserves_existing_extra_config(self):
        """User extra config is merged on top of existing extra config."""
        cfg = {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"existing_key": "existing_val"},
        }
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        _merge_user_kv_config(cfg, user)

        assert cfg["kv_connector_extra_config"] == {
            "existing_key": "existing_val",
            "backends": ["LIBFABRIC"],
        }

    def test_overrides_kv_role(self):
        """User-provided kv_role overrides the generated one."""
        cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_producer",
        )
        _merge_user_kv_config(cfg, user)

        assert cfg["kv_role"] == "kv_producer"

    def test_overrides_module_path(self):
        """User-provided kv_connector_module_path overrides the generated one."""
        cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_module_path="custom.module.path",
        )
        _merge_user_kv_config(cfg, user)

        assert cfg["kv_connector_module_path"] == "custom.module.path"

    def test_no_extra_config_is_noop(self):
        """When user config has only defaults, connector_cfg is unchanged."""
        cfg = {"kv_connector": "NixlConnector", "kv_role": "kv_both"}
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
        )
        _merge_user_kv_config(cfg, user)

        assert "kv_connector_extra_config" not in cfg


# ---------------------------------------------------------------------------
# create_kv_transfer_config
# ---------------------------------------------------------------------------


class TestCreateKvTransferConfig:
    def test_no_connector_no_user_config_returns_none(self):
        """No --connector and no --kv-transfer-config returns None."""
        dynamo_cfg, engine_cfg = _make_configs(connector_list=[])
        assert create_kv_transfer_config(dynamo_cfg, engine_cfg) is None

    def test_no_connector_with_user_config_returns_none(self):
        """No --connector with --kv-transfer-config returns None (user config used as-is)."""
        user = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_both")
        dynamo_cfg, engine_cfg = _make_configs(
            connector_list=[], user_kv_transfer_config=user
        )
        assert create_kv_transfer_config(dynamo_cfg, engine_cfg) is None

    def test_single_connector_without_user_config(self):
        """--connector nixl alone produces a NixlConnector config."""
        dynamo_cfg, engine_cfg = _make_configs(connector_list=["nixl"])
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        assert result is not None
        assert result.kv_connector == "NixlConnector"
        assert result.kv_role == "kv_both"
        assert result.kv_connector_extra_config == {}

    def test_single_connector_with_user_config_merge(self):
        """--connector nixl + --kv-transfer-config merges extra config."""
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        dynamo_cfg, engine_cfg = _make_configs(
            connector_list=["nixl"], user_kv_transfer_config=user
        )
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        assert result is not None
        assert result.kv_connector == "NixlConnector"
        assert result.kv_connector_extra_config == {"backends": ["LIBFABRIC"]}

    def test_multi_connector_without_user_config(self):
        """--connector kvbm nixl produces a PdConnector with two sub-connectors."""
        dynamo_cfg, engine_cfg = _make_configs(connector_list=["kvbm", "nixl"])
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        assert result is not None
        assert result.kv_connector == "PdConnector"
        connectors = result.kv_connector_extra_config["connectors"]
        assert len(connectors) == 2
        assert connectors[0]["kv_connector"] == "DynamoConnector"
        assert connectors[1]["kv_connector"] == "NixlConnector"
        assert "kv_connector_extra_config" not in connectors[0]
        assert "kv_connector_extra_config" not in connectors[1]

    def test_multi_connector_merges_into_matching_only(self):
        """--connector kvbm nixl + --kv-transfer-config merges into nixl only."""
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        dynamo_cfg, engine_cfg = _make_configs(
            connector_list=["kvbm", "nixl"], user_kv_transfer_config=user
        )
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        assert result is not None
        assert result.kv_connector == "PdConnector"
        connectors = result.kv_connector_extra_config["connectors"]
        # kvbm should be untouched
        assert "kv_connector_extra_config" not in connectors[0]
        # nixl should have the merged config
        assert connectors[1]["kv_connector_extra_config"] == {"backends": ["LIBFABRIC"]}

    def test_multi_connector_preserves_kvbm_module_path(self):
        """kvbm connector retains its kv_connector_module_path after merge."""
        user = KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"backends": ["LIBFABRIC"]},
        )
        dynamo_cfg, engine_cfg = _make_configs(
            connector_list=["kvbm", "nixl"], user_kv_transfer_config=user
        )
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        assert result is not None
        connectors = result.kv_connector_extra_config["connectors"]
        assert (
            connectors[0]["kv_connector_module_path"]
            == "kvbm.vllm_integration.connector"
        )

    def test_lmcache_connector_config(self):
        """--connector lmcache produces an LMCacheConnectorV1 config."""
        dynamo_cfg, engine_cfg = _make_configs(connector_list=["lmcache"])
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        assert result is not None
        assert result.kv_connector == "LMCacheConnectorV1"
        assert result.kv_role == "kv_both"

    def test_unknown_connector_is_skipped(self):
        """Unknown connectors in the list are silently skipped."""
        dynamo_cfg, engine_cfg = _make_configs(connector_list=["nixl", "unknown"])
        result = create_kv_transfer_config(dynamo_cfg, engine_cfg)

        # Only nixl is recognized, so we get a single-connector result
        assert result is not None
        assert result.kv_connector == "NixlConnector"
