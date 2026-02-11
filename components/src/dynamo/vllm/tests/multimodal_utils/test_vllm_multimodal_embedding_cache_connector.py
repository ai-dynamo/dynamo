# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DynamoMultimodalEmbeddingCacheConnector version check."""

from unittest.mock import MagicMock, patch

import pytest

from dynamo.vllm.multimodal_utils import multimodal_embedding_cache_connector as mod

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


def _make_vllm_config(capacity_gb: float = 1.0) -> MagicMock:
    config = MagicMock()
    config.ec_transfer_config.ec_connector_extra_config = {
        "multimodal_embedding_cache_capacity_gb": capacity_gb,
    }
    return config


class TestVersionCheck:
    def test_warns_old_vllm(self):
        with (
            patch.object(mod, "_vllm_version", "0.16.5"),
            patch.object(mod.ECConnectorBase, "__init__", return_value=None),
            patch.object(mod.logger, "warning") as mock_warn,
        ):
            connector = mod.DynamoMultimodalEmbeddingCacheConnector(
                vllm_config=_make_vllm_config(),
                role=MagicMock(),
            )
            assert connector is not None
            mock_warn.assert_called_once()
            # logger.warning uses %-style: format string is args[0],
            # MINIMUM_VLLM_VERSION is args[1], _vllm_version is args[2].
            assert mock_warn.call_args[0][1] == mod.MINIMUM_VLLM_VERSION
            assert mock_warn.call_args[0][2] == "0.16.5"
