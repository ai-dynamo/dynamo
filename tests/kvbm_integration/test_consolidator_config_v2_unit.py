# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for kvbm.v2.vllm_integration.consolidator_config.

Regression coverage for the v2 (vLLM) consolidator auto-enable path: the
consolidator subscribes to vLLM's KV-event ZMQ endpoint, which only exists
when kv_events_config is set. Without it, get_consolidator_endpoints would
dereference a None kv_events_config and crash at startup, so
should_enable_consolidator must return False instead.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("kvbm", reason="kvbm package not installed")
pytest.importorskip("vllm", reason="vllm not installed")

from kvbm.v2.vllm_integration.consolidator_config import (  # noqa: E402
    should_enable_consolidator,
)


def _eligible_vllm_config(kv_events_config):
    """A VllmConfig-shaped object that satisfies every consolidator
    precondition except (optionally) kv_events_config."""
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="DynamoConnector",
            kv_connector_extra_config={},
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        kv_events_config=kv_events_config,
    )


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.kvbm
@pytest.mark.gpu_0
class TestV2ShouldEnableConsolidator:
    def test_disabled_when_kv_events_unconfigured(self):
        """Regression: eligible KVBM config but kv_events_config=None must
        disable the consolidator (previously returned True, then crashed in
        get_consolidator_endpoints on `kv_events_config.endpoint`)."""
        assert should_enable_consolidator(_eligible_vllm_config(None)) is False

    def test_enabled_when_kv_events_configured(self):
        cfg = _eligible_vllm_config(SimpleNamespace(endpoint="tcp://127.0.0.1:5557"))
        assert should_enable_consolidator(cfg) is True
