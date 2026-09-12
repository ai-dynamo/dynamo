# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.vllm import snapshot
from dynamo.vllm.publisher import StatLoggerFactory

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_prepare_snapshot_preserves_engine_stat_loggers(monkeypatch):
    snapshot_config = Mock()
    snapshot_config.run_lifecycle = AsyncMock(return_value=True)
    monkeypatch.setattr(snapshot.SnapshotConfig, "from_env", lambda: snapshot_config)
    monkeypatch.setattr(snapshot, "configure_snapshot_capture_env", Mock())

    engine_client = Mock(checkpoint_prepare=Mock(), checkpoint_restore=Mock())
    engine_setup = (
        engine_client,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    setup_vllm_engine = Mock(return_value=engine_setup)
    config = SimpleNamespace(
        headless=False,
        embedding_worker=False,
        engine_args=SimpleNamespace(enable_sleep_mode=False),
    )

    controller = await snapshot.prepare_snapshot_engine(config, setup_vllm_engine)

    assert controller is not None
    restored_engine, stat_logger_factory = controller.engine
    assert restored_engine is engine_setup
    assert isinstance(stat_logger_factory, StatLoggerFactory)
    setup_vllm_engine.assert_called_once_with(config, stat_logger_factory)
    assert config.engine_args.enable_sleep_mode is True
