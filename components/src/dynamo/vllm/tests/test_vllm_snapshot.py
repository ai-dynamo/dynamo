# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.common.snapshot.restore_context import (
    apply_snapshot_restore_env,
    write_snapshot_restore_context,
)
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


@pytest.mark.parametrize(
    "restore_pod_name, expected",
    [
        ("tenant-a,alias-a", ("tenant-a", ["alias-a"])),
        (None, ("captured", [])),
    ],
)
def test_restore_served_model_name(monkeypatch, tmp_path, restore_pod_name, expected):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", str(tmp_path))
    if restore_pod_name is None:
        monkeypatch.delenv("DYN_SERVED_MODEL_NAME", raising=False)
    else:
        monkeypatch.setenv("DYN_SERVED_MODEL_NAME", restore_pod_name)
    write_snapshot_restore_context()
    # The restored process still carries the env it was captured with.
    monkeypatch.setenv("DYN_SERVED_MODEL_NAME", "captured")
    apply_snapshot_restore_env()
    config = SimpleNamespace(served_model_name="captured", served_model_aliases=[])

    snapshot.apply_restore_served_model_name(config)

    assert (config.served_model_name, config.served_model_aliases) == expected
