# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared unified-backend runner."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
    exc_type=ImportError,
)

import dynamo.common.backend.run as run_module  # noqa: E402
from dynamo.common.backend.worker import WorkerConfig  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


async def test_start_uses_engine_from_args_by_default():
    engine_cls = MagicMock()
    engine = MagicMock()
    worker_config = WorkerConfig(namespace="test")
    engine_cls.from_args = AsyncMock(return_value=(engine, worker_config))
    worker = MagicMock()
    worker.run = AsyncMock()

    with patch.object(run_module, "Worker", return_value=worker) as worker_cls:
        await run_module._start(engine_cls, ["--model", "test-model"])

    engine_cls.from_args.assert_awaited_once_with(["--model", "test-model"])
    worker_cls.assert_called_once_with(engine, worker_config)
    worker.run.assert_awaited_once_with()


async def test_start_uses_supplied_engine_factory():
    engine_cls = MagicMock()
    engine_cls.from_args = AsyncMock()
    engine = MagicMock()
    worker_config = WorkerConfig(namespace="test")
    engine_factory = AsyncMock(return_value=(engine, worker_config))
    worker = MagicMock()
    worker.run = AsyncMock()

    with patch.object(run_module, "Worker", return_value=worker) as worker_cls:
        await run_module._start(
            engine_cls,
            ["--model", "test-model"],
            engine_factory=engine_factory,
        )

    engine_factory.assert_awaited_once_with(["--model", "test-model"])
    engine_cls.from_args.assert_not_called()
    worker_cls.assert_called_once_with(engine, worker_config)
    worker.run.assert_awaited_once_with()
