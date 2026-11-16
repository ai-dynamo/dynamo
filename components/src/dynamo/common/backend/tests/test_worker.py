# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Python ``Worker`` shim."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
    exc_type=ImportError,
)

from dynamo.common.backend import worker as worker_module  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


async def test_run_keeps_host_logging_when_configure_logging_is_false(monkeypatch):
    rust_worker = MagicMock(run=AsyncMock())
    monkeypatch.setattr(
        worker_module._backend, "Worker", MagicMock(return_value=rust_worker)
    )
    # Registered so the variable run() exports is restored.
    monkeypatch.delenv("DYN_RESPONSE_PLANE", raising=False)
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    host_handler = logging.NullHandler()
    root.addHandler(host_handler)
    root.setLevel(logging.WARNING)
    try:
        config = worker_module.WorkerConfig(namespace="test", configure_logging=False)
        await worker_module.Worker(MagicMock(), config).run()

        assert root.handlers == [*saved_handlers, host_handler]
        assert root.level == logging.WARNING
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)
