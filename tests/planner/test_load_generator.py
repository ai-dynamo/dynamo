# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit test for LoadGenerator subprocess management (DYN-2086).

Validates that aiperf timeouts kill the entire process group via os.killpg,
preventing orphaned child processes from holding pipe FDs and causing hangs.
"""

import asyncio
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from utils.load_generator import LoadGenerator

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_timeout_kills_process_group(tmp_path):
    """On timeout, the entire process group must be killed via os.killpg."""
    target_pid = 99999
    generator = LoadGenerator()

    mock_proc = MagicMock()
    mock_proc.pid = target_pid
    mock_proc.returncode = -9
    mock_proc.wait = AsyncMock(return_value=-9)

    async def fake_exec(*args, **kwargs):
        return mock_proc

    async def fake_wait_for(coro, timeout=None):
        if hasattr(coro, "close"):
            coro.close()
        raise asyncio.TimeoutError()

    async def _run():
        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch("asyncio.wait_for", side_effect=fake_wait_for),
            patch("os.killpg") as mock_killpg,
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                await generator.generate_load(1.0, 1, str(tmp_path))

            mock_killpg.assert_called_once_with(target_pid, signal.SIGKILL)

    asyncio.run(_run())
