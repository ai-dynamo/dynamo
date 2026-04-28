# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import signal
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.multimodal.sweep import server as srv
from benchmarks.multimodal.sweep.server import ServerManager

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def _mgr_with_mock_proc(mock_proc: MagicMock) -> ServerManager:
    mgr = ServerManager()
    mgr._process = mock_proc
    return mgr


def test_stop_sends_sigint_not_sigterm() -> None:
    """stop() must send SIGINT directly to the wrapper (not SIGTERM, not killpg).

    SIGINT is the documented signal that the wrapper's `trap cleanup INT TERM`
    forwards to nsys for graceful finalize. Broadcasting SIGTERM via killpg
    would race the trap and may skip nsys finalize.
    """
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 12345
    proc.wait.return_value = 0
    mgr = _mgr_with_mock_proc(proc)

    with patch.object(srv, "time") as mock_time:
        mgr.stop()
        mock_time.sleep.assert_called_with(srv._LEASE_DRAIN_SECS)

    proc.send_signal.assert_called_once_with(signal.SIGINT)
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_stop_waits_full_graceful_window() -> None:
    """stop() must wait at least the wrapper's 150 s cleanup budget."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 12345
    proc.wait.return_value = 0
    mgr = _mgr_with_mock_proc(proc)

    with patch.object(srv, "time"):
        mgr.stop()

    proc.wait.assert_called_once()
    timeout = proc.wait.call_args.kwargs.get("timeout") or proc.wait.call_args.args[0]
    assert timeout >= 150, f"wait timeout {timeout}s is below wrapper's 150s budget"


def test_stop_tree_kills_on_timeout() -> None:
    """If the wrapper doesn't exit, stop() escalates to tree-kill."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 12345
    proc.wait.side_effect = [
        subprocess.TimeoutExpired("wrapper", srv._GRACEFUL_STOP_TIMEOUT_SECS),
        0,
    ]
    mgr = _mgr_with_mock_proc(proc)

    with (
        patch.object(srv, "_tree_kill") as mock_tk,
        patch.object(srv, "time"),
    ):
        mgr.stop()

    mock_tk.assert_called_once_with(12345)


def test_stop_drains_lease_after_clean_exit() -> None:
    """The trailing sleep must equal _LEASE_DRAIN_SECS so the next config
    cannot observe stale etcd state from the prior config's lease."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 12345
    proc.wait.return_value = 0
    mgr = _mgr_with_mock_proc(proc)

    with patch.object(srv, "time") as mock_time:
        mgr.stop()

    sleep_calls = [c.args[0] for c in mock_time.sleep.call_args_list]
    assert srv._LEASE_DRAIN_SECS in sleep_calls
    assert srv._LEASE_DRAIN_SECS >= 10, "lease drain must exceed primary lease TTL"


def test_stop_is_noop_when_not_running() -> None:
    mgr = ServerManager()
    assert mgr._process is None
    mgr.stop()  # must not raise


def test_descendants_walks_proc_tree() -> None:
    """_descendants returns the transitive child PIDs from /proc."""
    fake_tree = {1000: 1, 2000: 1000, 3000: 2000, 4000: 999}

    def fake_read_ppid(pid: int) -> int | None:
        return fake_tree.get(pid)

    with (
        patch.object(srv, "_read_ppid", side_effect=fake_read_ppid),
        patch.object(srv.os.path, "isdir", return_value=True),
        patch.object(
            srv.os, "listdir", return_value=[str(p) for p in fake_tree.keys()]
        ),
    ):
        out = sorted(srv._descendants(1000))

    assert out == [2000, 3000]
