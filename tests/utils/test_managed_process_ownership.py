# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership inspection must neither hide workers nor signal reused PIDs."""

import subprocess
import unittest
from unittest.mock import Mock, patch

import psutil
import pytest

from tests.utils.managed_process import ManagedProcess, ManagedProcessStopError

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


class ProcessOwnershipTests(unittest.TestCase):
    def managed(self):
        return ManagedProcess(
            command=["unused"],
            terminate_all_matching_process_names=False,
            graceful_shutdown=True,
        )

    def test_unreadable_unrelated_process_does_not_block_shutdown(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=None))
        managed._pgid = 100
        with (
            patch.object(managed, "_remember_children"),
            patch("psutil.process_iter", return_value=[Mock(pid=200)]),
            patch("os.getsid", side_effect=PermissionError),
        ):
            self.assertEqual(managed._owned_processes(), [])

    def test_owned_status_denied_remains_owned_and_is_not_clean(self):
        managed = self.managed()
        child = Mock(pid=200)
        child.status.side_effect = psutil.AccessDenied(child.pid)
        managed._owned_children[child.pid] = child
        with self.assertRaises(ManagedProcessStopError):
            managed._owned_processes()
        self.assertIs(managed._owned_children[child.pid], child)

    def test_session_scan_refreshes_cached_pid_identity(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=None))
        managed._pgid = 100
        stale = Mock(pid=200)
        stale.is_running.return_value = False
        managed._owned_children[200] = stale
        fresh = Mock(pid=200)
        fresh.status.return_value = psutil.STATUS_SLEEPING
        with (
            patch.object(managed, "_remember_children"),
            patch("psutil.process_iter", return_value=[stale]),
            patch("psutil.Process", return_value=fresh),
            patch("os.getsid", return_value=100),
        ):
            self.assertEqual(managed._owned_processes(), [fresh])
        fresh.create_time.assert_called_once()
        self.assertIs(managed._owned_children[200], fresh)

    def test_foreign_session_pid_is_never_adopted(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=None))
        managed._pgid = 100
        foreign = Mock(pid=200)
        with (
            patch.object(managed, "_remember_children"),
            patch("psutil.process_iter", return_value=[foreign]),
            patch("psutil.Process") as process,
            patch("os.getsid", return_value=300),
        ):
            self.assertEqual(managed._owned_processes(), [])
        process.assert_not_called()
        foreign.kill.assert_not_called()

    def test_reused_session_leader_prevents_adopting_foreign_workers(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=0))
        managed._pgid = 100
        with (
            patch("os.getsid", return_value=100),
            patch("psutil.process_iter") as processes,
        ):
            self.assertEqual(managed._owned_processes(), [])
        processes.assert_not_called()

    def test_denied_session_worker_identity_is_not_silently_skipped(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=None))
        managed._pgid = 100
        child = Mock(pid=200)
        child.create_time.side_effect = psutil.AccessDenied(child.pid)
        with (
            patch.object(managed, "_remember_children"),
            patch("psutil.process_iter", return_value=[Mock(pid=200)]),
            patch("psutil.Process", return_value=child),
            patch("os.getsid", return_value=100),
            self.assertRaises(ManagedProcessStopError),
        ):
            managed._owned_processes()
        managed.proc = None
        managed._pgid = None
        with self.assertRaises(ManagedProcessStopError):
            managed.stop_gracefully()
        child.kill.assert_not_called()

    def test_denied_child_identity_never_becomes_clean_on_retry(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=None))
        child = Mock(pid=200)
        child.create_time.side_effect = psutil.AccessDenied(child.pid)
        with patch("psutil.Process") as process:
            process.return_value.children.return_value = [child]
            with self.assertRaises(ManagedProcessStopError):
                managed._remember_children()
        managed.proc = None
        with self.assertRaises(ManagedProcessStopError):
            managed.stop_gracefully()
        child.kill.assert_not_called()

    def test_denied_worker_kill_is_reported_and_retains_ownership(self):
        managed = self.managed()
        managed.proc = Mock(pid=100, poll=Mock(return_value=0))
        child = Mock(pid=200)
        child.kill.side_effect = psutil.AccessDenied(child.pid)
        managed._owned_children[child.pid] = child
        with (
            patch.object(managed, "_remember_children"),
            patch.object(managed, "_owned_processes", return_value=[child]),
            patch.object(
                managed,
                "_wait_owned_processes",
                side_effect=subprocess.TimeoutExpired("workers", 1),
            ),
            patch.object(managed, "_stop_started_processes"),
            self.assertRaises(ManagedProcessStopError),
        ):
            managed.stop_gracefully(worker_timeout=0)
        self.assertIs(managed._owned_children[child.pid], child)


if __name__ == "__main__":
    unittest.main()
