# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ManagedProcess teardown behavior.

Verifies that __exit__ / _terminate_process_group correctly kills process
trees under various scenarios: simple children, deep trees, children that
create their own process groups, and xdist-safe mode skipping stragglers.

All test processes are lightweight shell/python one-liners that sleep;
no GPU or network resources are needed.

IMPORTANT: Never use generic command names like "sleep" as stragglers or
command names with terminate_all_matching_process_names=True — that kills
container infrastructure (tail -f, sleep in docker-init, etc.).
Always use unique markers scoped to the test invocation.
"""

import os
import signal
import subprocess
import time
import uuid

import psutil
import pytest

from tests.utils.managed_process import ManagedProcess

pytestmark = [
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.pre_merge,
]


def _unique_marker() -> str:
    """Per-call unique marker that won't collide across xdist workers."""
    return f"__mp_test_{uuid.uuid4().hex[:12]}__"


def _pid_alive(pid: int) -> bool:
    """Check whether a PID is still running (zombies count as dead)."""
    try:
        p = psutil.Process(pid)
        return p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _wait_for_pid_death(pid: int, timeout: float = 10.0) -> bool:
    """Poll until PID is dead or timeout. Returns True if dead."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.1)
    return False


def _collect_tree_pids(root_pid: int) -> set[int]:
    """Return {root_pid} union all descendant PIDs."""
    pids = set()
    try:
        parent = psutil.Process(root_pid)
        pids.add(root_pid)
        for child in parent.children(recursive=True):
            pids.add(child.pid)
    except psutil.NoSuchProcess:
        pass
    return pids


def _wait_for_tree(
    root_pid: int, min_count: int, timeout: float = 3.0, poll: float = 0.1
) -> set[int]:
    """Poll until the process tree has at least min_count members."""
    deadline = time.monotonic() + timeout
    pids: set[int] = set()
    while time.monotonic() < deadline:
        pids = _collect_tree_pids(root_pid)
        if len(pids) >= min_count:
            return pids
        time.sleep(poll)
    return pids


def _bash_sleep_cmd(marker: str, tag: str = "") -> list[str]:
    """Return a bash command that sleeps 300s with an embedded unique marker.
    The trailing `: noexit` prevents bash from exec-ing into sleep
    (which would lose the marker from the cmdline)."""
    return ["bash", "-c", f": {marker}{tag}; sleep 300; : noexit"]


# ---------------------------------------------------------------------------
# Scenario 1: Simple process with children — all should die on __exit__
# ---------------------------------------------------------------------------
class TestSimpleProcessTree:
    def test_parent_and_children_killed(self, tmp_path):
        """A parent that forks children; all should be dead after __exit__."""
        marker = _unique_marker()
        mp = ManagedProcess(
            command=[
                "bash",
                "-c",
                f": {marker}; sleep 300 & sleep 300 & wait",
            ],
            timeout=10,
            display_output=False,
            terminate_all_matching_process_names=False,
            log_dir=str(tmp_path),
        )

        with mp:
            assert mp.proc is not None
            root_pid = mp.proc.pid
            tree_pids = _wait_for_tree(root_pid, min_count=2)
            assert len(tree_pids) >= 2, f"Expected parent + children, got {tree_pids}"

        for pid in tree_pids:
            assert _wait_for_pid_death(
                pid, timeout=10
            ), f"PID {pid} still alive after teardown"


# ---------------------------------------------------------------------------
# Scenario 2: Deep process tree (grandchildren)
# ---------------------------------------------------------------------------
class TestDeepProcessTree:
    def test_grandchildren_killed(self, tmp_path):
        """Parent -> child -> grandchild; all should be dead after __exit__."""
        marker = _unique_marker()
        mp = ManagedProcess(
            command=[
                "bash",
                "-c",
                f": {marker}; bash -c 'bash -c \"sleep 300\" & wait' & wait",
            ],
            timeout=10,
            display_output=False,
            terminate_all_matching_process_names=False,
            log_dir=str(tmp_path),
        )

        with mp:
            assert mp.proc is not None
            root_pid = mp.proc.pid
            tree_pids = _wait_for_tree(root_pid, min_count=3)
            assert (
                len(tree_pids) >= 3
            ), f"Expected parent + child + grandchild, got {tree_pids}"

        for pid in tree_pids:
            assert _wait_for_pid_death(
                pid, timeout=10
            ), f"PID {pid} still alive after teardown"


# ---------------------------------------------------------------------------
# Scenario 3: Child creates its own process group (setpgid)
# ---------------------------------------------------------------------------
class TestChildWithOwnProcessGroup:
    def test_child_in_own_pgid_killed(self, tmp_path):
        """A child that calls setpgid(0,0) to leave the parent's group
        should still be killed via the snapshotted pgid set."""
        script = (
            "import os, time; "
            "pid = os.fork(); "
            "_ = (os.setpgid(0, 0), time.sleep(300)) if pid == 0 else "
            "(time.sleep(0.3), time.sleep(300))"
        )
        mp = ManagedProcess(
            command=["python3", "-c", script],
            timeout=10,
            display_output=False,
            terminate_all_matching_process_names=False,
            log_dir=str(tmp_path),
        )

        with mp:
            assert mp.proc is not None
            root_pid = mp.proc.pid
            tree_pids = _wait_for_tree(root_pid, min_count=2)
            assert len(tree_pids) >= 2, f"Expected parent + child, got {tree_pids}"

            child_pids = tree_pids - {root_pid}
            parent_pgid = os.getpgid(root_pid)
            found_separate_pgid = False
            for cpid in child_pids:
                try:
                    if os.getpgid(cpid) != parent_pgid:
                        found_separate_pgid = True
                        break
                except (ProcessLookupError, OSError):
                    pass
            if not found_separate_pgid:
                pytest.skip("Child didn't get a separate pgid (OS-dependent)")

        for pid in tree_pids:
            assert _wait_for_pid_death(
                pid, timeout=10
            ), f"PID {pid} still alive after teardown (separate pgid scenario)"


# ---------------------------------------------------------------------------
# Scenario 4: xdist-safe mode skips _cleanup_stragglers
# ---------------------------------------------------------------------------
class TestXdistSafeSkipsStragglers:
    def test_stragglers_not_killed_in_xdist_mode(self, tmp_path):
        """With terminate_all_matching_process_names=False, _cleanup_stragglers
        should NOT kill unrelated processes matching the straggler pattern."""
        marker = _unique_marker()
        bystander = subprocess.Popen(
            _bash_sleep_cmd(marker, "bystander"),
            start_new_session=True,
        )
        bystander_pid = bystander.pid

        try:
            mp = ManagedProcess(
                command=_bash_sleep_cmd(marker, "main"),
                timeout=10,
                display_output=False,
                terminate_all_matching_process_names=False,
                straggler_commands=[marker],
                log_dir=str(tmp_path),
            )

            with mp:
                pass

            assert _pid_alive(
                bystander_pid
            ), "Bystander was killed even though xdist-safe mode was on"
        finally:
            try:
                os.killpg(os.getpgid(bystander_pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            # Reap the zombie so it doesn't linger in the process table
            # for the rest of the pytest session.
            try:
                bystander.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(bystander_pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

    def test_stragglers_killed_when_not_xdist_mode(self, tmp_path):
        """With terminate_all_matching_process_names=True, _cleanup_stragglers
        SHOULD kill processes matching the straggler pattern."""
        marker = _unique_marker()
        victim_tag = f"{marker}_victim"
        launcher_tag = f"{marker}_launcher"

        bystander = subprocess.Popen(
            ["bash", "-c", f": {victim_tag}; sleep 300; : noexit"],
            start_new_session=True,
        )
        bystander_pid = bystander.pid

        try:
            mp = ManagedProcess(
                command=["bash", "-c", f": {launcher_tag}; sleep 1"],
                timeout=10,
                display_output=False,
                display_name=launcher_tag,
                terminate_all_matching_process_names=True,
                straggler_commands=[victim_tag],
                log_dir=str(tmp_path),
            )

            with mp:
                time.sleep(0.5)

            assert _wait_for_pid_death(
                bystander_pid, timeout=10
            ), "Bystander with matching straggler command should have been killed"
        finally:
            try:
                os.killpg(os.getpgid(bystander_pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            # Reap the zombie so it doesn't linger in the process table
            # for the rest of the pytest session.
            try:
                bystander.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(bystander_pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass


# ---------------------------------------------------------------------------
# Scenario 5: Process already dead before __exit__
# ---------------------------------------------------------------------------
class TestAlreadyDeadProcess:
    def test_exit_handles_dead_process(self, tmp_path):
        """If the process exits on its own before __exit__, teardown should
        not raise."""
        mp = ManagedProcess(
            command=["bash", "-c", "exit 0"],
            timeout=10,
            display_output=False,
            terminate_all_matching_process_names=False,
            log_dir=str(tmp_path),
        )

        with mp:
            time.sleep(0.5)
        # No exception = pass


# ---------------------------------------------------------------------------
# Scenario 6: SIGTERM grace period — process that traps SIGTERM
# ---------------------------------------------------------------------------
class TestSigtermGracePeriod:
    def test_process_gets_sigterm_grace_before_sigkill(self, tmp_path):
        """A process that handles SIGTERM and takes a moment to exit should
        get the grace period, not be immediately SIGKILLed.

        Uses a Python child that writes a "ready" file after installing its
        SIGTERM handler, so we don't race against interpreter startup."""
        marker_file = str(tmp_path / "got_sigterm")
        ready_file = str(tmp_path / "ready")
        script = (
            "import os, signal, time, pathlib; "
            f"marker = pathlib.Path('{marker_file}'); "
            f"ready = pathlib.Path('{ready_file}'); "
            "signal.signal(signal.SIGTERM, "
            "lambda *_: (marker.touch(), os._exit(0))); "
            "ready.touch(); "
            "[time.sleep(0.1) for _ in iter(int, 1)]"
        )
        mp = ManagedProcess(
            command=["python3", "-c", script],
            timeout=10,
            display_output=False,
            terminate_all_matching_process_names=False,
            log_dir=str(tmp_path),
        )

        with mp:
            assert mp.proc is not None
            deadline = time.monotonic() + 5.0
            while not os.path.exists(ready_file):
                assert time.monotonic() < deadline, "Child never became ready"
                time.sleep(0.05)

        assert os.path.exists(
            marker_file
        ), "Process was SIGKILLed before SIGTERM handler could run"
