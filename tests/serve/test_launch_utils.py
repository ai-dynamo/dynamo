# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import signal
import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.timeout(120),
]

LAUNCH_UTILS = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
LAUNCH_DIR = Path(__file__).parents[2] / "examples/backends/vllm/launch"

LABELLED_SCRIPTS = (
    ("disagg_multimodal_epd.sh", ("frontend", "encode", "prefill", "decode")),
    ("disagg_multimodal_p_d.sh", ("frontend", "prefill", "decode")),
)

# Stands in for a worker that stays up. Blocks on an fd rather than a timer,
# so the surviving workers cost no wall-clock time and cannot exit first.
STAY_UP = "tail -f /dev/null"

# The one process that leaves. It has to outlive the `wait` it is meant to
# wake, so this is a short timed exit rather than an immediate one: a worker
# reaped before the wait starts is dropped from bash's jobs table, which the
# wait cannot then report on.
LEAVE_WITH = "(sleep 0.2; exit {code})"

UNNAMED = "A background process exited with code"

# `wait -n -p`, the only way to learn which child exited, arrived in bash 5.1.
# launch_utils.sh still supports 4.3, where every exit stays unnamed, so the
# label assertions have to follow whichever bash the runner installed.
_VERSINFO = subprocess.run(
    ["bash", "-c", 'printf "%s %s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"'],
    capture_output=True,
    text=True,
    check=True,
    timeout=60,
).stdout.split()
BASH_NAMES_WORKERS = tuple(int(part) for part in _VERSINFO) >= (5, 1)


def _run_script(body: str) -> subprocess.CompletedProcess:
    """Run a bash snippet in its own process group.

    wait_any_exit signals the whole process group on the way out, so the
    snippet must not share one with pytest. A snippet that hangs never
    reaches that signal, so the timeout path has to take the group down
    here or the workers it backgrounded outlive the test run.
    """
    process = subprocess.Popen(
        ["bash", "-c", f"set -e\nsource {shlex.quote(str(LAUNCH_UTILS))}\n{body}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.communicate(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        raise

    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def test_wait_any_exit_names_the_labelled_worker_that_left() -> None:
    result = _run_script(
        f"""
        {STAY_UP} & dyn_track_worker frontend
        {LEAVE_WITH.format(code=7)} & dyn_track_worker prefill
        {STAY_UP} & dyn_track_worker decode
        wait_any_exit
        """
    )

    assert result.returncode == 7, result.stderr
    assert "exited with code 7" in result.stdout, result.stdout
    if BASH_NAMES_WORKERS:
        assert "Worker 'prefill' (pid" in result.stdout, result.stdout
        assert "Worker 'frontend'" not in result.stdout, result.stdout
        assert "Worker 'decode'" not in result.stdout, result.stdout
    else:
        assert UNNAMED in result.stdout, result.stdout


def test_wait_any_exit_keeps_the_generic_message_without_labels() -> None:
    result = _run_script(
        f"""
        {STAY_UP} &
        {LEAVE_WITH.format(code=3)} &
        wait_any_exit
        """
    )

    assert result.returncode == 3, result.stderr
    assert f"{UNNAMED} 3" in result.stdout, result.stdout


def test_wait_any_exit_never_names_an_untracked_process() -> None:
    """Keep every label out of the report unless that worker is the one that left.

    Naming a worker that is still up would send the reader after the wrong
    process, so an untracked exit has to stay on the generic message even
    while labelled workers are running.
    """
    result = _run_script(
        f"""
        {STAY_UP} & dyn_track_worker frontend
        {STAY_UP} & dyn_track_worker decode
        {LEAVE_WITH.format(code=5)} &
        wait_any_exit
        """
    )

    assert result.returncode == 5, result.stderr
    assert "Worker '" not in result.stdout, result.stdout
    assert f"{UNNAMED} 5" in result.stdout, result.stdout


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("surviving_worker", [False, True])
def test_wait_any_exit_reports_worker_that_exited_during_startup(
    exit_code: int, surviving_worker: bool
) -> None:
    """A saved PID retains its status even after Bash drops its job entry.

    Explicitly waiting for the early worker makes this ordering deterministic;
    Bash allows a second wait by PID to retrieve the saved status. A surviving
    worker must not hide the early exit, and an empty job table is not an error.
    """
    result = _run_script(
        f"""
        (exit {exit_code}) & dyn_track_worker early
        early_pid=$!
        wait "$early_pid" || :
        {f"{STAY_UP} & dyn_track_worker later" if surviving_worker else ":"}
        wait_any_exit
        """
    )

    assert result.returncode == exit_code, result.stderr
    assert f"exited with code {exit_code}" in result.stdout, result.stdout
    assert "Worker 'later'" not in result.stdout, result.stdout
    if BASH_NAMES_WORKERS:
        assert "Worker 'early' (pid" in result.stdout, result.stdout
    else:
        assert UNNAMED in result.stdout, result.stdout


def test_wait_any_exit_prefers_startup_failure_over_success(tmp_path: Path) -> None:
    """Launch order does not determine the associative array's traversal order.

    Assign exit codes after registering both PIDs so the successful child is
    considered first. Stopping at the first completed child must fail this test.
    """
    first_pipe = tmp_path / "first-worker"
    second_pipe = tmp_path / "second-worker"
    os.mkfifo(first_pipe)
    os.mkfifo(second_pipe)
    result = _run_script(
        f"""
        exec 3<> {shlex.quote(str(first_pipe))}
        exec 4<> {shlex.quote(str(second_pipe))}
        (read -r code <&3; exit "$code") & dyn_track_worker first
        first_pid=$!
        (read -r code <&4; exit "$code") & dyn_track_worker second
        second_pid=$!
        tracked_pids=("${{!DYN_TRACKED_WORKERS[@]}}")
        if [[ "${{tracked_pids[0]}}" == "$first_pid" ]]; then
            dyn_track_worker frontend "$first_pid"
            dyn_track_worker prefill "$second_pid"
            printf '0\\n' >&3
            printf '7\\n' >&4
        else
            dyn_track_worker frontend "$second_pid"
            dyn_track_worker prefill "$first_pid"
            printf '7\\n' >&3
            printf '0\\n' >&4
        fi
        wait "$first_pid" || :
        wait "$second_pid" || :
        wait_any_exit
        """
    )

    assert result.returncode == 7, result.stderr
    assert "exited with code 7" in result.stdout, result.stdout
    assert "Worker 'frontend'" not in result.stdout, result.stdout
    if BASH_NAMES_WORKERS:
        assert "Worker 'prefill' (pid" in result.stdout, result.stdout
    else:
        assert UNNAMED in result.stdout, result.stdout


@pytest.mark.parametrize("script_name,expected_labels", LABELLED_SCRIPTS)
def test_disagg_multimodal_labels_every_background_process(
    script_name: str, expected_labels: tuple[str, ...]
) -> None:
    """Keep a label on every process these launch scripts background.

    A worker added later without a label would go back to an anonymous exit
    code in the nightly log, which is what this check prevents.
    """
    lines = (LAUNCH_DIR / script_name).read_text().splitlines()
    backgrounded = [
        index
        for index, line in enumerate(lines)
        if line.rstrip().endswith("&") and not line.rstrip().endswith("&&")
    ]
    labelled = [
        (index, line.split()[1])
        for index, line in enumerate(lines)
        if line.startswith("dyn_track_worker ")
    ]

    assert len(backgrounded) == len(expected_labels), backgrounded
    # dyn_track_worker reads $!, so it has to sit on the line directly after
    # the command it names. A call moved anywhere else keeps both the count
    # and the order below while labelling some other process.
    assert [index for index, _ in labelled] == [index + 1 for index in backgrounded]
    assert tuple(label for _, label in labelled) == expected_labels
