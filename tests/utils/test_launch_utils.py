# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the shell helpers in examples/common/launch_utils.sh.

These run the real helper under bash, so they fail if the launch scripts stop
reporting which child exited -- the first thing read when a launch dies in CI.

The message is the surface under test, not a diagnostic: the parallel-test
orchestrator scrapes this line, so its wording is the contract.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

# Each test runs in well under a second. The timeout is sized above the 60s
# subprocess timeout below so a stuck helper surfaces as the subprocess error,
# which names the harness output, rather than as a bare pytest timeout.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(90),
    # setsid is util-linux; it is absent on macOS, where these helpers are not
    # used. Skipping beats a failure that says nothing about the helper.
    pytest.mark.skipif(
        shutil.which("setsid") is None, reason="setsid is required to isolate the pgid"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_UTILS = REPO_ROOT / "examples" / "common" / "launch_utils.sh"

EXIT_LINE = re.compile(r"A background process exited with code (\d+)(.*)")


def _run_wait_any_exit(body: str, tmp_path: Path) -> tuple[int, str]:
    """Run *body* then wait_any_exit in its own process group.

    wait_any_exit signals its whole process group on the way out, so the script
    gets a group of its own rather than taking pytest down with it.
    """
    script = tmp_path / "harness.sh"
    sourced = shlex.quote(str(LAUNCH_UTILS))
    script.write_text(f"#!/bin/bash\nsource {sourced}\n{body}\nwait_any_exit\n")
    script.chmod(0o755)

    completed = subprocess.run(
        ["setsid", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    match = EXIT_LINE.search(completed.stdout)
    assert match is not None, f"no exit line in output: {completed.stdout!r}"
    return int(match.group(1)), match.group(2)


def test_wait_any_exit_names_a_plain_background_child(tmp_path: Path) -> None:
    """The ordinary case: a backgrounded command is named by its command line."""
    code, detail = _run_wait_any_exit("sleep 0.05 &\nsleep 30 &", tmp_path)

    assert code == 0
    assert "sleep 0.05" in detail


def test_wait_any_exit_names_a_background_pipeline_member(tmp_path: Path) -> None:
    """A pipeline member must be named too.

    `jobs -l` gives a pipeline one line per member, and every line after the
    first carries the pid in the first column with no job spec or state. Those
    are exactly the pids `wait -n -p` reports for the launch scripts that
    background `... | tee <log>` pipelines, so reading only the second column
    would drop the command and leave a bare pid in the log.
    """
    code, detail = _run_wait_any_exit("sleep 0.05 | sleep 0.15 &\nsleep 30 &", tmp_path)

    assert code == 0
    assert "sleep 0.15" in detail


def test_wait_any_exit_reports_a_failing_child_exit_code(tmp_path: Path) -> None:
    """The exit code is the child's, not the helper's."""
    code, _ = _run_wait_any_exit("bash -c 'exit 7' &\nsleep 30 &", tmp_path)

    assert code == 7
