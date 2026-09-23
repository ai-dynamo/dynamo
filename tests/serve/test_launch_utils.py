# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell-level checks for examples/common/launch_utils.sh process reporting."""

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

# The disaggregated multimodal scripts whose workers take a NixlConnector
# KV-transfer config, and the label each backgrounded process must carry.
LABELLED_SCRIPTS = {
    "disagg_multimodal_epd.sh": ["frontend", "encode", "prefill", "decode"],
    "disagg_multimodal_p_d.sh": ["frontend", "prefill", "decode"],
}

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
    snippet must not share one with pytest.
    """
    return subprocess.run(
        ["bash", "-c", f"set -e\nsource {LAUNCH_UTILS}\n{body}"],
        capture_output=True,
        text=True,
        check=False,
        start_new_session=True,
        timeout=60,
    )


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


@pytest.mark.parametrize(
    "script_name,expected_labels", sorted(LABELLED_SCRIPTS.items())
)
def test_disagg_multimodal_labels_every_background_process(
    script_name: str, expected_labels: list[str]
) -> None:
    """Keep a label on every process these launch scripts background.

    A worker added later without a label would go back to an anonymous exit
    code in the nightly log, which is what this check prevents.
    """
    lines = (LAUNCH_DIR / script_name).read_text().splitlines()
    backgrounded = [
        line
        for line in lines
        if line.rstrip().endswith("&") and not line.rstrip().endswith("&&")
    ]
    labels = [line.split()[1] for line in lines if line.startswith("dyn_track_worker ")]

    assert len(backgrounded) == len(expected_labels), backgrounded
    assert labels == expected_labels
