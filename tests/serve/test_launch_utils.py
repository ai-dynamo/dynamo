# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell-level checks for examples/common/launch_utils.sh process reporting."""

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

LAUNCH_UTILS = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
EPD_SCRIPT = (
    Path(__file__).parents[2] / "examples/backends/vllm/launch/disagg_multimodal_epd.sh"
)


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


def test_wait_any_exit_names_the_labelled_process_that_left() -> None:
    """Report which labelled process exited, not just its exit code."""
    result = _run_script(
        """
        sleep 30 & dyn_track_worker frontend
        (sleep 0.2; exit 7) & dyn_track_worker prefill
        sleep 30 & dyn_track_worker decode
        wait_any_exit
        """
    )

    assert result.returncode == 7, result.stderr
    assert "Worker 'prefill' (pid" in result.stdout, result.stdout
    assert "exited with code 7" in result.stdout, result.stdout
    assert "Worker 'frontend'" not in result.stdout, result.stdout
    assert "Worker 'decode'" not in result.stdout, result.stdout


def test_wait_any_exit_keeps_the_generic_message_without_labels() -> None:
    """Leave unlabelled launch scripts on the original message."""
    result = _run_script(
        """
        sleep 30 &
        (sleep 0.2; exit 3) &
        wait_any_exit
        """
    )

    assert result.returncode == 3, result.stderr
    assert "A background process exited with code 3" in result.stdout, result.stdout


def test_disagg_multimodal_epd_labels_every_background_process() -> None:
    """Keep a label on each of the four processes the E/P/D script starts."""
    script = EPD_SCRIPT.read_text()
    backgrounded = [line for line in script.splitlines() if line.rstrip().endswith("&")]
    labels = [
        line.split()[1]
        for line in script.splitlines()
        if line.startswith("dyn_track_worker ")
    ]

    assert len(backgrounded) == 4, backgrounded
    assert labels == ["frontend", "encode", "prefill", "decode"]
