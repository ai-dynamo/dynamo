# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell-level checks for examples/common/launch_utils.sh process reporting."""

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

LAUNCH_UTILS = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
LAUNCH_DIR = Path(__file__).parents[2] / "examples/backends/vllm/launch"

# The disaggregated multimodal scripts whose workers take a NixlConnector
# KV-transfer config, and the label each backgrounded process must carry.
LABELLED_SCRIPTS = {
    "disagg_multimodal_epd.sh": ["frontend", "encode", "prefill", "decode"],
    "disagg_multimodal_p_d.sh": ["frontend", "prefill", "decode"],
}


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
