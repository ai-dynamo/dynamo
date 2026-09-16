# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

REPO_ROOT = Path(__file__).parents[4]
WORKFLOW = REPO_ROOT / "benchmarks/multimodal/sweep/workflows/vllm_serve.sh"
CONFIG = REPO_ROOT / (
    "benchmarks/multimodal/sweep/experiments/embedding_cache/vllm_serve.yaml"
)


def test_embedding_cache_sweep_selects_only_requested_arms() -> None:
    config = yaml.safe_load(CONFIG.read_text())

    assert [item["label"] for item in config["configs"]] == [
        "vllm-serve",
        "vllm-serve-native-ec",
    ]
    assert config["env"]["DYN_DISABLE_NSYS"] == "1"


def _run_workflow(tmp_path: Path, *args: str, enable_nsys: bool = False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vllm = bin_dir / "vllm"
    vllm.write_text("#!/bin/bash\nprintf '[fake-vllm] %s\\n' \"$*\"\n")
    vllm.chmod(0o755)
    nsys_bin = bin_dir / "nsys"
    nsys_bin.write_text(
        "#!/bin/bash\n"
        "printf '[fake-nsys] %s\\n' \"$*\"\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ $1 == vllm ]]; then exec "$@"; fi\n'
        "  shift\n"
        "done\n"
        "exit 2\n"
    )
    nsys_bin.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_HOME": str(REPO_ROOT),
            "DYN_DISABLE_NSYS": "0" if enable_nsys else "1",
            "DYN_NSYS_BIN": str(nsys_bin),
            "DYN_NSYS_DIR": str(tmp_path / "nsys"),
            "DYN_NSYS_TMPDIR": str(tmp_path / "staging"),
            "DYN_NSYS_TRACE": "cuda,nvtx",
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )

    result = subprocess.run(
        ["bash", str(WORKFLOW), "--model", "test-model", *args],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    return result


@pytest.mark.skipif(sys.platform == "darwin", reason="workflow requires GNU readlink")
def test_vllm_workflow_profiles_only_when_enabled(tmp_path: Path) -> None:
    result = _run_workflow(tmp_path, "--max-num-seqs", "2", enable_nsys=True)

    assert "[fake-nsys] profile --trace=cuda,nvtx" in result.stdout
    assert "[fake-vllm] serve test-model" in result.stdout
    assert "--max-num-seqs 2" in result.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="workflow requires GNU readlink")
def test_vllm_workflow_launches_without_profiler(tmp_path: Path) -> None:
    result = _run_workflow(tmp_path, "--max-num-seqs", "2")

    assert "[fake-nsys]" not in result.stdout
    assert "[fake-vllm] serve test-model" in result.stdout
    assert "--port 8000" in result.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="workflow requires GNU readlink")
def test_vllm_workflow_builds_dynamo_ec_config(tmp_path: Path) -> None:
    result = _run_workflow(tmp_path, "--multimodal-embedding-cache-capacity-gb", "4")

    assert "[fake-vllm] serve test-model" in result.stdout
    assert "DynamoMultimodalEmbeddingCacheConnector" in result.stdout
    assert '"multimodal_embedding_cache_capacity_gb": 4' in result.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="workflow requires bash 4.3")
def test_vllm_workflow_requires_model() -> None:
    result = subprocess.run(["bash", str(WORKFLOW)], capture_output=True, text=True)

    assert result.returncode != 0
    assert "--model is required" in result.stderr


@pytest.mark.skipif(sys.platform == "darwin", reason="workflow requires GNU setsid")
def test_vllm_workflow_kills_server_group_after_grace(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    child_pid_file = tmp_path / "child.pid"
    vllm = bin_dir / "vllm"
    vllm.write_text(
        "#!/bin/bash\n"
        'echo $$ > "$DYN_TEST_CHILD_PID_FILE"\n'
        "trap '' INT TERM\n"
        "while true; do sleep 60; done\n"
    )
    vllm.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "DYNAMO_HOME": str(REPO_ROOT),
            "DYN_DISABLE_NSYS": "1",
            "DYN_SERVER_SHUTDOWN_GRACE_SECONDS": "1",
            "DYN_TEST_CHILD_PID_FILE": str(child_pid_file),
            "PATH": f"{bin_dir}:{env['PATH']}",
        }
    )
    process = subprocess.Popen(
        ["bash", str(WORKFLOW), "--model", "test-model"],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not child_pid_file.exists():
            time.sleep(0.05)
        assert child_pid_file.exists()
        child_pid = int(child_pid_file.read_text())

        os.killpg(process.pid, signal.SIGTERM)
        process.communicate(timeout=5)

        assert process.returncode == 0
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
