# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]
LAUNCHER = Path(__file__).resolve().parents[2] / "lib/sidecar/sglang/launch/encoded.sh"


@pytest.mark.parametrize(
    ("options", "devices", "groups"),
    [
        (["--single-gpu"], "GPU-X", ["GPU-X", "GPU-X"]),
        (
            ["--tp-size", "4", "--colocate", "--split-pd"],
            "6,3,2,5",
            ["6,3,2,5", "6,3,2,5", "6,3,2,5"],
        ),
        (
            [
                "--tp-size",
                "2",
                "--encoder-tp-size",
                "1",
                "--encoders",
                "2",
                "--split-pd",
            ],
            "6,4,3,8,0,1",
            ["6", "4", "3,8", "0,1"],
        ),
        (
            ["--encoder-tp-size", "1", "--", "--tensor-parallel-size=2"],
            None,
            ["0", "1,2"],
        ),
        (["--tp-size", "4", "--colocate"], "0,1", []),
    ],
)
def test_epd_gpu_placement(tmp_path, options, devices, groups):
    # Capture actual native launch arguments without starting any model or GPU runtime.
    recorder = tmp_path / "record"
    recorder.write_text(
        """#!/bin/bash
if [[ "$*" == *sglang.launch_server* ]]; then
    while (($#)); do
        case "$1" in
            --tp-size) tp="$2"; shift ;;
            --nccl-port) port="$2"; shift ;;
        esac
        shift
    done
    printf '%s %s\\n' "$CUDA_VISIBLE_DEVICES" "$tp" > "$RECORDS/$port"
fi
sleep 0.3
"""
    )
    recorder.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "RECORDS": str(tmp_path),
        "DYNAMO_PYTHON": str(recorder),
        "SGLANG_PYTHON": str(recorder),
        "DYNAMO_SIDECAR_BIN": str(recorder),
        "DYN_LAUNCH_WAIT_ALL": "1",
    }
    if devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = devices
    result = subprocess.run(
        [str(LAUNCHER), *options],
        env=env,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=20,
    )
    if groups:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == 2
        assert "Need 4 visible GPUs" in result.stderr
    launches = [file.read_text().strip() for file in sorted(tmp_path.glob("[0-9]*"))]
    assert launches == [f"{group} {len(group.split(','))}" for group in groups]


@pytest.mark.parametrize("existing_key", [False, True])
def test_epd_workers_share_handoff_key(tmp_path, existing_key):
    # Each E/P/D process observes the file while the launcher's children are live.
    recorder = tmp_path / "record"
    recorder.write_text(
        f"#!{sys.executable}\n"
        """import os, sys
from pathlib import Path
if "sglang.launch_server" in sys.argv:
    assert os.environ["SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF"] == "1"
    assert os.environ["SGLANG_RUST_SERVER"] == ("0" if "--encoder-only" in sys.argv else "1")
    key = Path(os.environ["SGLANG_ENCODER_HANDOFF_KEY_FILE"])
    assert len(key.read_bytes()) == 32
    port = sys.argv[sys.argv.index("--nccl-port") + 1]
    (Path(os.environ["RECORDS"]) / port).write_text(str(key))
"""
    )
    recorder.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "RECORDS": str(tmp_path),
        "DYNAMO_PYTHON": str(recorder),
        "SGLANG_PYTHON": str(recorder),
        "DYNAMO_SIDECAR_BIN": str(recorder),
        "DYN_LAUNCH_WAIT_ALL": "1",
    }
    supplied = tmp_path / "existing.key"
    if existing_key:
        supplied.write_bytes(os.urandom(32))
        env["SGLANG_ENCODER_HANDOFF_KEY_FILE"] = str(supplied)
    subprocess.run(
        [str(LAUNCHER), "--split-pd", "--single-gpu"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        start_new_session=True,
        timeout=20,
    )
    paths = [Path(result.read_text()) for result in tmp_path.glob("[0-9]*")]
    assert len(paths) == 3 and len(set(paths)) == 1
    if existing_key:
        assert paths[0] == supplied and supplied.is_file()
    else:
        assert not paths[0].exists()
