# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
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
            ["--tp-size", "4", "--colocate"],
            "6,3,2,5",
            ["6,3,2,5", "6,3,2,5"],
        ),
        (
            ["--tp-size", "2", "--encoder-tp-size", "1", "--encoders", "2"],
            "6,4,3,8",
            ["6", "4", "3,8"],
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


def test_epd_roles_follow_the_native_encoder_contract(tmp_path):
    # E is the plain HTTP encoder; only the language worker runs the Rust
    # server in external-encoder mode. Each role records what it was given.
    recorder = tmp_path / "record"
    recorder.write_text(
        f"#!{sys.executable}\n"
        """import json, os, sys
from pathlib import Path
records = Path(os.environ["RECORDS"])
argv = sys.argv[1:]
if "sglang.launch_server" in argv:
    assert argv[argv.index("--encoder-transfer-backend") + 1] == "zmq_to_scheduler"
    assert "--grpc-mode" not in argv and "--disaggregation-mode" not in argv
    if "--encoder-only" in argv:
        assert "SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF" not in os.environ
        assert "SGLANG_RUST_SERVER" not in os.environ
        assert "--grpc-port" not in argv and "--language-only" not in argv
        role = "encoder"
    else:
        assert os.environ["SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF"] == "1"
        assert os.environ["SGLANG_RUST_SERVER"] == "1"
        assert "--language-only" in argv and "--grpc-port" in argv
        role = "language"
    port = argv[argv.index("--port") + 1]
    (records / f"engine-{role}-{port}").write_text("")
elif "--grpc-endpoint" in argv:
    name = "sidecar-" + os.environ["DYN_SYSTEM_PORT"]
    (records / name).write_text(json.dumps(argv))
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
        # The launcher must not pass the language worker's mode to E.
        "SGLANG_RUST_SERVER": "1",
        "SGLANG_ENABLE_EXTERNAL_ENCODER_HANDOFF": "1",
    }
    subprocess.run(
        [
            str(LAUNCHER),
            "--encoders",
            "2",
            "--single-gpu",
            "--model",
            "test/model",
            "--",
            "--served-model-name",
            "vision",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        start_new_session=True,
        timeout=20,
    )
    engines = sorted(path.name for path in tmp_path.glob("engine-*"))
    assert engines == [
        "engine-encoder-30003",
        "engine-encoder-30006",
        "engine-language-30009",
    ]
    sidecars = [
        json.loads(path.read_text()) for path in sorted(tmp_path.glob("sidecar-*"))
    ]
    assert len(sidecars) == 3
    for port, sidecar in zip((30003, 30006), sidecars):
        assert sidecar == [
            "--disaggregation-mode",
            "encode",
            "--grpc-endpoint",
            f"127.0.0.1:{port}",
            "--encoder-model-path",
            "test/model",
            "--encoder-served-model-name",
            "vision",
        ]
    assert sidecars[2] == ["--grpc-endpoint", "127.0.0.1:30010", "--route-to-encoder"]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--split-pd", "--single-gpu"], "--split-pd is not supported"),
        (["--encoder-transfer-backend", "nixl"], "Unknown option"),
    ],
)
def test_epd_launcher_rejects_unsupported_topologies(options, message):
    result = subprocess.run(
        [str(LAUNCHER), *options],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert message in result.stderr
