# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the CPU encoder pipeline through the public frontend and native sidecars."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DRIVER = (
    Path(__file__).resolve().parents[2] / "lib/sidecar/sglang/tests/encoder_pipeline.py"
)
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.model(MODEL),
]


# The driver bounds itself at 200 s; a run takes well under a minute.
@pytest.mark.timeout(240)
def test_sglang_encoder_pipeline(tmp_path):
    pytest.importorskip("grpc_tools", reason="generates the SGLang API stubs")

    from huggingface_hub import snapshot_download

    # Only the frontend tokenizer and model card are used; SGLang is faked.
    model = snapshot_download(MODEL, allow_patterns=["*.json", "*.txt", "*.jinja"])
    sidecar = os.getenv("DYNAMO_SIDECAR_BIN") or shutil.which("dynamo-sglang-sidecar")
    subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--frontend-python",
            os.getenv("DYNAMO_PYTHON", sys.executable),
            "--model",
            model,
            "--logs",
            str(tmp_path),
            *(["--sidecar", sidecar] if sidecar else []),
        ],
        check=True,
        timeout=220,
    )
