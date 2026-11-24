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
    pytest.mark.gpu_0,
    pytest.mark.model(MODEL),
]


@pytest.mark.parametrize("split", [False, True], ids=["e-pd", "e-p-d"])
def test_sglang_encoder_pipeline(split, tmp_path):
    info = pytest.importorskip(
        "sglang.encoder.v1.encoder_pb2", reason="requires SGLang's staged encoder API"
    )
    if not {"staged_transfer_backend", "staged_layout_id"}.issubset(
        info.GetEncoderInfoResponse.DESCRIPTOR.fields_by_name
    ):
        pytest.skip(
            "installed SGLang does not advertise staged encoder transport and layout"
        )
    pytest.importorskip(
        "sglang.test.staged_encoder", reason="requires SGLang's staged encoder fixture"
    )

    from huggingface_hub import snapshot_download

    # Only the frontend tokenizer and model card are used; E/P/D are model-free.
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
            "--native-generate",
            *(["--sidecar", sidecar] if sidecar else []),
            *(["--split"] if split else []),
        ],
        check=True,
        timeout=300,
    )
