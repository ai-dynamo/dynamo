# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_vllm_installs_s3fs_with_dependencies() -> None:
    dockerfile = (ROOT / "container/templates/vllm_runtime.Dockerfile").read_text()

    marker = "\"$(grep '^s3fs' /tmp/requirements.vllm.txt)\""
    assert marker in dockerfile
    install_prefix = dockerfile.split(marker, 1)[0].rsplit("uv pip install", 1)[1]
    assert "--no-deps" not in install_prefix
    assert "import aiobotocore, fsspec, s3fs" in dockerfile
    assert "get_filesystem_class('s3')" in dockerfile
