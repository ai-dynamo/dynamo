# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

DOCKERFILE = Path(__file__).with_name("Dockerfile")


def _stage(dockerfile: str, start: str, end: str | None = None) -> str:
    stage = dockerfile.split(start, 1)[1]
    return stage if end is None else stage.split(end, 1)[0]


def test_criu_builds_against_oldest_supported_placeholder_libc():
    dockerfile = DOCKERFILE.read_text()

    assert "FROM ubuntu:22.04 AS criu-builder" in dockerfile
    assert "FROM ubuntu:24.04 AS criu-builder" not in dockerfile


def test_runtime_variants_select_available_gnutls_package():
    dockerfile = DOCKERFILE.read_text()
    stages = (
        _stage(
            dockerfile,
            "FROM ${AGENT_BASE_IMAGE} AS agent",
            "FROM ${BASE_IMAGE} AS placeholder",
        ),
        _stage(dockerfile, "FROM ${BASE_IMAGE} AS placeholder"),
    )

    for stage in stages:
        assert "apt-cache show libgnutls30t64" in stage
        assert "echo libgnutls30t64" in stage
        assert "echo libgnutls30" in stage
        assert '"${gnutls_package}"' in stage
