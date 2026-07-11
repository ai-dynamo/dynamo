# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for focused HA Valkey Kubernetes sweep tests."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

from benchmarks.router.kubernetes.valkey_sweep import model as sweep_model


DIGEST_IMAGE = "registry.example/dynamo@sha256:" + "a" * 64
IMAGE_ID = "registry.example/dynamo@sha256:" + "b" * 64
BINDING = {
    "image": DIGEST_IMAGE,
    "core_revision": "c" * 40,
    "core_dirty": False,
    "valkey_revision": sweep_model.VALKEY_GIT_REVISION,
    "image_ids": [IMAGE_ID],
}
REAL_AIPERF_SUMMARY = json.loads(
    (
        Path(__file__).parent
        / "kubernetes"
        / "valkey_sweep"
        / "fixtures"
        / "aiperf-0.10-summary.json"
    ).read_text(encoding="utf-8")
)


def completed(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


def aiperf_metrics(requests: int, urls: list[str] | None = None) -> dict:
    metrics = deepcopy(REAL_AIPERF_SUMMARY)
    metrics["request_count"]["avg"] = requests
    if urls is not None:
        metrics["input_config"]["endpoint"]["urls"] = urls
    return metrics
