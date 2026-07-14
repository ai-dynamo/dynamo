# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang-managed native remote-backend launcher."""

import sys
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo.sglang import sidecar  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_main_forwards_sglang_argv(monkeypatch):
    native_launcher = MagicMock()
    monkeypatch.setattr(sidecar, "_run_sglang_remote", native_launcher)

    sidecar.main(["--sglang-endpoint", "http://127.0.0.1:30001"])

    native_launcher.assert_called_once_with(
        ["--sglang-endpoint", "http://127.0.0.1:30001"]
    )


def test_main_uses_process_argv_by_default(monkeypatch):
    native_launcher = MagicMock()
    monkeypatch.setattr(sidecar, "_run_sglang_remote", native_launcher)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dynamo.sglang.sidecar",
            "--sglang-endpoint",
            "http://127.0.0.1:30001",
        ],
    )

    sidecar.main()

    native_launcher.assert_called_once_with(
        ["--sglang-endpoint", "http://127.0.0.1:30001"]
    )
