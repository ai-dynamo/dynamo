# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang-managed native sidecar launcher."""

import importlib
import sys
from unittest.mock import MagicMock, call

import pytest

pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run `maturin develop` first",
)

from dynamo import sglang as sglang_package  # noqa: E402
from dynamo.runtime import logging as runtime_logging  # noqa: E402
from dynamo.sglang import sidecar  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_main_forwards_sglang_argv(monkeypatch):
    lifecycle = MagicMock()
    monkeypatch.setattr(
        sidecar, "configure_dynamo_logging", lifecycle.configure_logging
    )
    monkeypatch.setattr(sidecar, "_run_sglang_sidecar", lifecycle.run_sidecar)

    sidecar.main(["--sglang-endpoint", "http://127.0.0.1:30001"])

    assert lifecycle.mock_calls == [
        call.configure_logging(service_name="dynamo.sglang.sidecar"),
        call.run_sidecar(["--sglang-endpoint", "http://127.0.0.1:30001"]),
    ]


def test_main_uses_process_argv_by_default(monkeypatch):
    logging_config = MagicMock()
    native_launcher = MagicMock()
    monkeypatch.setattr(sidecar, "configure_dynamo_logging", logging_config)
    monkeypatch.setattr(sidecar, "_run_sglang_sidecar", native_launcher)
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

    logging_config.assert_called_once_with(service_name="dynamo.sglang.sidecar")
    native_launcher.assert_called_once_with(
        ["--sglang-endpoint", "http://127.0.0.1:30001"]
    )


def test_import_does_not_configure_logging(monkeypatch):
    logging_config = MagicMock()

    with monkeypatch.context() as context:
        context.setattr(runtime_logging, "configure_dynamo_logging", logging_config)
        context.setattr(sglang_package, "sidecar", sidecar)
        context.delitem(sys.modules, "dynamo.sglang.sidecar")
        importlib.import_module("dynamo.sglang.sidecar")

    logging_config.assert_not_called()
