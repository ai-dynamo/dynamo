# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the optional benchmark package out of unrelated backend collection."""

import importlib.util
import os
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1]
_MISSING = tuple(
    name
    for name in ("dynamo_decision_perf", "aiperf")
    if importlib.util.find_spec(name) is None
)
collect_ignore_glob = ["test_*.py"] if _MISSING else []


def pytest_configure(config):
    explicit = (
        Path.cwd().resolve().is_relative_to(_PACKAGE)
        or config.rootpath.resolve().is_relative_to(_PACKAGE)
        or any(
            Path(arg.split("::", 1)[0]).resolve().is_relative_to(_PACKAGE)
            for arg in config.args
        )
        or any(
            os.getenv(key) == "1"
            for key in (
                "DECISION_PERF_INTEGRATION",
                "DECISION_PERF_MOCKER_INTEGRATION",
            )
        )
    )
    if _MISSING and explicit:
        raise pytest.UsageError(
            "Decision API instrument dependencies missing: "
            + ", ".join(_MISSING)
            + ". Install benchmarks/decision_api[test] in the isolated benchmark environment."
        )
