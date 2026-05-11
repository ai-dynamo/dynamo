# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Testbed pytest configuration.

Three responsibilities, in strict order:

1. Install a stub for ``dynamo._core`` if the maturin-built native binding
   isn't available on this machine. This MUST happen before any test module
   imports ``dynamo.planner.*`` (which transitively pulls in the runtime).
2. Block real K8s writes via session-scoped autouse fixture.
3. Register testbed-specific markers so ``--strict-markers`` is happy.
"""

import importlib.util
import pathlib
import sys

# ---------------------------------------------------------------------------
# Step 1 — install the dynamo._core stub at conftest collection time.
#
# We load the stub module via importlib *bypassing* the dynamo.planner
# package, otherwise we'd trigger the exact import chain we're trying to
# avoid (planner.__init__ → connectors.kubernetes_api → dynamo.runtime
# → dynamo._core).
# ---------------------------------------------------------------------------
_STUB_PATH = pathlib.Path(__file__).parent / "_runtime_stub.py"
_spec = importlib.util.spec_from_file_location("_runtime_stub_isolated", _STUB_PATH)
_stub_mod = importlib.util.module_from_spec(_spec)
sys.modules["_runtime_stub_isolated"] = _stub_mod
_spec.loader.exec_module(_stub_mod)
_stub_installed = _stub_mod.install_stub_if_needed()

import pytest
from unittest.mock import patch


def pytest_configure(config: pytest.Config) -> None:
    """Register testbed-only markers."""
    config.addinivalue_line(
        "markers", "testbed: synthetic-metrics power planner stress testbed"
    )
    config.addinivalue_line(
        "markers",
        "gamma: γ-class scenarios (need mocker / dynamo.llm with native binding)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip γ-class tests when only the ``dynamo._core`` stub is installed.

    γ-class scenarios depend on the real Rust mocker (`PlannerReplayBridge`),
    which only exists when the maturin-built native binding is on
    ``sys.path``. ``importorskip("dynamo.llm")`` doesn't work here because
    ``dynamo.llm`` is pure-Python and importable even with the stub: we have
    to look at the stub flag explicitly.
    """
    if not _stub_installed:
        return
    skip_marker = pytest.mark.skip(
        reason="γ-class test requires real dynamo._core (mocker / "
        "PlannerReplayBridge); only the testbed stub is installed on "
        "this machine."
    )
    for item in items:
        if "gamma" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True, scope="session")
def _block_real_k8s_writes():
    """Monkeypatch K8s pod-write paths to raise — defense in depth."""
    with patch(
        "kubernetes.client.CoreV1Api.patch_namespaced_pod",
        side_effect=RuntimeError("Testbed: real K8s writes are forbidden"),
    ), patch(
        "kubernetes.client.CoreV1Api.create_namespaced_pod",
        side_effect=RuntimeError("Testbed: real K8s writes are forbidden"),
    ):
        yield
