# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for the parity harness.

E2E tests need session-scoped server boots: per `(impl, parser_family)`
we boot at most once and reuse across cases. They're gated behind the
`e2e` marker; run with:

    pytest tests/parity/parser/ -m e2e -v
"""

from __future__ import annotations

from typing import Iterator

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end test that boots a real server")


@pytest.fixture(scope="session")
def e2e_server_cache() -> dict:
    """Cache of (impl, family) -> base_url for the current session."""
    return {}


@pytest.fixture(scope="session")
def e2e_server_lifecycles() -> Iterator[list]:
    """Stack of context managers to exit at session end (LIFO)."""
    cms: list = []
    yield cms
    for cm in reversed(cms):
        try:
            cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001 — best-effort teardown
            print(f"[parity] cleanup error: {e}")
