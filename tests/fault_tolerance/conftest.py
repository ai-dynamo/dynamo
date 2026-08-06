# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory-wide configuration for the fault-tolerance suites."""

from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every fault-tolerance test ``topology_dependent``.

    These tests verify what happens when specific parts of a deployment fail:
    they SIGKILL a chosen worker process or pod, pause a rank, tail a worker's
    log for a migration marker, or compare metrics across two origins. None of
    that is expressible against a single frontend URL, so they need a
    deployment handle by construction and are exempt from the
    deployment-agnostic rule. See tests/README.md "Deployment-agnostic tests".
    """
    # pytest hands every conftest the WHOLE session item list, not just the
    # items under this directory, so filter explicitly. Without this the marker
    # silently lands on every test in the repo.
    for item in items:
        if _THIS_DIR in item.path.parents:
            item.add_marker(pytest.mark.topology_dependent)
