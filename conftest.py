# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Root conftest that applies to all tests in the repository.

Behavior:
- Auto-applies hierarchical markers before marker filtering:
  - tests marked as pre_merge will additionally get marked as post_merge and nightly
- Markers are declared in pyproject.toml (tool.pytest.ini_options.markers).
- Allows opting out by setting the env var DYNAMO_DISABLE_MARKER_IMPLICATIONS=1.
"""

import os
from typing import Sequence

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    """
    Enforce hierarchical marker relationships before marker filtering.
    - pre_merge implies pre_merge, post_merge and nightly
    - post_merge implies post_merge and nightly
    """
    # Provide an escape hatch if any external runner needs to disable implications.
    if os.getenv("DYNAMO_DISABLE_MARKER_IMPLICATIONS") == "1":
        return

    for item in items:
        marker_names = {m.name for m in item.iter_markers()}

        # pre_merge implies post_merge
        if "pre_merge" in marker_names and "post_merge" not in marker_names:
            item.add_marker("post_merge")
            marker_names.add("post_merge")

        # post_merge (or pre_merge) implies nightly
        if (
            "post_merge" in marker_names or "pre_merge" in marker_names
        ) and "nightly" not in marker_names:
            item.add_marker("nightly")
