# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Root conftest that applies to all tests in the repository.
Auto-applies the following marker hierarchy:
- pre_merge ⇒ post_merge + nightly
- post_merge ⇒ nightly
"""

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session, config, items):
    """
    Enforce hierarchical marker relationships before marker filtering.
    - pre_merge implies post_merge and nightly
    - post_merge implies nightly
    """
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
