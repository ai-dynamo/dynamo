# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory-wide configuration for the multimodal-router suites."""

from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every multimodal-router test ``topology_dependent``.

    Like the text router suites, these assert on which worker handled a
    request and on multimodal-hash routing decisions, not on the response body
    alone. See tests/README.md "Deployment-agnostic tests".
    """
    # pytest hands every conftest the WHOLE session item list, not just the
    # items under this directory, so filter explicitly. Without this the marker
    # silently lands on every test in the repo.
    for item in items:
        if _THIS_DIR in item.path.parents:
            item.add_marker(pytest.mark.topology_dependent)
