# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo scheduling markers for the migrated Spica test suite."""

from pathlib import Path

import pytest

_INTEGRATION_TESTS = {
    "test_load_predictor_sweep_integration.py",
    "test_replay_integration.py",
}
_TEST_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply Dynamo's required scheduling markers to Spica tests."""
    for item in items:
        item_path = Path(str(item.path))
        if _TEST_ROOT not in item_path.parents:
            continue

        item.add_marker(pytest.mark.gpu_0)
        item.add_marker(pytest.mark.planner)
        item.add_marker(pytest.mark.pre_merge)
        # Pinned third-party dependencies still use these deprecated APIs. Keep
        # warning-as-error enabled for every other warning in the Spica suite.
        item.add_marker(
            pytest.mark.filterwarnings(
                r"ignore:jax\.core\.Primitive is deprecated.*:DeprecationWarning:(equinox|jax)(\..*)?"
            )
        )
        item.add_marker(
            pytest.mark.filterwarnings(
                r"ignore:datetime\.datetime\.utcnow\(\) is deprecated.*:DeprecationWarning:google\.protobuf\.internal\.well_known_types"
            )
        )
        if item_path.name in _INTEGRATION_TESTS:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.timeout(300))
        else:
            item.add_marker(pytest.mark.unit)
