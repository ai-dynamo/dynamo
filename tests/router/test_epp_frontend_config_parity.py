# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Frontend vs Runtime EPP effective-config coverage for issue #16592.

Standard /epp hardcodes gRPC 9002 and health 9003. This module is not
host-network portable concurrent coverage: run it serially in a dedicated
network container. pre_merge is the dedicated serial/own-container CI
selector together with runtime_epp, not generic runtime-image coverage.

Inventory and completeness guards are fail-closed and knowingly incomplete
for this increment. They are ordinary failures, not skip/xfail. Gap
attempts share the same runner and stay ordinary tests.
"""

import pytest

from tests.router.epp_config_contract import (
    PROFILE_NAMES,
    assert_coverage_complete,
    assert_inventory_matches_manifest,
    assert_manifest_bindings_match_collected_tests,
    assert_required_sources_exist,
    run_effective_config_profile,
)
from tests.utils.constants import ROUTER_MODEL_NAME

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.runtime_epp,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(360),
]


def test_required_config_sources_exist():
    """Fail if the routing CLI or Rust config sources are absent."""
    assert_required_sources_exist()


def test_discovered_identifiers_have_decisions():
    """New or stale CLI/env/schema names fail until a reviewed row exists."""
    assert_inventory_matches_manifest()


def test_fe_honored_coverage_is_complete():
    """Pending or undecided FE-honored rows are not full-issue completion."""
    assert_coverage_complete()


def test_manifest_bindings_match_collected_tests(request):
    """Observer/profile metadata must name a collected profile test."""
    assert_manifest_bindings_match_collected_tests(
        item.nodeid for item in request.session.items
    )


@pytest.mark.parametrize("profile_name", list(PROFILE_NAMES))
def test_effective_config_profile(
    request, predownload_tokenizers, tmp_path, monkeypatch, profile_name
):
    """Compare native FE to_json with the owned EPP post-parse init event.

    Gap attempts are ordinary profiles. Setup, parse, warning, and FE
    sentinel failures stay hard AssertionError.
    """
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_effective_config_profile(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        profile_name,
        EppFrontendParityStack,
    )
