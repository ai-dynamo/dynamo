# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Frontend vs Runtime EPP effective-config coverage for issue #16592.

Standard /epp hardcodes gRPC 9002 and health 9003. This module is not
host-network portable concurrent coverage: run it serially in a dedicated
network container. pre_merge is the dedicated serial/own-container CI
selector together with runtime_epp, not generic runtime-image coverage.

Inventory and completeness guards are fail-closed and knowingly incomplete
for this increment. They are ordinary failures, not skip/xfail. The two
measured gaps are strict xfails of their dedicated mismatch subclasses.
Setup, warning, FE sentinel, parse, and other mismatches stay ordinary
hard failures. Other profiles stay ordinary tests. Admission/enforce
ignored-compatibility cases, the five conditional scalar probes, and the
positional-search probe are ordinary tests, not xfail. Remote-indexer and
session TTL rows bind existing collected Items rather than new tests.
Overlap refresh is an ordinary owned-process queue-INFO plus plain-chat
token-parity test, not xfail.
"""

import pytest

from tests.router.epp_config_contract import (
    ADMISSION_CONTROL_CASE_IDS,
    CONDITIONAL_SCALAR_PROBES,
    ENFORCE_DISAGG_CASE_IDS,
    PROFILE_NAMES,
    ConditionalJsonConfigMismatch,
    LoadAwarePresetConfigMismatch,
    assert_coverage_complete,
    assert_inventory_matches_manifest,
    assert_manifest_bindings_match_collected_tests,
    assert_required_sources_exist,
    run_conditional_scalar_not_fe_honored,
    run_effective_config_profile,
    run_ignored_admission_control,
    run_ignored_enforce_disagg,
    run_overlap_refresh_effective_config,
    run_positional_search_not_fe_honored,
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
    """Observer rows must bind collected Items and exact measured xfails."""
    assert_manifest_bindings_match_collected_tests(request.session.items)


def _effective_config_profile_case(profile_name: str):
    if profile_name == "load_aware_preset":
        return pytest.param(
            profile_name,
            marks=pytest.mark.xfail(
                strict=True,
                raises=LoadAwarePresetConfigMismatch,
                reason=(
                    "issue #16592: measured FE load-aware preset vs EPP "
                    "ignoring DYN_ROUTER_LOAD_AWARE"
                ),
            ),
        )
    if profile_name == "conditional_json":
        return pytest.param(
            profile_name,
            marks=pytest.mark.xfail(
                strict=True,
                raises=ConditionalJsonConfigMismatch,
                reason=(
                    "issue #16592: measured FE conditional JSON vs EPP "
                    "default scalar config"
                ),
            ),
        )
    return profile_name


@pytest.mark.parametrize(
    "profile_name",
    [_effective_config_profile_case(name) for name in PROFILE_NAMES],
)
def test_effective_config_profile(
    request, predownload_tokenizers, tmp_path, monkeypatch, profile_name
):
    """Compare native FE to_json with the owned EPP post-parse init event.

    load_aware_preset and conditional_json are measured strict gaps.
    Setup, parse, warning, and FE sentinel failures stay hard AssertionError.
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


@pytest.mark.parametrize("case_id", ADMISSION_CONTROL_CASE_IDS)
def test_ignored_admission_control(monkeypatch, case_id):
    """Admission-control CLI/env must warn and leave native kwargs/JSON unchanged."""
    run_ignored_admission_control(monkeypatch, case_id)


@pytest.mark.parametrize("case_id", ENFORCE_DISAGG_CASE_IDS)
def test_ignored_enforce_disagg(monkeypatch, case_id):
    """Enforce-disagg CLI/env must warn and leave native kwargs/JSON unchanged."""
    run_ignored_enforce_disagg(monkeypatch, case_id)


@pytest.mark.parametrize(
    ("env_name", "sentinel"),
    CONDITIONAL_SCALAR_PROBES,
    ids=[name for name, _ in CONDITIONAL_SCALAR_PROBES],
)
def test_conditional_scalar_not_fe_honored(monkeypatch, env_name, sentinel):
    """Each EPP-only conditional scalar env must leave FE native JSON unchanged."""
    run_conditional_scalar_not_fe_honored(monkeypatch, env_name, sentinel)


def test_positional_search_not_fe_honored(monkeypatch):
    """DYN_ROUTER_POSITIONAL_SEARCH_MODE must leave FE native JSON unchanged."""
    run_positional_search_not_fe_honored(monkeypatch)


def test_overlap_refresh_effective_config(
    request, predownload_tokenizers, tmp_path, monkeypatch
):
    """Disable overlap refresh on both owned processes and compare queue INFO."""
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_overlap_refresh_effective_config(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
    )
