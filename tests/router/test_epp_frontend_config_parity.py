# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Frontend vs Runtime EPP effective-config coverage for issue #16592.

Standard /epp hardcodes gRPC 9002 and health 9003. This module is not
host-network portable concurrent coverage: run it serially in a dedicated
network container. pre_merge is the dedicated serial/own-container CI
selector together with runtime_epp, not generic runtime-image coverage.

This is a staged partial configuration suite. Explicit pending rows and
their reasons are the unmeasured list. Inventory, identifier-ownership,
and collected-Item binding checks remain ordinary hard failures, not
skip/xfail. The two measured profile gaps, the decode-threshold consumer gap, and the
session-prefix config gap are strict xfails of their dedicated mismatch
subclasses. Setup, warning, FE sentinel, parse, and other mismatches stay
ordinary hard failures. Other profiles stay ordinary tests.
Admission/enforce ignored-compatibility cases, the five conditional scalar
probes, and the positional-search probe are ordinary tests, not xfail.
Remote-indexer and session TTL rows bind existing collected Items rather
than new tests. Overlap refresh is an ordinary owned-process queue-INFO
plus plain-chat token-parity test, not xfail. Keyed tracking and
approximate-policy core-metrics rows are ordinary unverified-candidate
tests, not xfail. The 31-byte size control is parameterized over
independent frontend/epp arms. active_decode_blocks_threshold is a
measured strict xfail of ActiveDecodeThresholdConfigMismatch after
default-None and configured 0.25 remote-load observation.
enable_session_prefix_index is a measured strict xfail of
SessionPrefixIndexConfigMismatch after omitted/false/true and dual-invalid
YAML controls on env-true/kv-false FE reject vs EPP accept. shared_cache_type
is an ordinary owned HiCache client-construction test, not xfail. Dynamic
block-size cross-input cases bind kv_cache_block_size as not-FE-honored
Dynamic worker-card geometry, not CLI support. Seven input/startup/capability
rows use source-backed unsupported_input/unsupported_capability decisions
with one ordinary source-capability Item; prefill abs/frac also bind three
ordinary FE admin /busy_threshold boundary profiles.
"""

import pytest

from tests.router.epp_config_contract import (
    ADMISSION_CONTROL_CASE_IDS,
    APPROXIMATE_POLICY_NAMES,
    CONDITIONAL_SCALAR_PROBES,
    ENFORCE_DISAGG_CASE_IDS,
    KEYED_TRACKING_PROFILE,
    KEYED_TRACKING_SIZE_ARMS,
    PREFILL_ADMISSION_PROFILES,
    PROFILE_NAMES,
    ActiveDecodeThresholdConfigMismatch,
    ConditionalJsonConfigMismatch,
    LoadAwarePresetConfigMismatch,
    SessionPrefixIndexConfigMismatch,
    assert_inventory_matches_manifest,
    assert_manifest_bindings_match_collected_tests,
    assert_required_sources_exist,
    run_conditional_scalar_not_fe_honored,
    run_effective_config_profile,
    run_ignored_admission_control,
    run_ignored_enforce_disagg,
    run_input_startup_capability_decisions,
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
    # epp_parity imports gRPC/grpc_tools; pytest imports this module before
    # marker deselection, so keep that import selected-test-only.
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
    # epp_parity imports gRPC/grpc_tools; pytest imports this module before
    # marker deselection, so keep that import selected-test-only.
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_overlap_refresh_effective_config(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
    )


@pytest.mark.parametrize("profile_name", [KEYED_TRACKING_PROFILE])
def test_keyed_tracking_runtime_config(
    request, predownload_tokenizers, tmp_path, monkeypatch, profile_name
):
    """Consume keyed-xxh3-v1 on both real routers and compare hash/ID."""
    from tests.router.common import run_keyed_tracking_runtime_config
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_keyed_tracking_runtime_config(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
        profile_name,
    )


@pytest.mark.parametrize("arm", list(KEYED_TRACKING_SIZE_ARMS))
def test_keyed_tracking_key_file_size(
    request, predownload_tokenizers, tmp_path, monkeypatch, arm
):
    """One owned consumer arm must reject a 31-byte public tracking key file."""
    from tests.router.common import run_keyed_tracking_key_file_size
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_keyed_tracking_key_file_size(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
        arm,
    )


@pytest.mark.parametrize("policy_name", list(APPROXIMATE_POLICY_NAMES))
def test_approximate_cache_policy_metrics(
    request, predownload_tokenizers, tmp_path, monkeypatch, policy_name
):
    """Read configured/effective core gauges for LRU and omitted-env TTL."""
    from tests.router.common import run_approximate_cache_policy_metrics
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_approximate_cache_policy_metrics(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
        policy_name,
    )


@pytest.mark.xfail(
    strict=True,
    raises=ActiveDecodeThresholdConfigMismatch,
    reason=(
        "issue #16592: FE configured 0.25 vs EPP monitor None with default "
        "None control; effective remote-load consumer scope, not a proven "
        "admission-pressure boundary"
    ),
)
def test_active_decode_threshold_consumer(
    request, predownload_tokenizers, tmp_path, monkeypatch
):
    """Observe default None and configured 0.25 decode-threshold load events."""
    from tests.router.common import run_active_decode_threshold_consumer
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_active_decode_threshold_consumer(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
    )


def test_shared_cache_type_consumer(
    request, predownload_tokenizers, tmp_path, monkeypatch
):
    """Observe omitted none and explicit hicache shared-cache construction."""
    from tests.router.common import run_shared_cache_type_consumer
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_shared_cache_type_consumer(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
    )


@pytest.mark.xfail(
    strict=True,
    raises=SessionPrefixIndexConfigMismatch,
    reason=(
        "issue #16592: measured env-true/kv-false FE validation rejection vs "
        "EPP accept-and-route; YAML dual-reject control; input/config scope only"
    ),
)
def test_session_prefix_index_config_parity(
    request, predownload_tokenizers, tmp_path, monkeypatch
):
    """Measure FE session-prefix env validation reject vs EPP accept-and-route."""
    from tests.router.common import run_session_prefix_index_config_parity
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_session_prefix_index_config_parity(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
    )


@pytest.mark.parametrize(
    "worker_block_size,frontend_cli_block_size",
    [
        pytest.param(32, 8, id="worker_card32_frontend_cli8"),
        pytest.param(32, 64, id="worker_card32_frontend_cli64"),
        pytest.param(16, 64, id="worker_card16_frontend_cli64"),
    ],
)
def test_dynamic_block_size_consumer(
    request,
    predownload_tokenizers,
    tmp_path,
    worker_block_size,
    frontend_cli_block_size,
):
    """Attribute Dynamic FE model block size to the worker card across CLI sizes."""
    from tests.router.common import run_dynamic_block_size_consumer
    from tests.router.epp_parity import EppFrontendRoutingParityStack

    _ = predownload_tokenizers
    run_dynamic_block_size_consumer(
        request,
        tmp_path,
        ROUTER_MODEL_NAME,
        EppFrontendRoutingParityStack,
        worker_block_size=worker_block_size,
        frontend_cli_block_size=frontend_cli_block_size,
    )


def test_input_startup_capability_decisions(monkeypatch):
    """Source-backed unsupported input/capability decisions for seven settings."""
    run_input_startup_capability_decisions(monkeypatch)


@pytest.mark.parametrize("profile_name", list(PREFILL_ADMISSION_PROFILES))
def test_prefill_admission_input_boundaries(
    request, predownload_tokenizers, tmp_path, monkeypatch, profile_name
):
    """GET /busy_threshold original FE handles for default/abs8192/frac0.75."""
    from tests.router.common import run_prefill_admission_input_boundaries
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    run_prefill_admission_input_boundaries(
        request,
        tmp_path,
        monkeypatch,
        ROUTER_MODEL_NAME,
        EppFrontendParityStack,
        profile_name,
    )
