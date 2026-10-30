# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Frontend HTTP vs Runtime EPP input-family coverage for issue #16592.

Standard /epp hardcodes gRPC 9002 and health 9003. This module is not
host-network portable concurrent coverage: run it serially in a dedicated
network container. pre_merge is the dedicated serial/own-container CI
selector together with runtime_epp, not generic runtime-image coverage.
"""

import pytest

from tests.router.common import SessionAffinityParityMismatch
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


@pytest.mark.parametrize(
    "family",
    (
        "cache_salt",
        "routing_constraints",
        pytest.param(
            "session_id",
            marks=pytest.mark.xfail(
                strict=True,
                raises=SessionAffinityParityMismatch,
                reason=(
                    "Runtime EPP session affinity parity gap: "
                    "https://github.com/ai-dynamo/dynamo/issues/16592"
                ),
            ),
        ),
    ),
)
def test_epp_frontend_inputs_parity(request, predownload_tokenizers, tmp_path, family):
    """Compare native Frontend and Runtime EPP on cache salt, taints, and session."""
    from tests.router.common import (
        frontend_epp_cache_salt_case,
        frontend_epp_constraints_cold_case,
        frontend_epp_first_selection_cases,
        run_frontend_epp_cache_salt_parity,
        run_frontend_epp_routing_constraints_parity,
        run_frontend_epp_session_id_parity,
    )
    from tests.router.epp_parity import (
        INDEXER_MODE_APPROXIMATE,
        EppFrontendRoutingParityStack,
    )

    _ = predownload_tokenizers
    if family == "cache_salt":
        case = frontend_epp_cache_salt_case(ROUTER_MODEL_NAME)
        with EppFrontendRoutingParityStack(
            request,
            tmp_path,
            ROUTER_MODEL_NAME,
            indexer_mode=INDEXER_MODE_APPROXIMATE,
            enable_worker_topology=True,
        ) as stack:
            run_frontend_epp_cache_salt_parity(stack, case)
        return
    if family == "routing_constraints":
        cases = frontend_epp_first_selection_cases(ROUTER_MODEL_NAME)
        cold = frontend_epp_constraints_cold_case(ROUTER_MODEL_NAME)
        with EppFrontendRoutingParityStack(
            request,
            tmp_path,
            ROUTER_MODEL_NAME,
            enable_worker_topology=True,
        ) as stack:
            run_frontend_epp_routing_constraints_parity(stack, cases, cold)
        return
    cases = frontend_epp_first_selection_cases(ROUTER_MODEL_NAME)
    with EppFrontendRoutingParityStack(
        request,
        tmp_path,
        ROUTER_MODEL_NAME,
        session_affinity_ttl_secs=300,
    ) as stack:
        run_frontend_epp_session_id_parity(stack, cases)


@pytest.mark.parametrize(
    "kind",
    ("body_strict", "header_strict", "header_normal", "sentinel_fallback"),
)
def test_epp_frontend_priority_parity(request, predownload_tokenizers, tmp_path, kind):
    """Compare FE/EPP queue admission under a held 1P+1D prefill slot."""
    from tests.router.common import (
        run_frontend_epp_priority_parity,
        run_frontend_epp_priority_sentinel_fallback,
    )
    from tests.router.epp_parity import EppFrontendPriorityParityStack

    _ = predownload_tokenizers
    with EppFrontendPriorityParityStack(request, tmp_path, ROUTER_MODEL_NAME) as stack:
        if kind == "sentinel_fallback":
            run_frontend_epp_priority_sentinel_fallback(stack, ROUTER_MODEL_NAME)
            return
        run_frontend_epp_priority_parity(stack, ROUTER_MODEL_NAME, kind)
