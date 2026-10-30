# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Frontend HTTP vs Runtime EPP first-selection coverage for issue #16592.

Standard /epp hardcodes gRPC 9002 and health 9003. This module is not
host-network portable concurrent coverage: run it serially in a dedicated
network container. pre_merge is the dedicated serial/own-container CI
selector together with runtime_epp, not generic runtime-image coverage.
"""

import pytest

from tests.router.common import (
    frontend_epp_first_selection_cases,
    run_frontend_epp_first_selection_parity,
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


def test_epp_frontend_first_selection_parity(request, predownload_tokenizers, tmp_path):
    """Compare native Frontend and Runtime EPP first selections on a frozen cache.

    One launch warms opposite long prefixes through a served remote indexer.
    Each paired comparison then starts a fresh FE/EPP consumer pair and checks
    typed worker/rank plus tokens. A live negative compares FE R_A with EPP R_B.
    """
    # epp_parity imports gRPC/grpc_tools; pytest imports this module before
    # marker deselection, so keep that import selected-test-only.
    from tests.router.epp_parity import EppFrontendRoutingParityStack

    _ = predownload_tokenizers
    cases = frontend_epp_first_selection_cases(ROUTER_MODEL_NAME)
    with EppFrontendRoutingParityStack(request, tmp_path, ROUTER_MODEL_NAME) as stack:
        run_frontend_epp_first_selection_parity(stack, cases)
