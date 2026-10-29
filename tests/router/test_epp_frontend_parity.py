# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Frontend HTTP vs Runtime EPP token-parity coverage for issue #16592.

Standard /epp hardcodes gRPC 9002 and health 9003. This module is not
host-network portable concurrent coverage: run it serially in a dedicated
network container. pre_merge is the dedicated serial/own-container CI
selector together with runtime_epp, not generic runtime-image coverage.
"""

import pytest

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


def test_epp_frontend_token_parity(request, predownload_tokenizers, tmp_path):
    """Compare native Frontend query_instance_id tokens to real Runtime EPP mutations.

    Cases (one launch): plain chat, text completion, system chat, multiturn
    chat, then a live negative control that changes one arm's content.

    Uses TCP request plane and ZMQ event plane so nats-server/etcd binaries
    are not prerequisites.
    """
    from tests.router.common import (
        frontend_epp_token_parity_cases,
        run_frontend_epp_token_parity,
    )
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    cases = frontend_epp_token_parity_cases(ROUTER_MODEL_NAME)
    with EppFrontendParityStack(request, tmp_path, ROUTER_MODEL_NAME) as stack:
        run_frontend_epp_token_parity(stack, cases, include_negative_control=True)


def test_epp_frontend_tools_token_parity(request, predownload_tokenizers, tmp_path):
    """Compare native Frontend and Runtime EPP tokens for tool schema and history.

    Cases (one launch): tool schema declaration and assistant tool_calls plus
    tool reply history. The runner also checks same-arm schema/omission
    sensitivity and live negative controls. This covers input rendering only.
    """
    from tests.router.common import (
        frontend_epp_tools_token_parity_cases,
        run_frontend_epp_tools_token_parity,
    )
    from tests.router.epp_parity import EppFrontendParityStack

    _ = predownload_tokenizers
    cases = frontend_epp_tools_token_parity_cases(ROUTER_MODEL_NAME)
    with EppFrontendParityStack(request, tmp_path, ROUTER_MODEL_NAME) as stack:
        run_frontend_epp_tools_token_parity(stack, cases)
