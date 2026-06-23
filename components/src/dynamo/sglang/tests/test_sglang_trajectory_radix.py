# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _handler(*, enabled: bool = True):
    handler = DecodeWorkerHandler.__new__(DecodeWorkerHandler)
    handler.enable_session_radix_cache = enabled
    return handler


def test_trajectory_id_becomes_sglang_session_tag():
    handler = _handler()
    request = {"agent_context": {"trajectory_id": "trajectory-1"}}

    assert handler._session_kwargs(request) == {
        "session_params": {"id": "trajectory-1"}
    }
    assert _handler(enabled=False)._session_kwargs(request) == {}
