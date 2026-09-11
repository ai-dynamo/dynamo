# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.fault_tolerance.deploy.worker_names import get_worker_service_name

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


@pytest.mark.parametrize(
    ("backend", "deploy_type", "expected"),
    [
        ("vllm", "agg", "worker"),
        ("vllm", "disagg", "decode"),
        ("sglang", "agg", "decode"),
        ("trtllm", "agg", "TRTLLMWorker"),
        ("trtllm", "disagg", "decode"),
    ],
)
def test_get_worker_service_name(backend, deploy_type, expected):
    assert get_worker_service_name(backend, deploy_type) == expected
