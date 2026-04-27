# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sglang health-check payload shape tests.

Asserts the canary HEALTH_CHECK_KEY marker is layered onto the disagg payload
(which the prefill handler reads), absent from the decode/agg payload (which
no handler reads), and survives DYN_HEALTH_CHECK_PAYLOAD env overrides.
"""

import json

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang.health_check import (
    SglangDisaggHealthCheckPayload,
    SglangHealthCheckPayload,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_disagg_payload_has_marker():
    assert SglangDisaggHealthCheckPayload().to_dict()[HEALTH_CHECK_KEY] is True


def test_decode_payload_has_no_marker():
    # Decode/agg handler doesn't read the marker; payload stays unmarked.
    assert HEALTH_CHECK_KEY not in SglangHealthCheckPayload().to_dict()


def test_disagg_env_override_preserves_marker(monkeypatch):
    """DYN_HEALTH_CHECK_PAYLOAD must not drop the canary marker."""
    monkeypatch.setenv(
        "DYN_HEALTH_CHECK_PAYLOAD",
        json.dumps(
            {
                "token_ids": [1],
                "sampling_options": {"temperature": 0.0},
                "stop_conditions": {"max_tokens": 1},
            }
        ),
    )
    assert SglangDisaggHealthCheckPayload().to_dict()[HEALTH_CHECK_KEY] is True
