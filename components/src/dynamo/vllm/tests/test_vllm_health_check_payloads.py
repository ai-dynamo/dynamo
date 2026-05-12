# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vllm health-check payload shape tests.

Asserts the canary HEALTH_CHECK_KEY marker is layered onto each Vllm probe
payload via the to_dict() override and survives DYN_HEALTH_CHECK_PAYLOAD env
overrides. No vllm handler branches on the marker today; this is wire-format
parity with trtllm/sglang for any future marker-gated behavior.
"""

import json

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.vllm.health_check import (
    VllmHealthCheckPayload,
    VllmOmniHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

PAYLOAD_CLASSES = [
    VllmHealthCheckPayload,
    VllmPrefillHealthCheckPayload,
    VllmOmniHealthCheckPayload,
]


@pytest.mark.parametrize("cls", PAYLOAD_CLASSES, ids=lambda c: c.__name__)
def test_payload_has_marker(cls):
    assert cls().to_dict()[HEALTH_CHECK_KEY] is True


@pytest.mark.parametrize("cls", PAYLOAD_CLASSES, ids=lambda c: c.__name__)
def test_env_override_preserves_marker(monkeypatch, cls):
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
    assert cls().to_dict()[HEALTH_CHECK_KEY] is True
