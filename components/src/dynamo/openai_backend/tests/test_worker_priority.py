# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def forward_priority_hint():
    from dynamo.openai_backend.worker import _forward_priority_hint

    return _forward_priority_hint


def _launcher_args() -> SimpleNamespace:
    return SimpleNamespace(
        model="test-model",
        served_model_name=None,
        engine_host="127.0.0.1",
        engine_port=30000,
        api_prefix="/v1",
        health_path="/health",
    )


@pytest.mark.parametrize(
    ("multiplier", "expected"),
    [(1, 17), (-1, -17)],
)
def test_forwards_raw_text_worker_priority(
    forward_priority_hint, multiplier: int, expected: int
):
    request = {"nvext": {"agent_hints": {"priority": 17}}}

    forward_priority_hint(request, multiplier)

    assert request["priority"] == expected


def test_normalized_routing_priority_takes_precedence(forward_priority_hint):
    request = {
        "routing": {"priority": 23},
        "nvext": {"agent_hints": {"priority": 17}},
    }

    forward_priority_hint(request, 1)

    assert request["priority"] == 23


@pytest.mark.parametrize("payload", [{}, {"nvext": {}}, {"routing": {}}])
def test_missing_priority_defaults_to_zero(forward_priority_hint, payload: dict):
    forward_priority_hint(payload, 1)

    assert payload["priority"] == 0


def test_invalid_priority_defaults_to_zero(forward_priority_hint):
    request = {"nvext": {"agent_hints": {"priority": "17"}}}

    forward_priority_hint(request, 1)

    assert request["priority"] == 0


def test_disabled_priority_forwarding_leaves_request_unchanged(forward_priority_hint):
    request = {"nvext": {"agent_hints": {"priority": 17}}}

    forward_priority_hint(request, None)

    assert "priority" not in request


def test_sglang_launcher_preserves_priority_direction():
    from dynamo.openai_backend.sglang import _worker_command

    assert _worker_command(_launcher_args())[-2:] == [
        "--priority-multiplier",
        "1",
    ]


def test_vllm_launcher_reverses_priority_direction():
    from dynamo.openai_backend.vllm import _worker_command

    assert _worker_command(_launcher_args())[-2:] == [
        "--priority-multiplier",
        "-1",
    ]
