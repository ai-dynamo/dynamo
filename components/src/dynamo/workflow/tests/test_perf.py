# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging

import pytest

from dynamo.workflow.perf import WorkflowPerfTracer

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def test_disabled_tracer_emits_nothing(caplog: pytest.LogCaptureFixture) -> None:
    tracer = WorkflowPerfTracer(enabled=False, sample_every=1)

    with caplog.at_level(logging.INFO):
        tracer.emit(logging.getLogger(__name__), "workflow.stage", "attempt")

    assert not caplog.records


def test_enabled_tracer_emits_joinable_json(caplog: pytest.LogCaptureFixture) -> None:
    tracer = WorkflowPerfTracer(enabled=True, sample_every=1)

    with caplog.at_level(logging.INFO):
        tracer.emit(
            logging.getLogger(__name__),
            "nixl.import",
            "lease-1",
            bytes=1024,
            wait_ms=2.5,
        )

    message = caplog.records[0].getMessage()
    payload = json.loads(message.removeprefix("workflow_perf "))
    assert payload == {
        "event": "nixl.import",
        "trace_id": "lease-1",
        "bytes": 1024,
        "wait_ms": 2.5,
    }


def test_sampling_is_stable() -> None:
    tracer = WorkflowPerfTracer(enabled=True, sample_every=7)

    first = [tracer.samples(f"request-{index}") for index in range(100)]
    second = [tracer.samples(f"request-{index}") for index in range(100)]

    assert first == second
    assert any(first)
    assert not all(first)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"DYN_WORKFLOW_PERF_TRACE": "sometimes"}, "must be one of"),
        (
            {
                "DYN_WORKFLOW_PERF_TRACE": "1",
                "DYN_WORKFLOW_PERF_SAMPLE_EVERY": "0",
            },
            "positive integer",
        ),
    ],
)
def test_environment_validation(environment: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WorkflowPerfTracer.from_environment(environment)
