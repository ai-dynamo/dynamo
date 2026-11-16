# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SGLangDisaggRouterMetricsPayload``."""

from unittest.mock import MagicMock

import pytest

from tests.utils import payloads as payloads_module
from tests.utils.payloads import SGLangDisaggRouterMetricsPayload

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

_REQUEST_COUNTER = "dynamo_component_requests_total"


def _request_sample(
    count: int,
    *,
    component: str = "prefill",
    endpoint: str = "generate",
    worker_id: str,
) -> str:
    return (
        f'{_REQUEST_COUNTER}{{dynamo_component="{component}",'
        f'dynamo_endpoint="{endpoint}",worker_id="{worker_id}"}} {count}\n'
    )


def _primary_metrics(*request_samples: str) -> str:
    """Return the common metrics required from the primary prefill worker."""
    return "".join(
        [
            *request_samples,
            "dynamo_component_uptime_seconds 10\n",
            "dynamo_component_total_blocks 0\n",
            "dynamo_component_gpu_cache_usage_percent 0.5\n",
            "dynamo_component_model_load_time_seconds 1\n",
            "dynamo_component_test_metric_one 1\n",
            "dynamo_component_test_metric_two 1\n",
        ]
    )


def _payload(system_ports: list[int]) -> SGLangDisaggRouterMetricsPayload:
    return SGLangDisaggRouterMetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        port=system_ports[0],
        system_ports=system_ports,
        min_num_requests=6,
        timeout=7,
    )


def _mock_secondary_metrics(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> tuple[MagicMock, MagicMock]:
    response = MagicMock()
    response.text = content
    mock_get = MagicMock(return_value=response)
    monkeypatch.setattr(payloads_module.requests, "get", mock_get)
    return mock_get, response


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_validate_aggregates_filtered_request_samples_across_prefill_workers(
    dynamo_dynamic_ports,
    monkeypatch,
):
    primary_port, secondary_port = dynamo_dynamic_ports.system_ports
    primary_content = _primary_metrics(
        _request_sample(1, worker_id="primary-a"),
        _request_sample(2, worker_id="primary-b"),
        _request_sample(100, endpoint="health", worker_id="primary-distractor"),
    )
    secondary_content = "".join(
        [
            _request_sample(3, worker_id="secondary"),
            _request_sample(100, component="decode", worker_id="secondary-distractor"),
        ]
    )
    mock_get, response = _mock_secondary_metrics(monkeypatch, secondary_content)

    _payload([primary_port, secondary_port]).validate(None, primary_content)

    mock_get.assert_called_once_with(
        f"http://localhost:{secondary_port}/metrics",
        timeout=7,
    )
    response.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
@pytest.mark.parametrize(
    "secondary_content",
    [
        "",
        _request_sample(3, component="decode", worker_id="secondary"),
    ],
    ids=["missing", "mismatched-labels"],
)
def test_validate_rejects_missing_or_mismatched_worker_metric(
    dynamo_dynamic_ports,
    monkeypatch,
    secondary_content,
):
    primary_port, secondary_port = dynamo_dynamic_ports.system_ports
    primary_content = _primary_metrics(
        _request_sample(3, worker_id="primary"),
    )
    mock_get, response = _mock_secondary_metrics(monkeypatch, secondary_content)

    with pytest.raises(AssertionError) as exc_info:
        _payload([primary_port, secondary_port]).validate(None, primary_content)

    assert f"Metric {_REQUEST_COUNTER}" in str(exc_info.value)
    assert "was not found" in str(exc_info.value)
    assert f"port {secondary_port}" in str(exc_info.value)
    mock_get.assert_called_once_with(
        f"http://localhost:{secondary_port}/metrics",
        timeout=7,
    )
    response.raise_for_status.assert_called_once_with()
