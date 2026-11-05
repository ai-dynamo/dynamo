# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the user-facing backpressure example through its Python utilities."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

from tests.utils.constants import ROUTER_MODEL_NAME
from tests.utils.output_paths import resolve_test_output_path
from tests.utils.port_utils import ServicePorts

# Each case uses reserved ports and its own discovery directory and processes.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.model(ROUTER_MODEL_NAME),
    pytest.mark.timeout(120),  # 3x the observed maximum of about 40s per case.
]

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/router/admission-control/backpressure/run.py"
)


@pytest.fixture(scope="module")
def example() -> ModuleType:
    # The example directory contains a hyphen, so load the actual file by path.
    spec = importlib.util.spec_from_file_location(
        "router_backpressure_example", EXAMPLE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backpressure example: {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.usefixtures("predownload_tokenizers")
@pytest.mark.parametrize("num_system_ports", [3], indirect=True)
@pytest.mark.parametrize("backpressure", [False, True], ids=["off", "on"])
def test_backpressure(
    request: pytest.FixtureRequest,
    example: ModuleType,
    backpressure: bool,
    dynamo_dynamic_ports: ServicePorts,
) -> None:
    """The same deployment and client used by run.py demonstrate backpressure."""
    output = Path(resolve_test_output_path(request.node.name))
    output.mkdir(parents=True, exist_ok=True)
    run_output = Path(tempfile.mkdtemp(prefix="example-", dir=output))
    result = asyncio.run(
        example.run_demo(
            backpressure=backpressure,
            output_dir=run_output,
            model=ROUTER_MODEL_NAME,
            ports=(
                dynamo_dynamic_ports.frontend_port,
                *dynamo_dynamic_ports.system_ports,
            ),
        )
    )

    healthy, removed, slow = result.windows
    reduced_capacity = (removed, slow)
    assert all(
        abs(window.sent_rps / example.REQUEST_RATE - 1) < 0.1
        for window in result.windows
    ), result.windows
    healthy_mean_rps = sum(healthy.arrival_rps) / 3
    assert healthy_mean_rps > 0, healthy
    assert all(
        abs(rate / healthy_mean_rps - 1) < 0.1 for rate in healthy.arrival_rps
    ), healthy
    assert removed.arrival_rps[2] == 0, removed
    assert all(window.completed_rps > 0 for window in result.windows), result.windows
    failures = [
        outcome
        for outcome in result.outcomes
        if outcome.status != 200 and not result.is_shutdown_failure(outcome)
    ]
    assert not failures, failures[:5]
    assert not [
        pid for pid in result.process_pids if Path("/proc", str(pid)).exists()
    ], result.process_pids

    if backpressure:
        assert all(
            abs(window.arrival_rps[index] / healthy.arrival_rps[index] - 1) <= 0.25
            for window in reduced_capacity
            for index in (0, 1)
        ), result.windows
        assert 0.35 <= slow.arrival_rps[2] / healthy.arrival_rps[2] <= 0.65, (
            healthy,
            slow,
        )
        assert all(window.queue_mean > 1 for window in result.windows), result.windows
        assert all(window.queue_delta > 0 for window in result.windows), result.windows
        assert all(
            sum(window.arrival_rps) < 0.95 * window.sent_rps
            for window in result.windows
        ), result.windows
        # Fraction 1.0 admits one prefill beyond the worker's token budget.
        # Allow two more requests for first-token/response accounting overlap.
        inflight_limit = example.MAX_BATCHED_TOKENS // example.INPUT_TOKENS + 1 + 2
        assert all(
            window.inflight_max <= inflight_limit for window in result.windows
        ), result.windows
    else:
        assert all(
            abs(rate / (healthy.sent_rps / 3) - 1) < 0.1 for rate in healthy.arrival_rps
        ), healthy
        assert all(
            abs(sum(window.arrival_rps) / window.sent_rps - 1) < 0.1
            for window in result.windows
        ), result.windows
        assert all(
            window.inflight_delta > 0 for window in result.windows
        ), result.windows
        assert all(window.queue_mean < 1 for window in result.windows), result.windows
