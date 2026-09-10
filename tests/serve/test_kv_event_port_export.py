# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-worker DYN_VLLM_KV_EVENT_PORT export in ``_prepare_deployment``.

ZMQ KV-event ports are host-wide. A launch script that binds one for every
worker needs one harness-allocated port per worker, or two deployments
scheduled concurrently on the same host bind the same port and one loses.
"""

from pathlib import Path
from typing import Any

import pytest

from tests.serve.common import _cleanup_prepared_deployment, _prepare_deployment
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.port_utils import ServicePorts, reserved_ports

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _FakeNode:
    def get_closest_marker(self, name: str) -> None:
        return None


class _FakeRequest:
    """Stands in for the pytest ``request`` fixture.

    ``_prepare_deployment`` only reaches into ``request.node.get_closest_marker``
    to read the KV-cache-budget markers; returning ``None`` selects the
    no-marker path.
    """

    node = _FakeNode()


def _config(directory: str) -> EngineConfig:
    return EngineConfig(
        name="kv-event-port-export",
        directory=directory,
        script_name="agg_multimodal_router.sh",
        model="test-model",
        marks=[],
        request_payloads=[],
    )


def _prepared_env(ports: ServicePorts, directory: str) -> tuple[dict, int]:
    prep = _prepare_deployment(
        _config(directory), _FakeRequest(), ports=ports, extra_env=None
    )
    try:
        return dict(prep.merged_env), len(prep.extra_allocated_ports)
    finally:
        _cleanup_prepared_deployment(prep)


@pytest.fixture(autouse=True)
def _no_xdist_stagger(monkeypatch: Any) -> None:
    # _prepare_deployment sleeps 15s per xdist worker index to spread vLLM
    # startup; this test never starts an engine, so skip the wait.
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)


def _service_ports(system: list, frontend: int, kv_event: int) -> ServicePorts:
    return ServicePorts(
        frontend_port=frontend,
        system_ports=system,
        kv_event_port=kv_event,
        nixl_side_channel_ports=[],
    )


def test_three_workers_each_get_a_distinct_kv_event_port(tmp_path: Path) -> None:
    with reserved_ports(5, DynamoPortRange.SERVE.value) as pool:
        frontend, kv_event, *system = pool
        env, extra_count = _prepared_env(
            _service_ports(system, frontend, kv_event), str(tmp_path)
        )

    exported = [env.get(f"DYN_VLLM_KV_EVENT_PORT{i}") for i in (1, 2, 3)]
    assert all(
        p is not None for p in exported
    ), f"every declared worker needs its own KV-event port, got {exported}"
    assert len(set(exported)) == 3, f"KV-event ports must not repeat: {exported}"

    # Worker 1 reuses the port the fixture already allocated; workers 2 and 3
    # are freshly allocated and must be handed to cleanup so they are released.
    assert exported[0] == str(kv_event)
    assert extra_count == 2

    system_ports = {env[f"DYN_SYSTEM_PORT{i}"] for i in (1, 2, 3)}
    assert system_ports.isdisjoint(exported)


def test_one_worker_still_gets_the_numbered_kv_event_port(tmp_path: Path) -> None:
    # The launch scripts read only DYN_VLLM_KV_EVENT_PORT{i}, so a one-worker
    # deployment without PORT1 silently falls back to the script's literal.
    with reserved_ports(3, DynamoPortRange.SERVE.value) as pool:
        frontend, kv_event, system = pool
        env, extra_count = _prepared_env(
            _service_ports([system], frontend, kv_event), str(tmp_path)
        )

    assert env["DYN_VLLM_KV_EVENT_PORT1"] == str(kv_event)
    assert extra_count == 0
