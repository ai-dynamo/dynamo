# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local-launch addressing and marker derivation."""

import pytest

from tests.serve.common import (
    _assert_topology_marker,
    _bind_payload_to_ports,
    marks_for_config,
    topology_dependent_reason,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payloads import ChatPayload, MetricsPayload

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
]

FRONTEND = 30001
SYSTEM = [30002, 30003]


def _chat(**kwargs):
    kwargs.setdefault("body", {})
    kwargs.setdefault("expected_response", [])
    kwargs.setdefault("expected_log", [])
    return ChatPayload(**kwargs)


def _config(payloads, name="cfg"):
    return EngineConfig(
        name=name,
        directory="/tmp",
        marks=[],
        request_payloads=payloads,
        model="m",
        script_name="agg.sh",
    )


# --- port binding ------------------------------------------------------------


def test_ordinary_payload_targets_the_frontend_port():
    bound = _bind_payload_to_ports(_chat(), frontend_port=FRONTEND, system_ports=SYSTEM)
    assert bound.port == FRONTEND
    assert bound.url() == f"http://localhost:{FRONTEND}/v1/chat/completions"


def test_binding_does_not_mutate_the_shared_instance():
    payload = _chat(port=DefaultPort.FRONTEND.value)
    _bind_payload_to_ports(payload, frontend_port=FRONTEND, system_ports=SYSTEM)
    assert payload.port == DefaultPort.FRONTEND.value


@pytest.mark.parametrize(
    "placeholder,expected_index",
    [(DefaultPort.SYSTEM1.value, 0), (DefaultPort.SYSTEM2.value, 1)],
)
def test_metrics_payload_maps_system_port_placeholders(placeholder, expected_index):
    payload = MetricsPayload(
        body={}, expected_response=[], expected_log=[], port=placeholder
    )
    bound = _bind_payload_to_ports(payload, frontend_port=FRONTEND, system_ports=SYSTEM)
    assert bound.port == SYSTEM[expected_index]


def test_metrics_payload_can_target_the_frontend():
    payload = MetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        port=DefaultPort.FRONTEND.value,
    )
    bound = _bind_payload_to_ports(payload, frontend_port=FRONTEND, system_ports=SYSTEM)
    assert bound.port == FRONTEND


def test_missing_system_port_is_a_configuration_error():
    payload = MetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        port=DefaultPort.SYSTEM2.value,
    )
    with pytest.raises(RuntimeError, match="SYSTEM_PORT2"):
        _bind_payload_to_ports(payload, frontend_port=FRONTEND, system_ports=[30002])


def test_extra_system_ports_are_mapped_and_literals_preserved():
    payload = _chat()
    payload.system_ports = [DefaultPort.SYSTEM1.value, DefaultPort.SYSTEM2.value, 9999]
    bound = _bind_payload_to_ports(payload, frontend_port=FRONTEND, system_ports=SYSTEM)
    assert bound.system_ports == [SYSTEM[0], SYSTEM[1], 9999]


# --- topology_dependent derivation -------------------------------------------


def test_response_only_config_is_deployment_agnostic():
    assert topology_dependent_reason(_config([_chat()])) is None


def test_log_assertions_make_a_config_topology_dependent():
    config = _config([_chat(expected_log=[r"KV hit rate"])])
    assert "logs" in topology_dependent_reason(config)


def test_worker_metrics_scrape_makes_a_config_topology_dependent():
    payload = MetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        port=DefaultPort.SYSTEM1.value,
    )
    assert "/metrics" in topology_dependent_reason(_config([payload]))


def test_frontend_metrics_scrape_stays_deployment_agnostic():
    """The frontend /metrics endpoint is reachable through the base URL."""
    payload = MetricsPayload(
        body={},
        expected_response=[],
        expected_log=[],
        port=DefaultPort.FRONTEND.value,
    )
    assert topology_dependent_reason(_config([payload])) is None


def test_system_port_addressing_makes_a_config_topology_dependent():
    payload = _chat()
    payload.system_ports = [DefaultPort.SYSTEM1.value]
    assert "system ports" in topology_dependent_reason(_config([payload]))


def test_any_coupled_payload_marks_the_whole_config():
    config = _config([_chat(), _chat(expected_log=["x"])])
    assert topology_dependent_reason(config) is not None


def _mark_names(marks):
    return {m.name for m in marks}


def test_marks_include_the_model_marker_and_nothing_else_when_agnostic():
    names = _mark_names(marks_for_config("cfg", _config([_chat()])))
    assert "model" in names
    assert "topology_dependent" not in names


def test_marks_add_topology_dependent_when_coupled():
    config = _config([_chat(expected_log=["x"])])
    assert "topology_dependent" in _mark_names(marks_for_config("cfg", config))


def test_marks_preserve_config_declared_marks():
    config = _config([_chat()])
    config.marks = [pytest.mark.gpu_1]
    assert "gpu_1" in _mark_names(marks_for_config("cfg", config))


def test_worker_health_gating_makes_a_config_topology_dependent():
    """Probing a worker's /health is not reachable behind a frontend URL."""
    config = _config([_chat()])
    config.health_check_workers = True
    assert "/health" in topology_dependent_reason(config)


class _DeferredLogPayload(ChatPayload):
    """Mirrors UuidPassthroughChatPayload: expected_log is filled per iteration."""

    def declares_log_assertions(self) -> bool:
        return True


def test_log_assertions_declared_after_collection_still_count():
    """expected_log is empty at collection but the payload says it will assert."""
    payload = _DeferredLogPayload(body={}, expected_response=[], expected_log=[])
    assert payload.expected_log == []
    assert "logs" in topology_dependent_reason(_config([payload]))


# --- the runtime guard -------------------------------------------------------


class _Node:
    def __init__(self, marked: bool) -> None:
        self._marked = marked

    def get_closest_marker(self, name):
        return object() if (self._marked and name == "topology_dependent") else None


class _Request:
    def __init__(self, marked: bool) -> None:
        self.node = _Node(marked)


def test_guard_passes_for_a_deployment_agnostic_config():
    _assert_topology_marker(_config([_chat()]), _Request(marked=False))


def test_guard_passes_when_a_coupled_config_is_marked():
    config = _config([_chat(expected_log=["x"])])
    _assert_topology_marker(config, _Request(marked=True))


def test_guard_names_the_reason_and_the_marker_when_unmarked():
    config = _config([_chat(expected_log=["x"])], name="agg_router")
    # pytest.fail raises an outcome, not a plain Exception.
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _assert_topology_marker(config, _Request(marked=False))

    message = str(excinfo.value)
    assert "agg_router" in message, "names the offending config"
    assert "asserts on server logs" in message, "names why it is coupled"
    assert "topology_dependent" in message, "names the marker to add"
