# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local-launch addressing and marker derivation."""

import pytest

from tests.serve.common import (
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
