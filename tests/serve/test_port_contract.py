# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

from tests.serve.common import _cleanup_prepared_deployment, _prepare_deployment
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.port_utils import ServicePorts, reserved_ports

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _RequestNode:
    def get_closest_marker(self, _name: str) -> None:
        return None


class _Request:
    node = _RequestNode()


def _config(directory: str) -> EngineConfig:
    return EngineConfig(
        name="port-contract",
        directory=directory,
        script_name="agg_multimodal_router.sh",
        model="test-model",
        marks=[],
        request_payloads=[],
    )


def _prepare(ports: ServicePorts, directory: str):
    """Build and clean up a deployment while returning its environment."""
    prepared = _prepare_deployment(
        _config(directory), _Request(), ports=ports, extra_env=None
    )
    try:
        return dict(prepared.merged_env)
    finally:
        _cleanup_prepared_deployment(prepared)


def test_prepared_environment_exports_complete_worker_port_vectors(
    tmp_path: Path,
) -> None:
    """Export one managed system, KV-event, and NIXL port per worker."""
    with reserved_ports(10, DynamoPortRange.SERVE.value) as allocated:
        frontend = allocated[0]
        system_ports = allocated[1:4]
        kv_event_ports = allocated[4:7]
        nixl_ports = allocated[7:10]
        ports = ServicePorts(
            frontend_port=frontend,
            system_ports=system_ports,
            kv_event_ports=kv_event_ports,
            nixl_side_channel_ports=nixl_ports,
        )
        env = _prepare(ports, str(tmp_path))

    assert env["DYN_MANAGED_PORTS"] == "1"
    assert env["DYN_SYSTEM_PORT"] == str(system_ports[0])
    assert "DYN_VLLM_KV_EVENT_PORT" not in env
    assert [env[f"DYN_SYSTEM_PORT{i}"] for i in range(1, 4)] == [
        str(port) for port in system_ports
    ]
    assert [env[f"DYN_VLLM_KV_EVENT_PORT{i}"] for i in range(1, 4)] == [
        str(port) for port in kv_event_ports
    ]
    assert [env[f"DYN_VLLM_NIXL_SIDE_CHANNEL_PORT{i}"] for i in range(1, 4)] == [
        str(port) for port in nixl_ports
    ]


def test_dyn_port_accepts_managed_ephemeral_system_port() -> None:
    """Allow zero for a managed system port so the runtime can bind ephemerally."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_MANAGED_PORTS": "1", "DYN_SYSTEM_PORT1": "0"},
        )

    assert result.returncode == 0
    assert result.stdout.strip() == "0"


def test_dyn_port_requires_indexed_values_in_managed_mode() -> None:
    """Reject a managed launch when a required indexed port is absent."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(2, DynamoPortRange.SERVE.value) as allocated:
        command = (
            f"source {launch_utils}; "
            "DYN_MANAGED_PORTS=1; "
            f"dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}; "
            f"dyn_port DYN_SYSTEM_PORT 2 {allocated[1]}"
        )
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": str(allocated[0])},
        )

    assert result.returncode != 0
    assert result.stdout.splitlines() == [str(allocated[0])]
    assert "DYN_SYSTEM_PORT2" in result.stderr


def test_dyn_port_keeps_standalone_fallback() -> None:
    """Use the supplied fallback when a standalone launch has no override."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={},
        )

    assert result.returncode == 0
    assert result.stdout.strip() == str(allocated[0])


def test_dyn_port_keeps_explicit_standalone_system_port() -> None:
    """Preserve an explicit system-port override for standalone launches."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": str(allocated[0])},
        )

    assert result.returncode == 0
    assert result.stdout.strip() == str(allocated[0])


def test_dyn_port_rejects_invalid_explicit_value() -> None:
    """Reject malformed standalone port overrides."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": "not-a-port"},
        )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_rejects_invalid_managed_value() -> None:
    """Reject malformed managed port overrides."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_MANAGED_PORTS": "1", "DYN_SYSTEM_PORT1": "not-a-port"},
        )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_rejects_out_of_range_value() -> None:
    """Reject numeric port overrides outside the runtime range."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 {allocated[0]}",
            ],
            capture_output=True,
            text=True,
            check=False,
            env={"DYN_SYSTEM_PORT1": "65536"},
        )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_rejects_out_of_range_fallback() -> None:
    """Reject a standalone fallback outside the runtime port range."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {launch_utils}; dyn_port DYN_SYSTEM_PORT 1 65536",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={},
    )

    assert result.returncode != 0
    assert "DYN_SYSTEM_PORT1" in result.stderr


def test_dyn_port_accepts_high_non_system_port() -> None:
    """Allow valid TCP ports above the system-status server's signed range."""
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"source {launch_utils}; dyn_port DYN_VLLM_KV_EVENT_PORT 1 20000",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"DYN_VLLM_KV_EVENT_PORT1": "40000"},
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "40000"


@pytest.fixture
def mm_router_preflight(tmp_path: Path) -> Path:
    """Run the real launcher's port checks, stopping before any services start."""
    source = (
        Path(__file__)
        .parents[2]
        .joinpath("examples/backends/sglang/launch/agg_multimodal_router.sh")
    )
    script = tmp_path / "examples/backends/sglang/launch/agg_multimodal_router.sh"
    script.parent.mkdir(parents=True)
    script.write_text(source.read_text())
    common = tmp_path / "examples/common"
    common.mkdir(parents=True)
    (common / "gpu_utils.sh").write_text("")
    # Reaching the banner proves validation finished. Exit before GPU helpers,
    # worker processes, readiness polling, or the process-group cleanup trap.
    (common / "launch_utils.sh").write_text("print_launch_banner() { exit 99; }\n")
    return script


@pytest.mark.sglang
@pytest.mark.multimodal
@pytest.mark.timeout(10)
@pytest.mark.parametrize("worker", [1, 2])
@pytest.mark.parametrize(
    "value",
    [None, "", "0", "00000", "-1", "abc", "1+2", "65536", "18446744073709551617"],
)
def test_mm_router_rejects_invalid_managed_nccl_port(
    mm_router_preflight: Path, worker: int, value: str | None
) -> None:
    """Reject every invalid spare port before even the first worker starts."""
    env = {
        "DYN_MANAGED_PORTS": "1",
        "DYN_SYSTEM_PORT3": "1",
        "DYN_SYSTEM_PORT4": "65535",
    }
    variable = f"DYN_SYSTEM_PORT{2 + worker}"
    if value is None:
        del env[variable]
    else:
        env[variable] = value
    result = subprocess.run(
        ["bash", str(mm_router_preflight)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert variable in result.stderr
    assert "expected" in result.stderr


@pytest.mark.sglang
@pytest.mark.multimodal
@pytest.mark.timeout(10)
@pytest.mark.parametrize("managed", [False, True])
def test_mm_router_accepts_nccl_port_boundaries_and_standalone(
    mm_router_preflight: Path, managed: bool
) -> None:
    """Accept TCP port boundaries in managed mode and no ports in standalone mode."""
    # These values are only validated; no process binds them.
    env = (
        {
            "DYN_MANAGED_PORTS": "1",
            "DYN_SYSTEM_PORT3": "00001",
            "DYN_SYSTEM_PORT4": "65535",
        }
        if managed
        else {}
    )
    result = subprocess.run(
        ["bash", str(mm_router_preflight)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 99, result.stderr
