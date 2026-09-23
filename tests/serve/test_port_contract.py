# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import pytest

from tests.serve.common import _cleanup_prepared_deployment, _prepare_deployment
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.port_utils import ServicePorts, reserved_ports

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

_E_PD_LAUNCHER = (
    Path(__file__).parents[2]
    / "examples/backends/vllm/launch/disagg_multimodal_e_pd.sh"
)

# Records one worker's ports, then stays alive: the launcher tears down its
# whole process group when a child exits, racing later workers out of the record.
_STUB_ENGINE = """#!/bin/bash
printf '%s\\t%s\\t%s\\n' \
    "${DYN_SYSTEM_PORT:-}" "${VLLM_NIXL_SIDE_CHANNEL_PORT:-}" "$*" >> "$PORT_RECORD"
sleep 2
"""


def _run_e_pd_launcher(
    tmp_path: Path, env: Dict[str, str]
) -> Tuple[subprocess.CompletedProcess, Dict[str, Dict[str, str]]]:
    """Launch the E+PD script against a stub engine and collect per-worker ports.

    Returns the finished process plus, keyed by role, the system port, the NIXL
    side-channel port, and the KV-event endpoint port each process received.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_text(_STUB_ENGINE)
    stub.chmod(0o755)
    record = tmp_path / "ports.tsv"

    result = subprocess.run(
        ["bash", str(_E_PD_LAUNCHER), "--single-gpu"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        # The launcher's cleanup trap signals its own process group. Without a
        # new session that group is pytest's.
        start_new_session=True,
        env={
            "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
            "PORT_RECORD": str(record),
            **env,
        },
    )

    workers: Dict[str, Dict[str, str]] = {}
    lines = record.read_text().splitlines() if record.exists() else []
    for line in lines:
        system_port, nixl_port, args = line.split("\t")
        if "--disaggregation-mode encode" in args:
            role = "encode"
        elif "--disaggregation-mode pd" in args:
            role = "pd"
        else:
            role = "frontend"
        endpoint = re.search(r"tcp://\*:(\d+)", args)
        workers[role] = {
            "system": system_port,
            "nixl": nixl_port,
            "kv": endpoint.group(1) if endpoint else "",
        }
    return result, workers


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


def test_e_pd_launcher_isolates_every_managed_worker_port(tmp_path: Path) -> None:
    """Hand the E+PD encode and PD workers their own injected ports."""
    with reserved_ports(6, DynamoPortRange.SERVE.value) as allocated:
        system_ports = allocated[0:2]
        kv_event_ports = allocated[2:4]
        nixl_ports = allocated[4:6]
        result, workers = _run_e_pd_launcher(
            tmp_path,
            {
                "DYN_MANAGED_PORTS": "1",
                # The harness aliases the unindexed name to worker 1's port.
                "DYN_SYSTEM_PORT": str(system_ports[0]),
                "DYN_SYSTEM_PORT1": str(system_ports[0]),
                "DYN_SYSTEM_PORT2": str(system_ports[1]),
                "DYN_VLLM_KV_EVENT_PORT1": str(kv_event_ports[0]),
                "DYN_VLLM_KV_EVENT_PORT2": str(kv_event_ports[1]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1": str(nixl_ports[0]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2": str(nixl_ports[1]),
            },
        )

    assert result.returncode == 0, result.stderr
    assert set(workers) == {"frontend", "encode", "pd"}, result.stdout
    assert workers["encode"] == {
        "system": str(system_ports[0]),
        "nixl": str(nixl_ports[0]),
        "kv": str(kv_event_ports[0]),
    }
    assert workers["pd"] == {
        "system": str(system_ports[1]),
        "nixl": str(nixl_ports[1]),
        "kv": str(kv_event_ports[1]),
    }
    # The frontend must not bind a worker's system port.
    assert workers["frontend"]["system"] not in {str(port) for port in system_ports}


def test_e_pd_launcher_fails_fast_on_missing_managed_port(tmp_path: Path) -> None:
    """Refuse to start E+PD workers on shared defaults when a port is absent."""
    with reserved_ports(5, DynamoPortRange.SERVE.value) as allocated:
        system_ports = allocated[0:2]
        kv_event_ports = allocated[2:4]
        result, workers = _run_e_pd_launcher(
            tmp_path,
            {
                "DYN_MANAGED_PORTS": "1",
                "DYN_SYSTEM_PORT": str(system_ports[0]),
                "DYN_SYSTEM_PORT1": str(system_ports[0]),
                "DYN_SYSTEM_PORT2": str(system_ports[1]),
                "DYN_VLLM_KV_EVENT_PORT1": str(kv_event_ports[0]),
                "DYN_VLLM_KV_EVENT_PORT2": str(kv_event_ports[1]),
                # Worker 2's NIXL side-channel port is deliberately missing.
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1": str(allocated[4]),
            },
        )

    assert result.returncode != 0
    assert "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2" in result.stderr
    assert "encode" not in workers
    assert "pd" not in workers


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
