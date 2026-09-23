# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Sequence, Tuple

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

# Records one worker's ports, then stays alive until every role has recorded:
# the launcher tears down its whole process group when a child exits, racing
# later workers out of the record. The bound only caps a launch that never
# starts all three; the launcher's own cleanup trap reaps this stub first.
_STUB_ROLES = 3
_STUB_ENGINE = f"""#!/bin/bash
# Only a module launch is a worker; the launcher also runs the interpreter for
# its own checks, and those have to reach the real one.
case " $* " in
    *" -m "*) ;;
    *) exec python3 "$@" ;;
esac
printf '%s\\t%s\\t%s\\n' \
    "${{DYN_SYSTEM_PORT:-}}" "${{VLLM_NIXL_SIDE_CHANNEL_PORT:-}}" "$*" >> "$PORT_RECORD"
for _ in $(seq 200); do
    [ "$(wc -l < "$PORT_RECORD")" -ge {_STUB_ROLES} ] && exit 0
    sleep 0.05
done
"""


def _run_e_pd_launcher(
    tmp_path: Path, env: Dict[str, str], extra_args: Sequence[str] = ()
) -> Tuple[subprocess.CompletedProcess, Dict[str, Dict[str, str]]]:
    """Launch the E+PD script against a stub engine and collect per-worker ports.

    Returns the finished process plus, keyed by role, the system port, the NIXL
    side-channel port, the KV-event endpoint port, and the raw command line each
    process received.
    """
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_text(_STUB_ENGINE)
    stub.chmod(0o755)
    record = tmp_path / "ports.tsv"

    result = subprocess.run(
        ["bash", str(_E_PD_LAUNCHER), "--single-gpu", *extra_args],
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
            "args": args,
        }
    return result, workers


def _ports(worker: Dict[str, str]) -> Dict[str, str]:
    """Drop the raw command line so a comparison stays exhaustive over ports."""
    return {key: worker[key] for key in ("system", "nixl", "kv")}


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


# Backstop above _run_e_pd_launcher's own 120s subprocess bound, so that bound
# reports the hang first and this only stops pytest waiting forever.
@pytest.mark.timeout(180)
def test_e_pd_launcher_isolates_every_managed_worker_port(tmp_path: Path) -> None:
    """Give each worker the vector index for its role, and the frontend none."""
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
    assert _ports(workers["encode"]) == {
        "system": str(system_ports[0]),
        "nixl": str(nixl_ports[0]),
        "kv": str(kv_event_ports[0]),
    }
    assert _ports(workers["pd"]) == {
        "system": str(system_ports[1]),
        "nixl": str(nixl_ports[1]),
        "kv": str(kv_event_ports[1]),
    }
    assert workers["frontend"]["system"] not in {str(port) for port in system_ports}


@pytest.mark.timeout(180)
def test_e_pd_launcher_fails_fast_on_missing_managed_port(tmp_path: Path) -> None:
    """Stop the launch before any worker starts, rather than share a default."""
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
                # PORT2 omitted on purpose.
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1": str(allocated[4]),
            },
        )

    assert result.returncode != 0
    assert "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2" in result.stderr
    assert "encode" not in workers
    assert "pd" not in workers


@pytest.mark.timeout(180)
def test_e_pd_launcher_refuses_managed_kv_events_override(tmp_path: Path) -> None:
    """Stop a passthrough config whose endpoint is not the reserved KV port."""
    with reserved_ports(7, DynamoPortRange.SERVE.value) as allocated:
        result, workers = _run_e_pd_launcher(
            tmp_path,
            {
                "DYN_MANAGED_PORTS": "1",
                "DYN_SYSTEM_PORT": str(allocated[0]),
                "DYN_SYSTEM_PORT1": str(allocated[0]),
                "DYN_SYSTEM_PORT2": str(allocated[1]),
                "DYN_VLLM_KV_EVENT_PORT1": str(allocated[2]),
                "DYN_VLLM_KV_EVENT_PORT2": str(allocated[3]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1": str(allocated[4]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2": str(allocated[5]),
            },
            extra_args=[
                "--kv-events-config",
                f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{allocated[6]}"}}',
            ],
        )

    assert result.returncode != 0
    assert "DYN_VLLM_KV_EVENT_PORT2" in result.stderr
    assert "pd" not in workers


# Two ways the reserved endpoint can appear ahead of the one that takes effect:
# nested in another object, and as an earlier duplicate key, which json.loads
# discards in favour of the last. Nothing here has to be a config vLLM accepts —
# the launcher decides before vLLM runs.
@pytest.mark.parametrize("decoy", ["nested", "duplicate"])
@pytest.mark.timeout(180)
def test_e_pd_launcher_refuses_managed_kv_events_endpoint_decoy(
    tmp_path: Path, decoy: str
) -> None:
    """Read the endpoint that takes effect, not the first one in the text."""
    with reserved_ports(7, DynamoPortRange.SERVE.value) as allocated:
        reserved = f'"endpoint":"tcp://*:{allocated[3]}"'
        first = f'"nested":{{{reserved}}}' if decoy == "nested" else reserved
        config = (
            '{"publisher":"zmq",' f"{first}," f'"endpoint":"tcp://*:{allocated[6]}"}}'
        )
        result, workers = _run_e_pd_launcher(
            tmp_path,
            {
                "DYN_MANAGED_PORTS": "1",
                "DYN_SYSTEM_PORT": str(allocated[0]),
                "DYN_SYSTEM_PORT1": str(allocated[0]),
                "DYN_SYSTEM_PORT2": str(allocated[1]),
                "DYN_VLLM_KV_EVENT_PORT1": str(allocated[2]),
                "DYN_VLLM_KV_EVENT_PORT2": str(allocated[3]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1": str(allocated[4]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2": str(allocated[5]),
            },
            extra_args=["--kv-events-config", config],
        )

    assert result.returncode != 0
    assert str(allocated[6]) in result.stderr
    assert "pd" not in workers


# Escaping a solidus is legal JSON and decodes to the same endpoint, so the
# launcher has to decode rather than match text. Both flag spellings argparse
# accepts are covered too, since the launcher matches the flag itself.
@pytest.mark.parametrize("escape_slashes", [False, True])
@pytest.mark.parametrize("separate_value", [True, False])
@pytest.mark.timeout(180)
def test_e_pd_launcher_keeps_managed_kv_events_on_the_reserved_port(
    tmp_path: Path, separate_value: bool, escape_slashes: bool
) -> None:
    """Keep a passthrough config that leaves the endpoint on the reserved port."""
    with reserved_ports(6, DynamoPortRange.SERVE.value) as allocated:
        scheme = "tcp:\\/\\/" if escape_slashes else "tcp://"
        config = (
            '{"publisher":"zmq","topic":"kv-events",'
            f'"endpoint":"{scheme}*:{allocated[3]}",'
            '"enable_kv_cache_events":true}'
        )
        result, workers = _run_e_pd_launcher(
            tmp_path,
            {
                "DYN_MANAGED_PORTS": "1",
                "DYN_SYSTEM_PORT": str(allocated[0]),
                "DYN_SYSTEM_PORT1": str(allocated[0]),
                "DYN_SYSTEM_PORT2": str(allocated[1]),
                "DYN_VLLM_KV_EVENT_PORT1": str(allocated[2]),
                "DYN_VLLM_KV_EVENT_PORT2": str(allocated[3]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1": str(allocated[4]),
                "DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2": str(allocated[5]),
            },
            extra_args=(
                ["--kv-events-config", config]
                if separate_value
                else [f"--kv-events-config={config}"]
            ),
        )

    assert result.returncode == 0, result.stderr
    # Read from the raw command line rather than the parsed "kv" field, which
    # only recognises the unescaped spelling.
    assert f"*:{allocated[3]}" in workers["pd"]["args"]
    # The generated config sets no enable_kv_cache_events, and vLLM would keep
    # only one of the two, so reading the flag back means the caller's copy is
    # the one that survived.
    assert "enable_kv_cache_events" in workers["pd"]["args"]
    assert workers["pd"]["args"].count("--kv-events-config") == 1


@pytest.mark.timeout(180)
def test_e_pd_launcher_keeps_standalone_kv_events_passthrough(tmp_path: Path) -> None:
    """Drop the generated config standalone, where the caller's has always won."""
    # _run_e_pd_launcher reports the first endpoint in the argument list. The
    # allocator's range is disjoint from the launcher's standalone KV-event
    # default, so reading this port back can only mean the generated option,
    # which would precede it, was dropped.
    with reserved_ports(1, DynamoPortRange.SERVE.value) as allocated:
        result, workers = _run_e_pd_launcher(
            tmp_path,
            {},
            extra_args=[
                "--kv-events-config",
                f'{{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:{allocated[0]}"}}',
            ],
        )

    assert result.returncode == 0, result.stderr
    assert workers["pd"]["kv"] == str(allocated[0])


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
