# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for production GMS process and listener topology."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from types import SimpleNamespace

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.cli import args as cli_args
from gpu_memory_service.cli import runner, server
from gpu_memory_service.cli.args import parse_args
from gpu_memory_service.common import cuda_utils
from gpu_memory_service.common.snapshot_profile import SNAPSHOT_PROFILE_ENV
from gpu_memory_service.common.utils import (
    ENV_SERVER_DEVICE_UUID,
    ENV_SERVER_EXPECTED_GPU_UUIDS,
    ENV_SERVER_GPU_UUID_ISOLATION,
    get_socket_path_for_uuid,
)
from gpu_memory_service.server import allocations as server_allocations
from gpu_memory_service.server import rpc as server_rpc

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_child_command_launches_default_multi_tag_runner():
    assert server._child_command(3) == [
        sys.executable,
        "-m",
        "gpu_memory_service",
        "--device",
        "3",
    ]


def test_child_launch_isolates_uuid_and_remaps_device_without_mutating_parent():
    parent_env = {
        "CUDA_VISIBLE_DEVICES": "GPU-parent-0,GPU-parent-1",
        "UNCHANGED": "value",
    }
    original = parent_env.copy()

    command, child_env = server._child_launch(
        5,
        device_uuid="GPU-assigned",
        environ=parent_env,
    )

    assert command == [
        sys.executable,
        "-m",
        "gpu_memory_service",
        "--device",
        "0",
    ]
    assert child_env is not parent_env
    assert child_env["CUDA_VISIBLE_DEVICES"] == "GPU-assigned"
    assert child_env[ENV_SERVER_DEVICE_UUID] == "GPU-assigned"
    assert child_env["UNCHANGED"] == "value"
    assert parent_env == original


def test_child_launch_feature_off_preserves_prior_command_and_inheritance():
    command, child_env = server._child_launch(
        5,
        environ={"CUDA_VISIBLE_DEVICES": "GPU-parent"},
    )

    assert command[-1] == "5"
    assert child_env is None


def test_resolve_visible_device_uuids_prefers_full_uuid_and_preserves_order():
    available = ["GPU-aaaa-0000", "GPU-bbbb-0000", "GPU-cccc-0000"]

    assert server._resolve_visible_device_uuids(
        available,
        "GPU-cccc,GPU-aaaa",
    ) == ["GPU-cccc-0000", "GPU-aaaa-0000"]


@pytest.mark.parametrize(
    ("visibility", "match"),
    [
        ("0,1", "ambiguous without a UUID allocation"),
        ("GPU-missing", "does not uniquely identify"),
        ("GPU-aaaa,GPU-aaaa", "duplicate GPU UUID"),
        ("MIG-instance", "full GPUs are required"),
    ],
)
def test_resolve_visible_device_uuids_rejects_ambiguous_assignments(
    visibility,
    match,
):
    with pytest.raises(ValueError, match=match):
        server._resolve_visible_device_uuids(
            ["GPU-aaaa-0000", "GPU-bbbb-0000"],
            visibility,
        )


def test_assigned_device_uuids_resolves_cuda_ordinals_through_nvidia_uuids(
    monkeypatch,
):
    available = ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]
    monkeypatch.setattr(server, "list_device_uuids", lambda: available)

    assert server._assigned_device_uuids(
        {
            "NVIDIA_VISIBLE_DEVICES": "GPU-cccc,GPU-aaaa",
            "CUDA_VISIBLE_DEVICES": "1",
        }
    ) == ["GPU-aaaa"]


@pytest.mark.parametrize(
    "cuda_visibility",
    ["0,1", "all", "GPU-bbbb,GPU-aaaa"],
)
def test_assigned_device_uuids_preserves_nvidia_uuid_allocation_order(
    monkeypatch,
    cuda_visibility,
):
    available = ["GPU-aaaa", "GPU-bbbb"]
    monkeypatch.setattr(server, "list_device_uuids", lambda: available)

    assert server._assigned_device_uuids(
        {
            "NVIDIA_VISIBLE_DEVICES": "GPU-bbbb,GPU-aaaa",
            "CUDA_VISIBLE_DEVICES": cuda_visibility,
        }
    ) == ["GPU-bbbb", "GPU-aaaa"]


def test_assigned_device_uuids_accepts_exact_eight_device_dra_allocation(
    monkeypatch,
):
    available = [f"GPU-{device}" for device in range(8)]
    monkeypatch.setattr(server, "list_device_uuids", lambda: available)

    assert server._assigned_device_uuids(
        {
            "NVIDIA_VISIBLE_DEVICES": "all",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            ENV_SERVER_EXPECTED_GPU_UUIDS: ",".join(available),
        }
    ) == available


@pytest.mark.parametrize("nvidia_visibility", ["all", "void"])
def test_assigned_device_uuids_accepts_declared_all_eight_dra_allocation(
    monkeypatch,
    nvidia_visibility,
):
    available = [f"GPU-{device}" for device in range(8)]
    monkeypatch.setattr(server, "list_device_uuids", lambda: available)

    assert server._assigned_device_uuids(
        {
            "NVIDIA_VISIBLE_DEVICES": nvidia_visibility,
            ENV_SERVER_EXPECTED_GPU_UUIDS: ",".join(available),
        }
    ) == available


def test_assigned_device_uuids_restricts_host_wide_nvml_to_declared_dra_uuids(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-host-0", "GPU-host-1", "GPU-host-2", "GPU-host-3"],
    )

    assert server._assigned_device_uuids(
        {
            "NVIDIA_VISIBLE_DEVICES": "all",
            ENV_SERVER_EXPECTED_GPU_UUIDS: "GPU-host-3,GPU-host-1",
        }
    ) == ["GPU-host-3", "GPU-host-1"]


def test_assigned_device_uuids_uses_declared_dra_uuids_without_nvidia_visibility(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-host-0", "GPU-host-1", "GPU-host-2"],
    )

    assert server._assigned_device_uuids(
        {
            ENV_SERVER_EXPECTED_GPU_UUIDS: "GPU-host-2,GPU-host-0",
            "CUDA_VISIBLE_DEVICES": "1",
        }
    ) == ["GPU-host-0"]


def test_assigned_device_uuids_rejects_nvidia_outside_declared_dra_allocation(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-host-0", "GPU-host-1", "GPU-host-2"],
    )

    with pytest.raises(ValueError, match="contradicts the declared DRA allocation"):
        server._assigned_device_uuids(
            {
                "NVIDIA_VISIBLE_DEVICES": "GPU-host-2",
                ENV_SERVER_EXPECTED_GPU_UUIDS: "GPU-host-0,GPU-host-1",
            }
        )


@pytest.mark.parametrize("cuda_visibility", [None, "", "none", "void"])
def test_assigned_device_uuids_treats_nvidia_none_as_empty(
    monkeypatch,
    cuda_visibility,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-aaaa", "GPU-bbbb"],
    )
    environ = {"NVIDIA_VISIBLE_DEVICES": "none"}
    if cuda_visibility is not None:
        environ["CUDA_VISIBLE_DEVICES"] = cuda_visibility

    assert server._assigned_device_uuids(environ) == []


def test_assigned_device_uuids_rejects_cuda_uuid_when_nvidia_visibility_is_none(
    monkeypatch,
):
    monkeypatch.setattr(server, "list_device_uuids", lambda: ["GPU-aaaa"])

    with pytest.raises(ValueError, match="contradicts NVIDIA_VISIBLE_DEVICES=none"):
        server._assigned_device_uuids(
            {
                "NVIDIA_VISIBLE_DEVICES": "none",
                "CUDA_VISIBLE_DEVICES": "GPU-aaaa",
            }
        )


def test_assigned_device_uuids_rejects_contradictory_nvidia_and_cuda_uuids(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"],
    )

    with pytest.raises(ValueError, match="does not uniquely identify"):
        server._assigned_device_uuids(
            {
                "NVIDIA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb",
                "CUDA_VISIBLE_DEVICES": "GPU-cccc",
            }
        )


def test_assigned_device_uuids_restricts_host_wide_nvml_to_explicit_allocation(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-host-0", "GPU-host-1", "GPU-host-2", "GPU-host-3"],
    )

    assert server._assigned_device_uuids(
        {
            "NVIDIA_VISIBLE_DEVICES": "GPU-host-3,GPU-host-1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    ) == ["GPU-host-3"]


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"NVIDIA_VISIBLE_DEVICES": "all"},
        {"NVIDIA_VISIBLE_DEVICES": "void"},
        {
            "NVIDIA_VISIBLE_DEVICES": "all",
            "CUDA_VISIBLE_DEVICES": "0,1",
        },
        {"CUDA_VISIBLE_DEVICES": "0,1"},
    ],
)
def test_assigned_device_uuids_rejects_ambiguous_host_wide_visibility(
    monkeypatch,
    environ,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-aaaa", "GPU-bbbb"],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        server._assigned_device_uuids(environ)


def test_assigned_device_uuids_rejects_host_ordinals_without_cuda_mapping(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-aaaa", "GPU-bbbb"],
    )

    with pytest.raises(ValueError, match="NVIDIA_VISIBLE_DEVICES is ambiguous"):
        server._assigned_device_uuids({"NVIDIA_VISIBLE_DEVICES": "4,7"})


def test_assigned_device_uuids_rejects_nvidia_ordinals_with_cuda_uuid(
    monkeypatch,
):
    monkeypatch.setattr(
        server,
        "list_device_uuids",
        lambda: ["GPU-aaaa", "GPU-bbbb"],
    )

    with pytest.raises(ValueError, match="NVIDIA_VISIBLE_DEVICES is ambiguous"):
        server._assigned_device_uuids(
            {
                "NVIDIA_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "GPU-bbbb",
            }
        )


def test_server_main_feature_off_does_not_construct_child_environment(monkeypatch):
    popen_calls = []

    class Process:
        pid = 123

    monkeypatch.delenv(ENV_SERVER_GPU_UUID_ISOLATION, raising=False)
    monkeypatch.setattr(server, "list_devices", lambda: [3])
    monkeypatch.setattr(
        server.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)) or Process(),
    )
    monkeypatch.setattr(server, "_supervise", lambda _: 0)
    monkeypatch.setattr(server.signal, "signal", lambda *_: None)

    with pytest.raises(SystemExit) as exit_info:
        server.main()

    assert exit_info.value.code == 0
    assert popen_calls == [
        (([sys.executable, "-m", "gpu_memory_service", "--device", "3"],), {})
    ]


def test_server_main_feature_on_launches_one_isolated_child_per_uuid(
    monkeypatch,
    caplog,
):
    uuids = ["GPU-aaaa", "GPU-bbbb"]
    popen_calls = []
    parent_before = None

    class Process:
        def __init__(self, pid):
            self.pid = pid

    monkeypatch.setenv(ENV_SERVER_GPU_UUID_ISOLATION, "1")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", ",".join(uuids))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv(ENV_SERVER_EXPECTED_GPU_UUIDS, raising=False)
    parent_before = dict(os.environ)
    monkeypatch.setattr(server, "list_device_uuids", lambda: uuids)
    monkeypatch.setattr(
        server,
        "list_devices",
        lambda: pytest.fail("feature-on path used legacy list_devices"),
    )

    def popen(command, *, env):
        popen_calls.append((command, env))
        return Process(1000 + len(popen_calls))

    monkeypatch.setattr(server.subprocess, "Popen", popen)
    monkeypatch.setattr(server, "_supervise", lambda _: 0)
    monkeypatch.setattr(server.signal, "signal", lambda *_: None)

    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as exit_info:
        server.main()

    assert exit_info.value.code == 0
    assert [command for command, _ in popen_calls] == [
        [sys.executable, "-m", "gpu_memory_service", "--device", "0"],
        [sys.executable, "-m", "gpu_memory_service", "--device", "0"],
    ]
    child_envs = [env for _, env in popen_calls]
    assert child_envs[0] is not child_envs[1]
    for child_env, uuid in zip(child_envs, uuids):
        assert child_env is not os.environ
        assert child_env["CUDA_VISIBLE_DEVICES"] == uuid
        assert child_env[ENV_SERVER_DEVICE_UUID] == uuid
    assert dict(os.environ) == parent_before
    assert "physical_uuid=GPU-aaaa child_device=0 pid=1001" in caplog.text
    assert "physical_uuid=GPU-bbbb child_device=0 pid=1002" in caplog.text


@pytest.mark.asyncio
async def test_runner_passes_physical_uuid_to_profiled_feature_off_server(
    monkeypatch,
):
    servers = []

    class RPCServer:
        def __init__(self, *_args, **kwargs):
            servers.append(kwargs)

    monkeypatch.setenv(SNAPSHOT_PROFILE_ENV, "1")
    monkeypatch.delenv(ENV_SERVER_DEVICE_UUID, raising=False)
    monkeypatch.setattr(runner, "list_device_uuids", lambda: ["GPU-a", "GPU-b"])
    monkeypatch.setattr(runner, "GMSRPCServer", RPCServer)
    monkeypatch.setattr(runner, "run_servers", lambda _servers: asyncio.sleep(0))
    monkeypatch.setattr(
        cli_args,
        "get_socket_path",
        lambda device, tag: f"/sockets/{device}-{tag}.sock",
    )
    configs = parse_args(["--device", "1"])

    await runner.serve_configs(configs)

    assert len(servers) == 2
    assert all(server_config["device"] == 1 for server_config in servers)
    assert all(
        server_config["physical_uuid"] == "GPU-b" for server_config in servers
    )


@pytest.mark.asyncio
async def test_server_profile_records_physical_uuid_pid_and_child_ordinal(
    monkeypatch,
    tmp_path,
    caplog,
):
    class AsyncServer:
        async def serve_forever(self):
            return

    async def start_unix_server(*_args, **_kwargs):
        return AsyncServer()

    monkeypatch.setenv(SNAPSHOT_PROFILE_ENV, "1")
    monkeypatch.setenv(ENV_SERVER_DEVICE_UUID, "GPU-physical")
    monkeypatch.setattr(server_rpc.os, "getpid", lambda: 4321)
    monkeypatch.setattr(
        server_allocations,
        "cuda_ensure_initialized",
        lambda: None,
    )
    monkeypatch.setattr(
        server_allocations,
        "cumem_get_allocation_granularity",
        lambda _device: 4096,
    )
    monkeypatch.setattr(
        server_rpc.asyncio,
        "start_unix_server",
        start_unix_server,
    )

    with caplog.at_level(logging.INFO):
        rpc_server = server_rpc.GMSRPCServer(
            str(tmp_path / "gms.sock"),
            device=0,
            service="weights",
        )
        await rpc_server.serve()

    payloads = [
        json.loads(record.getMessage().removeprefix("GMS_SNAPSHOT_PROFILE "))
        for record in caplog.records
        if record.getMessage().startswith("GMS_SNAPSHOT_PROFILE ")
    ]
    attributed = [
        payload
        for payload in payloads
        if payload["phase"]
        in {"server_cu_init", "allocation_manager_ready", "socket_ready"}
    ]
    assert {payload["phase"] for payload in attributed} == {
        "server_cu_init",
        "allocation_manager_ready",
        "socket_ready",
    }
    assert all(payload["physical_uuid"] == "GPU-physical" for payload in attributed)
    assert all(payload["pid"] == 4321 for payload in attributed)
    assert all(payload["device"] == 0 for payload in attributed)


class _Process:
    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True


def test_supervisor_terminates_siblings_when_child_exits():
    processes = [_Process(exit_code=17), _Process()]

    assert server._supervise(processes) == 17
    assert processes[1].terminated

    # A clean exit (poll() returning 0) is an exit, not "still running".
    clean = [_Process(exit_code=0), _Process()]
    assert server._supervise(clean) == 0
    assert clean[1].terminated


def test_parse_args_defaults_to_one_config_per_production_tag(monkeypatch):
    # get_socket_path queries the GPU UUID through NVML; stub the hardware.
    monkeypatch.setattr(
        cli_args,
        "get_socket_path",
        lambda device, tag: f"/sockets/{device}-{tag}.sock",
    )

    configs = parse_args(["--device", "3"])

    assert [config.tag for config in configs] == ["weights", "kv_cache"]
    assert [config.socket_path for config in configs] == [
        "/sockets/3-weights.sock",
        "/sockets/3-kv_cache.sock",
    ]
    assert all(config.device == 3 for config in configs)

    (config,) = parse_args(["--device", "3", "--tag", "kv_cache"])
    assert config.tag == "kv_cache"


def test_parse_args_uses_physical_uuid_socket_for_isolated_device(
    monkeypatch,
):
    monkeypatch.setenv(ENV_SERVER_DEVICE_UUID, "GPU-assigned")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned")
    monkeypatch.setattr(
        cli_args,
        "get_socket_path_for_uuid",
        lambda uuid, tag: f"/sockets/{uuid}-{tag}.sock",
    )

    configs = parse_args(["--device", "0"])

    assert [config.socket_path for config in configs] == [
        "/sockets/GPU-assigned-weights.sock",
        "/sockets/GPU-assigned-kv_cache.sock",
    ]
    assert all(config.device == 0 for config in configs)


def test_physical_uuid_socket_path_uses_shared_socket_directory(monkeypatch):
    monkeypatch.setenv("GMS_SOCKET_DIR", "/gms-intrapod-control")

    assert get_socket_path_for_uuid("GPU-assigned", "weights") == (
        "/gms-intrapod-control/gms_GPU-assigned_weights.sock"
    )


def test_device_memory_info_prefers_physical_uuid(monkeypatch):
    calls = []
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: calls.append(("uuid", uuid)) or "handle",
        nvmlDeviceGetHandleByIndex=lambda device: calls.append(("index", device)),
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(free=17, total=23),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    assert cuda_utils.device_memory_info(
        0,
        device_uuid="GPU-assigned",
    ) == (17, 23)
    assert calls == ["init", ("uuid", "GPU-assigned"), "shutdown"]


def test_parse_args_single_tag_honors_explicit_socket_path():
    (config,) = parse_args(
        ["--device", "3", "--tag", "weights", "--socket-path", "/run/gms.sock"]
    )

    assert config.socket_path == "/run/gms.sock"


@pytest.mark.parametrize(
    "argv",
    [
        ["--device", "3", "--tag", "weight"],
        ["--device", "3", "--tag", "weights", "--tag", "bogus"],
    ],
)
def test_parse_args_rejects_unknown_tags(argv, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)

    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        # Default tags: one socket path cannot serve both listeners.
        ["--device", "3", "--socket-path", "/run/gms.sock"],
        # Explicit multiple tags with one socket path.
        [
            "--device",
            "3",
            "--tag",
            "weights",
            "--tag",
            "kv_cache",
            "--socket-path",
            "/run/gms.sock",
        ],
    ],
)
def test_parse_args_rejects_socket_path_for_multiple_tags(argv, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)

    assert "requires exactly one --tag" in capsys.readouterr().err


def test_parse_args_rejects_duplicate_tags(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--device", "3", "--tag", "weights", "--tag", "weights"])

    assert "must be unique" in capsys.readouterr().err


@pytest.mark.timeout(10)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("crash", "match"),
    [
        pytest.param(True, "listener failed", id="listener-crash"),
        pytest.param(False, "stopped unexpectedly", id="clean-stop"),
    ],
)
async def test_server_stop_cancels_other_listener(crash, match):
    both_started = asyncio.Event()
    started = 0
    sibling_cancelled = asyncio.Event()

    class Server:
        def __init__(self, stops: bool) -> None:
            self.stops = stops

        async def serve(self) -> None:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await both_started.wait()
            if self.stops:
                if crash:
                    raise RuntimeError("listener failed")
                return  # A clean return must still be fail-closed.
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

    with pytest.raises(RuntimeError, match=match):
        await runner.run_servers([Server(stops=True), Server(stops=False)])

    assert sibling_cancelled.is_set()
