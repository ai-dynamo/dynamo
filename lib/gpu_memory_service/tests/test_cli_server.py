# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for production GMS process and listener topology."""

from __future__ import annotations

import asyncio
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
from gpu_memory_service.common.utils import (
    ENV_SERVER_DEVICE_UUID,
    ENV_SERVER_GPU_UUID_ISOLATION,
    get_socket_path_for_uuid,
)

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


@pytest.mark.parametrize("cuda_visibility", ["0,1", "all"])
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


def test_assigned_device_uuids_accepts_identity_ordinal_allocation(monkeypatch):
    available = ["GPU-aaaa", "GPU-bbbb", "GPU-cccc"]
    monkeypatch.setattr(server, "list_device_uuids", lambda: available)

    assert server._assigned_device_uuids(
        {"CUDA_VISIBLE_DEVICES": "0,1,2"}
    ) == available


def test_assigned_device_uuids_uses_dra_visible_nvml_allocation(monkeypatch):
    uuids = [f"GPU-{device}" for device in range(8)]
    monkeypatch.setattr(server, "list_device_uuids", lambda: uuids)

    assert server._assigned_device_uuids(
        {"NVIDIA_VISIBLE_DEVICES": "void"}
    ) == uuids


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
