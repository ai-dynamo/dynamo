# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from dynamo_decision_perf import serving
from dynamo_decision_perf.workloads import MODEL_REVISION

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def fake_runtime(monkeypatch):
    events = []

    class Process:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.command = kwargs.get("command", [])
            self.port = 23456 if kwargs.get("kind") == "etcd" else 23457
            self.proc = SimpleNamespace(pid=12345)
            self.terminate_all_matching_process_names = False

        def __enter__(self):
            events.append(("start", self))
            return self

        def __exit__(self, *args):
            events.append(("stop", self))

    @contextmanager
    def ports(count, *args):
        yield list(range(22000, 22000 + count))

    monkeypatch.setattr(
        serving,
        "_dependencies",
        lambda: SimpleNamespace(
            ManagedProcess=Process,
            EtcdServer=lambda *a, **kw: Process(kind="etcd", **kw),
            NatsServer=lambda *a, **kw: Process(kind="nats", **kw),
            reserved_ports=ports,
            check_http_ok=lambda _: True,
            check_health_ready=lambda _: True,
            model_registered=lambda *a, **kw: True,
        ),
    )
    return events


def config(tmp_path, **kwargs):
    model_path = tmp_path / MODEL_REVISION
    model_path.mkdir(exist_ok=True)
    return serving.ServingConfig(
        model_path=model_path,
        model="Qwen/Qwen3.8-27B",
        log_dir=tmp_path / "logs",
        **kwargs,
    )


def test_mocker_owns_only_its_processes_and_cleans_on_failure(tmp_path, fake_runtime):
    with (
        pytest.raises(RuntimeError, match="caller"),
        serving.serve(config(tmp_path, workers=2)) as handle,
    ):
        assert handle.worker_count == 2
        assert len(handle.monitor_pids) == 3
        assert handle.base_url.startswith("http://127.0.0.1:")
        raise RuntimeError("caller failure")
    starts = [process for action, process in fake_runtime if action == "start"]
    stops = [process for action, process in fake_runtime if action == "stop"]
    assert stops == starts[::-1]
    assert len(starts) == 5
    assert all(not p.terminate_all_matching_process_names for p in starts)
    workers = [p for p in starts if "dynamo.mocker" in p.command]
    assert len(workers) == 2
    assert all(p.env["CUDA_VISIBLE_DEVICES"] == "" for p in workers)
    assert all(p.env["HF_HUB_OFFLINE"] == "1" for p in workers)


def test_owned_worker_system_api_is_always_loopback(
    tmp_path, fake_runtime, monkeypatch
):
    monkeypatch.setenv("DYN_SYSTEM_HOST", "0.0.0.0")
    with serving.serve(config(tmp_path)):
        pass
    workers = [
        p
        for action, p in fake_runtime
        if action == "start" and "dynamo.mocker" in p.command
    ]
    assert all(p.env["DYN_SYSTEM_HOST"] == "127.0.0.1" for p in workers)


@pytest.mark.parametrize("failure", [SystemExit, KeyboardInterrupt])
def test_interrupt_during_startup_cleans_current_owned_process(
    tmp_path, fake_runtime, monkeypatch, failure
):
    dependencies = serving._dependencies()
    process_type = dependencies.ManagedProcess
    original_enter = process_type.__enter__

    def interrupted_enter(process):
        original_enter(process)
        if "dynamo.mocker" in process.command:
            raise failure()
        return process

    monkeypatch.setattr(process_type, "__enter__", interrupted_enter)
    monkeypatch.setattr(serving, "_dependencies", lambda: dependencies)
    with pytest.raises(failure), serving.serve(config(tmp_path)):
        pass
    starts = [process for action, process in fake_runtime if action == "start"]
    stops = [process for action, process in fake_runtime if action == "stop"]
    assert stops == starts[::-1]


@pytest.mark.parametrize("approval", [None, 0, -1, 4.1, float("nan"), float("inf")])
def test_gpu_requires_finite_explicit_budget_before_startup(
    tmp_path, fake_runtime, approval
):
    with (
        pytest.raises(ValueError, match="GPU"),
        serving.serve(config(tmp_path, mode="sglang", approve_gpu_hours=approval)),
    ):
        pass
    assert fake_runtime == []


def test_gpu_commands_preserve_serial_profile(tmp_path, fake_runtime, monkeypatch):
    monkeypatch.setattr(serving, "version", lambda name: "0.5.19")

    @contextmanager
    def no_alarm(hours):
        assert hours == 0.25
        yield

    monkeypatch.setattr(serving, "_gpu_deadline", no_alarm)
    with serving.serve(config(tmp_path, mode="native", approve_gpu_hours=0.25)):
        pass
    processes = [p for action, p in fake_runtime if action == "start"]
    assert len(processes) == 1
    command = processes[0].command
    for flag, expected in (
        ("--tp", "1"),
        ("--pp-size", "1"),
        ("--max-running-requests", "1"),
        ("--context-length", "2048"),
        ("--max-total-tokens", "2048"),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--disable-overlap-schedule" in command
    assert "--disable-piecewise-cuda-graph" in command
    assert "--disable-cuda-graph" in command
    assert command[command.index("--mem-fraction-static") + 1] == "0.75"


def test_gpu_rejects_other_version_and_multiple_devices(
    tmp_path, fake_runtime, monkeypatch
):
    monkeypatch.setattr(serving, "version", lambda name: "0.5.21")
    with (
        pytest.raises(ValueError, match="0.5.19"),
        serving.serve(config(tmp_path, mode="sglang", approve_gpu_hours=1)),
    ):
        pass
    with (
        pytest.raises(ValueError, match="one GPU"),
        serving.serve(config(tmp_path, mode="sglang", approve_gpu_hours=1, gpu="0,1")),
    ):
        pass
    assert fake_runtime == []


def test_no_output_overwrite_or_unpinned_model(tmp_path, fake_runtime):
    cfg = config(tmp_path)
    cfg.log_dir.mkdir()
    with pytest.raises(FileExistsError), serving.serve(cfg):
        pass
    with (
        pytest.raises(ValueError, match="revision"),
        serving.serve(
            serving.ServingConfig(
                model_path=tmp_path, model="model", log_dir=tmp_path / "fresh"
            )
        ),
    ):
        pass
    assert fake_runtime == []


def test_gpu_deadline_restores_alarm_without_sleeping(monkeypatch):
    timers, handlers = [], []
    monkeypatch.setattr(serving.signal, "getitimer", lambda kind: (0, 0))
    monkeypatch.setattr(serving.signal, "getsignal", lambda kind: "old")
    monkeypatch.setattr(
        serving.signal, "signal", lambda kind, handler: handlers.append(handler)
    )
    monkeypatch.setattr(
        serving.signal, "setitimer", lambda kind, seconds: timers.append(seconds)
    )
    with pytest.raises(TimeoutError, match="deadline"), serving._gpu_deadline(0.01):
        handlers[0](0, None)
    assert timers == [36, 0]
    assert handlers[-1] == "old"


def test_gpu_does_not_replace_existing_alarm(monkeypatch):
    monkeypatch.setattr(serving.signal, "getitimer", lambda kind: (2, 0))
    with pytest.raises(ValueError, match="existing"), serving._gpu_deadline(1):
        pass


def test_sigterm_unwinds_owned_context_and_restores_handler(monkeypatch):
    handlers = []
    monkeypatch.setattr(serving.signal, "getsignal", lambda kind: "old")
    monkeypatch.setattr(
        serving.signal, "signal", lambda kind, handler: handlers.append(handler)
    )
    with pytest.raises(SystemExit), serving._termination_signal():
        handlers[0](15, None)
    assert handlers[-1] == "old"


@pytest.mark.parametrize(
    "kwargs", [{"mode": "unknown"}, {"workers": 0}, {"startup_timeout": 0}]
)
def test_invalid_config_has_no_side_effects(tmp_path, fake_runtime, kwargs):
    with pytest.raises(ValueError), serving.serve(config(tmp_path, **kwargs)):
        pass
    assert fake_runtime == []
