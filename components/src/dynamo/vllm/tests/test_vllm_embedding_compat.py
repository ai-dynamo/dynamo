# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned engine launch contracts and ownership of embedding listeners."""

import importlib
import json
import os
import socket
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm.config")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_worker_import_does_not_require_optional_embedding_api():
    module = importlib.import_module("dynamo.vllm.main")
    assert callable(module.main)


@pytest.fixture
def pool_contract(monkeypatch, tmp_path):
    processes = importlib.import_module("dynamo.vllm.embedding_worker_processes")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(_api_process_count=1, _api_process_rank=0)
    )
    addresses = SimpleNamespace(inputs=["input0", "input1"], outputs=["out0", "out1"])
    listeners = SimpleNamespace(
        addresses=addresses,
        inputs=[SimpleNamespace(socket=Mock()) for _ in range(2)],
        outputs=[SimpleNamespace(socket=Mock()) for _ in range(2)],
        cleanup=Mock(),
    )
    for idx in range(2):
        listeners.inputs[idx].socket.fileno.return_value = 10 + idx
        listeners.outputs[idx].socket.fileno.return_value = 20 + idx
    manager = Mock()
    launch = SimpleNamespace(
        engine_manager=manager, coordinator=None, addresses=addresses, tensor_queue=None
    )
    rpc_directory = Mock()
    rpc_directory.name = str(tmp_path)
    child = Mock(pid=123)
    child.poll.return_value = 0
    spawn = Mock(return_value=child)
    attach = Mock(return_value=(Mock(), config))
    monkeypatch.delenv(processes._ROLE_ENV, raising=False)
    monkeypatch.setattr(processes, "_short_rpc_directory", lambda: rpc_directory)
    monkeypatch.setattr(processes.Executor, "get_class", lambda _config: object)
    monkeypatch.setattr(processes.subprocess, "Popen", spawn)
    monkeypatch.setattr(processes, "_attach_client", attach)
    monkeypatch.setattr(processes.EmbeddingWorkerProcessGroup, "start_monitor", Mock())
    return SimpleNamespace(
        processes=processes,
        config=config,
        addresses=addresses,
        listeners=listeners,
        manager=manager,
        launch=launch,
        rpc_directory=rpc_directory,
        spawn=spawn,
        attach=attach,
    )


def _run_pool(contract):
    return contract.processes.create_shared_embedding_engine_client(
        vllm_config=contract.config,
        process_count=2,
        usage_context=None,
        stat_loggers=[],
        enable_log_requests=False,
        disable_log_stats=False,
    )


@pytest.mark.parametrize("style", ["legacy_tuple", "structured", "listeners"])
def test_pool_uses_matching_launch_contract_and_shares_one_engine(
    monkeypatch, pool_contract, style
):
    c = pool_contract
    calls = []

    # Exact signatures deliberately reject the stale extra argument. The old
    # test's *args mock hid a production incompatibility.
    @contextmanager
    def legacy_launch(config, executor, log_stats, addresses, process_count):
        calls.append((config, executor, log_stats, addresses, process_count))
        yield c.manager, None, c.addresses, None

    @contextmanager
    def structured_launch(config, executor, log_stats, addresses):
        calls.append((config, executor, log_stats, addresses))
        yield c.launch

    api = SimpleNamespace(get_engine_zmq_addresses=Mock(return_value=c.addresses))
    if style == "legacy_tuple":
        api.launch_core_engines = legacy_launch
    else:
        api.CoreEngineLaunch = SimpleNamespace
        api.launch_core_engines = structured_launch
    if style == "listeners":
        del api.get_engine_zmq_addresses
        api.bind_engine_zmq_listeners = Mock(return_value=c.listeners)
    monkeypatch.setattr(c.processes, "engine_utils", api, raising=False)

    client, _, group = _run_pool(c)
    assert client is c.attach.return_value[0]
    assert len(calls) == 1
    assert calls[0][3] is c.addresses
    assert c.spawn.call_count == 1
    assert group.engine_manager is c.manager
    if style == "listeners":
        api.bind_engine_zmq_listeners.assert_called_once_with(c.config, 2)
        assert c.spawn.call_args.kwargs["pass_fds"] == (11, 21)
        assert json.loads(
            c.spawn.call_args.kwargs["env"][c.processes._LISTENER_FDS_ENV]
        ) == [11, 21]
        assert (
            c.attach.call_args.kwargs["input_listener"] is c.listeners.inputs[0].socket
        )
        assert (
            c.attach.call_args.kwargs["output_listener"]
            is c.listeners.outputs[0].socket
        )
    else:
        assert "pass_fds" not in c.spawn.call_args.kwargs
    group.cleanup()
    group.cleanup()
    c.manager.shutdown.assert_called_once()
    c.rpc_directory.cleanup.assert_called_once()
    if style == "listeners":
        c.listeners.cleanup.assert_called_once()


def test_child_adopts_only_inherited_sockets_and_closes_on_attach_failure(
    monkeypatch,
):
    processes = importlib.import_module("dynamo.vllm.embedding_worker_processes")
    monkeypatch.setenv(processes._ROLE_ENV, processes._CHILD_ROLE)
    monkeypatch.setenv(processes._INDEX_ENV, "1")
    monkeypatch.setenv(
        processes._ENGINE_ADDRESSES_ENV,
        json.dumps(
            {"process_count": 2, "inputs": ["in0", "in1"], "outputs": ["out0", "out1"]}
        ),
    )
    with (
        socket.socket() as input_listener,
        socket.socket() as output_listener,
    ):
        fds = (os.dup(input_listener.fileno()), os.dup(output_listener.fileno()))
        monkeypatch.setenv(processes._LISTENER_FDS_ENV, json.dumps(fds))

        def failed_attach(*args, **kwargs):
            assert kwargs["input_listener"].fileno() == fds[0]
            assert kwargs["output_listener"].fileno() == fds[1]
            raise RuntimeError("attachment failed")

        monkeypatch.setattr(processes, "_attach_client", failed_attach)
        with pytest.raises(RuntimeError, match="attachment failed"):
            processes.create_shared_embedding_engine_client(
                vllm_config=None,
                process_count=2,
                usage_context=None,
                stat_loggers=[],
                enable_log_requests=False,
                disable_log_stats=False,
            )
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)


def test_cleanup_restores_rpc_environment_even_if_listener_cleanup_fails(
    monkeypatch, tmp_path
):
    processes = importlib.import_module("dynamo.vllm.embedding_worker_processes")
    rpc_directory = Mock()
    rpc_directory.name = str(tmp_path)
    monkeypatch.setenv(processes._RPC_BASE_PATH_ENV, rpc_directory.name)
    listeners = Mock()
    listeners.cleanup.side_effect = OSError("listener cleanup failed")
    with pytest.raises(OSError, match="listener cleanup failed"):
        processes._cleanup_rpc_transport(listeners, rpc_directory, "previous")
    rpc_directory.cleanup.assert_called_once()
    assert os.environ[processes._RPC_BASE_PATH_ENV] == "previous"


def test_group_cleanup_joins_monitor_after_listener_cleanup_failure(
    monkeypatch, tmp_path, caplog
):
    processes = importlib.import_module("dynamo.vllm.embedding_worker_processes")
    rpc_directory = Mock()
    rpc_directory.name = str(tmp_path)
    monkeypatch.setenv(processes._RPC_BASE_PATH_ENV, rpc_directory.name)
    listeners = Mock()
    listeners.cleanup.side_effect = OSError("listener cleanup failed")
    group = processes.EmbeddingWorkerProcessGroup(
        children=[],
        engine_manager=Mock(),
        rpc_directory=rpc_directory,
        previous_rpc_base_path=None,
        listeners=listeners,
    )
    group._monitor_thread = Mock()
    group._monitor_thread.is_alive.return_value = True
    group.cleanup()
    group._monitor_thread.join.assert_called_once_with(timeout=1.0)
    rpc_directory.cleanup.assert_called_once()
    assert processes._RPC_BASE_PATH_ENV not in os.environ
    assert "listener cleanup failed" in caplog.text


def test_pool_closes_bound_listeners_when_child_spawn_fails(monkeypatch, pool_contract):
    c = pool_contract

    @contextmanager
    def launch(config, executor, log_stats, addresses):
        yield c.launch

    monkeypatch.setattr(
        c.processes,
        "engine_utils",
        SimpleNamespace(
            CoreEngineLaunch=SimpleNamespace,
            bind_engine_zmq_listeners=Mock(return_value=c.listeners),
            launch_core_engines=launch,
        ),
        raising=False,
    )
    c.spawn.side_effect = OSError("cannot spawn embedding child")
    with pytest.raises(OSError, match="cannot spawn embedding child"):
        _run_pool(c)
    c.manager.shutdown.assert_called_once()
    c.listeners.cleanup.assert_called_once()
    c.rpc_directory.cleanup.assert_called_once()
