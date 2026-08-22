# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for embedding endpoint processes sharing one EngineCore."""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from vllm.usage.usage_lib import UsageContext

from dynamo.vllm import embedding_worker_processes as processes

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            _api_process_count=1,
            _api_process_rank=0,
        )
    )


def test_child_environment_offsets_system_port_by_index(monkeypatch):
    monkeypatch.setenv("DYN_SYSTEM_PORT", "19401")
    env = processes._child_environment(
        process_count=4,
        process_index=2,
        addresses_json='{"process_count": 4}',
        parent_pid=123,
    )

    assert env[processes._ROLE_ENV] == processes._CHILD_ROLE
    assert env[processes._INDEX_ENV] == "2"
    assert env[processes._PARENT_PID_ENV] == "123"
    # The parent keeps 19401 as index 0, so children start one above it.
    assert env["DYN_SYSTEM_PORT"] == "19403"
    assert env["PYTHONUNBUFFERED"] == "1"


@pytest.mark.parametrize("raw", ["-1", "0", "not-a-port"])
def test_child_environment_preserves_non_positive_system_port(monkeypatch, raw):
    """-1 means disabled and 0 means "pick an ephemeral port".

    Overriding either would start a listener the operator did not ask for, or
    discard an explicit request for ephemeral ports.
    """
    monkeypatch.setenv("DYN_SYSTEM_PORT", raw)
    env = processes._child_environment(
        process_count=2,
        process_index=1,
        addresses_json='{"process_count": 2}',
        parent_pid=123,
    )

    assert env["DYN_SYSTEM_PORT"] == raw


def test_child_environment_omits_system_port_when_parent_has_none(monkeypatch):
    monkeypatch.delenv("DYN_SYSTEM_PORT", raising=False)
    env = processes._child_environment(
        process_count=2,
        process_index=1,
        addresses_json='{"process_count": 2}',
        parent_pid=123,
    )

    assert "DYN_SYSTEM_PORT" not in env


def test_decode_child_addresses_validates_parent_count(monkeypatch):
    monkeypatch.setenv(processes._INDEX_ENV, "1")
    monkeypatch.setenv(
        processes._ENGINE_ADDRESSES_ENV,
        json.dumps(
            {
                "process_count": 2,
                "inputs": ["in0", "in1"],
                "outputs": ["out0", "out1"],
            }
        ),
    )
    assert processes._decode_child_addresses(2) == (
        1,
        ["in0", "in1"],
        ["out0", "out1"],
    )

    with pytest.raises(RuntimeError, match="does not match"):
        processes._decode_child_addresses(4)


def test_attach_client_sets_vllm_api_rank():
    client = Mock()
    config = _vllm_config()
    with patch.object(
        processes.AsyncLLM, "from_vllm_config", return_value=client
    ) as create:
        result, client_config = processes._attach_client(
            config,
            process_count=8,
            process_index=3,
            input_address="ipc://input",
            output_address="ipc://output",
            usage_context=UsageContext.OPENAI_API_SERVER,
            stat_loggers=[],
            enable_log_requests=False,
            disable_log_stats=False,
        )

    assert result is client
    assert client_config.parallel_config._api_process_count == 8
    assert client_config.parallel_config._api_process_rank == 3
    assert config.parallel_config._api_process_count == 1
    create.assert_called_once()
    kwargs = create.call_args.kwargs
    assert kwargs["client_addresses"] == {
        "input_address": "ipc://input",
        "output_address": "ipc://output",
    }
    assert kwargs["client_count"] == 8
    assert kwargs["client_index"] == 3


def test_stat_loggers_are_precreated_on_calling_thread():
    config = _vllm_config()
    logger = Mock()
    factory = Mock(return_value=logger)

    wrappers = processes._precreate_stat_logger_factories(config, [factory])

    factory.assert_called_once_with(config, 0)
    assert len(wrappers) == 1
    assert wrappers[0](_vllm_config(), 7) is logger


def test_process_group_cleanup_is_ordered_and_idempotent(monkeypatch):
    child = Mock(pid=101)
    child.poll.side_effect = [None, 0, 0]
    engine_manager = Mock()
    rpc_directory = Mock(name="rpc_directory")
    rpc_directory.name = "/tmp/dynamo-vllm-rpc-test"
    monkeypatch.setenv(processes._RPC_BASE_PATH_ENV, rpc_directory.name)

    group = processes.EmbeddingWorkerProcessGroup(
        children=[(1, child)],
        engine_manager=engine_manager,
        rpc_directory=rpc_directory,
        previous_rpc_base_path="/tmp/original-rpc",
    )
    group.cleanup(timeout=2.0)
    group.cleanup(timeout=2.0)

    child.terminate.assert_called_once()
    engine_manager.shutdown.assert_called_once_with(timeout=2.0)
    rpc_directory.cleanup.assert_called_once()
    assert os.environ[processes._RPC_BASE_PATH_ENV] == "/tmp/original-rpc"


def test_process_group_reports_unexpected_child_exit():
    child = Mock(pid=102)
    child.poll.return_value = 17
    callback = Mock()
    rpc_directory = Mock()
    rpc_directory.name = "/tmp/dynamo-vllm-rpc-test"
    group = processes.EmbeddingWorkerProcessGroup(
        children=[(2, child)],
        engine_manager=Mock(),
        rpc_directory=rpc_directory,
        previous_rpc_base_path=None,
        parent_failure_callback=callback,
    )

    group.start_monitor()
    group._monitor_thread.join(timeout=1.0)

    callback.assert_called_once_with(2, 17)
    group._stopping.set()


def test_parent_launches_n_minus_one_children_and_one_engine(monkeypatch):
    process_count = 4
    addresses = SimpleNamespace(
        inputs=[f"in{i}" for i in range(process_count)],
        outputs=[f"out{i}" for i in range(process_count)],
    )
    engine_manager = Mock()
    rpc_directory = Mock()
    rpc_directory.name = "/tmp/dynamo-vllm-rpc-test"
    children = []

    def create_child(*_args, **_kwargs):
        child = Mock(pid=200 + len(children))
        child.poll.side_effect = [None, 0, 0]
        children.append(child)
        return child

    @contextmanager
    def launch_context(*_args, **_kwargs):
        yield engine_manager, None, addresses, None

    parent_client = Mock()
    parent_config = _vllm_config()
    monkeypatch.delenv(processes._ROLE_ENV, raising=False)
    monkeypatch.setattr(processes, "_short_rpc_directory", lambda: rpc_directory)
    monkeypatch.setattr(processes, "get_engine_zmq_addresses", lambda *_: addresses)
    monkeypatch.setattr(processes, "launch_core_engines", launch_context)
    monkeypatch.setattr(processes.Executor, "get_class", lambda _config: object)
    monkeypatch.setattr(processes.subprocess, "Popen", create_child)
    monkeypatch.setattr(
        processes,
        "_attach_client",
        lambda *_args, **_kwargs: (parent_client, parent_config),
    )
    monkeypatch.setattr(
        processes.EmbeddingWorkerProcessGroup, "start_monitor", lambda _self: None
    )

    client, client_config, group = processes.create_shared_embedding_engine_client(
        vllm_config=_vllm_config(),
        process_count=process_count,
        usage_context=UsageContext.OPENAI_API_SERVER,
        stat_loggers=[],
        enable_log_requests=False,
        disable_log_stats=False,
    )

    assert client is parent_client
    assert client_config is parent_config
    assert group is not None
    assert len(children) == process_count - 1
    assert group.engine_manager is engine_manager
    group.cleanup()


def test_child_attaches_without_launching_engine(monkeypatch):
    monkeypatch.setenv(processes._ROLE_ENV, processes._CHILD_ROLE)
    monkeypatch.setenv(processes._INDEX_ENV, "1")
    monkeypatch.setenv(
        processes._ENGINE_ADDRESSES_ENV,
        json.dumps(
            {
                "process_count": 2,
                "inputs": ["in0", "in1"],
                "outputs": ["out0", "out1"],
            }
        ),
    )
    child_client = Mock()
    child_config = _vllm_config()
    attach = Mock(return_value=(child_client, child_config))
    monkeypatch.setattr(processes, "_attach_client", attach)

    client, config, group = processes.create_shared_embedding_engine_client(
        vllm_config=_vllm_config(),
        process_count=2,
        usage_context=UsageContext.OPENAI_API_SERVER,
        stat_loggers=[],
        enable_log_requests=False,
        disable_log_stats=True,
    )

    assert client is child_client
    assert config is child_config
    assert group is None
    assert attach.call_args.kwargs["process_index"] == 1
    assert attach.call_args.kwargs["input_address"] == "in1"
    assert attach.call_args.kwargs["output_address"] == "out1"
