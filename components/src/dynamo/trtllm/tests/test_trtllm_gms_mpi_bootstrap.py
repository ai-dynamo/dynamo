# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GMS MPI-worker bootstrap patch.

Exercises the Dynamo-side monkey-patch that propagates setup_gms() into
TensorRT-LLM's mpi4py.futures.MPIPoolExecutor child processes. Uses sys.modules
stubs so the test runs without tensorrt_llm, mpi4py, or CUDA.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
]


@pytest.fixture
def stubbed_env(monkeypatch):
    """Stub tensorrt_llm.llmapi.mpi_session and mpi4py.futures before importing bootstrap."""
    captured_kwargs: dict = {}
    call_log: list[str] = []

    class FakeComm:
        def __init__(self):
            self.rank = 0

        def Get_rank(self):
            return self.rank

        def Barrier(self):
            call_log.append(f"barrier:{self.rank}")

    fake_comm = FakeComm()

    class FakeMPI:
        COMM_WORLD = fake_comm

    class FakeMPIPoolExecutor:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    class FakeMpiPoolSession:
        def __init__(self, n_workers: int):
            self.n_workers = n_workers
            self.mpi_pool = None

    mpi4py_module = types.ModuleType("mpi4py")
    mpi4py_futures_module = types.ModuleType("mpi4py.futures")
    mpi4py_futures_module.MPIPoolExecutor = FakeMPIPoolExecutor
    mpi4py_module.futures = mpi4py_futures_module
    mpi4py_module.MPI = FakeMPI

    trtllm_module = types.ModuleType("tensorrt_llm")
    trtllm_llmapi_module = types.ModuleType("tensorrt_llm.llmapi")
    trtllm_mpi_session_module = types.ModuleType("tensorrt_llm.llmapi.mpi_session")
    trtllm_torch_module = types.ModuleType("tensorrt_llm._torch")
    trtllm_pyexecutor_module = types.ModuleType("tensorrt_llm._torch.pyexecutor")
    trtllm_pyexecutor_creator_module = types.ModuleType(
        "tensorrt_llm._torch.pyexecutor.py_executor_creator"
    )
    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(empty_cache=lambda: None)

    class FakePyExecutor:
        def __init__(self, name: str = "executor"):
            self.name = name

        def start_worker(self):
            call_log.append(f"start_worker:{self.name}")

    class FakePyTorchModelEngine:
        def __init__(self, *args, **kwargs):
            call_log.append("model_engine_init")

    def fake_create_py_executor_instance(*args, name: str = "executor", **kwargs):
        return FakePyExecutor(name)

    def fake_create_py_executor(*args, **kwargs):
        call_log.append("create_py_executor")
        executor = fake_create_py_executor_instance()
        executor.start_worker()
        return executor

    trtllm_pyexecutor_creator_module.PyExecutor = FakePyExecutor
    trtllm_pyexecutor_creator_module.PyTorchModelEngine = FakePyTorchModelEngine
    trtllm_pyexecutor_creator_module.create_py_executor_instance = (
        fake_create_py_executor_instance
    )
    trtllm_pyexecutor_creator_module.create_py_executor = fake_create_py_executor
    trtllm_mpi_session_module.MpiPoolSession = FakeMpiPoolSession
    trtllm_module.llmapi = trtllm_llmapi_module
    trtllm_llmapi_module.mpi_session = trtllm_mpi_session_module
    trtllm_module._torch = trtllm_torch_module
    trtllm_torch_module.pyexecutor = trtllm_pyexecutor_module
    trtllm_pyexecutor_module.py_executor_creator = trtllm_pyexecutor_creator_module

    monkeypatch.setitem(sys.modules, "mpi4py", mpi4py_module)
    monkeypatch.setitem(sys.modules, "mpi4py.futures", mpi4py_futures_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", trtllm_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", trtllm_llmapi_module)
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm.llmapi.mpi_session", trtllm_mpi_session_module
    )
    monkeypatch.setitem(sys.modules, "tensorrt_llm._torch", trtllm_torch_module)
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm._torch.pyexecutor", trtllm_pyexecutor_module
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.py_executor_creator",
        trtllm_pyexecutor_creator_module,
    )

    # Force re-import so install_mpi_worker_bootstrap rebinds the stub's class.
    sys.modules.pop("gpu_memory_service.integrations.trtllm.mpi_bootstrap", None)
    bootstrap = importlib.import_module(
        "gpu_memory_service.integrations.trtllm.mpi_bootstrap"
    )
    bootstrap._bootstrap_installed = False
    bootstrap._extra_config_json = None
    bootstrap._executor_finalize_hook_installed = False
    bootstrap._shadow_activation_fd = None

    yield {
        "bootstrap": bootstrap,
        "FakeMpiPoolSession": FakeMpiPoolSession,
        "captured_kwargs": captured_kwargs,
        "py_executor_creator": trtllm_pyexecutor_creator_module,
        "call_log": call_log,
        "fake_comm": fake_comm,
    }


def test_install_patches_mpi_pool_session_start_method(stubbed_env):
    bootstrap = stubbed_env["bootstrap"]
    FakeMpiPoolSession = stubbed_env["FakeMpiPoolSession"]
    original = (
        FakeMpiPoolSession._start_mpi_pool
        if hasattr(FakeMpiPoolSession, "_start_mpi_pool")
        else None
    )

    bootstrap.install_mpi_worker_bootstrap()

    patched = FakeMpiPoolSession._start_mpi_pool
    assert patched is not original
    assert callable(patched)


def test_install_is_idempotent(stubbed_env):
    bootstrap = stubbed_env["bootstrap"]
    FakeMpiPoolSession = stubbed_env["FakeMpiPoolSession"]

    bootstrap.install_mpi_worker_bootstrap()
    first = FakeMpiPoolSession._start_mpi_pool

    bootstrap.install_mpi_worker_bootstrap()
    assert FakeMpiPoolSession._start_mpi_pool is first


def test_start_mpi_pool_injects_initializer_and_extra_config(stubbed_env, monkeypatch):
    bootstrap = stubbed_env["bootstrap"]
    FakeMpiPoolSession = stubbed_env["FakeMpiPoolSession"]
    captured_kwargs = stubbed_env["captured_kwargs"]

    monkeypatch.setenv("DYN_GMS_TRTLLM_ENABLED", "1")
    monkeypatch.setenv("DYN_GMS_SHADOW_MODE", "1")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("FAILOVER_LOCK_PATH", "/tmp/failover.lock")
    monkeypatch.setenv("TRTLLM_FOO", "bar")
    monkeypatch.setenv("TLLM_BAZ", "qux")

    bootstrap.set_extra_config({"gms_read_only": True})
    bootstrap.install_mpi_worker_bootstrap()

    session = FakeMpiPoolSession(n_workers=4)
    session._start_mpi_pool()

    assert captured_kwargs["max_workers"] == 4
    assert captured_kwargs["initializer"] is bootstrap.worker_init_hook
    env_snapshot, extra_config_json = captured_kwargs["initargs"]
    assert env_snapshot == {
        "DYN_GMS_TRTLLM_ENABLED": "1",
        "DYN_GMS_SHADOW_MODE": "1",
        "ENGINE_ID": "1",
        "FAILOVER_LOCK_PATH": "/tmp/failover.lock",
    }
    assert extra_config_json == '{"gms_read_only": true}'
    # TRTLLM_* / TLLM_* vars plus propagated env must flow through executor env=.
    assert captured_kwargs["env"]["TRTLLM_FOO"] == "bar"
    assert captured_kwargs["env"]["TLLM_BAZ"] == "qux"
    assert captured_kwargs["env"]["DYN_GMS_TRTLLM_ENABLED"] == "1"
    assert captured_kwargs["env"]["FAILOVER_LOCK_PATH"] == "/tmp/failover.lock"


def test_worker_init_hook_restores_env_and_calls_setup_gms(stubbed_env, monkeypatch):
    bootstrap = stubbed_env["bootstrap"]

    # Child environment starts without the propagated vars — simulates fresh interpreter.
    for key in (
        "DYN_GMS_TRTLLM_ENABLED",
        "DYN_GMS_SHADOW_MODE",
        "ENGINE_ID",
        "GMS_SOCKET_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    import gpu_memory_service.integrations.trtllm as trtllm_gms

    fake_setup_gms = MagicMock()
    monkeypatch.setattr(trtllm_gms, "setup_gms", fake_setup_gms)

    env_snapshot = {
        "DYN_GMS_TRTLLM_ENABLED": "1",
        "DYN_GMS_SHADOW_MODE": "1",
        "ENGINE_ID": "3",
    }
    bootstrap.worker_init_hook(env_snapshot, '{"gms_read_only": true}')

    import os

    assert os.environ["ENGINE_ID"] == "3"
    assert os.environ["DYN_GMS_SHADOW_MODE"] == "1"
    fake_setup_gms.assert_called_once_with({"gms_read_only": True})


def test_worker_init_hook_no_op_when_gms_disabled(stubbed_env, monkeypatch):
    bootstrap = stubbed_env["bootstrap"]

    monkeypatch.delenv("DYN_GMS_TRTLLM_ENABLED", raising=False)
    import gpu_memory_service.integrations.trtllm as trtllm_gms

    fake_setup_gms = MagicMock()
    monkeypatch.setattr(trtllm_gms, "setup_gms", fake_setup_gms)

    bootstrap.worker_init_hook({"ENGINE_ID": "1"}, None)

    fake_setup_gms.assert_not_called()


def test_worker_init_hook_installs_child_finalize_before_start_worker(
    stubbed_env, monkeypatch
):
    bootstrap = stubbed_env["bootstrap"]
    py_executor_creator = stubbed_env["py_executor_creator"]
    call_log = stubbed_env["call_log"]

    class FakePyExecutor:
        def __init__(self, name: str):
            self.name = name

        def start_worker(self):
            call_log.append(f"start_worker:{self.name}")

    def fake_create_py_executor(*args, **kwargs):
        call_log.append("temp_executor_created")
        py_executor_creator.create_py_executor_instance(name="temp")
        call_log.append("final_executor_created")
        executor = py_executor_creator.create_py_executor_instance(name="final")
        executor.start_worker()
        return executor

    py_executor_creator.PyExecutor = FakePyExecutor
    py_executor_creator.create_py_executor_instance = (
        lambda *args, name="executor", **kwargs: FakePyExecutor(name)
    )
    py_executor_creator.create_py_executor = fake_create_py_executor

    import gpu_memory_service.integrations.trtllm as trtllm_gms

    fake_setup_gms = MagicMock(
        side_effect=lambda extra: call_log.append(f"setup:{extra}")
    )
    fake_finalize = MagicMock(side_effect=lambda: call_log.append("finalize"))
    monkeypatch.setattr(trtllm_gms, "setup_gms", fake_setup_gms)
    monkeypatch.setattr(trtllm_gms, "finalize_pending_gms_write", fake_finalize)

    env_snapshot = {"DYN_GMS_TRTLLM_ENABLED": "1", "ENGINE_ID": "0"}
    bootstrap.worker_init_hook(
        env_snapshot, '{"gms_delay_commit_until_engine_init": true}'
    )

    executor = py_executor_creator.create_py_executor()

    assert executor.name == "final"
    assert fake_setup_gms.call_count == 1
    assert fake_finalize.call_count == 1
    assert call_log == [
        "setup:{'gms_delay_commit_until_engine_init': True}",
        "temp_executor_created",
        "final_executor_created",
        "finalize",
        "start_worker:final",
    ]


def test_worker_init_hook_skips_child_finalize_hook_without_delay_flag(
    stubbed_env, monkeypatch
):
    bootstrap = stubbed_env["bootstrap"]
    py_executor_creator = stubbed_env["py_executor_creator"]

    original_create_py_executor = py_executor_creator.create_py_executor
    import gpu_memory_service.integrations.trtllm as trtllm_gms

    fake_setup_gms = MagicMock()
    fake_finalize = MagicMock()
    monkeypatch.setattr(trtllm_gms, "setup_gms", fake_setup_gms)
    monkeypatch.setattr(trtllm_gms, "finalize_pending_gms_write", fake_finalize)

    bootstrap.worker_init_hook(
        {"DYN_GMS_TRTLLM_ENABLED": "1", "ENGINE_ID": "0"}, '{"gms_read_only": true}'
    )

    assert py_executor_creator.create_py_executor is original_create_py_executor
    py_executor_creator.create_py_executor()
    fake_finalize.assert_not_called()


def test_worker_init_hook_standby_gates_after_model_engine_before_kv_build(
    stubbed_env, monkeypatch
):
    bootstrap = stubbed_env["bootstrap"]
    py_executor_creator = stubbed_env["py_executor_creator"]
    call_log = stubbed_env["call_log"]

    def fake_create_py_executor(*args, **kwargs):
        call_log.append("create_py_executor")
        py_executor_creator.PyTorchModelEngine()
        call_log.append("after_model_engine")
        py_executor_creator.create_py_executor_instance(name="final")
        call_log.append("after_kv_build")
        return object()

    py_executor_creator.create_py_executor = fake_create_py_executor

    import gpu_memory_service.integrations.trtllm as trtllm_gms

    fake_setup_gms = MagicMock(
        side_effect=lambda extra: call_log.append(f"setup:{extra}")
    )
    monkeypatch.setattr(trtllm_gms, "setup_gms", fake_setup_gms)
    monkeypatch.setattr(
        bootstrap,
        "_wait_for_shadow_activation",
        lambda: call_log.append("wait_for_activation"),
    )

    env_snapshot = {
        "DYN_GMS_TRTLLM_ENABLED": "1",
        "DYN_GMS_SHADOW_MODE": "1",
        "ENGINE_ID": "1",
        "FAILOVER_LOCK_PATH": "/tmp/failover.lock",
    }
    bootstrap.worker_init_hook(env_snapshot, '{"gms_read_only": true}')

    py_executor_creator.create_py_executor()

    assert fake_setup_gms.call_count == 1
    assert call_log == [
        "setup:{'gms_read_only': True}",
        "create_py_executor",
        "model_engine_init",
        "wait_for_activation",
        "after_model_engine",
        "after_kv_build",
    ]


def test_worker_init_hook_is_picklable(monkeypatch):
    """mpi4py.futures requires the initializer to be picklable-by-reference."""
    import pickle
    import types

    torch_module = types.ModuleType("torch")
    torch_module.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    from gpu_memory_service.integrations.trtllm.mpi_bootstrap import worker_init_hook

    # Function imported by module path; pickle roundtrip resolves back to the same object.
    assert pickle.loads(pickle.dumps(worker_init_hook)) is worker_init_hook
