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

    trtllm_module = types.ModuleType("tensorrt_llm")
    trtllm_llmapi_module = types.ModuleType("tensorrt_llm.llmapi")
    trtllm_mpi_session_module = types.ModuleType("tensorrt_llm.llmapi.mpi_session")
    trtllm_mpi_session_module.MpiPoolSession = FakeMpiPoolSession
    trtllm_module.llmapi = trtllm_llmapi_module
    trtllm_llmapi_module.mpi_session = trtllm_mpi_session_module

    monkeypatch.setitem(sys.modules, "mpi4py", mpi4py_module)
    monkeypatch.setitem(sys.modules, "mpi4py.futures", mpi4py_futures_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm", trtllm_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", trtllm_llmapi_module)
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm.llmapi.mpi_session", trtllm_mpi_session_module
    )

    # Force re-import so install_mpi_worker_bootstrap rebinds the stub's class.
    sys.modules.pop("gpu_memory_service.integrations.trtllm.mpi_bootstrap", None)
    bootstrap = importlib.import_module(
        "gpu_memory_service.integrations.trtllm.mpi_bootstrap"
    )
    bootstrap._bootstrap_installed = False
    bootstrap._extra_config_json = None

    yield {
        "bootstrap": bootstrap,
        "FakeMpiPoolSession": FakeMpiPoolSession,
        "captured_kwargs": captured_kwargs,
    }


def test_install_patches_mpi_pool_session_start_method(stubbed_env):
    bootstrap = stubbed_env["bootstrap"]
    FakeMpiPoolSession = stubbed_env["FakeMpiPoolSession"]
    original = FakeMpiPoolSession._start_mpi_pool if hasattr(
        FakeMpiPoolSession, "_start_mpi_pool"
    ) else None

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


def test_start_mpi_pool_injects_initializer_and_extra_config(
    stubbed_env, monkeypatch
):
    bootstrap = stubbed_env["bootstrap"]
    FakeMpiPoolSession = stubbed_env["FakeMpiPoolSession"]
    captured_kwargs = stubbed_env["captured_kwargs"]

    monkeypatch.setenv("DYN_GMS_TRTLLM_ENABLED", "1")
    monkeypatch.setenv("DYN_GMS_SHADOW_MODE", "1")
    monkeypatch.setenv("ENGINE_ID", "1")
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
    }
    assert extra_config_json == '{"gms_read_only": true}'
    # TRTLLM_* / TLLM_* vars plus propagated env must flow through executor env=.
    assert captured_kwargs["env"]["TRTLLM_FOO"] == "bar"
    assert captured_kwargs["env"]["TLLM_BAZ"] == "qux"
    assert captured_kwargs["env"]["DYN_GMS_TRTLLM_ENABLED"] == "1"


def test_worker_init_hook_restores_env_and_calls_setup_gms(
    stubbed_env, monkeypatch
):
    bootstrap = stubbed_env["bootstrap"]

    # Child environment starts without the propagated vars — simulates fresh interpreter.
    for key in (
        "DYN_GMS_TRTLLM_ENABLED",
        "DYN_GMS_SHADOW_MODE",
        "ENGINE_ID",
        "GMS_SOCKET_DIR",
    ):
        monkeypatch.delenv(key, raising=False)

    fake_setup_gms = MagicMock()
    fake_trtllm_pkg = types.ModuleType("gpu_memory_service.integrations.trtllm")
    fake_trtllm_pkg.setup_gms = fake_setup_gms
    monkeypatch.setitem(
        sys.modules, "gpu_memory_service.integrations.trtllm", fake_trtllm_pkg
    )

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
    fake_setup_gms = MagicMock()
    fake_trtllm_pkg = types.ModuleType("gpu_memory_service.integrations.trtllm")
    fake_trtllm_pkg.setup_gms = fake_setup_gms
    monkeypatch.setitem(
        sys.modules, "gpu_memory_service.integrations.trtllm", fake_trtllm_pkg
    )

    bootstrap.worker_init_hook({"ENGINE_ID": "1"}, None)

    fake_setup_gms.assert_not_called()


def test_worker_init_hook_is_picklable():
    """mpi4py.futures requires the initializer to be picklable-by-reference."""
    import pickle

    from gpu_memory_service.integrations.trtllm.mpi_bootstrap import (
        worker_init_hook,
    )

    # Function imported by module path; pickle roundtrip resolves back to the same object.
    assert pickle.loads(pickle.dumps(worker_init_hook)) is worker_init_hook
