# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""check_fork_safe: when --worker-start-method fork is refused."""

import importlib
from types import SimpleNamespace

import pytest

from dynamo.vllm import fork_safety

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def vllm_main():
    """dynamo.vllm.main, imported when a test runs.

    The pre-commit marker report collects this file without vLLM's
    dependencies, so the module cannot be imported at collection.
    """
    return importlib.import_module("dynamo.vllm.main")


@pytest.fixture
def process(monkeypatch, vllm_main):
    """The process state check_fork_safe looks at; safe to fork by default."""
    state = SimpleNamespace(cuda=False, python_threads=[[]], os_threads=["python3"])
    monkeypatch.setattr(vllm_main, "cuda_is_initialized", lambda: state.cuda)
    # Each look at the Python threads takes the next list; the last one repeats.
    monkeypatch.setattr(
        fork_safety,
        "other_python_threads",
        lambda: state.python_threads.pop(0)
        if len(state.python_threads) > 1
        else state.python_threads[0],
    )
    monkeypatch.setattr(fork_safety, "os_thread_names", lambda: state.os_threads)
    monkeypatch.setattr(fork_safety.time, "sleep", lambda _seconds: None)
    return state


def test_fork_check_passes_without_cuda_other_threads_or_the_runtime(
    process, vllm_main
):
    process.os_threads = ["python3", "jemalloc_bg_thd"]
    vllm_main.check_fork_safe()


def test_fork_check_refuses_when_cuda_is_initialized(process, vllm_main):
    process.cuda = True
    with pytest.raises(RuntimeError, match="CUDA is already initialized"):
        vllm_main.check_fork_safe()


def test_fork_check_waits_for_a_thread_that_is_about_to_finish(process, vllm_main):
    process.python_threads = [["import-helper"], ["import-helper"], []]
    vllm_main.check_fork_safe()
    assert process.python_threads == [[]]


def test_fork_check_refuses_while_another_python_thread_runs(
    process, vllm_main, monkeypatch
):
    monkeypatch.setattr(fork_safety, "THREAD_SETTLE_SECONDS", 0.0)
    process.python_threads = [["holds-a-lock"]]
    with pytest.raises(RuntimeError, match="Python threads are running: holds-a-lock"):
        vllm_main.check_fork_safe()


def test_fork_check_refuses_once_the_runtime_has_started(process, vllm_main):
    process.os_threads = ["python3", "tokio-rt-worker", "tokio-rt-worker"]
    with pytest.raises(
        RuntimeError, match=r"Dynamo's runtime is running \(2 threads\)"
    ):
        vllm_main.check_fork_safe()
