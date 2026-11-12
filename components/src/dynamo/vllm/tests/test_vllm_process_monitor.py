import multiprocessing
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import psutil
import pytest

from dynamo.vllm.process_monitor import LocalEngineProcesses

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.fault_tolerance,
]


def engine_with_handles(handles, elastic=False):
    return SimpleNamespace(
        engine_core=SimpleNamespace(
            resources=SimpleNamespace(engine_manager=SimpleNamespace(processes=handles))
        ),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(enable_elastic_ep=elastic)
        ),
    )


def test_captures_core_and_execution_workers_not_transient_helpers():
    core = MagicMock()
    worker = MagicMock()
    helper = MagicMock()
    worker.name.return_value = "VLLM::Worker_TP0"
    helper.name.return_value = "compile_worker"
    core.children.return_value = [worker, helper]
    with patch("dynamo.vllm.process_monitor.psutil.Process", return_value=core):
        monitor = LocalEngineProcesses(engine_with_handles([SimpleNamespace(pid=123)]))

    assert monitor.processes == [core, worker]
    core.create_time.assert_called_once_with()
    worker.create_time.assert_called_once_with()


def test_elastic_worker_membership_is_not_treated_as_process_failure():
    core = MagicMock()
    with patch("dynamo.vllm.process_monitor.psutil.Process", return_value=core):
        monitor = LocalEngineProcesses(
            engine_with_handles([SimpleNamespace(pid=123)], elastic=True)
        )

    assert monitor.processes == [core]
    core.children.assert_not_called()


@pytest.mark.parametrize("condition", ["exited", "zombie", "reused", "missing"])
def test_detects_dead_execution_process(condition):
    monitor = object.__new__(LocalEngineProcesses)
    process = MagicMock(pid=123)
    process.is_running.return_value = condition not in ("exited", "reused")
    process.status.return_value = psutil.STATUS_ZOMBIE
    if condition == "missing":
        process.status.side_effect = psutil.NoSuchProcess(123)
    monitor.processes = [process]

    with pytest.raises(RuntimeError, match="123 exited"):
        monitor.check()


def test_nonlocal_engine_does_not_scan_unrelated_processes():
    engine = engine_with_handles([])
    engine.engine_core.resources.engine_manager = None
    with patch("dynamo.vllm.process_monitor.psutil.Process") as process:
        monitor = LocalEngineProcesses(engine)
        monitor.check()
    process.assert_not_called()


def test_real_process_exit_detected_without_engine_rpc():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        monitor = LocalEngineProcesses(
            engine_with_handles([SimpleNamespace(pid=child.pid)])
        )
        monitor.check()
        child.terminate()
        child.wait(timeout=5)
        with pytest.raises(RuntimeError, match="exited"):
            monitor.check()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires fork"
)
def test_real_unreaped_zombie_is_not_alive():
    parent, child = multiprocessing.Pipe()

    def exit_child():
        child.send(os.getpid())
        child.recv()
        os._exit(0)

    process = multiprocessing.get_context("fork").Process(target=exit_child)
    try:
        process.start()
        assert parent.poll(5)
        process_id = parent.recv()
        monitor = LocalEngineProcesses(
            engine_with_handles([SimpleNamespace(pid=process_id)])
        )
        child_identity = psutil.Process(process_id)
        parent.send(True)
        deadline = time.monotonic() + 5
        while child_identity.status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline, "Child did not exit"
            time.sleep(0.01)
        with pytest.raises(RuntimeError, match="exited"):
            monitor.check()
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()
        parent.close()
        child.close()
