# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dynamo.vllm.launcher: fork-safety checks, fallback, supervision and cleanup.

The supervision tests run the real launcher in a subprocess, with its engine
module replaced by a small fake engine written to tmp_path.
"""

import os
import select
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from dynamo.vllm import launcher

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Runs the launcher with the fake engine and the given pre-import modules.
WRAPPER = """
import os, sys
from dynamo.vllm import fork_safety, launcher
launcher.ENGINE_MODULE = "fake_engine"
fork_safety.THREAD_SETTLE_SECONDS = float(os.environ["FAKE_SETTLE_SECONDS"])
modules = [m for m in sys.argv[1].split(",") if m]
launcher.preimport_steps = lambda: [
    (m, lambda m=m: __import__(m)) for m in modules
]
launcher.main(sys.argv[2:])
"""

FAKE_ENGINE = """
import os, signal, sys, time

mode = os.environ["FAKE_ENGINE_MODE"]
print("FAKE start pid=%d argv=%s" % (os.getpid(), sys.argv[1:]), flush=True)
if mode.startswith("exit:"):
    sys.exit(int(mode.split(":")[1]))
if mode == "kill-self":
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "first-dies":
    marker = os.path.join(os.environ["FAKE_DIR"], "first")
    if not os.path.exists(marker):
        open(marker, "w").close()
        sys.exit(3)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print("FAKE waiting", flush=True)
    signal.pause()
if mode == "leaves-orphans":
    for detach in (False, True):
        pid = os.fork()
        if pid == 0:
            if detach:
                os.setsid()
            time.sleep(600)
            os._exit(0)
        print("FAKE child %d" % pid, flush=True)
    sys.exit(5)
"""

SHORT_THREAD = """
import threading, time
threading.Thread(target=time.sleep, args=(0.1,), name="short").start()
"""

LONG_THREAD = """
import threading
threading.Thread(target=threading.Event().wait, name="leaky", daemon=True).start()
"""


class Launcher:
    def __init__(self, fake_dir, mode, preimport="", launcher_args=(), settle=30.0):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join([str(fake_dir), *sys.path]),
            "FAKE_ENGINE_MODE": mode,
            "FAKE_DIR": str(fake_dir),
            "FAKE_SETTLE_SECONDS": str(settle),
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-c", WRAPPER, preimport, *launcher_args, "--", "-x"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.lines: list[str] = []
        self._pending = ""

    def read_until(self, text, timeout=30.0) -> str:
        # Raw reads: a buffered readline() can take several lines off the pipe
        # and leave select() waiting for data that has already arrived.
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while True:
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self.lines.append(line + "\n")
                if text in line:
                    return line
            remaining = deadline - time.monotonic()
            ready = remaining > 0 and select.select([fd], [], [], remaining)[0]
            chunk = os.read(fd, 65536) if ready else b""
            if not chunk:
                raise AssertionError(f"{text!r} not seen:\n{self.output}")
            self._pending += chunk.decode()

    def finish(self, timeout=30.0) -> int:
        out, _ = self.proc.communicate(timeout=timeout)
        self.lines.append(self._pending + out.decode())
        self._pending = ""
        return self.proc.returncode

    @property
    def output(self) -> str:
        return "".join(self.lines)


def _kill_processes_of(fake_dir) -> None:
    """Kill every process started for this test: engines leave the launcher's group."""
    marker = f"FAKE_DIR={fake_dir}".encode()
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as environ:
                if marker in environ.read().split(b"\0"):
                    os.kill(int(entry), signal.SIGKILL)
        except OSError:  # gone, or not ours
            continue


@pytest.fixture
def launch(tmp_path):
    """Start launchers with the fake engine; nothing they start outlives the test."""
    (tmp_path / "fake_engine").mkdir()
    (tmp_path / "fake_engine" / "__init__.py").write_text("")
    (tmp_path / "fake_engine" / "__main__.py").write_text(FAKE_ENGINE)
    (tmp_path / "short_thread.py").write_text(SHORT_THREAD)
    (tmp_path / "long_thread.py").write_text(LONG_THREAD)
    started: list[Launcher] = []

    def start(mode, **kwargs) -> Launcher:
        started.append(Launcher(tmp_path, mode, **kwargs))
        return started[-1]

    yield start
    _kill_processes_of(tmp_path)
    for run in started:
        run.proc.kill()
        run.proc.wait()
        run.proc.stdout.close()


@pytest.mark.unit
def test_engine_args_follow_the_separator():
    args, engine_args = launcher.parse_args(
        ["--restart-min-uptime", "5", "--", "--model", "m", "--", "x"]
    )
    assert args.restart_min_uptime == 5.0
    assert engine_args == ["--model", "m", "--", "x"]


@pytest.mark.unit
@pytest.mark.parametrize("value", ["nan", "-1", "inf"])
def test_restart_min_uptime_must_be_a_duration(value):
    with pytest.raises(SystemExit):
        launcher.parse_args([f"--restart-min-uptime={value}", "--", "-x"])


@pytest.mark.unit
def test_a_running_python_thread_is_a_hazard():
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name="holds-a-lock")
    thread.start()
    try:
        assert any("holds-a-lock" in p for p in launcher.fork_hazards())
    finally:
        release.set()
        thread.join()


@pytest.mark.unit
def test_initialized_cuda_is_a_hazard(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert "torch CUDA is initialized" in launcher.fork_hazards()


@pytest.mark.unit
def test_open_driver_descriptors_are_a_hazard(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "NVIDIA_DEVICE_PREFIX", str(tmp_path / "nvidia"))
    with open(tmp_path / "nvidia0", "w"), open(tmp_path / "nvidiactl", "w"):
        assert any(
            p.startswith("2 ") and "descriptors" in p for p in launcher.fork_hazards()
        )


@pytest.mark.integration
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "mode, status", [("exit:7", 7), ("exit:0", 0), ("kill-self", 137)]
)
def test_engine_that_exits_early_ends_the_launcher_with_its_status(
    launch, mode, status
):
    run = launch(mode)
    assert run.finish() == status
    assert run.output.count("FAKE start") == 1
    assert "argv=['-x']" in run.output


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_engine_that_ends_with_status_zero_is_not_replaced(launch):
    run = launch("exit:0", launcher_args=["--restart-min-uptime=0"])
    assert run.finish() == 0
    assert run.output.count("FAKE start") == 1


@pytest.mark.integration
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    # The fake engine exits 0 on SIGTERM and leaves SIGINT to Python's default.
    "signum, status",
    [(signal.SIGTERM, 0), (signal.SIGINT, 130)],
)
def test_engine_that_ran_long_enough_is_replaced_and_shutdown_reaches_it(
    launch, signum, status
):
    run = launch("first-dies", launcher_args=["--restart-min-uptime=0"])
    first = run.read_until("FAKE start")
    run.read_until("starting a replacement")
    second = run.read_until("FAKE start")
    run.read_until("FAKE waiting")
    run.proc.send_signal(signum)
    assert run.finish() == status
    assert first != second
    assert run.output.count("FAKE start") == 2


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_a_dead_engines_processes_are_killed_and_reaped(launch):
    run = launch("leaves-orphans")
    children = [int(run.read_until("FAKE child").split()[-1]) for _ in range(2)]
    assert run.finish() == 5
    for pid in children:  # one in the engine's group, one that left it
        assert not os.path.exists(f"/proc/{pid}"), run.output


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_a_thread_that_finishes_after_its_import_does_not_block_forking(launch):
    run = launch("exit:0", preimport="short_thread")
    assert run.finish() == 0
    assert "started engine" in run.output
    assert "not fork-safe" not in run.output


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_unsafe_pre_import_falls_back_to_a_plain_start(launch):
    run = launch("exit:0", preimport="long_thread", settle=0.2)
    assert run.finish() == 0
    assert "not fork-safe after long_thread" in run.output
    assert "leaky" in run.output
    # The plain engine replaces the launcher (same pid) and gets the arguments.
    assert f"FAKE start pid={run.proc.pid} argv=['-x']" in run.output
    assert "started engine" not in run.output


@pytest.mark.unit
def test_supervisor_refuses_to_fork_once_unsafe(monkeypatch):
    monkeypatch.setattr(launcher, "fork_hazards", lambda: ["2 threads (x)"])
    monkeypatch.setattr(launcher.os, "fork", lambda: pytest.fail("forked"))
    monkeypatch.setattr(launcher.ctypes, "CDLL", lambda *_a, **_k: _Libc())
    monkeypatch.setattr(launcher.signal, "signal", lambda *_a: None)
    monkeypatch.setattr(launcher.signal, "pthread_sigmask", lambda *_a: None)
    assert launcher.Supervisor(["-x"], 0.0).run() == 1


class _Libc:
    @staticmethod
    def prctl(*_args):
        return 0
