# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dynamo.vllm.launcher: fork-safety checks, fallback, supervision and liveness.

The supervision tests run the real launcher in a subprocess, with its engine
module replaced by a small fake engine written to tmp_path.
"""

import http.server
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
from tests.utils.constants import DynamoPortRange
from tests.utils.port_utils import allocate_port, deallocate_port

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Runs the launcher with the fake engine and the given pre-import modules.
WRAPPER = """
import os, sys, threading
from dynamo.vllm import fork_safety, launcher
launcher.ENGINE_MODULE = "fake_engine"
if os.environ.get("FAKE_BACKGROUND_THREAD"):
    threading.Thread(target=threading.Event().wait, name="agent", daemon=True).start()
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
print(
    "FAKE start pid=%d argv=%s forked=%s blocked=%s"
    % (
        os.getpid(),
        sys.argv[1:],
        "dynamo.vllm.launcher" in sys.modules,
        sorted(signal.pthread_sigmask(signal.SIG_BLOCK, [])),
    ),
    flush=True,
)


def serve_live(answer):
    import http.server, threading

    class Live(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(answer())
            self.end_headers()

        def log_message(self, *_):
            pass

    address = ("127.0.0.1", int(os.environ["FAKE_PORT"]))
    server = http.server.HTTPServer(address, Live)
    threading.Thread(target=server.serve_forever, daemon=True).start()


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
if mode == "first-stops-answering":
    # The first engine answers one check and then fails them; later ones are live.
    marker = os.path.join(os.environ["FAKE_DIR"], "first")
    first = not os.path.exists(marker)
    open(marker, "w").close()
    answers = iter([200])
    serve_live(lambda: next(answers, 503) if first else 200)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print("FAKE serving", flush=True)
    signal.pause()
if mode == "shutdown-during-a-check":
    # The second check asks the launcher to shut down while it is in flight,
    # and fails, as a draining engine's endpoint does.
    checks = iter([200])

    def answer():
        status = next(checks, 503)
        if status == 503:
            os.kill(os.getppid(), signal.SIGTERM)
        return status

    serve_live(answer)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print("FAKE serving", flush=True)
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
    def __init__(
        self,
        fake_dir,
        mode,
        preimport="",
        launcher_args=(),
        settle=30.0,
        port=0,
        background_thread=False,
    ):
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join([str(fake_dir), *sys.path]),
            "FAKE_ENGINE_MODE": mode,
            "FAKE_DIR": str(fake_dir),
            "FAKE_SETTLE_SECONDS": str(settle),
            "FAKE_PORT": str(port),
        }
        if background_thread:
            env["FAKE_BACKGROUND_THREAD"] = "1"
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
    # Forked from the launcher's modules, with no signal left blocked.
    assert "argv=['-x'] forked=True blocked=[]" in run.output


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
def test_unsafe_pre_import_restarts_the_launcher_without_it(launch):
    run = launch("exit:0", preimport="long_thread", settle=0.2)
    assert run.finish() == 0
    assert "not fork-safe after long_thread" in run.output
    assert "leaky" in run.output
    assert "restarting without pre-import" in run.output
    # The launcher keeps its pid across the restart and still supervises: the
    # engine is a new interpreter in another process.
    assert "started engine" in run.output
    assert "argv=['-x'] forked=False blocked=[]" in run.output
    assert f"FAKE start pid={run.proc.pid} " not in run.output


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_without_pre_import_the_launcher_need_not_be_fork_safe(launch):
    run = launch("exit:0", launcher_args=["--no-preimport"], background_thread=True)
    assert run.finish() == 0
    assert "argv=['-x'] forked=False blocked=[]" in run.output
    assert "safe to fork" not in run.output


@pytest.mark.unit
def test_restart_keeps_how_the_launcher_was_invoked(monkeypatch):
    commands = []
    monkeypatch.setattr(launcher.os, "execv", lambda _exe, cmd: commands.append(cmd))
    monkeypatch.setattr(
        launcher.sys,
        "orig_argv",
        ["python3", "-X", "faulthandler", "-m", "dynamo.vllm.launcher", "--", "-x"],
    )

    launcher._restart_without_preimport(["--", "-x"])
    # Arguments that are not the end of the command line: start the module anew.
    launcher._restart_without_preimport(["--restart-min-uptime", "5", "--", "-x"])

    assert commands == [
        ["python3", "-X", "faulthandler", "-m", "dynamo.vllm.launcher"]
        + ["--no-preimport", "--", "-x"],
        [sys.executable, "-m", "dynamo.vllm.launcher", "--no-preimport"]
        + ["--restart-min-uptime", "5", "--", "-x"],
    ]


@pytest.fixture
def engine_port():
    port = allocate_port(DynamoPortRange.SERVE)
    yield port
    deallocate_port(port)


def _liveness_args(port, live_file, threshold):
    return [
        "--restart-min-uptime=0",
        f"--liveness-url=http://127.0.0.1:{port}/live",
        "--liveness-period=0.05",
        "--liveness-timeout=5",
        f"--liveness-failure-threshold={threshold}",
        "--startup-timeout=30",
        f"--liveness-file={live_file}",
    ]


def _uptime() -> float:
    with open("/proc/uptime") as uptime:
        return float(uptime.read().split()[0])


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_a_hung_engine_is_killed_and_replaced(launch, engine_port, tmp_path):
    live_file = tmp_path / "live"
    run = launch(
        "first-stops-answering",
        launcher_args=_liveness_args(engine_port, live_file, threshold=3),
        port=engine_port,
    )
    run.read_until("FAKE serving")
    run.read_until("is hung (3 liveness check(s) failed in a row, last: HTTP 503)")
    run.read_until("exited with 137")
    run.read_until("FAKE serving")
    assert int(live_file.read_text()) > _uptime()
    run.proc.send_signal(signal.SIGTERM)
    assert run.finish() == 0
    assert run.output.count("is hung") == 1, run.output


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_an_engine_that_is_shutting_down_is_not_hung(launch, engine_port, tmp_path):
    run = launch(
        "shutdown-during-a-check",
        launcher_args=_liveness_args(engine_port, tmp_path / "live", threshold=1),
        port=engine_port,
    )
    assert run.finish() == 0
    assert "is hung" not in run.output
    assert run.output.count("FAKE start") == 1


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


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Probe:
    """Answers checks from a script of results; a failed check takes its timeout."""

    def __init__(self, clock, results):
        self.clock = clock
        self.results = list(results)
        self.calls: list[float] = []

    def __call__(self, url, timeout) -> str | None:
        self.calls.append(self.clock.now)
        failure = None if self.results.pop(0) else "HTTP 503"
        if failure is not None:
            self.clock.now += timeout
        return failure


# The fake uptime runs this far ahead of the fake monotonic clock.
_BOOT = 1_000_000


def _liveness(clock, probe, tmp_path=None, **kwargs):
    params = dict(period=5.0, timeout=4.0, failure_threshold=3, startup_timeout=60.0)
    params.update(kwargs)
    return launcher.EngineLiveness(
        "http://127.0.0.1:9090/live",
        liveness_file=None if tmp_path is None else str(tmp_path / "live"),
        probe=probe,
        clock=clock,
        uptime=lambda: clock.now + _BOOT,
        **params,
    )


def _run_checks(liveness, clock, count):
    """Advance the clock to each due check and run it; return the first reason."""
    for _ in range(count):
        clock.now += liveness.seconds_to_next_check()
        reason = liveness.hung()
        if reason is not None:
            return reason
    return None


def _deadline(tmp_path) -> int:
    return int((tmp_path / "live").read_text())


@pytest.mark.unit
def test_checks_run_once_per_period():
    clock = _Clock()
    probe = _Probe(clock, [True] * 3)
    liveness = _liveness(clock, probe)
    liveness.engine_started()
    assert liveness.hung() is None  # not due yet
    assert probe.calls == []
    assert _run_checks(liveness, clock, 3) is None
    assert probe.calls == [1005.0, 1010.0, 1015.0]


@pytest.mark.unit
def test_a_live_engine_is_hung_after_the_threshold_of_failures_in_a_row():
    clock = _Clock()
    # One failure, recovery, then three in a row.
    probe = _Probe(clock, [True, False, True, False, False, False])
    liveness = _liveness(clock, probe)
    liveness.engine_started()
    assert _run_checks(liveness, clock, 5) is None
    assert _run_checks(liveness, clock, 1) == (
        "3 liveness check(s) failed in a row, last: HTTP 503"
    )


@pytest.mark.unit
def test_a_starting_engine_counts_as_live_until_the_startup_timeout(tmp_path):
    clock = _Clock()
    probe = _Probe(clock, [False] * 20)
    liveness = _liveness(clock, probe, tmp_path, startup_timeout=60.0)
    liveness.engine_started()
    deadlines = []
    for _ in range(6):  # checks at 5, 14, 23, 32, 41 and 50 s: still starting
        assert _run_checks(liveness, clock, 1) is None
        deadlines.append(_deadline(tmp_path))
    assert deadlines == sorted(set(deadlines)), "each failed startup check renews"
    assert _run_checks(liveness, clock, 1) == (
        "not live 63 s after it started, last: HTTP 503"
    )


@pytest.mark.unit
def test_the_liveness_file_outlasts_the_launchers_own_hang_detection(tmp_path):
    clock = _Clock()
    probe = _Probe(clock, [True] + [False] * 3)
    liveness = _liveness(clock, probe, tmp_path)
    liveness.engine_started()
    assert _run_checks(liveness, clock, 1) is None
    deadline = _deadline(tmp_path)
    # Every failing check takes its full timeout; none of them renews the file.
    assert _run_checks(liveness, clock, 3) is not None
    assert _deadline(tmp_path) == deadline
    # What is left is the time to kill the engine and start the next one.
    assert deadline - (clock.now + _BOOT) >= launcher.REPLACE_ALLOWANCE_SECONDS


@pytest.mark.unit
def test_a_new_engine_starts_a_fresh_startup_window(tmp_path):
    clock = _Clock()
    probe = _Probe(clock, [True, False, False])
    liveness = _liveness(clock, probe, tmp_path)
    liveness.engine_started()
    assert _run_checks(liveness, clock, 3) is None
    assert liveness.failures == 2
    clock.now += 100
    liveness.engine_started()
    assert not liveness.live and liveness.failures == 0
    assert _deadline(tmp_path) == int(clock.now + _BOOT + liveness.renewal)


@pytest.mark.unit
def test_the_launcher_counts_as_live_while_it_prepares(tmp_path):
    clock = _Clock()
    liveness = _liveness(clock, _Probe(clock, []), tmp_path, startup_timeout=600.0)
    liveness.launcher_starting()
    assert _deadline(tmp_path) == int(clock.now + _BOOT + 600)


@pytest.mark.unit
def test_a_liveness_file_that_cannot_be_written_fails_at_start_only(tmp_path, caplog):
    clock = _Clock()
    liveness = _liveness(clock, _Probe(clock, [True]), tmp_path / "missing")
    with pytest.raises(OSError):
        liveness.launcher_starting()
    # Later the engine is still supervised; the missed renewal is logged.
    liveness.engine_started()
    assert _run_checks(liveness, clock, 1) is None
    assert "cannot renew" in caplog.text


class _Status(http.server.BaseHTTPRequestHandler):
    """Answers /<status>; /slow sends its answer one byte every 50 ms."""

    def do_GET(self):
        self.server.paths.append(self.path)
        if self.path == "/slow":
            for byte in b"HTTP/1.1 200 OK\r\n\r\n":
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.05)
            return
        self.send_response(int(self.path.split("?")[0].strip("/")))
        self.end_headers()

    def log_message(self, *_args):
        pass


@pytest.fixture
def status_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _Status)
    server.paths = []
    thread = threading.Thread(target=server.serve_forever, args=(0.01,))
    thread.start()
    yield server
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_probe_counts_answers_as_a_kubelet_does(status_server, engine_port):
    url = f"http://127.0.0.1:{status_server.server_port}"
    assert launcher.probe_http(f"{url}/200?full=true", 5.0) is None
    assert status_server.paths[-1] == "/200?full=true"
    assert launcher.probe_http(f"{url}/302", 5.0) is None
    assert launcher.probe_http(f"{url}/503", 5.0) == "HTTP 503"
    # The timeout bounds the whole exchange, not each read.
    assert launcher.probe_http(f"{url}/slow", 0.2) == "timed out"
    closed = f"http://127.0.0.1:{engine_port}/live"
    assert launcher.probe_http(closed, 5.0) == "Connection refused"


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        ["--liveness-file=live"],  # needs --liveness-url
        ["--liveness-url=https://127.0.0.1:9090/live"],
        ["--liveness-url=127.0.0.1:9090/live"],
        ["--liveness-url=http://127.0.0.1:port/live"],
        ["--liveness-period=0"],
        ["--liveness-period=nan"],
        ["--liveness-timeout=-1"],
        ["--startup-timeout=inf"],
        ["--liveness-failure-threshold=0"],
    ],
)
def test_liveness_options_are_checked_when_parsed(options):
    with pytest.raises(SystemExit):
        launcher.parse_args([*options, "--", "-x"])
