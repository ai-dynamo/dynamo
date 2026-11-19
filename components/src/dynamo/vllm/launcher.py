# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start ``dynamo.vllm`` engines by fork from a pre-imported parent.

Run as the engine container's main process::

    python -m dynamo.vllm.launcher [options] -- <dynamo.vllm arguments>

An engine spends tens of seconds importing torch, vLLM and their dependencies
before it loads anything. The launcher imports them once, stays free of CUDA
and threads, and forks each engine from that state, so an engine that replaces
a dead one skips the imports. When an import leaves the process unsafe to fork,
it restarts itself without the imports and starts each engine in a new
interpreter instead.

The launcher stays alive while engines come and go: it forwards SIGTERM and
SIGINT to the engine, adopts and reaps the engine's orphaned processes, and
starts a replacement when an engine that ran for a while fails or hangs. An
engine that exits early, with status 0, or after a shutdown signal ends the
launcher with its exit status.
"""

import argparse
import atexit
import ctypes
import gc
import importlib
import logging
import math
import os
import runpy
import signal
import socket
import sys
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Sequence

from dynamo.vllm import fork_safety

logger = logging.getLogger(__name__)

ENGINE_MODULE = "dynamo.vllm"
# What the engine's API process and vLLM's EngineCore import. GPU worker modules
# and vLLM plugins are left to the processes that load them: in vLLM 0.33 some
# query the device when imported (for example the FP8 kernel support checks).
ENGINE_MODULES = (
    "torch",
    "vllm",
    "vllm.v1.engine.async_llm",
    "vllm.v1.engine.core",
    "vllm.v1.executor.multiproc_executor",
)
NVIDIA_DEVICE_PREFIX = "/dev/nvidia"
# The OS thread of a Python thread that has finished exits a moment later.
OS_THREAD_EXIT_SECONDS = 1.0
SHUTDOWN_SIGNALS = (signal.SIGTERM, signal.SIGINT)
# How long killing a hung engine and starting its replacement may take before
# the launcher stops counting as live.
REPLACE_ALLOWANCE_SECONDS = 10.0
_PR_SET_CHILD_SUBREAPER = 36


def _nvidia_fds() -> int:
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            count += os.readlink(f"/proc/self/fd/{fd}").startswith(NVIDIA_DEVICE_PREFIX)
        except OSError:  # the listing's own descriptor
            continue
    return count


def fork_hazards() -> list[str]:
    """Why a child forked now could not use the GPU or could deadlock.

    A child of a process with a CUDA context or open driver descriptors cannot
    use the GPU; a child keeps only the forking thread, so a lock any other
    thread (Python or native, such as a driver thread) holds stays held.
    """
    problems = []
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        problems.append("torch CUDA is initialized")
    nvidia_fds = _nvidia_fds()
    if nvidia_fds:
        problems.append(f"{nvidia_fds} {NVIDIA_DEVICE_PREFIX}* descriptors are open")
    threads = fork_safety.os_thread_names()
    if len(threads) > 1:
        names = [f"{n} x{c}" for n, c in sorted(Counter(threads).items())]
        names += [f"Python thread {n!r}" for n in fork_safety.other_python_threads()]
        problems.append(f"{len(threads)} threads ({', '.join(names)})")
    return problems


def _freeze_gc() -> None:
    # Engines collect garbage during start-up; frozen objects are skipped, so
    # they do not rescan the launcher's heap, and its pages stay shared.
    gc.collect()
    gc.freeze()
    # Extension modules report objects that are still frozen at exit as leaks.
    atexit.register(gc.unfreeze)


def preimport_steps() -> list[tuple[str, Callable]]:
    steps: list[tuple[str, Callable]] = [
        (module, lambda module=module: importlib.import_module(module))
        for module in ENGINE_MODULES
    ]
    steps.append(("gc freeze", _freeze_gc))
    return steps


def _settle_threads() -> None:
    """Wait for threads that an import started and that finish on their own."""
    if not fork_safety.other_python_threads():
        return
    if fork_safety.wait_for_python_threads(fork_safety.THREAD_SETTLE_SECONDS):
        return
    deadline = time.monotonic() + OS_THREAD_EXIT_SECONDS
    while len(fork_safety.os_thread_names()) > 1 and time.monotonic() < deadline:
        time.sleep(0.01)


def prepare(steps: Sequence[tuple[str, Callable]]) -> list[str]:
    """Run the pre-import steps; return why forking is unsafe (empty: safe).

    Stops at the first step after which the process is no longer fork-safe.
    """
    for name, step in steps:
        start = time.monotonic()
        step()
        _settle_threads()
        problems = fork_hazards()
        logger.info("Launcher: %s in %.2f s", name, time.monotonic() - start)
        if problems:
            return [f"after {name}: {problem}" for problem in problems]
    return []


def _restart_without_preimport(argv: Sequence[str]) -> None:
    """Replace this process with a launcher that skips the pre-import.

    A fresh interpreter sheds whatever the unsafe import left behind (driver
    handles, threads, memory). It starts each engine as a new interpreter.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    # Keep how the launcher was invoked (interpreter flags, -m or a script)
    # when its own arguments end the command line.
    if argv and sys.orig_argv[-len(argv) :] == list(argv):
        command = [*sys.orig_argv[: -len(argv)], "--no-preimport", *argv]
    else:
        command = [
            sys.executable,
            "-m",
            f"{ENGINE_MODULE}.launcher",
            "--no-preimport",
            *argv,
        ]
    os.execv(sys.executable, command)


def _exit_code(status: int) -> int:
    code = os.waitstatus_to_exitcode(status)
    return 128 - code if code < 0 else code


def _run_engine(engine_args: Sequence[str]) -> None:
    """In the forked child: become the engine, as ``python -m dynamo.vllm`` is.

    Does not return. The engine's SystemExit or error leaves through the
    interpreter's normal exit, so its exit handlers and finalizers run.
    """
    # Own session, so the launcher can kill the engine's whole tree.
    os.setsid()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.pthread_sigmask(signal.SIG_SETMASK, ())
    sys.argv = [sys.argv[0], *engine_args]
    runpy.run_module(ENGINE_MODULE, run_name="__main__", alter_sys=True)
    sys.exit(0)


def _exec_engine(engine_args: Sequence[str]) -> None:
    """In the forked child: become a new ``python -m dynamo.vllm``.

    Does not return. Nothing but these calls runs between fork and exec, so
    the launcher need not be fork-safe to start an engine this way.
    """
    try:
        os.setsid()
        for signum in SHUTDOWN_SIGNALS:
            signal.signal(signum, signal.SIG_DFL)
        signal.pthread_sigmask(signal.SIG_SETMASK, ())
        os.execv(sys.executable, [sys.executable, "-m", ENGINE_MODULE, *engine_args])
    except OSError as exc:
        print(f"Launcher: cannot start the engine: {exc}", file=sys.stderr, flush=True)
    finally:
        # Only reached when the exec failed.
        os._exit(127)


def _uptime() -> float:
    """Seconds since boot. Unlike the wall clock, it is never stepped."""
    with open("/proc/uptime") as uptime:
        return float(uptime.read().split()[0])


def probe_http(url: str, timeout: float) -> str | None:
    """One HTTP GET; None on a 2xx or 3xx answer, else why the check failed.

    Counts answers as a kubelet probe does and, like it, bounds the whole
    exchange by ``timeout``. No proxy is consulted: the engine listens in the
    same pod.
    """
    parts = urllib.parse.urlsplit(url)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    request = (
        f"GET {target} HTTP/1.1\r\nHost: {parts.netloc}\r\nConnection: close\r\n\r\n"
    )
    deadline = time.monotonic() + timeout
    status_line = b""
    try:
        address = (parts.hostname, parts.port or 80)
        with socket.create_connection(address, timeout=timeout) as sock:
            sock.sendall(request.encode())
            while b"\r\n" not in status_line:
                sock.settimeout(max(deadline - time.monotonic(), 0.001))
                chunk = sock.recv(256)
                if not chunk:
                    break
                status_line += chunk
    except TimeoutError:
        return "timed out"
    except OSError as exc:
        return exc.strerror or str(exc)
    fields = status_line.split(None, 2)
    if len(fields) < 2 or not fields[0].startswith(b"HTTP/") or not fields[1].isdigit():
        return "no HTTP answer"
    status = int(fields[1])
    return None if 200 <= status < 400 else f"HTTP {status}"


class EngineLiveness:
    """Decides when a running engine is hung, the way kubelet probes would.

    Until the engine first answers its liveness endpoint it is starting and is
    given ``startup_timeout`` seconds; after that, ``failure_threshold`` failed
    checks in a row mean it is hung. A parked shadow answers ``/live``, so it is
    live.

    While the launcher prepares, or the engine is starting or live, the
    deadline in ``liveness_file`` is renewed for an exec probe, as an uptime
    in seconds. It outlasts the launcher's own hang detection, so it expires
    only when the launcher stops running or cannot renew it.
    """

    def __init__(
        self,
        url: str,
        period: float,
        timeout: float,
        failure_threshold: int,
        startup_timeout: float,
        liveness_file: str | None = None,
        probe: Callable[[str, float], str | None] = probe_http,
        clock: Callable[[], float] = time.monotonic,
        uptime: Callable[[], float] = _uptime,
    ):
        self.url = url
        self.period = period
        self.timeout = timeout
        self.failure_threshold = failure_threshold
        self.startup_timeout = startup_timeout
        self.liveness_file = liveness_file
        # The slowest hang detection (every check timing out), then the time to
        # kill the engine and start the next one.
        self.renewal = failure_threshold * (period + timeout) + max(
            2 * period, REPLACE_ALLOWANCE_SECONDS
        )
        self._probe = probe
        self._clock = clock
        self._uptime = uptime
        self.started = clock()
        self.next_check = self.started + period
        self.live = False
        self.failures = 0

    def launcher_starting(self) -> None:
        """Count as live while the launcher prepares, as long as an engine may.

        Kubelet may read the file before the first engine starts. Raises
        OSError if the file cannot be written.
        """
        if self.liveness_file is not None:
            self._write(self.liveness_file, self.startup_timeout)

    def engine_started(self) -> None:
        self.started = self._clock()
        self.next_check = self.started + self.period
        self.live = False
        self.failures = 0
        self._renew()

    def seconds_to_next_check(self) -> float:
        return max(0.0, self.next_check - self._clock())

    def hung(self) -> str | None:
        """Run a check if one is due; return why the engine is hung, or None."""
        if self._clock() < self.next_check:
            return None
        failure = self._probe(self.url, self.timeout)
        now = self._clock()
        self.next_check = now + self.period
        if failure is None:
            self.live, self.failures = True, 0
        elif self.live:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                return f"{self.failures} liveness check(s) failed in a row, last: {failure}"
            return None
        elif now - self.started >= self.startup_timeout:
            return (
                f"not live {now - self.started:.0f} s after it started, last: {failure}"
            )
        self._renew()
        return None

    def _renew(self) -> None:
        if self.liveness_file is None:
            return
        try:
            self._write(self.liveness_file, self.renewal)
        except OSError as exc:
            # Supervision goes on; the deadline running out is the escalation.
            logger.error("Launcher: cannot renew %s: %s", self.liveness_file, exc)

    def _write(self, path: str, seconds_ahead: float) -> None:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(f"{math.ceil(self._uptime() + seconds_ahead)}\n")
        os.replace(tmp, path)


def _child_pids() -> list[int]:
    me = os.getpid()
    children = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            # Bytes: a command name need not be valid UTF-8.
            with open(f"/proc/{entry}/stat", "rb") as stat:
                # The command name may contain spaces; fields resume after ")".
                fields = stat.read().rsplit(b")", 1)[1].split()
        except OSError:  # exited since the listing
            continue
        if int(fields[1]) == me:
            children.append(int(entry))
    return children


def _kill_leftovers(engine_pid: int) -> None:
    """Kill and reap what is left of a dead engine: its group and any orphan.

    The launcher is a child subreaper, so every orphaned engine process is
    re-parented to it; killing until it has no children leaves nothing behind.
    """
    try:
        os.killpg(engine_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    while True:
        for pid in _child_pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return


def _exit_on_signal(signum: int, _frame) -> None:
    sys.exit(128 + signum)


class Supervisor:
    """Starts engines and keeps one running until shutdown or an early exit."""

    def __init__(
        self,
        engine_args: Sequence[str],
        restart_min_uptime: float,
        preimported: bool = True,
        liveness: EngineLiveness | None = None,
    ):
        self.engine_args = list(engine_args)
        self.restart_min_uptime = restart_min_uptime
        self.preimported = preimported
        self.liveness = liveness
        self.engine_pid: int | None = None
        self.stop_signal: int | None = None

    def _forward(self, signum: int, _frame) -> None:
        self.stop_signal = signum
        if self.engine_pid is not None:
            # Only the engine's main process, as Kubernetes signals a container:
            # the engine shuts its own processes down in order.
            try:
                os.kill(self.engine_pid, signum)
            except ProcessLookupError:
                pass

    def _start_engine(self) -> int:
        # Output still buffered here would be written again by every engine.
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            if self.preimported:
                _run_engine(self.engine_args)
            _exec_engine(self.engine_args)
        return pid

    def _wait_for_engine(self, engine_pid: int) -> int | None:
        """Wait for the engine to exit, killing it if it hangs.

        Returns its exit status, or None if it did not exit after the kill.
        """
        kill_deadline = None
        while True:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == engine_pid:
                self.engine_pid = None
                return _exit_code(status)
            if pid:
                logger.debug("Launcher: reaped orphaned engine process %d", pid)
                continue
            if kill_deadline is not None:
                remaining = kill_deadline - time.monotonic()
                if remaining <= 0:
                    return None
                signal.sigtimedwait({signal.SIGCHLD}, remaining)
                continue
            # An engine shutting down may stop answering; it is not hung.
            if self.liveness is None or self.stop_signal is not None:
                signal.sigwaitinfo({signal.SIGCHLD})
                continue
            reason = self.liveness.hung()
            # The check takes time: a shutdown signal may have arrived during it.
            if reason is not None and self.stop_signal is None:
                logger.error(
                    "Launcher: engine (pid %d) is hung (%s); killing it",
                    engine_pid,
                    reason,
                )
                try:
                    os.killpg(engine_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                kill_deadline = time.monotonic() + REPLACE_ALLOWANCE_SECONDS
                continue
            signal.sigtimedwait({signal.SIGCHLD}, self.liveness.seconds_to_next_check())

    def run(self) -> int:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")
        # An inherited "ignore" would discard the engine's exit status.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        # Taken with sigtimedwait in _wait_for_engine.
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGCHLD})
        for signum in SHUTDOWN_SIGNALS:
            signal.signal(signum, self._forward)
        while True:
            # A shutdown signal is held from this check until the engine's pid
            # is known, so it ends the launcher here or reaches the engine.
            signal.pthread_sigmask(signal.SIG_BLOCK, SHUTDOWN_SIGNALS)
            if self.stop_signal is not None:
                return 128 + self.stop_signal
            # An engine started as a new interpreter does not inherit this state.
            problems = fork_hazards() if self.preimported else []
            if problems:
                logger.error(
                    "Launcher: no longer safe to fork (%s); exiting",
                    "; ".join(problems),
                )
                return 1
            started = time.monotonic()
            pid = self._start_engine()
            self.engine_pid = pid
            signal.pthread_sigmask(signal.SIG_UNBLOCK, SHUTDOWN_SIGNALS)
            if self.liveness is not None:
                self.liveness.engine_started()
            logger.info("Launcher: started engine (pid %d)", pid)
            code = self._wait_for_engine(pid)
            if code is None:
                # The liveness file is no longer renewed either.
                logger.error(
                    "Launcher: engine (pid %d) did not exit within %.0f s of "
                    "SIGKILL; exiting so that the container restarts",
                    pid,
                    REPLACE_ALLOWANCE_SECONDS,
                )
                return 1
            uptime = time.monotonic() - started
            _kill_leftovers(pid)
            # An engine that ends on its own (status 0) is done, not failed.
            if (
                self.stop_signal is not None
                or code == 0
                or uptime < self.restart_min_uptime
            ):
                logger.info(
                    "Launcher: engine (pid %d) exited with %d after %.1f s; exiting",
                    pid,
                    code,
                    uptime,
                )
                return code
            logger.warning(
                "Launcher: engine (pid %d) exited with %d after %.1f s; "
                "starting a replacement",
                pid,
                code,
                uptime,
            )


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    argv = list(argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.vllm.launcher",
        usage="%(prog)s [options] -- <dynamo.vllm arguments>",
        description=__doc__.split("\n\n")[0],
        allow_abbrev=False,
    )
    parser.add_argument(
        "--restart-min-uptime",
        type=float,
        default=60.0,
        help="Replace an engine that fails after running at least this many "
        "seconds; an engine that exits sooner ends the launcher with its status "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--no-preimport",
        action="store_true",
        help="Start each engine in a new interpreter instead of forking it from "
        "pre-imported modules. Used automatically when pre-import is not fork-safe.",
    )
    parser.add_argument(
        "--liveness-url",
        help="The engine's liveness endpoint, e.g. http://127.0.0.1:9090/live. "
        "When set, an engine that stops answering it is killed and handled like "
        "a failed engine.",
    )
    parser.add_argument(
        "--liveness-period",
        type=float,
        default=5.0,
        help="Seconds between checks (default: %(default)s).",
    )
    parser.add_argument(
        "--liveness-timeout",
        type=float,
        default=4.0,
        help="Seconds a check may take (default: %(default)s).",
    )
    parser.add_argument(
        "--liveness-failure-threshold",
        type=int,
        default=3,
        help="Failed checks in a row after which a live engine counts as hung "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=7200.0,
        help="Seconds an engine may take to first answer --liveness-url "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--liveness-file",
        help="Keep the uptime (seconds since boot, as in /proc/uptime) until which "
        "the launcher counts as live in this file, for an exec liveness probe. "
        "Requires --liveness-url.",
    )
    args = parser.parse_args(argv[:split])
    if not math.isfinite(args.restart_min_uptime) or args.restart_min_uptime < 0:
        parser.error("--restart-min-uptime must be a non-negative number of seconds")
    for option in ("liveness_period", "liveness_timeout", "startup_timeout"):
        value = getattr(args, option)
        if not math.isfinite(value) or value <= 0:
            flag = "--" + option.replace("_", "-")
            parser.error(f"{flag} must be a positive number of seconds")
    if args.liveness_failure_threshold < 1:
        parser.error("--liveness-failure-threshold must be at least 1")
    if args.liveness_file and not args.liveness_url:
        parser.error("--liveness-file requires --liveness-url")
    if args.liveness_url and not _is_http_url(args.liveness_url):
        parser.error("--liveness-url must be an http:// URL with a host")
    return args, argv[split + 1 :]


def _is_http_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    try:
        parts.port
    except ValueError:  # not a port number
        return False
    return parts.scheme == "http" and bool(parts.hostname)


def main(argv: Sequence[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    args, engine_args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # A container's main process has no default signal actions. Until an engine
    # runs there is nothing to shut down, so a shutdown signal ends the launcher.
    for signum in SHUTDOWN_SIGNALS:
        signal.signal(signum, _exit_on_signal)
    liveness = None
    if args.liveness_url:
        liveness = EngineLiveness(
            args.liveness_url,
            args.liveness_period,
            args.liveness_timeout,
            args.liveness_failure_threshold,
            args.startup_timeout,
            args.liveness_file,
        )
        liveness.launcher_starting()
    if not args.no_preimport:
        problems = prepare(preimport_steps())
        if problems:
            logger.warning(
                "Launcher: not fork-safe %s; restarting without pre-import",
                "; ".join(problems),
            )
            _restart_without_preimport(argv)
    supervisor = Supervisor(
        engine_args,
        args.restart_min_uptime,
        preimported=not args.no_preimport,
        liveness=liveness,
    )
    sys.exit(supervisor.run())


if __name__ == "__main__":
    main()
