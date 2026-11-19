# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start ``dynamo.vllm`` engines by fork from a pre-imported parent.

Run as the engine container's main process::

    python -m dynamo.vllm.launcher [--restart-min-uptime SECONDS] -- <dynamo.vllm arguments>

An engine spends tens of seconds importing torch, vLLM and their dependencies
before it loads anything. The launcher imports them once, stays free of CUDA
and threads, and forks each engine from that state, so an engine that replaces
a dead one skips the imports. When an import leaves the process unsafe to fork,
it runs a plain ``python -m dynamo.vllm`` instead.

The launcher stays alive while engines come and go: it forwards SIGTERM and
SIGINT to the engine, adopts and reaps the engine's orphaned processes, and
starts a replacement when an engine that ran for a while fails. An engine that
exits early, with status 0, or after a shutdown signal ends the launcher with
its exit status. An engine that hangs is not detected.
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
import sys
import time
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


def exec_plain_start(engine_args: Sequence[str]) -> None:
    """Replace this process with ``python -m dynamo.vllm`` (same PID)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-m", ENGINE_MODULE, *engine_args])


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
    signal.pthread_sigmask(signal.SIG_UNBLOCK, SHUTDOWN_SIGNALS)
    sys.argv = [sys.argv[0], *engine_args]
    runpy.run_module(ENGINE_MODULE, run_name="__main__", alter_sys=True)
    sys.exit(0)


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
    """Forks engines and keeps one running until shutdown or an early exit."""

    def __init__(self, engine_args: Sequence[str], restart_min_uptime: float):
        self.engine_args = list(engine_args)
        self.restart_min_uptime = restart_min_uptime
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
            _run_engine(self.engine_args)
        return pid

    def _wait_for_engine(self, engine_pid: int) -> int:
        while True:
            pid, status = os.waitpid(-1, 0)
            if pid == engine_pid:
                self.engine_pid = None
                return _exit_code(status)
            logger.debug("Launcher: reaped orphaned engine process %d", pid)

    def run(self) -> int:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")
        # An inherited "ignore" would discard the engine's exit status.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        for signum in SHUTDOWN_SIGNALS:
            signal.signal(signum, self._forward)
        while True:
            # A shutdown signal is held from this check until the engine's pid
            # is known, so it ends the launcher here or reaches the engine.
            signal.pthread_sigmask(signal.SIG_BLOCK, SHUTDOWN_SIGNALS)
            if self.stop_signal is not None:
                return 128 + self.stop_signal
            problems = fork_hazards()
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
            logger.info("Launcher: started engine (pid %d)", pid)
            code = self._wait_for_engine(pid)
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
        usage="%(prog)s [--restart-min-uptime SECONDS] -- <dynamo.vllm arguments>",
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
    args = parser.parse_args(argv[:split])
    if not math.isfinite(args.restart_min_uptime) or args.restart_min_uptime < 0:
        parser.error("--restart-min-uptime must be a non-negative number of seconds")
    return args, argv[split + 1 :]


def main(argv: Sequence[str] | None = None) -> None:
    args, engine_args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # A container's main process has no default signal actions. Until an engine
    # runs there is nothing to shut down, so a shutdown signal ends the launcher.
    for signum in SHUTDOWN_SIGNALS:
        signal.signal(signum, _exit_on_signal)
    problems = prepare(preimport_steps())
    if problems:
        logger.warning(
            "Launcher: not fork-safe %s; starting dynamo.vllm without pre-import",
            "; ".join(problems),
        )
        exec_plain_start(engine_args)
    sys.exit(Supervisor(engine_args, args.restart_min_uptime).run())


if __name__ == "__main__":
    main()
