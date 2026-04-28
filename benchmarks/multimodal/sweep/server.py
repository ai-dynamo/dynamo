# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


# Shutdown protocol timings.
#
# dynamo_serve.sh's cleanup() loops up to 300 s waiting for both nsys
# instances to finalize; vllm_serve.sh has the same 300 s budget for its
# single nsys. Add a 30 s margin for python's
# graceful_shutdown_with_discovery (5 s grace + drain + runtime.shutdown),
# which runs in parallel with nsys finalize.
_GRACEFUL_STOP_TIMEOUT_SECS = 330

# After the wrapper exits, let etcd revoke the primary lease (TTL = 10 s in
# lib/runtime/src/transports/etcd.rs) before the next config starts. Without
# this drain, the next frontend can observe stale v1/instances and v1/mdc
# entries from the prior backend → readiness false-positive and routing to
# dead TCP ports.
_LEASE_DRAIN_SECS = 12


class ServerManager:
    """Manages the lifecycle of a serving backend launched via a bash script.

    Uses ``setsid`` so the server gets its own process group, allowing clean
    shutdown without killing the orchestrator.
    """

    def __init__(self, port: int = 8000, timeout: int = 600) -> None:
        self.port = port
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        workflow_script: str,
        model: str,
        extra_args: Optional[List[str]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        """Launch the workflow script and block until the model is served."""
        if self.is_running:
            raise RuntimeError("Server is already running. Call stop() first.")

        script = Path(workflow_script)
        if not script.is_file():
            raise FileNotFoundError(f"Workflow script not found: {script}")

        model_flag = "--model-path" if "trtllm" in str(script) else "--model"
        cmd = ["bash", str(script), model_flag, model]
        if extra_args:
            cmd.extend(extra_args)

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)

        print(f"Launching: {' '.join(cmd)}", flush=True)
        self._process = subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
        )

        self.wait_for_ready(model)

    def wait_for_ready(self, model: str) -> None:
        """Poll /v1/models until the expected model name appears."""
        import urllib.error
        import urllib.request

        url = f"http://localhost:{self.port}/v1/models"
        deadline = time.monotonic() + self.timeout

        print(
            f"Waiting for server at {url} to list model '{model}' "
            f"(timeout: {self.timeout}s)...",
            flush=True,
        )

        while time.monotonic() < deadline:
            rc = self._process.poll() if self._process is not None else None
            if rc is not None:
                raise RuntimeError(
                    f"Workflow script exited with code {rc} before server ready; "
                    "see stderr above."
                )
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read().decode()
                    if model in body:
                        print("Server is ready (model registered).", flush=True)
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(5)

        self.stop()
        raise TimeoutError(f"Server did not become ready within {self.timeout}s")

    def stop(self) -> None:
        """Stop the server, finalize nsys, and let the etcd lease drain.

        Protocol (matches the trap design in
        benchmarks/multimodal/sweep/workflows/{vllm,dynamo}_serve.sh):

        1. SIGINT directly to the wrapper script (NOT killpg). The wrapper's
           ``trap cleanup INT TERM`` then forwards SIGINT to each nsys —
           nsys's documented graceful-finalize signal. Broadcasting SIGTERM
           via killpg races the trap and may hit nsys with an undocumented
           direct SIGTERM, skipping finalize and leaving an unrecoverable
           .qdstrm.
        2. Wait up to ``_GRACEFUL_STOP_TIMEOUT_SECS`` for the wrapper to exit
           naturally. That covers python's graceful_shutdown_with_discovery
           + both nsys finalizes within the wrapper's own 300 s budget.
        3. On timeout, tree-kill: walk /proc to enumerate every descendant
           PID — including subprocesses that called setsid() and escaped the
           wrapper's pgid (vLLM's EngineCore). killpg alone leaves those
           orphaned, holding /dev/shm regions, GPU memory, and the etcd
           lease keepalive task. That orphan state is what poisons the next
           config's startup.
        4. Sleep ``_LEASE_DRAIN_SECS`` so etcd revokes the primary 10 s lease
           before the next config's frontend enumerates discovery.
        """
        if self._process is None:
            return

        pid = self._process.pid
        print(f"Stopping server (PID {pid})...", flush=True)

        # (1) Targeted SIGINT to the wrapper.
        try:
            self._process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

        # (2) Wait for the wrapper-driven cleanup to finish.
        try:
            self._process.wait(timeout=_GRACEFUL_STOP_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            # (3) Escalate.
            print(
                f"  graceful stop timed out after {_GRACEFUL_STOP_TIMEOUT_SECS}s; "
                f"tree-killing descendants",
                flush=True,
            )
            _tree_kill(pid)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        print(f"Server stopped (PID {pid}).", flush=True)
        self._process = None

        # (4) Lease drain.
        time.sleep(_LEASE_DRAIN_SECS)


def _read_ppid(pid: int) -> Optional[int]:
    """Return the parent PID of ``pid`` from /proc, or None if unreadable.

    /proc/<pid>/stat fields are space-separated, but field 2 (``comm``) is
    parenthesised and can contain spaces or parens. Parse by locating the
    final ')' and reading from there.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    close_idx = stat.rfind(")")
    if close_idx < 0:
        return None
    after = stat[close_idx + 1 :].split()
    if len(after) < 2:
        return None
    try:
        return int(after[1])
    except ValueError:
        return None


def _descendants(root_pid: int) -> List[int]:
    """Walk /proc to collect every descendant PID of ``root_pid``.

    Captures setsid'd subprocesses that have escaped the original pgid —
    those are unreachable via os.killpg() but still show up in /proc with
    their parent (or PID 1, after reparenting). Linux-only; on non-Linux
    callers should fall back to killpg/kill at the root.
    """
    if not os.path.isdir("/proc"):
        return []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []

    children: Dict[int, List[int]] = {}
    for entry in entries:
        if not entry.isdigit():
            continue
        pid_int = int(entry)
        ppid = _read_ppid(pid_int)
        if ppid is None:
            continue
        children.setdefault(ppid, []).append(pid_int)

    out: List[int] = []
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for c in children.get(p, []):
            out.append(c)
            stack.append(c)
    return out


def _tree_kill(root_pid: int) -> None:
    """SIGKILL ``root_pid`` and every descendant, regardless of pgid."""
    for child_pid in _descendants(root_pid):
        try:
            os.kill(child_pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    # Belt: also reap the pgid in case the /proc walk missed something.
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.kill(root_pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
