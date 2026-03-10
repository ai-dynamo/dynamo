# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import psutil


class ServerManager:
    """Manages the lifecycle of a serving backend launched via a bash script.

    Uses ``setsid`` so the server gets its own process group, allowing clean
    shutdown without killing the orchestrator.

    Cleanup follows the pattern from PR #7122: snapshot child processes before
    killing the parent, then explicitly kill any orphaned children that escaped
    the process group (e.g. TRT-LLM engine workers started via MPI in separate
    PGIDs).
    """

    def __init__(self, port: int = 8000, timeout: int = 600) -> None:
        self.port = port
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[open] = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(
        self,
        workflow_script: str,
        model: str,
        extra_args: Optional[List[str]] = None,
        env_overrides: Optional[dict] = None,
        log_dir: Optional[Path] = None,
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

        stdout_target = subprocess.DEVNULL
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "server.log"
            print(f"Server logs -> {log_path}", flush=True)
            self._log_file = open(log_path, "w")
            stdout_target = self._log_file

        print(f"Launching: {' '.join(cmd)}", flush=True)
        self._process = subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
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
            if not self.is_running:
                raise RuntimeError(
                    "Server process exited unexpectedly during startup "
                    f"(exit code {self._process.returncode})."
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
        """Stop the server by killing its process group and orphaned children.

        TRT-LLM spawns engine workers via MPI in separate process groups.
        os.killpg() only reaches the parent's PGID, leaving MPI workers alive
        with GPU memory allocated. We snapshot children before killing the
        parent, then explicitly kill any survivors (PR #7122 pattern).
        """
        if self._process is None:
            return

        pid = self._process.pid
        print(f"Stopping server (PID {pid})...", flush=True)

        # 1. Snapshot all child processes BEFORE killing the parent.
        #    MPI workers run in separate PGIDs — after the parent dies they
        #    become orphans reparented to init, invisible via parent PID.
        orphan_candidates = []
        try:
            parent = psutil.Process(pid)
            orphan_candidates = parent.children(recursive=True)
            print(
                f"  Snapshotted {len(orphan_candidates)} child processes",
                flush=True,
            )
        except psutil.NoSuchProcess:
            pass

        # 2. SIGTERM the process group
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                self._process.terminate()
            except (ProcessLookupError, PermissionError):
                pass

        # 3. Wait for graceful shutdown
        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            # 4. Escalate to SIGKILL
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        # 5. Kill any orphaned child processes that survived the process
        #    group kill (MPI workers, engine cores in different PGIDs).
        killed_orphans = 0
        for child in orphan_candidates:
            try:
                if child.is_running():
                    print(
                        f"  Killing orphaned child: PID {child.pid} "
                        f"name={child.name()}",
                        flush=True,
                    )
                    child.kill()
                    child.wait(timeout=5)
                    killed_orphans += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
        if killed_orphans:
            print(f"  Killed {killed_orphans} orphaned child processes", flush=True)

        print(f"Server stopped (PID {pid}).", flush=True)
        self._process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

        # Wait for GPU memory to be released
        time.sleep(10)
