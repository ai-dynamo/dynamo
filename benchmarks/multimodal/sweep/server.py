# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional


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
        env_overrides: Optional[dict] = None,
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
        """Wait for model registration, then require one real inference."""
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
                        if self._chat_probe(model):
                            print(
                                "Server is ready (chat inference succeeded).",
                                flush=True,
                            )
                            return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(5)

        self.stop()
        raise TimeoutError(f"Server did not become ready within {self.timeout}s")

    def _chat_probe(self, model: str) -> bool:
        """Return true only after a minimal non-streaming generation succeeds."""
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "Reply ready."}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"http://localhost:{self.port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def stop(self) -> None:
        """Stop the server by killing its process group."""
        if self._process is None:
            return

        pid = self._process.pid
        print(f"Stopping server (PID {pid})...", flush=True)

        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                self._process.terminate()
            except (ProcessLookupError, PermissionError):
                pass

        try:
            self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self._process.wait(timeout=5)

        print(f"Server stopped (PID {pid}).", flush=True)
        self._process = None
        time.sleep(5)
