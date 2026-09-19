# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Optional


class ServerManager:
    """Manages the lifecycle of a serving backend launched via a bash script.

    Uses ``setsid`` so the server gets its own process group, allowing clean
    shutdown without killing the orchestrator.
    """

    def __init__(self, port: int = 8000, timeout: int = 600) -> None:
        self.port = port
        self.timeout = timeout
        self.terminate_timeout = 15.0
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
        env["DYN_HTTP_PORT"] = str(self.port)
        profiling = env.get("DYN_DISABLE_NSYS", "1") != "1"
        default_terminate_timeout = 300.0 if profiling else 15.0
        default_shutdown_grace = 150.0 if profiling else 10.0
        raw_terminate_timeout = env.get("DYN_SERVER_TERMINATE_TIMEOUT")
        raw_shutdown_grace = env.get("DYN_SERVER_SHUTDOWN_GRACE_SECONDS")
        self.terminate_timeout = float(
            raw_terminate_timeout or default_terminate_timeout
        )
        shutdown_grace = float(raw_shutdown_grace or default_shutdown_grace)
        if self.terminate_timeout <= 0:
            raise ValueError("DYN_SERVER_TERMINATE_TIMEOUT must be positive")
        if shutdown_grace <= 0:
            raise ValueError("DYN_SERVER_SHUTDOWN_GRACE_SECONDS must be positive")
        if self.terminate_timeout <= shutdown_grace:
            raise ValueError(
                "DYN_SERVER_TERMINATE_TIMEOUT must exceed "
                "DYN_SERVER_SHUTDOWN_GRACE_SECONDS"
            )

        print(f"Launching: {' '.join(cmd)}", flush=True)
        self._process = subprocess.Popen(
            cmd,
            start_new_session=True,
            env=env,
        )

        self.wait_for_ready(model)

    def wait_for_ready(self, model: str) -> None:
        """Poll /v1/models until the expected model name appears."""
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

    def validate_prefix_cache(
        self,
        model: str,
        user_text: str,
        min_cached_tokens: int,
        output_path: Path,
    ) -> dict[str, Any]:
        """Warm and verify the shared text prefix through chat completions."""
        if min_cached_tokens <= 0:
            raise ValueError("min_cached_tokens must be positive")

        url = f"http://localhost:{self.port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_text}],
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }

        def send() -> dict[str, Any]:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                raise RuntimeError(
                    f"prefix-cache probe failed with HTTP {exc.code}: {body}"
                ) from exc

        responses = [send(), send()]
        usages = [response.get("usage", {}) for response in responses]
        cached_tokens = (usages[1].get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        summary = {
            "minimum_cached_tokens": min_cached_tokens,
            "first_usage": usages[0],
            "second_usage": usages[1],
            "passed": cached_tokens >= min_cached_tokens,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if not summary["passed"]:
            raise RuntimeError(
                "prefix-cache probe cached "
                f"{cached_tokens} tokens; expected at least {min_cached_tokens}"
            )
        print(f"Prefix-cache probe passed: cached_tokens={cached_tokens}", flush=True)
        return summary

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
            self._process.wait(timeout=self.terminate_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self._process.wait(timeout=5)

        print(f"Server stopped (PID {pid}).", flush=True)
        self._process = None
        time.sleep(5)
