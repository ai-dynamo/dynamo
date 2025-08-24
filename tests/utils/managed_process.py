# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import psutil
import requests


def terminate_process(process, logger=logging.getLogger(), immediate_kill=False):
    try:
        logger.info("Terminating PID: %s name: %s", process.pid, process.name())
        if immediate_kill:
            logger.info("Sending Kill: %s %s", process.pid, process.name())
            process.kill()
        else:
            process.terminate()
    except psutil.AccessDenied:
        logger.warning("Access denied for PID %s", process.pid)
    except psutil.NoSuchProcess:
        logger.warning("PID %s no longer exists", process.pid)


def terminate_process_tree(
    pid, logger=logging.getLogger(), immediate_kill=False, timeout=10
):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            terminate_process(child, logger, immediate_kill)

        terminate_process(parent, logger, immediate_kill)

        for child in parent.children(recursive=True):
            try:
                child.wait(timeout)
            except psutil.TimeoutExpired:
                terminate_process(child, logger, immediate_kill=True)
        try:
            parent.wait(timeout)
        except psutil.TimeoutExpired:
            terminate_process(parent, logger, immediate_kill=True)

    except psutil.NoSuchProcess:
        # Process already terminated
        pass


@dataclass
class ManagedProcess:
    command: List[str]
    env: Optional[dict] = None
    health_check_ports: List[int] = field(default_factory=list)
    health_check_urls: List[Any] = field(default_factory=list)
    delayed_start: int = 0
    timeout: int = 300
    working_dir: Optional[str] = None
    display_output: bool = False
    data_dir: Optional[str] = None
    terminate_existing: bool = True
    stragglers: List[str] = field(default_factory=list)
    straggler_commands: List[str] = field(default_factory=list)
    log_dir: str = os.getcwd()

    _logger = logging.getLogger()
    _command_name = None
    _log_path = None
    _tee_proc = None
    _sed_proc = None

    def __enter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self._command_name = self.command[0]
            os.makedirs(self.log_dir, exist_ok=True)
            log_name = f"{self._command_name}.log.txt"
            self._log_path = os.path.join(self.log_dir, log_name)

            if self.data_dir:
                self._remove_directory(self.data_dir)

            self._terminate_existing()
            self._start_process()
            time.sleep(self.delayed_start)
            elapsed = self._check_ports(self.timeout)
            self._check_urls(self.timeout - elapsed)

            return self

        except Exception as e:
            self.__exit__(None, None, None)
            raise e

    def __exit__(self, exc_type, exc_val, exc_tb):
        process_list = [self.proc, self._tee_proc, self._sed_proc]
        for process in process_list:
            if process:
                if process.stdout:
                    process.stdout.close()
                if process.stdin:
                    process.stdin.close()
                terminate_process_tree(process.pid, self._logger)
                process.wait()
        if self.data_dir:
            self._remove_directory(self.data_dir)

        for ps_process in psutil.process_iter(["name", "cmdline"]):
            try:
                if ps_process.name() in self.stragglers:
                    self._logger.info(
                        "Terminating Straggler %s %s", ps_process.name(), ps_process.pid
                    )

                    terminate_process_tree(ps_process.pid, self._logger)
                for cmdline in self.straggler_commands:
                    if cmdline in " ".join(ps_process.cmdline()):
                        self._logger.info(
                            "Terminating Straggler Cmdline %s %s %s",
                            ps_process.name(),
                            ps_process.pid,
                            cmdline,
                        )
                        terminate_process_tree(ps_process.pid, self._logger)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have terminated or become inaccessible during iteration
                pass

    def _start_process(self):
        assert self._command_name
        assert self._log_path

        self._logger.info(
            "Running command: %s in %s",
            " ".join(self.command),
            self.working_dir or os.getcwd(),
        )

        stdin = subprocess.DEVNULL
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT

        if self.display_output:
            self.proc = subprocess.Popen(
                self.command,
                env=self.env or os.environ.copy(),
                cwd=self.working_dir,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,  # Isolate process group to prevent kill 0 from affecting parent
            )
            self._sed_proc = subprocess.Popen(
                ["sed", "-u", f"s/^/[{self._command_name.upper()}] /"],
                stdin=self.proc.stdout,
                stdout=subprocess.PIPE,
            )

            self._tee_proc = subprocess.Popen(
                ["tee", self._log_path], stdin=self._sed_proc.stdout
            )

        else:
            with open(self._log_path, "w", encoding="utf-8") as f:
                self.proc = subprocess.Popen(
                    self.command,
                    env=self.env or os.environ.copy(),
                    cwd=self.working_dir,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,  # Isolate process group to prevent kill 0 from affecting parent
                )

                self._sed_proc = subprocess.Popen(
                    ["sed", "-u", f"s/^/[{self._command_name.upper()}] /"],
                    stdin=self.proc.stdout,
                    stdout=f,
                )
            self._tee_proc = None

    def _remove_directory(self, path: str) -> None:
        """Remove a directory."""
        try:
            shutil.rmtree(path, ignore_errors=True)
        except (OSError, IOError) as e:
            self._logger.warning("Warning: Failed to remove directory %s: %s", path, e)

    def _log_tail_on_error(self, lines=20):
        """Print the last few lines of the log file when process dies."""
        if self._log_path and os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r") as f:
                    log_lines = f.readlines()
                    if log_lines:
                        self._logger.error(
                            "=== Last %d lines from %s ===",
                            min(lines, len(log_lines)),
                            self._log_path,
                        )
                        for line in log_lines[-lines:]:
                            self._logger.error(line.rstrip())
                        self._logger.error("=== End of log tail ===")
            except Exception as e:
                self._logger.warning("Could not read log file: %s", e)

    def _check_process_alive(self, context=""):
        """Check if the main process is still alive. Raises RuntimeError if dead."""
        if self.proc and self.proc.poll() is not None:
            returncode = self.proc.returncode
            self._logger.error(
                "Main server process died with exit code %d%s",
                returncode,
                f" {context}" if context else "",
            )
            # Try to get last few lines from log for debugging
            self._log_tail_on_error()
            raise RuntimeError(
                f"Main server process exited with code {returncode}{f' {context}' if context else ''}"
            )

    def _check_ports(self, timeout):
        elapsed = 0.0
        for port in self.health_check_ports:
            elapsed += self._check_port(port, timeout - elapsed)
        return elapsed

    def _check_port(self, port, timeout=30, sleep=0.1):
        """Check if a port is open on localhost."""
        start_time = time.time()
        self._logger.info("Checking Port: %s", port)
        elapsed = 0.0
        while elapsed < timeout:
            # Check if the main process is still alive
            self._check_process_alive(f"while waiting for port {port}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) == 0:
                    self._logger.info("SUCCESS: Check Port: %s", port)
                    return time.time() - start_time
            time.sleep(sleep)
            elapsed = time.time() - start_time
        self._logger.error("FAILED: Check Port: %s", port)
        raise RuntimeError("FAILED: Check Port: %s" % port)

    def _check_urls(self, timeout):
        elapsed = 0.0
        for url in self.health_check_urls:
            elapsed += self._check_url(url, timeout - elapsed)
        return elapsed

    def _check_url(self, url, timeout=30, sleep=1, log_interval=10):
        if isinstance(url, tuple):
            response_check = url[1]
            url = url[0]
        else:
            response_check = None
        start_time = time.time()
        self._logger.info("Checking URL %s", url)
        elapsed = 0.0
        attempt = 0
        last_log_time = 0.0

        while elapsed < timeout:
            self._check_process_alive("while waiting for health check")

            attempt += 1
            check_failed = False
            failure_reason = None

            try:
                response = requests.get(url, timeout=timeout - elapsed)
                if response.status_code == 200:
                    if response_check is None or response_check(response):
                        # Try to format JSON response nicely, otherwise show raw text
                        try:
                            response_data = response.json()
                            response_str = json.dumps(response_data, indent=2)
                            self._logger.info(
                                "SUCCESS: Check URL: %s (attempt=%d, elapsed=%.1fs)\nResponse:\n%s",
                                url,
                                attempt,
                                elapsed,
                                response_str,
                            )
                        except (json.JSONDecodeError, Exception):
                            # If not JSON or any error, show raw text (truncated if too long)
                            response_text = response.text
                            if len(response_text) > 500:
                                response_text = response_text[:500] + "... (truncated)"
                            self._logger.info(
                                "SUCCESS: Check URL: %s (attempt=%d, elapsed=%.1fs)\nResponse: %s",
                                url,
                                attempt,
                                elapsed,
                                response_text,
                            )
                        return time.time() - start_time
                    else:
                        check_failed = True
                        failure_reason = "custom check failed"
                else:
                    check_failed = True
                    failure_reason = f"status code {response.status_code}"
            except requests.RequestException as e:
                check_failed = True
                failure_reason = f"request exception: {e}"

            # Log progress every log_interval seconds for any failure
            if check_failed and elapsed - last_log_time >= log_interval:
                self._logger.info(
                    "Still waiting for URL %s (%s) (attempt=%d, elapsed=%.1fs)",
                    url,
                    failure_reason,
                    attempt,
                    elapsed,
                )
                last_log_time = elapsed

            time.sleep(sleep)
            elapsed = time.time() - start_time

        self._logger.error(
            "FAILED: Check URL: %s (attempts=%d, elapsed=%.1fs)", url, attempt, elapsed
        )
        raise RuntimeError("FAILED: Check URL: %s" % url)

    def _terminate_existing(self):
        if self.terminate_existing:
            for proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    if (
                        proc.name() == self._command_name
                        or proc.name() in self.stragglers
                    ):
                        self._logger.info(
                            "Terminating Existing %s %s", proc.name(), proc.pid
                        )

                        terminate_process_tree(proc.pid, self._logger)
                    for cmdline in self.straggler_commands:
                        if cmdline in " ".join(proc.cmdline()):
                            self._logger.info(
                                "Terminating Existing CmdLine %s %s %s",
                                proc.name(),
                                proc.pid,
                                proc.cmdline(),
                            )
                            terminate_process_tree(proc.pid, self._logger)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    # Process may have terminated or become inaccessible during iteration
                    pass


def main():
    with ManagedProcess(
        command=[
            "dynamo",
            "run",
            "in=http",
            "out=vllm",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        ],
        display_output=True,
        terminate_existing=True,
        health_check_ports=[8080],
        health_check_urls=["http://localhost:8080/v1/models"],
        timeout=10,
    ):
        time.sleep(60)
        pass


if __name__ == "__main__":
    main()
