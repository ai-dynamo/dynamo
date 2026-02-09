# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import psutil
import requests

from tests.utils.constants import DefaultPort
from tests.utils.port_utils import allocate_port, deallocate_port


def terminate_process(process, logger=logging.getLogger(), immediate_kill=False):
    """Terminate a single process.

    Kill Strategy:
    - immediate_kill=False: Send SIGTERM (graceful, signal 15)
    - immediate_kill=True:  Send SIGKILL (force, signal 9, aka kill -9)
    """
    try:
        logger.info("Terminating PID: %s name: %s", process.pid, process.name())
        if immediate_kill:
            logger.info("Sending SIGKILL (kill -9): %s %s", process.pid, process.name())
            process.kill()  # SIGKILL (9)
        else:
            logger.info("Sending SIGTERM (kill): %s %s", process.pid, process.name())
            process.terminate()  # SIGTERM (15)
    except psutil.AccessDenied:
        logger.warning("Access denied for PID %s", process.pid)
    except psutil.NoSuchProcess:
        logger.warning("PID %s no longer exists", process.pid)


def terminate_process_tree(
    pid, logger=logging.getLogger(), immediate_kill=False, timeout=10
):
    """Terminate a process and all its children.

    Kill Sequence:
    ==============
    1. Snapshot all children (recursive)
    2. Send SIGTERM to parent IMMEDIATELY (no delay)
    3. Wait up to `timeout` seconds (default 10s) for parent to exit
    4. If parent still alive after timeout: Send SIGKILL to parent (force)
    5. Send SIGTERM to all children IMMEDIATELY
    6. Wait up to `timeout` seconds for all processes to exit
    7. If any still alive after timeout: Send SIGKILL to remaining (force)

    Timeout Parameter:
    - Controls how long to WAIT AFTER SIGTERM before escalating to SIGKILL
    - NOT a delay before sending SIGTERM (SIGTERM is sent immediately)

    Summary: SIGTERM (immediate) → wait → SIGKILL if still alive
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # 1. Snapshot children before signaling parent
    children = parent.children(recursive=True)

    # 2. Terminate parent first (graceful)
    terminate_process(parent, logger, immediate_kill=immediate_kill)

    # 3. Wait for parent to exit
    try:
        parent.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        logger.warning("Parent process did not exit within timeout")
        terminate_process(parent, logger, immediate_kill=True)

    # 4. Terminate children if still alive
    for child in children:
        terminate_process(child, logger, immediate_kill=immediate_kill)

    # 5. Wait for all processes to exit
    all_procs = [parent] + children
    gone, alive = psutil.wait_procs(all_procs, timeout=timeout)

    # 6. Escalate remaining alive processes if needed
    for p in alive:
        terminate_process(p, logger, immediate_kill=True)
    psutil.wait_procs(alive, timeout=timeout)


@dataclass
class ManagedProcess:
    """Manages a subprocess with health checks and automatic cleanup.

    Parallel Execution Safety:
    -------------------------
    For pytest-xdist or parallel test execution, use terminate_all_matching_process_names=False:

        # ❌ NOT SAFE for parallel tests (kills ALL vllm processes system-wide):
        ManagedProcess(
            command=["vllm", "serve", "--port", "8000"],
            terminate_all_matching_process_names=True,  # Kills all vllm, including other tests!
            stragglers=["vllm"],
        )

        # ✅ SAFE for parallel tests (only kills what we launch):
        ManagedProcess(
            command=["vllm", "serve", "--port", "8000"],
            terminate_all_matching_process_names=False,  # Don't kill other processes
        )

    With terminate_all_matching_process_names=False and dynamic/randomized ports:
    - Each test gets unique ports, so zombies from previous runs won't conflict
    - ManagedProcess only terminates the process it launched (via self.proc.pid)
    - No risk of killing other tests' processes or developer's manual processes
    - Pytest will clean up any remaining processes when the test process exits
    """

    command: List[str]
    env: Optional[dict] = None
    health_check_ports: List[int] = field(default_factory=list)
    health_check_urls: List[Any] = field(default_factory=list)
    health_check_funcs: List[Any] = field(default_factory=list)
    delayed_start: int = 0
    timeout: int = 300
    working_dir: Optional[str] = None
    display_output: bool = False
    data_dir: Optional[str] = None
    # WARNING: terminate_all_matching_process_names=True is NOT safe for pytest-xdist
    # Kills ALL processes system-wide with matching name (other tests, dev processes, etc.)
    # Use False for parallel tests with dynamic ports to only kill launched process.
    # TODO: Change default to False once all tests use dynamic ports
    terminate_all_matching_process_names: bool = True
    stragglers: List[str] = field(default_factory=list)
    straggler_commands: List[str] = field(default_factory=list)
    log_dir: str = os.getcwd()
    display_name: Optional[str] = None

    # Ensure attributes exist even if startup fails early
    proc: Optional[subprocess.Popen] = None
    _pgid: Optional[int] = None

    _logger = logging.getLogger()
    _command_name = None
    _log_path = None
    _tee_proc = None
    _sed_proc = None

    @property
    def log_path(self):
        """Return the absolute path to the process log file if available."""
        return self._log_path

    def read_logs(self) -> str:
        """Read and return the entire contents of the process log file.

        Returns an empty string if the log file is not yet available.
        """
        try:
            if self._log_path and os.path.exists(self._log_path):
                with open(self._log_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            self._logger.warning("Could not read log file %s: %s", self._log_path, e)
        return ""

    def __enter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            # self._command_name = self.command[0]
            if self.display_name:
                self._command_name = self.display_name
            else:
                self._command_name = self.command[0]

            # Keep test logs out of the git working tree: many tests pass a relative
            # `log_dir` derived from `request.node.name`, which otherwise creates a large
            # number of untracked directories under the repo root during pytest runs.
            if not os.path.isabs(self.log_dir):
                log_root = os.environ.get(
                    "DYN_TEST_OUTPUT_PATH",
                    os.path.join(tempfile.gettempdir(), "dynamo_tests"),
                )
                self.log_dir = os.path.join(log_root, self.log_dir)

            os.makedirs(self.log_dir, exist_ok=True)
            log_name = f"{self._command_name}.log.txt"
            self._log_path = os.path.join(self.log_dir, log_name)

            if self.data_dir:
                self._remove_directory(self.data_dir)

            self._terminate_all_matching_process_names()  # Name-based cleanup (NOT xdist safe)
            self._start_process()
            time.sleep(self.delayed_start)
            elapsed = self._check_ports(self.timeout)
            self._check_urls(self.timeout - elapsed)
            self._check_funcs(self.timeout - elapsed)

            return self

        except Exception:
            try:
                self.__exit__(None, None, None)
            except Exception as cleanup_err:
                self._logger.warning(
                    "Error during cleanup in __enter__: %s", cleanup_err
                )
            raise

    def _cleanup_stragglers(self):
        """Clean up straggler processes - called during exit and signal handling"""
        try:
            if self.stragglers or self.straggler_commands:
                self._logger.info(
                    "Checking for straggler processes: stragglers=%s, straggler_commands=%s",
                    self.stragglers,
                    self.straggler_commands,
                )

            for ps_process in psutil.process_iter(["name", "cmdline"]):
                try:
                    process_name = ps_process.name()
                    if process_name in self.stragglers:
                        self._logger.info(
                            "Terminating Straggler %s %s", process_name, ps_process.pid
                        )
                        terminate_process_tree(ps_process.pid, self._logger)

                    # Check command line arguments
                    cmdline = ps_process.cmdline()
                    cmdline_str = " ".join(cmdline) if cmdline else ""
                    for straggler_cmd in self.straggler_commands:
                        if straggler_cmd in cmdline_str:
                            self._logger.info(
                                "Terminating Straggler Cmdline %s %s %s",
                                process_name,
                                ps_process.pid,
                                straggler_cmd,
                            )
                            terminate_process_tree(ps_process.pid, self._logger)
                            break  # Avoid terminating the same process multiple times
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    # Process may have terminated or become inaccessible during iteration
                    pass
                except Exception as e:
                    # Catch any other unexpected errors to ensure cleanup continues
                    self._logger.warning("Error checking process: %s", e)
        except Exception as e:
            # Ensure that any error in straggler cleanup doesn't prevent other cleanup
            self._logger.error("Error during straggler cleanup: %s", e)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup: Terminate launched processes.

        Termination Strategy (Graceful → Escalate to Force):
        =====================================================
        1. Send SIGTERM to process group immediately (no delay before SIGTERM)
        2. Wait up to 2s (poll every 0.1s) for processes to exit
        3. If still alive after 2s: Send SIGKILL (force kill)
        4. Terminate individual processes (self.proc, tee, sed):
           - Send SIGTERM immediately (no delay)
           - Wait up to 10s for exit
           - If still alive after 10s: Send SIGKILL (force kill)
        5. Clean up straggler processes (if configured)

        Signal Details:
        - SIGTERM (15): Graceful - allows cleanup handlers to run
        - SIGKILL (9): Force kill - immediate, cannot be caught

        Timeout Parameter:
        - Controls how long to WAIT AFTER SIGTERM before escalating to SIGKILL
        - NOT a delay before sending SIGTERM (SIGTERM is sent immediately)

        This ALWAYS runs regardless of terminate_all_matching_process_names setting.
        """
        try:
            self._terminate_process_group()

            process_list = [self.proc, self._tee_proc, self._sed_proc]
            for process in process_list:
                if process:
                    try:
                        if process.stdout:
                            process.stdout.close()
                        if process.stdin:
                            process.stdin.close()
                        terminate_process_tree(process.pid, self._logger)
                        process.wait()
                    except Exception as e:
                        self._logger.warning("Error terminating process: %s", e)
            if self.data_dir:
                self._remove_directory(self.data_dir)
        finally:
            # Always run straggler cleanup, even if interrupted
            self._cleanup_stragglers()

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
            # Capture the child's process group id for robust cleanup even if parent shell exits
            try:
                self._pgid = os.getpgid(self.proc.pid)
            except Exception as e:
                self._logger.warning("Could not get process group id: %s", e)
                self._pgid = None
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
                # Capture the child's process group id for robust cleanup even if parent shell exits
                try:
                    self._pgid = os.getpgid(self.proc.pid)
                except Exception as e:
                    self._logger.warning("Could not get process group id: %s", e)
                    self._pgid = None

                self._sed_proc = subprocess.Popen(
                    ["sed", "-u", f"s/^/[{self._command_name.upper()}] /"],
                    stdin=self.proc.stdout,
                    stdout=f,
                )
            self._tee_proc = None

    def _terminate_process_group(self, timeout: float = 2.0):
        """Terminate the entire process group/session started for the child.

        Kill Sequence:
        ==============
        1. Send SIGTERM to entire process group IMMEDIATELY (no delay)
        2. Wait up to `timeout` seconds (default 2s), polling every 0.1s
        3. If still alive after timeout: Send SIGKILL (force kill, immediate)

        Timeout Parameter:
        - Controls how long to WAIT AFTER SIGTERM before escalating to SIGKILL
        - NOT a delay before sending SIGTERM (SIGTERM is sent immediately)

        Process groups catch cases where the launcher shell exits and its
        children are reparented, leaving no parent PID to traverse, but they
        remain in the same process group.
        """
        if self._pgid is None:
            return
        try:
            self._logger.info("Terminating process group: %s", self._pgid)
            os.killpg(self._pgid, signal.SIGTERM)  # Step 1: Graceful SIGTERM
        except ProcessLookupError:
            return
        except Exception as e:
            self._logger.warning(
                "Error sending SIGTERM to process group %s: %s", self._pgid, e
            )
            return

        # Step 2: Poll for process exit instead of fixed sleep to minimize teardown time
        poll_interval = 0.1
        elapsed = 0.0
        while elapsed < timeout:
            try:
                # Check if any process in the group is still alive
                os.killpg(self._pgid, 0)  # Signal 0 = check existence (no kill)
            except ProcessLookupError:
                # Process group no longer exists - done
                return
            except Exception:
                # Other errors (e.g., permission) - assume done
                return
            time.sleep(poll_interval)
            elapsed += poll_interval

        # Step 3: Force kill if anything remains after timeout
        try:
            os.killpg(self._pgid, signal.SIGKILL)  # SIGKILL (kill -9) - immediate
        except ProcessLookupError:
            pass
        except Exception as e:
            self._logger.warning(
                "Error sending SIGKILL to process group %s: %s", self._pgid, e
            )

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

    def _check_url(self, url, timeout=30, sleep=1, log_interval=20):
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
            "TIMEOUT: Check URL: %s failed after %.1fs (attempts=%d, timeout=%.1fs)",
            url,
            elapsed,
            attempt,
            timeout,
        )
        raise RuntimeError(
            "TIMEOUT: Check URL: %s failed after %.1fs (timeout=%.1fs)"
            % (url, elapsed, timeout)
        )

    def _check_funcs(self, timeout):
        elapsed = 0.0
        for func in self.health_check_funcs:
            elapsed += self._check_func(func, timeout - elapsed)
        return elapsed

    def _check_func(self, func, timeout=30, sleep=1, log_interval=20):
        start_time = time.time()
        func_name = getattr(func, "__name__", str(func))
        self._logger.info("Running custom health check '%s'", func_name)
        elapsed = 0.0
        attempt = 0
        last_log_time = 0.0

        while elapsed < timeout:
            self._check_process_alive("while waiting for health check")

            attempt += 1
            check_failed = False
            failure_reason = None

            try:
                # Prefer functions that accept remaining timeout; fall back to no-arg call
                try:
                    result = func(timeout - elapsed)
                except TypeError:
                    result = func()

                if bool(result):
                    self._logger.info(
                        "SUCCESS: Custom health check '%s' passed (attempt=%d, elapsed=%.1fs)",
                        func_name,
                        attempt,
                        elapsed,
                    )
                    return time.time() - start_time
                else:
                    check_failed = True
                    failure_reason = "returned False"
            except Exception as e:
                check_failed = True
                failure_reason = f"exception: {e}"

            if check_failed and elapsed - last_log_time >= log_interval:
                self._logger.info(
                    "Still waiting on custom health check '%s' (%s) (attempt=%d, elapsed=%.1fs)",
                    func_name,
                    failure_reason,
                    attempt,
                    elapsed,
                )
                last_log_time = elapsed

            time.sleep(sleep)
            elapsed = time.time() - start_time

        self._logger.error(
            "FAILED: Custom health check '%s' (attempts=%d, elapsed=%.1fs)",
            func_name,
            attempt,
            elapsed,
        )
        raise RuntimeError("FAILED: Custom health check")

    def _terminate_all_matching_process_names(self):
        # WARNING: This method is NOT pytest-xdist safe! Kills ALL matching processes system-wide.
        # DANGER: When terminate_all_matching_process_names=True, this kills:
        #   - Processes with matching name (self._command_name)
        #   - Processes in stragglers list
        #   - Processes with matching command substring (straggler_commands)
        # It does NOT check:
        #   - Port numbers (kills processes on ALL ports, not just ours)
        #   - Process ownership (kills other tests' processes)
        #   - Working directory (kills unrelated processes)
        # RECOMMENDED: Use terminate_all_matching_process_names=False for parallel tests.
        # ManagedProcess already only kills what it launched (self.proc.pid) on exit.
        if self.terminate_all_matching_process_names:
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

    def is_running(self) -> bool:
        """Check if the process is still running"""
        return (
            hasattr(self, "proc") and self.proc is not None and self.proc.poll() is None
        )

    def get_pid(self) -> int | None:
        """Get the PID of the managed process."""
        return self.proc.pid if self.proc else None

    def subprocesses(self) -> list[psutil.Process]:
        """Find child processes of the current process."""
        if (
            not hasattr(self, "proc")
            or self.proc is None
            or self.proc.poll() is not None
        ):
            return []

        try:
            parent = psutil.Process(self.proc.pid)
            return parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return []


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend"""

    _logger = logging.getLogger()

    def __init__(
        self,
        request: Any,
        *,
        frontend_port: Optional[int] = None,
        router_mode: str = "round-robin",
        migration_limit: int = 0,
        extra_args: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
        # Default to false so pytest-xdist workers don't kill each other's frontends.
        terminate_all_matching_process_names: bool = False,
        display_name: Optional[str] = None,
    ):
        # TODO: Refactor remaining duplicate "DynamoFrontendProcess" helpers in tests to
        # use this shared implementation (and delete the copies):
        # - tests/fault_tolerance/cancellation/utils.py
        # - tests/fault_tolerance/migration/utils.py
        # - tests/fault_tolerance/etcd_ha/utils.py
        # - tests/fault_tolerance/test_vllm_health_check.py
        self._allocated_http_port: Optional[int] = None
        if frontend_port == 0:
            # Treat `0` as "allocate a random free port" for xdist-safe tests.
            # We allocate within the i16-safe range required by the Rust side.
            frontend_port = allocate_port(DefaultPort.FRONTEND.value)
            self._allocated_http_port = frontend_port

        # If frontend_port is unset, dynamo.frontend defaults to DefaultPort.FRONTEND.
        self.http_port = (
            DefaultPort.FRONTEND.value if frontend_port is None else int(frontend_port)
        )

        command = ["python", "-m", "dynamo.frontend", "--router-mode", router_mode]

        # dynamo.frontend defaults to 8000 when neither env nor flag is provided.
        if frontend_port is not None:
            command.extend(["--http-port", str(frontend_port)])
        # Migration limit is configured at the frontend level
        command.extend(["--migration-limit", str(migration_limit)])
        if extra_args:
            command.extend(extra_args)

        # Unset DYN_SYSTEM_PORT - frontend doesn't use system metrics server
        env = os.environ.copy()
        env.pop("DYN_SYSTEM_PORT", None)
        if extra_env:
            env.update(extra_env)

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
            self._logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            # Directory doesn't exist, which is fine
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_all_matching_process_names=terminate_all_matching_process_names,
            log_dir=log_dir,
            display_name=display_name,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self._allocated_http_port is not None:
                deallocate_port(self._allocated_http_port)
                self._allocated_http_port = None

    @property
    def frontend_port(self) -> int:
        """Back-compat alias for older tests that expect `frontend.frontend_port`."""
        return self.http_port


def main():
    # NOTE: This entrypoint is for manual testing/debugging of `ManagedProcess` only.
    # It is not used by the pytest suite.

    # Example: Safe for parallel tests with terminate_all_matching_process_names=False
    with ManagedProcess(
        command=["python", "-m", "dynamo.frontend", "--port", "8000"],
        display_output=True,
        terminate_all_matching_process_names=False,  # ✅ Safe - only kills what we launch
        health_check_ports=[8000],
        health_check_urls=["http://localhost:8000/v1/models"],
        timeout=10,
    ):
        time.sleep(60)
        pass


if __name__ == "__main__":
    main()
