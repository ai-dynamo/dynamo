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


@dataclass
class ManagedProcess:
    command: List[str]
    env: Optional[dict] = None
    health_check_ports: List[int] = field(default_factory=list)
    health_check_urls: List[Any] = field(default_factory=list)
    timeout: int = 300
    working_dir: Optional[str] = None
    display_output: bool = False
    data_dir: Optional[str] = None
    terminate_existing: bool = True
    stragglers: List[str] = field(default_factory=list)
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
            self.timeout -= self._check_ports(self.timeout)
            self.timeout -= self._check_urls(self.timeout)

            return self.proc

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
                self._terminate_process_tree(process.pid)
                process.wait()
        if self.data_dir:
            self._remove_directory(self.data_dir)

        for ps_process in psutil.process_iter(["name", "cmdline"]):
            if ps_process.name() in self.stragglers:
                self._terminate_process_tree(ps_process.pid)

    def _start_process(self):
        assert self._command_name
        assert self._log_path

        self._logger.info(
            f"Running command: {' '.join(self.command)} in {self.working_dir or os.getcwd()}"
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
            self._logger.warning(f"Warning: Failed to remove directory {path}: {e}")

    def _check_ports(self, timeout):
        time_taken = 0
        for port in self.health_check_ports:
            time_taken += self._check_port(port, timeout)
        return time_taken

    def _check_port(self, port, timeout=30, sleep=0.1):
        """Check if a port is open on localhost."""
        start_time = time.time()
        self._logger.info(f"Checking Port: {port}")
        while time.time() - start_time < timeout:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("localhost", port)) == 0:
                    self._logger.info(f"SUCCESS: Check Port:{port}")
                    return time.time() - start_time
            time.sleep(sleep)
        self._logger.error(f"FAILED: Check Port: {port}")
        raise RuntimeError(f"FAILED: Check Port: {port}")

    def _check_urls(self, timeout):
        time_taken = 0
        for url in self.health_check_urls:
            time_taken += self._check_url(url, timeout)
        return time_taken

    def _check_url(self, url, timeout=30, sleep=0.1):
        if isinstance(url, tuple):
            response_check = url[1]
            url = url[0]
        else:
            response_check = None
        start_time = time.time()
        self._logger.info(f"Checking URL {url}")
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    if response_check is None or response_check(response):
                        self._logger.info(f"SUCCESS: Check URL:{url}")
                        return time.time() - start_time
            except requests.RequestException as e:
                self._logger.warn(f"URL check failed: {e}")
            time.sleep(sleep)

        self._logger.error(f"FAILED: Check URL: {url}")
        raise RuntimeError(f"FAILED: Check URL: {url}")

    def _terminate_existing(self):
        if self.terminate_existing:
            self._logger.info(f"Terminating Existing {self._command_name}")
            for proc in psutil.process_iter(["name", "cmdline"]):
                if proc.name() == self._command_name or proc.name() in self.stragglers:
                    self._terminate_process_tree(proc.pid)

    def _terminate_process(self, process):
        try:
            self._logger.info(f"Terminating {process}")
            process.terminate()
        except psutil.AccessDenied:
            self._logger.warning(f"Access denied for PID {process.pid}")
        except psutil.NoSuchProcess:
            self._logger.warning(f"PID {process.pid} no longer exists")
        except psutil.TimeoutExpired:
            self._logger.warning(
                f"PID {process.pid} did not terminate in timeout, killing"
            )
            process.kill()

    def _terminate_process_tree(self, pid):
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            self._terminate_process(child)
        self._terminate_process(parent)


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
