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

import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import ClassVar, List, Optional


@dataclass
class ManagedProcess:
    command: List[str]
    env: Optional[dict] = None
    health_check: Optional[List[int | str]] = None
    timeout: int = 300
    cwd: Optional[str] = None
    output: bool = False
    data_dir: Optional[str] = None

    ensure_not_running: bool = False
    _processes: ClassVar[list[object]] = []

    def __enter__(self):
        if self.data_dir:
            self._remove_directory(self.data_dir)
        # if self.ensure_not_running:
        #     for proc in _processes:
        #         if proc.command == self.command:
        #             raise RuntimeError("Process is already running")

        print(f"Running command: {' '.join(self.command)} in {self.cwd or os.getcwd()}")

        stdin = subprocess.DEVNULL
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        if self.output:
            stdin = None
            stdout = None
            stderr = None

        self.proc = subprocess.Popen(
            self.command,
            env=self.env or os.environ.copy(),
            cwd=self.cwd,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        start_time = time.time()

        if self.check_ports:
            print(f"Waiting for ports: {self.check_ports}")
            while time.time() - start_time < self.timeout:
                if all(is_port_open(p) for p in self.check_ports):
                    print(f"All ports {self.check_ports} are ready")
                    break
                time.sleep(0.1)
            else:
                self.proc.terminate()
                raise TimeoutError(f"Ports {self.check_ports} not ready in time")

        ManagedProcess._processes.append(self)
        return self.proc

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _processes
        if self.proc:
            print(f"Terminating process: {self.command[0]}")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Process did not terminate gracefully, killing it")
                self.proc.kill()
                self.proc.wait()
            ManagedProcess._processes.remove(self)
        if self.data_dir:
            self._cleanup_directory(self.data_dir)

    def _remove_directory(path: str) -> None:
        """Remove a directory."""
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Failed to remove directory {path}: {e}")

    def _is_port_open(port: int) -> bool:
        """Check if a port is open on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _check_health(url_or_port, timeout):
        pass


def main():
    with ManagedProcess(
        command=["dynamo", "run", "out=vllm", "Qwen/Qwen2.5-3B-Instruct"], output=True
    ) as mp:
        time.sleep(60)
        pass


if __name__ == "__main__":
    main()
